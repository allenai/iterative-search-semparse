from typing import Dict, Optional, List

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util


@Model.register("latent_alignment_shared_char")
class LatentAlignmentSharedChar(Model):
    def __init__(self, vocab: Vocabulary,
                 utterance_embedder: TextFieldEmbedder,
                 logical_form_embedder: TextFieldEmbedder,
                 char_embedder: TextFieldEmbedder,
                 utterance_encoder: Seq2SeqEncoder,
                 normalize_by_len: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.utterance_embedder = utterance_embedder
        self.logical_form_embedder = logical_form_embedder
        self.char_embedder = char_embedder
        self.utterance_encoder = utterance_encoder

        self.normalize_by_len = normalize_by_len
        self.translation_layer = Linear(
                self.char_embedder.get_output_dim() + self.logical_form_embedder.get_output_dim(),
                self.utterance_encoder.get_output_dim())

        self.mean_ranks = 0.0
        self.accuracy = 0.0
        self.hits3 = 0.0
        self.hits5 = 0.0
        self.hits10 = 0.0
        self.batches = 0.0
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                logical_forms: Dict[str, torch.LongTensor],
                utterance_string: List[str],
                logical_form_strings: List[List[str]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------

        Returns
        -------

        """
        # (batch_size, num_utterance_tokens, utterance_embedding_dim)
        # import pdb; pdb.set_trace();
        embedded_utterance_tokens = self.utterance_embedder({'tokens': utterance['tokens']})
        embedded_utterance_char_tokens = self.char_embedder({'token_characters': utterance['token_characters']})

        embedded_logical_forms = self.logical_form_embedder({'lf_tokens': logical_forms['lf_tokens']},
                                                            num_wrapping_dims=1)
        embedded_logical_forms_char_tokens = self.char_embedder(
                {'token_characters': logical_forms['lf_token_characters']}, num_wrapping_dims=1)

        embedded_utterance = torch.cat([embedded_utterance_tokens, embedded_utterance_char_tokens], dim=-1)
        embedded_logical_forms = torch.cat([embedded_logical_forms, embedded_logical_forms_char_tokens], dim=-1)

        utterance_mask = util.get_text_field_mask(utterance)
        encoded_utterance = self.utterance_encoder(embedded_utterance, utterance_mask)
        # Because we're just summing everything in the end, we can do the sum upfront to save some
        # time.
        # (batch_size, utterance_embedding_dim)
        encoded_utterance = encoded_utterance.sum(dim=1)
        # (batch_size, num_logical_forms, num_lf_tokens)
        logical_form_token_mask = util.get_text_field_mask(logical_forms, num_wrapping_dims=1)
        logical_form_lens = 1 + torch.sum(logical_form_token_mask, dim=-1)  # to avoid division by zero, add 1
        # (batch_size, num_logical_forms)
        logical_form_mask = logical_form_token_mask.sum(dim=-1).clamp(max=1)

        # (batch_size, num_logical_forms, lf_embedding_dim)

        encoded_logical_forms = embedded_logical_forms.sum(dim=2)

        # (batch_size, num_logical_forms, utterance_embedding_dim)
        predicted_embeddings = self.translation_layer(encoded_logical_forms)

        # (batch_size, num_logical_forms)
        similarities = torch.nn.functional.cosine_similarity(predicted_embeddings,
                                                             encoded_utterance.unsqueeze(1),
                                                             dim=2)
        if self.normalize_by_len:
            similarities = similarities / logical_form_lens.type(torch.cuda.FloatTensor)
        # Make sure masked logical forms aren't included in the max.
        similarities = util.replace_masked_values(similarities, logical_form_mask, -1e7)

        ranks = (similarities[:, 0].unsqueeze(1) < similarities)
        curr_ranks = ranks.sum(dim=-1)  # (32,) ranks
        hits = [(curr_ranks < k).sum().cpu().data.numpy() for k in [3, 5, 10]]
        self.hits3 += hits[0]
        self.hits5 += hits[1]
        self.hits10 += hits[2]
        self.mean_ranks += curr_ranks.sum(dim=0).cpu().data.numpy()
        self.batches += ranks.shape[0]

        # replace max with logsumexp
        # loss = -1.0*torch.logsumexp(similarities,dim=-1).sum()
        max_similarity, most_similar = similarities.max(dim=-1)
        loss = (1 - max_similarity).sum()

        self.accuracy += (most_similar == 0).sum().cpu().data.numpy()

        most_similar_strings = []
        for instance_most_similar, instance_logical_forms in zip(most_similar.tolist(), logical_form_strings):
            most_similar_strings.append(instance_logical_forms[instance_most_similar])
        return {"loss": loss, "most_similar": most_similar_strings, "utterance": utterance_string}

    @overrides
    def get_metrics(self, reset: bool = False):
        if self.batches == 0:
            return {'mean_rank': -1, 'accuracy': -1, 'hits3': -1, 'hits5': -1, 'hits10': -1}
        mean_rank = self.mean_ranks / self.batches
        mean_accuracy = self.accuracy / self.batches
        mean_hits3 = self.hits3 / self.batches
        mean_hits5 = self.hits5 / self.batches
        mean_hits10 = self.hits10 / self.batches

        if reset:
            self.mean_ranks = 0.0
            self.accuracy = 0.0
            self.hits3 = 0.0
            self.hits5 = 0.0
            self.hits10 = 0.0
            self.batches = 0.0
        return {'mean_rank': mean_rank, 'mean_accuracy': mean_accuracy, 'hits3': mean_hits3, 'hits5': mean_hits5,
                'hits10': mean_hits10}
