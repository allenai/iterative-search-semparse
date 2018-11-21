from collections import defaultdict
from typing import Dict, Optional, List

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.modules import MatrixAttention, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util


@Model.register("latent_ibm_model_1")
class LatentIbmModel1(Model):
    def __init__(self, vocab: Vocabulary,
                 utterance_embedder: TextFieldEmbedder,
                 logical_form_embedder: TextFieldEmbedder,
                 translation_layer: MatrixAttention,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.utterance_embedder = utterance_embedder
        self.logical_form_embedder = logical_form_embedder
        self.translation_layer = translation_layer

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
        # pylint: disable=arguments-differ,protected-access

        # For more complicated embeddings (char-CNN, etc.), we just need to create range vectors of
        # length vocab_size and apply the utterance embedder and LF embedder to them.

        # (token_vocab_size, utterance_embedding_dim)
        token_embeddings = self.utterance_embedder._token_embedders['tokens'].weight
        # (batch_size, num_utterance_tokens)
        token_indices = utterance['tokens']
        # (batch_size, num_utterance_tokens)
        utterance_mask = util.get_text_field_mask(utterance)

        batch_size, num_utterance_tokens = token_indices.size()
        token_vocab_size = token_embeddings.size(0)
        batch_range_vector = util.get_range_vector(batch_size, util.get_device_of(token_indices))

        # (lf_token_vocab_size, lf_embedding_dim)
        lf_token_embeddings = self.logical_form_embedder._token_embedders['lf_tokens'].weight
        # (batch_size, num_logical_forms, num_lf_tokens)
        lf_token_indices = logical_forms['lf_tokens']
        # (batch_size, num_logical_forms, num_lf_tokens)
        logical_form_token_mask = util.get_text_field_mask(logical_forms, num_wrapping_dims=1)
        # (batch_size, num_logical_forms)
        logical_form_mask = logical_form_token_mask.sum(dim=-1).clamp(max=1)

        _, num_logical_forms, num_lf_tokens = lf_token_indices.size()

        # (lf_token_vocab_size, token_vocab_size)
        translation_logits = self.translation_layer(lf_token_embeddings.unsqueeze(0),
                                                    token_embeddings.unsqueeze(0)).squeeze(0)
        # We could mask out the padding token before doing this normalization, but leaving it there
        # allows us to use it as a null alignment.  The UNK token kind of already does that, too,
        # though, because we're not actually ever using UNKs...
        translation_probs = torch.nn.functional.softmax(translation_logits, dim=-1)

        # At this point we've done all of our computation that will be trained.  With IBM model 1
        # we're just trying to update the translation probabilities we just computed.  Everything
        # below, up until we're actually computing a loss, is just using batch-level EM to get
        # target probability distributions that we can use as targets.

        # Now we get the probabilities that each LF token generated each utterance token, with some
        # complex indexing.
        combined_shape = (batch_size, num_logical_forms, num_lf_tokens, num_utterance_tokens)
        expanded_token_indices = token_indices.unsqueeze(1).unsqueeze(1).expand(combined_shape)
        expanded_lf_token_indices = lf_token_indices.unsqueeze(3).expand(combined_shape)

        # (batch_size, num_logical_forms, num_lf_tokens, num_utterance_tokens); p(u_token | lf_token)
        aligned_probs = translation_probs[expanded_lf_token_indices, expanded_token_indices]

        # With these token-level translation probabilities, we can compute
        # p(utterance | logical_form).  We use this to pick the best logical form, then we'll use
        # that logical form to do the E and M steps of IBM model 1.

        # (batch_size, num_logical_forms, num_utterance_tokens)
        utterance_token_probs = util.replace_masked_values(aligned_probs,
                                                           logical_form_token_mask.unsqueeze(3),
                                                           0).sum(dim=2)
        # (batch_size, num_logical_forms)
        prob_utterance_given_lf = util.replace_masked_values(utterance_token_probs,
                                                             utterance_mask.unsqueeze(1),
                                                             1).prod(dim=2)
        prob_utterance_given_lf = util.replace_masked_values(prob_utterance_given_lf, logical_form_mask, -1e7)

        # (batch_size,)
        _, best_lf_indices = prob_utterance_given_lf.max(dim=1)

        # Now, we get all of the relevant statistics for the best logical form.

        # (batch_size, num_lf_tokens, num_utterance_tokens)
        aligned_probs = aligned_probs[batch_range_vector, best_lf_indices, ...]

        # (batch_size, num_lf_tokens)
        lf_token_indices = lf_token_indices[batch_range_vector, best_lf_indices].cpu().numpy()

        # (batch_size, num_lf_tokens)
        logical_form_token_mask = logical_form_token_mask[batch_range_vector, best_lf_indices].cpu().numpy()
        utterance_mask = utterance_mask.cpu().numpy()

        ########
        # E STEP
        ########

        # (batch_size, num_utterance_tokens)
        utterance_normalization_constant = aligned_probs.sum(dim=1)
        # (batch_size, num_lf_tokens, num_utterance_tokens)
        normalized_probs = aligned_probs / utterance_normalization_constant.unsqueeze(1)

        # Easiest to compute this on the CPU, with python logic.  We'll be creating dictionaries
        # with vocab counts.
        normalized_probs = normalized_probs.detach().cpu().numpy()
        token_indices = token_indices.cpu().numpy()
        lf_token_counts = defaultdict(float)
        lf_translation_counts = defaultdict(lambda: defaultdict(float))
        for batch_index in range(batch_size):
            for lf_index in range(num_lf_tokens):
                if logical_form_token_mask[batch_index, lf_index] == 0:
                    continue
                lf_token = lf_token_indices[batch_index, lf_index]
                for token_index in range(num_utterance_tokens):
                    if utterance_mask[batch_index, token_index] == 0:
                        continue
                    utterance_token = token_indices[batch_index, token_index]
                    prob = normalized_probs[batch_index, lf_index, token_index]
                    lf_token_counts[lf_token] += prob
                    lf_translation_counts[lf_token][utterance_token] += prob

        ########
        # M STEP
        ########

        # We've computed our expected translation counts above for p(utterance_token | lf_token).
        # Here we'll compute a distribution given those counts and take a gradient step in that
        # direction.

        loss = torch.tensor(0.0).to(aligned_probs.device)  # pylint: disable=not-callable
        for lf_token, total_count in lf_token_counts.items():
            desired_token_probs = torch.zeros((token_vocab_size,)).to(aligned_probs.device)
            for utterance_token, count in lf_translation_counts[lf_token].items():
                desired_token_probs[utterance_token] = count / total_count
            loss += torch.nn.functional.kl_div(translation_probs[lf_token].log(),
                                               desired_token_probs)


        ranks = (prob_utterance_given_lf[:, 0].unsqueeze(1) < prob_utterance_given_lf)
        curr_ranks = ranks.sum(dim=-1) # (32,) ranks
        hits = [(curr_ranks < k).sum().cpu().data.numpy() for k in [3, 5, 10]]
        self.hits3 += hits[0]
        self.hits5 += hits[1]
        self.hits10 += hits[2]
        self.mean_ranks += curr_ranks.sum(dim=0).cpu().data.numpy()
        self.batches += ranks.shape[0]

        self.accuracy += (best_lf_indices == 0).sum().cpu().data.numpy()

        most_similar_strings = []
        for instance_most_similar, instance_logical_forms in zip(best_lf_indices.tolist(), logical_form_strings):
            most_similar_strings.append(instance_logical_forms[instance_most_similar])
        return {
                "loss": loss,
                "most_similar": most_similar_strings,
                "utterance": utterance_string,
                "all_similarities": prob_utterance_given_lf
                }

    @overrides
    def get_metrics(self, reset: bool = False):
        if self.batches == 0:
            return {'mean_rank' : -1, 'accuracy' : -1, 'hits3' : -1, 'hits5' : -1, 'hits10' : -1}
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
        return {
                'mean_rank' : mean_rank,
                'mean_accuracy' : mean_accuracy,
                'hits3' : mean_hits3,
                'hits5' : mean_hits5,
                'hits10' : mean_hits10
                }
