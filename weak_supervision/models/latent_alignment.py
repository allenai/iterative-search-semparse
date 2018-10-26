from typing import Dict, Optional, List

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util


@Model.register("latent_alignment")
class LatentAlignment(Model):
    def __init__(self, vocab: Vocabulary,
                 utterance_embedder: TextFieldEmbedder,
                 logical_form_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.utterance_embedder = utterance_embedder
        self.logical_form_embedder = logical_form_embedder
        self.translation_layer = Linear(self.logical_form_embedder.get_output_dim(),
                                        self.utterance_embedder.get_output_dim())

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
        embedded_utterance = self.utterance_embedder(utterance)

        # (batch_size, num_logical_forms, num_lf_tokens, lf_embedding_dim)
        embedded_logical_forms = self.logical_form_embedder(logical_forms, num_wrapping_dims=1)

        # (batch_size, num_logical_forms, num_lf_tokens)
        logical_form_token_mask = util.get_text_field_mask(logical_forms, num_wrapping_dims=1)
        # (batch_size, num_logical_forms)
        logical_form_mask = logical_form_token_mask.sum(dim=-1).clamp(max=1)

        # Because we're just summing everything in the end, we can do the sum upfront to save some
        # time.
        # (batch_size, utterance_embedding_dim)
        encoded_utterance = embedded_utterance.sum(dim=1)

        # (batch_size, num_logical_forms, lf_embedding_dim)
        encoded_logical_forms = embedded_logical_forms.sum(dim=2)

        # (batch_size, num_logical_forms, utterance_embedding_dim)
        predicted_embeddings = self.translation_layer(encoded_logical_forms)

        # (batch_size, num_logical_forms)
        similarities = torch.nn.functional.cosine_similarity(predicted_embeddings,
                                                             encoded_utterance.unsqueeze(1),
                                                             dim=2)

        # Make sure masked logical forms aren't included in the max.
        similarities = util.replace_masked_values(similarities, logical_form_mask, -1e7)

        max_similarity, most_similar = similarities.max(dim=-1)
        loss = (1 - max_similarity).sum()
        most_similar_strings = []
        for instance_most_similar, instance_logical_forms in zip(most_similar.tolist(), logical_form_strings):
            most_similar_strings.append(instance_logical_forms[instance_most_similar])
        return {"loss": loss, "most_similar": most_similar_strings, "utterance": utterance_string}
