import logging
import os
from functools import partial
from typing import Dict, List, Tuple, Set, Any

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.models.archival import load_archive, Archive
from allennlp.models.model import Model
from allennlp.models.semantic_parsing.wikitables.wikitables_semantic_parser import WikiTablesSemanticParser
from allennlp.modules import Attention, FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.semparse.type_declarations import wikitables_lambda_dcs as types
from allennlp.semparse.worlds import WikiTablesWorld
from allennlp.state_machines.states import GrammarBasedState
from weak_supervision.state_machines.trainers import DynamicMaximumMarginalLikelihood 
from allennlp.state_machines.trainers.decoder_trainer import DecoderTrainer

from allennlp.state_machines.transition_functions import LinkingTransitionFunction
from allennlp.training.metrics import Average

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("wikitables_dmml_parser")
class WikiTablesDMMLSemanticParser(WikiTablesSemanticParser):
    """
    A ``WikiTablesDMMLSemanticParser`` is a :class:`WikiTablesSemanticParser` that learns to search
    for logical forms that yield the correct denotations.

    Parameters
    ----------
    vocab : ``Vocabulary``
    question_embedder : ``TextFieldEmbedder``
        Embedder for questions. Passed to super class.
    action_embedding_dim : ``int``
        Dimension to use for action embeddings. Passed to super class.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input question. Passed to super class.
    entity_encoder : ``Seq2VecEncoder``
        The encoder to used for averaging the words of an entity. Passed to super class.
    attention : ``Attention``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the transition function.
    decoder_beam_size : ``int``
        Beam size to be used by the ExpectedRiskMinimization algorithm.
    decoder_num_finished_states : ``int``
        Number of finished states for which costs will be computed by the ExpectedRiskMinimization
        algorithm.
    max_decoding_steps : ``int``
        Maximum number of steps the decoder should take before giving up. Used both during training
        and evaluation. Passed to super class.
    add_action_bias : ``bool``, optional (default=True)
        If ``True``, we will learn a bias weight for each action that gets used when predicting
        that action, in addition to its embedding.  Passed to super class.
    normalize_beam_score_by_length : ``bool``, optional (default=False)
        Should we normalize the log-probabilities by length before renormalizing the beam? This was
        shown to work better for NML by Edunov et al., but that many not be the case for semantic
        parsing.
    use_neighbor_similarity_for_linking : ``bool``, optional (default=False)
        If ``True``, we will compute a max similarity between a question token and the `neighbors`
        of an entity as a component of the linking scores.  This is meant to capture the same kind
        of information as the ``related_column`` feature. Passed to super class.
    dropout : ``float``, optional (default=0)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer). Passed to super class.
    num_linking_features : ``int``, optional (default=10)
        We need to construct a parameter vector for the linking features, so we need to know how
        many there are.  The default of 10 here matches the default in the ``KnowledgeGraphField``,
        which is to use all ten defined features. If this is 0, another term will be added to the
        linking score. This term contains the maximum similarity value from the entity's neighbors
        and the question. Passed to super class.
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this. Passed to super
        class.
    tables_directory : ``str``, optional (default=/wikitables/)
        The directory to find tables when evaluating logical forms.  We rely on a call to SEMPRE to
        evaluate logical forms, and SEMPRE needs to read the table from disk itself.  This tells
        SEMPRE where to find the tables. Passed to super class.
    mml_model_file : ``str``, optional (default=None)
        If you want to initialize this model using weights from another model trained using MML,
        pass the path to the ``model.tar.gz`` file of that model here.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 question_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 entity_encoder: Seq2VecEncoder,
                 attention: Attention,
                 decoder_beam_size: int,
                 decoder_num_finished_states: int,
                 max_decoding_steps: int,
                 mixture_feedforward: FeedForward = None,
                 add_action_bias: bool = True,
                 normalize_beam_score_by_length: bool = False,
                 use_neighbor_similarity_for_linking: bool = False,
                 use_sampling: bool = False,
                 dropout: float = 0.0,
                 num_linking_features: int = 10,
                 rule_namespace: str = 'rule_labels',
                 tables_directory: str = '/wikitables/',
                 mml_model_file: str = None) -> None:
        use_similarity = use_neighbor_similarity_for_linking
        super().__init__(vocab=vocab,
                         question_embedder=question_embedder,
                         action_embedding_dim=action_embedding_dim,
                         encoder=encoder,
                         entity_encoder=entity_encoder,
                         max_decoding_steps=max_decoding_steps,
                         add_action_bias=add_action_bias,
                         use_neighbor_similarity_for_linking=use_similarity,
                         dropout=dropout,
                         num_linking_features=num_linking_features,
                         rule_namespace=rule_namespace,
                         tables_directory=tables_directory)
        # Not sure why mypy needs a type annotation for this!


        self._decoder_trainer: DynamicMaximumMarginalLikelihood = \
                 DynamicMaximumMarginalLikelihood(beam_size=decoder_beam_size,
                                          normalize_by_length=normalize_beam_score_by_length,
                                          max_decoding_steps=self._max_decoding_steps,
                                          max_num_finished_states=decoder_num_finished_states,
                                          sample_states=use_sampling)
        unlinked_terminals_global_indices = []
        self._decoder_step = LinkingTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
                                                       action_embedding_dim=action_embedding_dim,
                                                       input_attention=attention,
                                                       num_start_types=self._num_start_types,
                                                       predict_start_type_separately=True,
                                                       add_action_bias=self._add_action_bias,
                                                       mixture_feedforward=mixture_feedforward,
                                                       dropout=dropout)
        self.noop = 0.0
        self.search_hits = 0.0
        if mml_model_file is not None:
            if os.path.isfile(mml_model_file):
                archive = load_archive(mml_model_file)
                self._initialize_weights_from_archive(archive)
            else:
                # A model file is passed, but it does not exist. This is expected to happen when
                # you're using a trained DMML model to decode. But it may also happen if the path to
                # the file is really just incorrect. So throwing a warning.
                logger.warning("MML model file for initializing weights is passed, but does not exist."
                               " This is fine if you're just decoding.")

    def _initialize_weights_from_archive(self, archive: Archive) -> None:
        logger.info("Initializing weights from MML model.")
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        question_embedder_weight = "_question_embedder.token_embedder_tokens.weight"
        if question_embedder_weight not in archived_parameters or \
           question_embedder_weight not in model_parameters:
            raise RuntimeError("When initializing model weights from an MML model, we need "
                               "the question embedder to be a TokenEmbedder using namespace called "
                               "tokens.")
        for name, weights in archived_parameters.items():
            if name in model_parameters:
                if name == question_embedder_weight:
                    # The shapes of embedding weights will most likely differ between the two models
                    # because the vocabularies will most likely be different. We will get a mapping
                    # of indices from this model's token indices to the archived model's and copy
                    # the tensor accordingly.
                    vocab_index_mapping = self._get_vocab_index_mapping(archive.model.vocab)
                    archived_embedding_weights = weights.data
                    new_weights = model_parameters[name].data.clone()
                    for index, archived_index in vocab_index_mapping:
                        new_weights[index] = archived_embedding_weights[archived_index]
                    logger.info("Copied embeddings of %d out of %d tokens",
                                len(vocab_index_mapping), new_weights.size()[0])
                else:
                    new_weights = weights.data
                logger.info("Copying parameter %s", name)
                model_parameters[name].data.copy_(new_weights)

    def _get_vocab_index_mapping(self, archived_vocab: Vocabulary) -> List[Tuple[int, int]]:
        vocab_index_mapping: List[Tuple[int, int]] = []
        for index in range(self.vocab.get_vocab_size(namespace='tokens')):
            token = self.vocab.get_token_from_index(index=index, namespace='tokens')
            archived_token_index = archived_vocab.get_token_index(token, namespace='tokens')
            # Checking if we got the UNK token index, because we don't want all new token
            # representations initialized to UNK token's representation. We do that by checking if
            # the two tokens are the same. They will not be if the token at the archived index is
            # UNK.
            if archived_vocab.get_token_from_index(archived_token_index, namespace="tokens") == token:
                vocab_index_mapping.append((index, archived_token_index))
        return vocab_index_mapping

    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                table: Dict[str, torch.LongTensor],
                world: List[WikiTablesWorld],
                actions: List[List[ProductionRuleArray]],
                example_lisp_string: List[str],
                target_action_sequences: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the question ``TextField``. This will
           be passed through a ``TextFieldEmbedder`` and then through an encoder.
        table : ``Dict[str, torch.LongTensor]``
            The output of ``KnowledgeGraphField.as_array()`` applied on the table
            ``KnowledgeGraphField``.  This output is similar to a ``TextField`` output, where each
            entity in the table is treated as a "token", and we will use a ``TextFieldEmbedder`` to
            get embeddings for each entity.
        world : ``List[WikiTablesWorld]``
            We use a ``MetadataField`` to get the ``World`` for each input instance.  Because of
            how ``MetadataField`` works, this gets passed to us as a ``List[WikiTablesWorld]``,
        actions : ``List[List[ProductionRuleArray]]``
            A list of all possible actions for each ``World`` in the batch, indexed into a
            ``ProductionRuleArray`` using a ``ProductionRuleField``.  We will embed all of these
            and use the embeddings to determine which action to take at each timestep in the
            decoder.
        example_lisp_string : ``List[str]``
            The example (lisp-formatted) string corresponding to the given input.  This comes
            directly from the ``.examples`` file provided with the dataset.  We pass this to SEMPRE
            when evaluating denotation accuracy; it is otherwise unused.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenized question within a 'question_tokens' key.
        """

        outputs: Dict[str, Any] = {}
        rnn_state, grammar_state = self._get_initial_rnn_and_grammar_state(question,
                                                                           table,
                                                                           world,
                                                                           actions,
                                                                           outputs)

        batch_size = len(rnn_state)
        initial_score = rnn_state[0].hidden_state.new_zeros(batch_size)
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),  # type: ignore
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=rnn_state,
                                          grammar_state=grammar_state,
                                          possible_actions=actions,
                                          extras=example_lisp_string,
                                          debug_info=None)

        if not self.training:
            initial_state.debug_info = [[] for _ in range(batch_size)]

        outputs = self._decoder_trainer.decode(initial_state,  # type: ignore
                                               self._decoder_step, partial(self._get_state_cost, world) )

        best_final_states = outputs['best_final_states']
 
        self.search_hits += outputs["search_hits"]
        if outputs["noop"]:
            self.noop += 1.0
        if not self.training:
            self._compute_validation_outputs(actions,
                                             best_final_states,
                                             world,
                                             example_lisp_string,
                                             metadata,
                                             outputs)
        return outputs


    def _get_state_cost(self, worlds: List[WikiTablesWorld], state: GrammarBasedState) -> torch.Tensor:
        # returns True if the logical form gives the correct answer
        if not state.is_finished():
            raise RuntimeError("_get_state_cost() is not defined for unfinished states!")
        world = worlds[state.batch_indices[0]]

        # We use this as the denotation cost if the path is incorrect.
        action_history = state.action_history[0]
        batch_index = state.batch_indices[0]
        action_strings = [state.possible_actions[batch_index][i][0] for i in action_history]
        logical_form = world.get_logical_form(action_strings)
        lisp_string = state.extras[batch_index]
        if self._executor.evaluate_logical_form(logical_form, lisp_string):
            cost = 1.0
        else:
            cost = 0.0

        return torch.Tensor(cost)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        The base class returns a dict with dpd accuracy, denotation accuracy, and logical form
        percentage metrics. We add the agenda coverage metric here.
        """
        metrics = super().get_metrics(reset)
        metrics["noop"] = self.noop
        metrics["search_hits"] = self.search_hits
        if reset:
            self.noop = 0.0
            self.search_hits = 0.0
        return metrics
