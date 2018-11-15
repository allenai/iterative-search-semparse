from typing import Callable, Dict, List, TypeVar
from collections import defaultdict

import torch

from allennlp.nn import util as nn_util
from allennlp.state_machines.states import State
from allennlp.state_machines.trainers.decoder_trainer import DecoderTrainer
from allennlp.state_machines.transition_functions import TransitionFunction

StateType = TypeVar('StateType', bound=State)  # pylint: disable=invalid-name


class DynamicMaximumMarginalLikelihood(DecoderTrainer[Callable[[StateType], torch.Tensor]]):
    """

    Parameters
    ----------
    beam_size : ``int``
    noramlize_by_length : ``bool``
        Should the log probabilities be normalized by length before renormalizing them? Edunov et
        al. do this in their work.
    max_decoding_steps : ``int``
        The maximum number of steps we should take during decoding.
    max_num_decoded_sequences : ``int``, optional (default=1)
        Maximum number of sorted decoded sequences to return. Defaults to 1.
    max_num_finished_states : ``int``, optional (default = None)
        Maximum number of finished states to keep after search. This is to finished states as
        ``beam_size`` is to unfinished ones. Costs are computed for only these number of states per
        instance. If not set, we will keep all the finished states.
    """
    def __init__(self,
                 beam_size: int,
                 normalize_by_length: bool,
                 max_decoding_steps: int,
                 max_num_decoded_sequences: int = 1,
                 sample_states: bool = False,
                 max_num_finished_states: int = None) -> None:
        self._beam_size = beam_size
        self._normalize_by_length = normalize_by_length
        self._max_decoding_steps = max_decoding_steps
        self._max_num_decoded_sequences = max_num_decoded_sequences
        self._max_num_finished_states = max_num_finished_states
        self.sample_states = sample_states

    def decode(self,
               initial_state: State,
               transition_function: TransitionFunction,
               supervision: Callable[[StateType], torch.Tensor]) -> Dict[str, torch.Tensor]:
        reward_function = supervision


        if self.sample_states:
            finished_states = []
            for _ in range(self._max_num_finished_states):
                finished_states.extend(self._sample(initial_state, transition_function))
        else:
            finished_states = self._get_finished_states(initial_state, transition_function)

        states_by_batch_index: Dict[int, List[State]] = defaultdict(list)
        for state in finished_states:
            assert len(state.batch_indices) == 1
            batch_index = state.batch_indices[0]
            states_by_batch_index[batch_index].append(state)

        loss = initial_state.score[0].new_zeros(1)
        search_hits = 0.0
        for instance_states in states_by_batch_index.values():
            scores = [state.score[0].view(-1) for state in instance_states if reward_function(state) == 1]
            if not scores:
                loss += -nn_util.logsumexp(torch.cat(scores))
                search_hits += 1
        return {'loss' : loss / len(states_by_batch_index),
                'best_final_states' : self._get_best_final_states(finished_states),
                'noop' : search_hits == 0.0,
                'search_hits' : search_hits}


    def _sample(self,
                initial_state: State,
                transition_function: TransitionFunction) -> List[StateType]:

        finished_states = []
        states = [initial_state]
        num_steps = 0
        while states and num_steps < self._max_decoding_steps:
            next_states = []
            grouped_state = states[0].combine_states(states)
            # These states already come sorted.
            for next_state in transition_function.take_step(grouped_state, sample_states=True):
                if next_state.is_finished():
                    finished_states.append(next_state)
                else:
                    next_states.append(next_state)

            states = next_states
            num_steps += 1

        return finished_states


    def _get_finished_states(self,
                             initial_state: State,
                             transition_function: TransitionFunction) -> List[StateType]:
        finished_states = []
        states = [initial_state]
        num_steps = 0
        while states and num_steps < self._max_decoding_steps:
            next_states = []
            grouped_state = states[0].combine_states(states)
            # These states already come sorted.
            for next_state in transition_function.take_step(grouped_state):
                if next_state.is_finished():
                    finished_states.append(next_state)
                else:
                    next_states.append(next_state)

            states = self._prune_beam(states=next_states,
                                      beam_size=self._beam_size,
                                      sort_states=False)
            num_steps += 1
        if self._max_num_finished_states is not None:
            finished_states = self._prune_beam(states=finished_states,
                                               beam_size=self._max_num_finished_states,
                                               sort_states=True)
        return finished_states

    # TODO(pradeep): Move this method to state_machines.util
    @staticmethod
    def _prune_beam(states: List[State],
                    beam_size: int,
                    sort_states: bool = False) -> List[State]:
        """
        This method can be used to prune the set of unfinished states on a beam or finished states
        at the end of search. In the former case, the states need not be sorted because the all come
        from the same decoding step, which does the sorting. However, if the states are finished and
        this method is called at the end of the search, they need to be sorted because they come
        from different decoding steps.
        """
        states_by_batch_index: Dict[int, List[State]] = defaultdict(list)
        for state in states:
            assert len(state.batch_indices) == 1
            batch_index = state.batch_indices[0]
            states_by_batch_index[batch_index].append(state)
        pruned_states = []
        for _, instance_states in states_by_batch_index.items():
            if sort_states:
                scores = torch.cat([state.score[0].view(-1) for state in instance_states])
                _, sorted_indices = scores.sort(-1, descending=True)
                sorted_states = [instance_states[i] for i in sorted_indices.detach().cpu().numpy()]
                instance_states = sorted_states
            for state in instance_states[:beam_size]:
                pruned_states.append(state)
        return pruned_states


    def _get_best_final_states(self, finished_states: List[StateType]) -> Dict[int, List[StateType]]:
        """
        Returns the best finished states for each batch instance based on model scores. We return
        at most ``self._max_num_decoded_sequences`` number of sequences per instance.
        """
        batch_states: Dict[int, List[StateType]] = defaultdict(list)
        for state in finished_states:
            batch_states[state.batch_indices[0]].append(state)
        best_states: Dict[int, List[StateType]] = {}
        for batch_index, states in batch_states.items():
            # The time this sort takes is pretty negligible, no particular need to optimize this
            # yet.  Maybe with a larger beam size...
            finished_to_sort = [(-state.score[0].item(), state) for state in states]
            finished_to_sort.sort(key=lambda x: x[0])
            best_states[batch_index] = [state[1] for state in finished_to_sort[:self._beam_size]]
        return best_states
