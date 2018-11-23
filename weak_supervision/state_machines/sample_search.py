from collections import defaultdict
from typing import Dict, Generic, List, Mapping, Sequence, TypeVar

from allennlp.common.registrable import FromParams
from allennlp.state_machines.states import State
from allennlp.state_machines.transition_functions import TransitionFunction

StateType = TypeVar('StateType', bound=State)  # pylint: disable=invalid-name

class SampleSearch:
    def __init__(self, max_num_decoded_sequences):
        self._max_num_decoded_sequences = max_num_decoded_sequences

    def search(self,
               num_steps: int,
               initial_state: StateType,
               transition_function: TransitionFunction):

        all_states = []
        for _ in range(self._max_num_decoded_sequences):
            curr_unfinished_states, curr_finished_states = self._sample_states(initial_state,
                                                          transition_function,
                                                          num_steps)
            all_states.extend(curr_finished_states)
            all_states.extend(curr_unfinished_states)


        states_by_batch_index: Dict[int, List[State]] = defaultdict(list)
        for state in all_states:
            assert len(state.batch_indices) == 1
            batch_index = state.batch_indices[0]
            states_by_batch_index[batch_index].append(state)

        return states_by_batch_index
        
    def _sample_states(self,
                       initial_state: State,
                       transition_function: TransitionFunction,
                       max_steps: int):

        finished_states = []
        states = [initial_state]
        num_steps = 0
        while states and num_steps < max_steps:
            next_states = []
            grouped_state = states[0].combine_states(states)
            for next_state in transition_function.take_step(grouped_state, sample_states=True):
                if next_state.is_finished():
                    finished_states.append(next_state)
                else:
                    next_states.append(next_state)

            states = next_states
            num_steps += 1

        return states, finished_states

