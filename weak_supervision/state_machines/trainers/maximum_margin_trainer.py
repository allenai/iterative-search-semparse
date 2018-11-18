from overrides import  overrides
from weak_supervision.state_machines.trainers.dynamic_maximum_marginal_likelihood import DynamicMaximumMarginalLikelihood
from typing import Callable, Dict, List, TypeVar
from collections import defaultdict

import torch

from allennlp.state_machines.states import State


StateType = TypeVar('StateType', bound=State)  # pylint: disable=invalid-name

class MaxMarginTrainer(DynamicMaximumMarginalLikelihood):

    def __init__(self,
                 beam_size: int,
                 normalize_by_length: bool,
                 max_decoding_steps: int,
                 max_num_decoded_sequences: int = 1,
                 max_num_finished_states: int = None) -> None:
        self._beam_size = beam_size
        self._normalize_by_length = normalize_by_length
        self._max_decoding_steps = max_decoding_steps
        self._max_num_decoded_sequences = max_num_decoded_sequences
        self._max_num_finished_states = max_num_finished_states

    @overrides
    def process(self,
                 initial_state: State,
                 finished_states: List[State],
                 reward_function: Callable[[StateType], torch.Tensor]):
        states_by_batch_index: Dict[int, List[State]] = defaultdict(list)
        for state in finished_states:
            assert len(state.batch_indices) == 1
            batch_index = state.batch_indices[0]
            states_by_batch_index[batch_index].append(state)

        loss = initial_state.score[0].new_zeros(1)
        search_hits = 0.0
        for instance_states in states_by_batch_index.values():
            all_correct_scores = [state.score[0].view(-1) for state in instance_states if reward_function(state) == 1]
            if all_correct_scores:
                all_correct_scores = torch.cat(all_correct_scores)
                best_score, _ = torch.max(all_correct_scores, dim = 0)
                best_score += 1
                search_hits += 1
            else:
                continue 

            all_constraints = torch.cat([ (state.score[0] - best_score + reward_function(state)).view(-1)
                                          for state in instance_states])

            most_violating_score, _ = torch.max(all_constraints, dim =0)
            loss += torch.clamp(most_violating_score, min=0)


        return {'loss' : loss / len(states_by_batch_index),
                'best_final_states' : self._get_best_final_states(finished_states),
                'noop' : search_hits == 0.0,
                'search_hits' : search_hits}
