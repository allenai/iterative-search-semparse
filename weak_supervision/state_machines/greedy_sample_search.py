from collections import defaultdict
from typing import Dict, Generic, List, Mapping, Sequence, TypeVar
import numpy as np
import random

from allennlp.common.registrable import FromParams
from allennlp.state_machines.states import State
from allennlp.state_machines.transition_functions import TransitionFunction

StateType = TypeVar('StateType', bound=State)  # pylint: disable=invalid-name


class GreedyEpsilonBeamSearch(FromParams, Generic[StateType]):
    """

   This class implements GreedyEpsilonBeamSearch to balance exploration with exploitation
   from Guu et al. (2017) https://arxiv.org/pdf/1704.07926.pdf
    Parameters
    ----------
    beam_size : ``int``
        The beam size to use.
    per_node_beam_size : ``int``, optional (default = beam_size)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to `beam_size`. Setting this parameter
        to a number smaller than `beam_size` may give better results, as it can introduce
        more diversity into the search. See Freitag and Al-Onaizan 2017,
        "Beam Search Strategies for Neural Machine Translation".
    epsilon : mixing probability between random sampling and beam search
    """
    def __init__(self, beam_size: int, per_node_beam_size: int = None, epsilon: float = 0.0) -> None:
        self._beam_size = beam_size
        self._per_node_beam_size = per_node_beam_size or beam_size
        self._epsilon = epsilon

    def search(self,
               num_steps: int,
               initial_state: StateType,
               transition_function: TransitionFunction,
               keep_final_unfinished_states: bool = True) -> Mapping[int, Sequence[StateType]]:
        """
        Parameters
        ----------
        num_steps : ``int``
            How many steps should we take in our search?  This is an upper bound, as it's possible
            for the search to run out of valid actions before hitting this number, or for all
            states on the beam to finish.
        initial_state : ``StateType``
            The starting state of our search.  This is assumed to be `batched`, and our beam search
            is batch-aware - we'll keep ``beam_size`` states around for each instance in the batch.
        transition_function : ``TransitionFunction``
            The ``TransitionFunction`` object that defines and scores transitions from one state to the
            next.
        keep_final_unfinished_states : ``bool``, optional (default=True)
            If we run out of steps before a state is "finished", should we return that state in our
            search results?
        Returns
        -------
        best_states : ``Dict[int, List[StateType]]``
            This is a mapping from batch index to the top states for that instance.
        """
        finished_states: Dict[int, List[StateType]] = defaultdict(list)
        states = [initial_state]
        step_num = 1
        while states and step_num <= num_steps:
            next_states: Dict[int, List[StateType]] = defaultdict(list)
            grouped_state = states[0].combine_states(states)

            # Since we are doing greedy epsilon beam search, we first want to collect all next states for every example
            # before we process them
            for score, next_state in transition_function.take_step(grouped_state, max_actions=None, greedy_process=True):
                # NOTE: we're doing state.batch_indices[0] here (and similar things below),
                # hard-coding a group size of 1.  But, our use of `next_state.is_finished()`
                # already checks for that, as it crashes if the group size is not 1.
                batch_index = next_state.batch_indices[0]
                if next_state.is_finished():
                    finished_states[batch_index].append((score, next_state))
                else:
                    if step_num == num_steps and keep_final_unfinished_states:
                        finished_states[batch_index].append((score,next_state))
                    next_states[batch_index].append((score,next_state))

            states = []
            for batch_index, batch_states in next_states.items():
                # here we process all the batch states to return self._beam_size number of states per example
                # by using the greedy epsilon beam search method
                states.extend(self.greedy_epsilon_process(batch_states))
            step_num += 1
        best_states: Dict[int, Sequence[StateType]] = {}
        for batch_index, batch_states in finished_states.items():
            # The time this sort takes is pretty negligible, no particular need to optimize this
            # yet.  Maybe with a larger beam size...
            best_states[batch_index] = self.greedy_epsilon_process(batch_states)
        return best_states

    def greedy_epsilon_process(self, batch_states):
        # batch_states is all possible continuations for every state in the batch. We want to choose k=self._beam_size
        # number of new states out of this. For vanilla beam search (epsilon=0), this amounts to just sorting by their
        # log probabilities and picking the top-k, but for greedy epsilon beam search we first make k  binary decisions
        # and if decision[i] = 0, we choose the best unchosen state, otherwise out of the unchosen states, we pick
        # a state at random

        all_indices = list(range(len(batch_states)))
        all_indices.sort(key = lambda  idx : batch_states[idx][0] , reverse = True)
        # keep track of actions chosen so far
        chosen = [0 for _ in all_indices] 
        unchosen = set(all_indices)

        # keep track of current largest unchosen
        curr_top = 0
        decisions = np.random.binomial(size=min(len(batch_states), self._beam_size), n=1, p= self._epsilon)
        new_states = []
        for decision in decisions:
            # use the current highest
            if not decision: 
                _, state = batch_states[curr_top]
                chosen[curr_top] = 1
                unchosen.remove(curr_top)
            else:
                random_idx = random.sample(unchosen, 1)[0]
                unchosen.remove(random_idx)
                _, state = batch_states[random_idx]
                chosen[random_idx] = 1
            # restore the invariant
            while curr_top < len(chosen) and chosen[curr_top]: curr_top += 1
            new_states.append(state)

        return new_states


