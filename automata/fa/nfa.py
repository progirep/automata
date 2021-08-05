#!/usr/bin/env python3
"""Classes and methods for working with nondeterministic finite automata."""

import copy

import automata.base.exceptions as exceptions
import automata.fa.fa as fa
from collections import deque

from pydot import Edge, Node, Dot

class NFA(fa.FA):
    """A nondeterministic finite automaton."""

    def __init__(self, *, states, input_symbols, transitions,
                 initial_state, final_states):
        """Initialize a complete NFA."""
        self.states = states.copy()
        self.input_symbols = input_symbols.copy()
        self.transitions = copy.deepcopy(transitions)
        self.initial_state = initial_state
        self.final_states = final_states.copy()
        self.validate()
        
    def __str__(self):
        result = "NFA(\n"
        result += "  states = "+str(self.states)+",\n"
        result += "  input_symbols = "+str(self.input_symbols)+",\n"
        result += "  transitions = "+str(self.transitions)+",\n"
        result += "  initial_state = "+str(self.initial_state)+",\n"
        result += "  final_states = "+str(self.final_states)+"\n)"
        return result

    @classmethod
    def from_dfa(cls, dfa):
        """Initialize this NFA as one equivalent to the given DFA."""
        nfa_transitions = {}

        for start_state, paths in dfa.transitions.items():
            nfa_transitions[start_state] = {}
            for input_symbol, end_state in paths.items():
                nfa_transitions[start_state][input_symbol] = {end_state}

        return cls(
            states=dfa.states, input_symbols=dfa.input_symbols,
            transitions=nfa_transitions, initial_state=dfa.initial_state,
            final_states=dfa.final_states)

    @staticmethod
    def _stringify_states_unsorted(states):
        """Stringify the given set of states as a single state name."""
        return '('+','.join(states)+')'

    def _cross_product(self, other):
        """
        Creates a new NFA which is the cross product of BFAs self and other
        with an empty set of final states.
        """
        assert self.input_symbols == other.input_symbols
        states_a = list(self.states)
        states_b = list(other.states)
        new_states = {
            self._stringify_states_unsorted((a, b))
            for a in states_a for b in states_b
        }
        new_transitions = dict()
        for state_a, transitions_a in self.transitions.items():
            for state_b, transitions_b in other.transitions.items():
                new_state = self._stringify_states_unsorted(
                    (state_a, state_b)
                )
                new_transitions[new_state] = dict()
                for symbol in self.input_symbols:
                    new_transitions[new_state][symbol] = set([
                        self._stringify_states_unsorted(
                            (k1, k2)
                        )
                    for k1 in transitions_a[symbol] for k2 in transitions_b[symbol]])
        new_initial_state = self._stringify_states_unsorted(
            (self.initial_state, other.initial_state)
        )

        return NFA(
            states=new_states,
            input_symbols=self.input_symbols,
            transitions=new_transitions,
            initial_state=new_initial_state,
            final_states=set()
        )
            
            
    def intersection(self, other):
        """
        Takes as input two NFAs M1 and M2 which
        accept languages L1 and L2 respectively.
        Returns a DFA which accepts the intersection of L1 and L2.
        """
        new_nfa = self._cross_product(other)
        for state_a in self.final_states:
            for state_b in other.final_states:
                new_nfa.final_states.add(
                    self._stringify_states_unsorted((state_a, state_b))
                )
        return new_nfa            

    def _validate_transition_invalid_symbols(self, start_state, paths):
        for input_symbol in paths.keys():
            if input_symbol not in self.input_symbols and input_symbol != '':
                raise exceptions.InvalidSymbolError(
                    'state {} has invalid transition symbol {}'.format(
                        start_state, input_symbol))

    def _validate_transition_end_states(self, start_state, paths):
        """Raise an error if transition end states are invalid."""
        for end_states in paths.values():
            for end_state in end_states:
                if end_state not in self.states:
                    raise exceptions.InvalidStateError(
                        'end state {} for transition on {} is '
                        'not valid'.format(end_state, start_state))

    def validate(self):
        """Return True if this NFA is internally consistent."""
        for start_state, paths in self.transitions.items():
            self._validate_transition_invalid_symbols(start_state, paths)
            self._validate_transition_end_states(start_state, paths)
        self._validate_initial_state()
        self._validate_initial_state_transitions()
        self._validate_final_states()
        return True

    def _get_lambda_closure(self, start_state):
        """
        Return the lambda closure for the given state.

        The lambda closure of a state q is the set containing q, along with
        every state that can be reached from q by following only lambda
        transitions.
        """
        stack = []
        encountered_states = set()
        stack.append(start_state)

        while stack:
            state = stack.pop()
            if state not in encountered_states:
                encountered_states.add(state)
                if '' in self.transitions[state]:
                    stack.extend(self.transitions[state][''])

        return encountered_states

    def _get_next_current_states(self, current_states, input_symbol):
        """Return the next set of current states given the current set."""
        next_current_states = set()

        for current_state in current_states:
            symbol_end_states = self.transitions[current_state].get(
                input_symbol)
            if symbol_end_states:
                for end_state in symbol_end_states:
                    next_current_states.update(
                        self._get_lambda_closure(end_state))

        return next_current_states

    def _check_for_input_rejection(self, current_states):
        """Raise an error if the given config indicates rejected input."""
        if not (current_states & self.final_states):
            raise exceptions.RejectionException(
                'the NFA stopped on all non-final states ({})'.format(
                    ', '.join(str(state) for state in current_states)))

    def read_input_stepwise(self, input_str):
        """
        Check if the given string is accepted by this NFA.

        Yield the current configuration of the NFA at each step.
        """
        current_states = self._get_lambda_closure(self.initial_state)

        yield current_states
        for input_symbol in input_str:
            current_states = self._get_next_current_states(
                current_states, input_symbol)
            yield current_states

        self._check_for_input_rejection(current_states)
        
    def remove_unreachable_states(self):
        """Remove states which are not reachable from the initial state."""
        reachable_states = self._compute_reachable_states()
        unreachable_states = self.states - reachable_states
        for state in unreachable_states:
            self.states.remove(state)
            del self.transitions[state]
            if state in self.final_states:
                self.final_states.remove(state)
                
    def remove_states_with_empty_language(self):
        """Removes states from which no final states are reachable"""
        non_empty_states = set(self.final_states)
        oldLen = -1
        # Populate the set of states that can reach a final state
        while len(non_empty_states)!=oldLen:
            oldLen = len(non_empty_states)
            for state in self.states:
                if not state in non_empty_states:
                    for symbol, dst_states in self.transitions[state].items():
                        for dst_state in dst_states:
                            if dst_state in non_empty_states:
                                non_empty_states.add(state)
        # Remove the states with an empty language
        empty_states = self.states - non_empty_states
        for state in empty_states:
            if state==self.initial_state:
                # Empty language
                self.transitions[state] = {b : set([]) for b in self.input_symbols}
            else:
                self.states.remove(state)
                del self.transitions[state]
        # Remove all transitions to states with an empty language
        for state in self.states:
            for symbol, dst_states in self.transitions[state].items():
                self.transitions[state][symbol] = self.transitions[state][symbol] - empty_states


    def _compute_reachable_states(self):
        """Compute the states which are reachable from the initial state."""
        reachable_states = set()
        states_to_check = deque()
        states_to_check.append(self.initial_state)
        reachable_states.add(self.initial_state)
        while states_to_check:
            state = states_to_check.popleft()
            for symbol, dst_states in self.transitions[state].items():
                for dst_state in dst_states:
                    if dst_state not in reachable_states:
                        reachable_states.add(dst_state)
                        states_to_check.append(dst_state)
        return reachable_states        
        
    def show_diagram(self, path=None):
        """
            Creates the graph associated with this NFA
        """
        # Nodes are set of states

        graph = Dot(graph_type='digraph', rankdir='LR')
        nodes = {}
        for state in self.states:
            if state == self.initial_state:
                # color start state with green
                if state in self.final_states:
                    initial_state_node = Node(
                        state, style="filled", peripheries=2, fillcolor="green")
                else:
                    initial_state_node = Node(
                        state, style="filled", fillcolor="green")
                nodes[state] = initial_state_node
                graph.add_node(initial_state_node)
            else:
                if state in self.final_states:
                    state_node = Node(state,peripheries=2)
                else:
                    state_node = Node(state)
                nodes[state] = state_node
                graph.add_node(state_node)
        # adding edges
        for from_state, lookup in self.transitions.items():
            for to_label, to_states in lookup.items():
                for to_state in to_states:
                    graph.add_edge(Edge(
                        nodes[from_state],
                        nodes[to_state],
                        label=to_label
                    ))
        if path:
            graph.write_png(path)
        return graph
