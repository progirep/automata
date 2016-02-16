#!/usr/bin/env python3

import automata.automaton as automaton
import nose.tools as nose
from automata.nfa import NFA


class TestNFA():

    def setup(self):
        # NFA which matches "aaa" or any string where number of 'a's is an even
        # number greater than zero
        self.nfa = NFA(**{
            'states': {'q0', 'q1', 'q2', 'q3', 'q4', 'q5'},
            'symbols': {'a', 'b'},
            'transitions': {
                'q0': {'a': {'q1', 'q4'}},
                'q1': {'a': {'q2'}},
                'q2': {'a': {'q3'}},
                'q3': {},
                'q4': {'a': {'q5'}},
                'q5': {'a': {'q4'}}
            },
            'initial_state': 'q0',
            'final_states': {'q3', 'q5'}
        })

    def test_validate_automaton_missing_state(self):
        with nose.assert_raises(automaton.MissingStateError):
            del self.nfa.transitions['q1']
            self.nfa.validate_automaton()

    def test_validate_automaton_invalid_symbol(self):
        with nose.assert_raises(automaton.InvalidSymbolError):
            self.nfa.transitions['q1']['c'] = {'q2'}
            self.nfa.validate_automaton()

    def test_validate_automaton_invalid_state(self):
        with nose.assert_raises(automaton.InvalidStateError):
            self.nfa.transitions['q1']['a'] = {'q6'}
            self.nfa.validate_automaton()

    def test_validate_automaton_invalid_initial_state(self):
        with nose.assert_raises(automaton.InvalidStateError):
            self.nfa.initial_state = 'q6'
            self.nfa.validate_automaton()

    def test_validate_automaton_invalid_final_state(self):
        with nose.assert_raises(automaton.InvalidStateError):
            self.nfa.final_states = {'q6'}
            self.nfa.validate_automaton()