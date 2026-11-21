import unittest

from enigma import EnigmaMachine


class RotorSteppingTests(unittest.TestCase):
    def test_rightmost_rotor_advances_on_every_letter(self):
        machine = EnigmaMachine(positions=["A", "A", "A"])
        machine.encrypt("A")
        self.assertEqual([rotor.position for rotor in machine.rotors], [0, 0, 1])

    def test_turnover_and_double_step_use_notches_before_stepping(self):
        machine = EnigmaMachine(positions=["A", "D", "U"])
        for expected in ([0, 3, 21], [0, 4, 22], [1, 5, 23]):
            machine.encrypt("A")
            self.assertEqual([rotor.position for rotor in machine.rotors], expected)


if __name__ == "__main__":
    unittest.main()
