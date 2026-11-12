#!/usr/bin/env python3
"""Enigma cipher machine - pure Python implementation."""

import argparse
import sys
from typing import Dict, List, Optional, Tuple

# Enigma I rotor wirings (standard, ring A, position A)
ROTOR_WIRINGS = {
    "I":   "EKMFLGDQVZNTOWYHXUSPAIBRCJ",
    "II":  "AJDKSIRUXBLHWTMCQGZNPYFVOE",
    "III": "BDFHJLCPRTXVZNYEIWGAKMUSQO",
    "IV":  "ESOVPZJAYQUIRHXLNFTGKDCMWB",
    "V":   "VZBRGITYUPSDNHLXAWMJQOFECK",
}

# Reflector B wiring (standard)
REFLECTOR_B = "YRUHQSLDPXNGOKMIEBFZCWVJAT"

# Turnover notches for rotors I-V
TURNOVER_NOTCHES = {
    "I":   "Q",  # 16
    "II":  "E",  # 4
    "III": "V",  # 21
    "IV":  "J",  # 9
    "V":   "Z",  # 25
}


def char_to_idx(c: str) -> int:
    return ord(c.upper()) - ord('A')


def idx_to_char(i: int) -> str:
    return chr(i + ord('A'))


class Plugboard:
    """Pairwise letter swap box."""

    def __init__(self, pairs: List[Tuple[str, str]] = None):
        self._map: Dict[str, str] = {}
        if pairs:
            for a, b in pairs:
                self._map[a.upper()] = b.upper()
                self._map[b.upper()] = a.upper()

    def __call__(self, char: str) -> str:
        return self._map.get(char.upper(), char)

    def __repr__(self) -> str:
        return f"Plugboard({len(self._map)//2} pairs)"


class Reflector:
    """Fixed reflector (involution)."""

    def __init__(self, wiring: str = None):
        self._wiring = wiring or REFLECTOR_B

    def __call__(self, char: str) -> str:
        idx = char_to_idx(char)
        return self._wiring[idx]

    def __repr__(self) -> str:
        return "Reflector(B)"


class Rotor:
    """Single Enigma rotor."""

    def __init__(self, name: str, ring: str = "A", position: str = "A"):
        self.name = name
        self.ring = char_to_idx(ring)
        self.position = char_to_idx(position)
        self.wiring = ROTOR_WIRINGS[name]

        # Build reverse mapping
        self._rev: Dict[str, str] = {}
        for i, c in enumerate(self.wiring):
            self._rev[c] = idx_to_char(i)

    def advance(self) -> None:
        """Advance rotor by one step."""
        self.position = (self.position + 1) % 26

    def encrypt_right(self, char: str) -> str:
        """Encrypt right-to-left through rotor."""
        idx = char_to_idx(char)
        # Apply position offset
        shifted = (idx - self.position) % 26
        # Apply ring setting offset
        shifted = (shifted + self.ring) % 26
        # Get output from wiring
        out = char_to_idx(self.wiring[shifted])
        # Undo ring setting
        out = (out - self.ring) % 26
        # Undo position
        out = (out + self.position) % 26
        return idx_to_char(out)

    def encrypt_left(self, char: str) -> str:
        """Encrypt left-to-right through rotor."""
        idx = char_to_idx(char)
        # Apply position offset
        shifted = (idx - self.position) % 26
        # Apply ring setting offset
        shifted = (shifted + self.ring) % 26
        # Get output from reverse wiring
        out = char_to_idx(self._rev[idx_to_char(shifted)])
        # Undo ring setting
        out = (out - self.ring) % 26
        # Undo position
        out = (out + self.position) % 26
        return idx_to_char(out)

    def __repr__(self) -> str:
        return f"Rotor({self.name}, ring={idx_to_char(self.ring)}, pos={idx_to_char(self.position)})"


class EnigmaMachine:
    """Main Enigma machine."""

    DEFAULT_ROTORS = ["I", "II", "III"]
    DEFAULT_PLUGBOARD: List[Tuple[str, str]] = []

    def __init__(
        self,
        rotors: List[str] = None,
        positions: List[str] = None,
        plugboard_pairs: List[Tuple[str, str]] = None,
        reflector: str = "B",
    ):
        self.reflector = Reflector()
        self.plugboard = Plugboard(pairs=plugboard_pairs or self.DEFAULT_PLUGBOARD)

        # Build rotors
        self.rotors: List[Rotor] = []
        self.initial_positions: List[str] = []
        rotor_names = rotors or self.DEFAULT_ROTORS
        for i, name in enumerate(rotor_names):
            pos = positions[i] if positions and i < len(positions) else "A"
            self.initial_positions.append(pos)
            self.rotors.append(Rotor(name, "A", pos))

    def encrypt(self, text: str) -> str:
        """Encrypt text through the Enigma machine."""
        result = []
        for char in text.upper():
            if not char.isalpha():
                result.append(char)
                continue

            # Step rotors (rightmost advances first)
            self._step_rotors()

            # Plugboard
            c = self.plugboard(char)

            # Rotors right to left
            for rotor in reversed(self.rotors):
                c = rotor.encrypt_right(c)

            # Reflector
            c = self.reflector(c)

            # Rotors left to right
            for rotor in self.rotors:
                c = rotor.encrypt_left(c)

            # Plugboard again
            c = self.plugboard(c)

            result.append(c)

        return "".join(result)

    def _reset(self) -> None:
        """Reset rotors to initial positions (needed for decryption)."""
        for rotor, pos in zip(self.rotors, self.initial_positions):
            rotor.position = char_to_idx(pos)

    def decrypt(self, text: str) -> str:
        """Decrypt text. Enigma is reciprocal: reset rotors then encrypt again."""
        self._reset()
        return self.encrypt(text)

    def _step_rotors(self) -> None:
        """Advance the rightmost rotor, including turnover and double-stepping."""
        right = self.rotors[-1]
        right_at_notch = idx_to_char(right.position) == TURNOVER_NOTCHES[right.name]
        middle_at_notch = (
            len(self.rotors) > 2
            and idx_to_char(self.rotors[-2].position) == TURNOVER_NOTCHES[self.rotors[-2].name]
        )

        right.advance()
        if len(self.rotors) > 1 and (right_at_notch or middle_at_notch):
            self.rotors[-2].advance()
        if middle_at_notch:
            self.rotors[-3].advance()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enigma cipher machine - encrypt/decrypt text"
    )
    parser.add_argument(
        "--rotors", nargs="+", default=None,
        help="Rotor names: I II III IV V (default: I II III)",
    )
    parser.add_argument(
        "--positions", nargs="+", default=None,
        help="Rotor positions e.g. A B C (default: A A A)",
    )
    parser.add_argument(
        "--plugboard", nargs="+", default=None,
        help="Plugboard pairs: A B C D (swap A-B, C-D)",
    )
    parser.add_argument(
        "--text", default="HELLO WORLD",
        help="Text to encrypt/decrypt",
    )
    parser.add_argument(
        "--mode", choices=["encrypt", "decrypt"], default="encrypt",
        help="Operation mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse plugboard pairs
    pairs = None
    if args.plugboard:
        if len(args.plugboard) % 2 != 0:
            print("Error: plugboard pairs must be even count", file=sys.stderr)
            sys.exit(1)
        pairs = [(args.plugboard[i], args.plugboard[i+1])
                 for i in range(0, len(args.plugboard), 2)]

    # Build machine
    machine = EnigmaMachine(
        rotors=args.rotors,
        positions=args.positions,
        plugboard_pairs=pairs,
        reflector="B",
    )

    # Perform operation
    result = machine.encrypt(args.text) if args.mode == "encrypt" else machine.decrypt(args.text)
    print(result)


if __name__ == "__main__":
    main()
