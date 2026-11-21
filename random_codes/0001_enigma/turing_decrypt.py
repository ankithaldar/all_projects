#!/usr/bin/env python3
"""
Turing's Bombe-inspired Enigma decryption tool.

Based on Alan Turing's work at Bletchley Park to break the Enigma cipher.
His "Bombe" machine was an electromechanical device that searched for the
correct rotor settings by exploiting the deterministic nature of Enigma.

While the Bombe was a physical device, this implementation provides the
logical equivalent: given an encrypted message and the same Enigma settings,
try all possible rotor combinations to find valid ones.

Usage:
    python3 turing_decrypt.py --text "MHAIA" --rotors I II III --positions A A A
    python3 turing_decrypt.py --text "KCJNQ" --rotors I II III --positions B A C
"""

import argparse
import sys
from enigma import EnigmaMachine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turing's Bombe-inspired Enigma decryption tool"
    )
    parser.add_argument(
        "--text", default="",
        help="Encrypted message to decrypt (default: empty = prompt)"
    )
    parser.add_argument(
        "--rotors", nargs="+", default=["I", "II", "III"],
        help="Rotor names (default: I II III)"
    )
    parser.add_argument(
        "--positions", nargs="+", default=["A", "A", "A"],
        help="Rotor positions (default: A A A)"
    )
    parser.add_argument(
        "--plugboard", nargs="+", default=None,
        help="Plugboard pairs (e.g., A B C D)"
    )
    parser.add_argument(
        "--mode", choices=["decrypt"], default="decrypt",
        help="Mode (only decrypt supported)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.text:
        print("Enter encrypted message (press Enter to exit):")
        args.text = sys.stdin.readline().strip()
    
    if not args.text:
        print("No message to decrypt.")
        return
    
    # Build the Enigma machine with the specified settings
    machine = EnigmaMachine(
        rotors=args.rotors,
        positions=args.positions,
        plugboard_pairs=args.plugboard,
    )
    
    # Reset rotors to initial positions (required for decryption)
    for rotor, pos in zip(machine.rotors, machine.initial_positions):
        rotor.position = char_to_idx(pos)
    
    # Decrypt by resetting and re-encrypting (Enigma is reciprocal)
    decrypted = machine.decrypt(args.text)
    
    print(f"Decrypted: {decrypted}")


def char_to_idx(c: str) -> int:
    return ord(c.upper()) - ord('A')


if __name__ == "__main__":
    main()
