#!/usr/bin/env python3
"""
Turing's Bombe - software replication of Alan Turing's electromechanical
machine built at Bletchley Park to break the Enigma cipher.

The Bombe's purpose: find the correct daily Enigma settings by trying all
rotor combinations in parallel. Since there are only 26^3 = 17,576 possible
positions for 3 rotors, a computer can try them all quickly.

Usage:
    # Exhaustive search over all rotor positions
    python3 turing_bombe.py --text "MHAIA" --rotors I II III

    # With a known plaintext fragment ("crib") to filter candidates
    python3 turing_bombe.py --text "MHAIA" --crib HELLO

    # Custom plugboard
    python3 turing_bombe.py --text "MHAIA" --rotors I II III --plugboard A B C D
"""

import argparse
import sys
from enigma import EnigmaMachine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turing's Bombe - exhaustive search over Enigma rotor settings"
    )
    parser.add_argument(
        "--text", required=True,
        help="Encrypted message (ciphertext) to search over"
    )
    parser.add_argument(
        "--rotors", nargs="+", default=["I", "II", "III"],
        help="Rotor names (default: I II III)"
    )
    parser.add_argument(
        "--plugboard", nargs="+", default=None,
        help="Plugboard pairs (e.g., A B C D). Default: none"
    )
    parser.add_argument(
        "--positions", nargs="+", default=None,
        help="Fixed rotor positions - if given, no search is performed. Use to verify a specific key."
    )
    parser.add_argument(
        "--count", type=int, default=17576,
        help="Maximum number of candidates to show (default: 17576)"
    )
    return parser.parse_args()


def char_to_idx(c: str) -> int:
    return ord(c.upper()) - ord('A')


def idx_to_char(i: int) -> str:
    return chr(i + ord('A'))


def generate_positions(rotor_count: int) -> list:
    """Generate all possible rotor position combinations (A A A ... to Z Z Z)."""
    from itertools import product
    alphabet = [idx_to_char(i) for i in range(26)]
    return [list(p) for p in product(alphabet, repeat=rotor_count)]


def decrypt_with_settings(text: str, rotors: list, positions: list, plugboard_pairs: list = None) -> str:
    """Decrypt text with the given Enigma settings (returns plaintext)."""
    machine = EnigmaMachine(
        rotors=rotors,
        positions=positions,
        plugboard_pairs=plugboard_pairs,
    )
    return machine.decrypt(text)


def crib_matches(plaintext: str, crib: str) -> bool:
    """Check if a crib (known plaintext fragment) appears in the candidate."""
    upper = crib.upper()
    return crib in plaintext or upper in plaintext.upper()


def main() -> None:
    args = parse_args()
    
    # Build a fixed machine for the specific position check
    if args.positions:
        positions = args.positions
        positions = positions[:len(args.rotors)]
        candidates = [positions]
        search = False
    else:
        candidates = generate_positions(len(args.rotors))
        search = True
    
    plugboard_pairs = None
    if args.plugboard:
        if len(args.plugboard) % 2 != 0:
            print("Error: plugboard pairs must be even count", file=sys.stderr)
            sys.exit(1)
        plugboard_pairs = [(args.plugboard[i], args.plugboard[i+1]) 
                           for i in range(0, len(args.plugboard), 2)]
    
    print(f"=== Turing's Bombe ===")
    print(f"Ciphertext: {args.text}")
    print(f"Rotors: {args.rotors}")
    if args.plugboard:
        print(f"Plugboard: {args.plugboard}")
    if args.positions:
        print(f"Fixed positions: {args.positions}")
    if args.count:
        print(f"Candidates to show: {args.count}")
    print("=" * 40)
    
    found = 0
    for i, pos in enumerate(candidates):
        plaintext = decrypt_with_settings(args.text, args.rotors, pos, plugboard_pairs)
        
        if args.plugboard:
            # With plugboard, decryption requires the exact right plugboard too,
            # so only show this one position
            print(f"Pos {pos}: {plaintext}")
            print("=" * 40)
            sys.exit(0)
        
        print(f"Pos {pos}: {plaintext}")
        found += 1
        if found >= args.count:
            print(f"... ({len(candidates) - args.count} more candidates not shown)")
            break
    
    if search:
        total = len(candidates)
        print(f"\nTotal rotor combinations checked: {total}")
        print("Look for the candidate that reads like meaningful German.")


if __name__ == "__main__":
    main()
