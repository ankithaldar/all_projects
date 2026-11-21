# Enigma Cipher Machine (pure Python)

A faithful software model of the German **Enigma I** cipher machine, together
with Bletchley Park–inspired tools for attacking it. No third-party packages —
standard library only.

## What it is

`enigma.py` implements the full machine: the five standard Enigma I rotor
wirings (I–V), Reflector B, turnover notches with correct **double-stepping**,
a plugboard (Steckerbrett), rotor positions and ring settings. Encrypting and
decrypting are reciprocal operations: reset the rotors to the starting
positions and run the same text through again to recover the plaintext.

The two companion scripts explore the cryptanalysis side:

- `turing_bombe.py` — a software "Bombe" that exhaustively searches rotor
  positions for a ciphertext.
- `turing_decrypt.py` — decrypt a message when the settings are already known.

## Files

| File | Purpose |
| --- | --- |
| `enigma.py` | Machine, rotor/plugboard/reflector classes, rotor stepping and the encrypt/decrypt CLI. |
| `turing_bombe.py` | Brute-force search over rotor positions (17,576 for three rotors). |
| `turing_decrypt.py` | Decrypt with known settings, prompting for ciphertext if omitted. |
| `test_enigma.py` | `unittest` suite covering rotor advance, turnover and double-step. |
| `README.md` | This file. |

## How to run

All commands assume you are inside the `0001_enigma/` folder.

### Encrypt / decrypt (`enigma.py`)

```sh
python3 enigma.py --text "HELLO WORLD"
python3 enigma.py --text "HELLO WORLD" --rotors II IV V --positions A B C
python3 enigma.py --text "KCJNQ" --plugboard A B C D --mode decrypt
```

CLI options:

- `--rotors` — rotor names from `{I, II, III, IV, V}` (default `I II III`).
- `--positions` — starting rotor positions, e.g. `A B C` (default `A A A`).
- `--plugboard` — even number of letters forming swap pairs, e.g. `A B C D`
  swaps A↔B and C↔D (default: none).
- `--text` — text to process (default `HELLO WORLD`).
- `--mode` — `encrypt` (default) or `decrypt`.

Note: encryption is stateless with respect to the CLI in the sense that the
machine starts from the configured positions each run. To decrypt, supply the
same settings the message was encrypted with. Ring settings are supported by
the `Rotor` class in code but are not exposed as a CLI flag.

### Bombe search (`turing_bombe.py`)

```sh
# Exhaustive search over all 26^3 = 17,576 position combinations
python3 turing_bombe.py --text "MHAIA" --rotors I II III

# Check one specific key without searching
python3 turing_bombe.py --text "MHAIA" --positions A A A

# Cap the number of candidates printed
python3 turing_bombe.py --text "MHAIA" --count 25
```

Options: `--text` (required), `--rotors`, `--plugboard`, `--positions` and
`--count`. Every candidate plaintext is logged. A crib filter helper
(`crib_matches`) exists in the source, but there is no `--crib` CLI flag yet
even though the module docstring shows one.

### Known-key decryption (`turing_decrypt.py`)

```sh
python3 turing_decrypt.py --text "KCJNQ" --rotors I II III --positions B A C
python3 turing_decrypt.py            # prompts interactively for the ciphertext
```

Options: `--text` (empty ⇒ prompt), `--rotors`, `--positions`, `--plugboard`
and `--mode` (only `decrypt` is supported).

## Tests

From inside `0001_enigma/`:

```sh
python3 -m unittest test_enigma.py
```

The suite checks that the rightmost rotor advances on every keypress and that
turnover/double-stepping uses the notch positions correctly.

## Notes

- Non-alphabetic characters pass through unchanged; input is upper-cased
  internally.
- This is an educational model: it is not secure by modern standards and is
  intended for learning and experimentation only.
