#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Regenerate the label space from the competition's ``train.csv``.

The 264 species codes and their class indices are **data**, not code: the
index order is the label order everywhere, so it is written once and
never regenerated silently.  This script writes it from the competition's
own CSV, and records the CSV's SHA-256 in the ``source`` field so the
label space can be traced back to the file it came from.

The packaged ``bird_codes.json`` is the **recovered original** label
space, not a placeholder, so it is not overwritten unless ``--force`` is
given.  The common names (``Genus species_Common Name``) are not
derivable from the competition CSV and are written as an empty mapping.

Usage:

    python -m birdcall.scripts.build_bird_codes --train-csv train.csv
'''

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Optional

import pandas as pd

from birdcall.provenance import sha256_file

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PACKAGE_ROOT / 'config_params' / 'bird_codes.json'
CODE_COLUMN = 'ebird_code'


def read_codes(train_csv: Path) -> list[str]:
  '''Reads the sorted species codes of a competition CSV.

  Args:
    train_csv: The path of the CSV.

  Returns:
    The sorted species codes.

  Raises:
    FileNotFoundError: If the CSV does not exist.
    ValueError: If the CSV has no ``ebird_code`` column.
  '''
  path = Path(train_csv)
  if not path.exists():
    raise FileNotFoundError(f'Competition CSV not found: {path}')
  frame = pd.read_csv(path)
  if CODE_COLUMN not in frame.columns:
    raise ValueError(f'{path} has no {CODE_COLUMN!r} column; it has '
                     f'{list(frame.columns)}.')
  codes = frame[CODE_COLUMN].dropna().astype(str).unique().tolist()
  return sorted(codes)


def build_label_space(train_csv: Path) -> dict[str, Any]:
  '''Builds the label space document.

  Args:
    train_csv: The path of the competition CSV.

  Returns:
    The document: the source, the number of classes, the code to index
    map and the (empty) common name map.
  '''
  path = Path(train_csv)
  codes = read_codes(path)
  return {
      'source': f'{path.name} sha256:{sha256_file(path)}',
      'num_classes': len(codes),
      'codes': {code: index for index, code in enumerate(codes)},
      'labels': {},
  }


def write_label_space(document: dict[str, Any], output: Path,
                      force: bool = False) -> Path:
  '''Writes the label space document.

  Args:
    document: The document to write.
    output: The path of the file.
    force: Whether an existing file may be overwritten.

  Returns:
    The path the document was written to.

  Raises:
    FileExistsError: If the file exists and ``force`` is false.
  '''
  path = Path(output)
  if path.exists() and not force:
    raise FileExistsError(
        f'{path} already exists. It is the recovered original label space, '
        'so it is only replaced when --force is given.')
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(document, indent=2, sort_keys=True),
                  encoding='utf-8')
  return path


def build_parser() -> argparse.ArgumentParser:
  '''Builds the command line parser.

  Returns:
    The parser.
  '''
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--train-csv', type=Path, required=True,
                      help='The path of the competition train.csv.')
  parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                      help='The path of the label space JSON. Defaults to '
                           f'the packaged one: {DEFAULT_OUTPUT}.')
  parser.add_argument('--force', action='store_true',
                      help='Overwrite an existing label space file.')
  return parser


def main(argv: Optional[list[str]] = None) -> int:
  '''Regenerates the label space.

  Args:
    argv: The command line arguments, or ``None`` for the real ones.

  Returns:
    The exit code.
  '''
  args = build_parser().parse_args(argv)
  document = build_label_space(args.train_csv)
  try:
    path = write_label_space(document, args.output, force=args.force)
  except FileExistsError as error:
    raise SystemExit(f'ERROR: {error}') from error
  print(f'Wrote {document["num_classes"]} codes to {path}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
