#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The seeded fold generator.

The original's fold CSVs are gone: they were data artefacts, produced
offline by two independent runs, and never committed.  The exact
original split is therefore unrecoverable, and this script does not
pretend to recover it.  It produces a **new, reproducible** split: the
same seed gives byte-identical CSVs, and the manifest records the seed,
the source CSV's SHA-256 and the SHA-256 of every generated file, so the
ensemble can be re-derived.

The split is stratified on the **primary** ``ebird_code`` only.  The
secondary labels are deliberately not used: the sampler keys on the
primary species, so stratifying on the union of primary and secondary
would make the per-class counts that drive the split differ from the
counts the sampler balances.

Usage:

    python -m birdcall.scripts.make_folds --train-csv train.csv \\
        --out-dir folds --folds 4 --seed 42 --family final_sed
'''

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from birdcall.provenance import sha256_file

# The two fold families of the original.  The name is the output
# subdirectory, and it is also the name the experiment configurations
# are grouped under.
FAMILIES = ('final_sed', 'final_sed_5_fold')

CODE_COLUMN = 'ebird_code'


def assign_folds(frame: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
  '''Assigns every row to a fold, stratified on the primary species.

  The rows of one species are shuffled with the seeded generator and
  then dealt round robin over the folds, so a species with several
  recordings is spread over the folds instead of landing in one.

  Args:
    frame: The competition rows.
    folds: The number of folds.
    seed: The seed of the split.

  Returns:
    The fold index of every row, in the order of the rows.

  Raises:
    ValueError: If the number of folds is not positive.
  '''
  if folds < 1:
    raise ValueError(f'A split needs at least one fold, got {folds}.')
  generator = np.random.default_rng(seed)
  assignment = np.zeros(len(frame), dtype=np.int64)
  codes = frame[CODE_COLUMN].to_numpy()
  for code in sorted(set(codes.tolist())):
    positions = np.flatnonzero(codes == code)
    for rank, position in enumerate(generator.permutation(len(positions))):
      assignment[positions[position]] = rank % folds
  return assignment


def write_folds(frame: pd.DataFrame, assignment: np.ndarray, out_dir: Path,
                folds: int) -> list[Path]:
  '''Writes the per-fold train and test CSVs.

  The rows keep the order of the source CSV, so the output is a pure
  function of the input and the seed.

  Args:
    frame: The competition rows.
    assignment: The fold index of every row.
    out_dir: The directory the CSVs are written to.
    folds: The number of folds.

  Returns:
    The paths of the written files.
  '''
  out_dir.mkdir(parents=True, exist_ok=True)
  written = []
  for fold in range(folds):
    test = frame.iloc[np.flatnonzero(assignment == fold)]
    train = frame.iloc[np.flatnonzero(assignment != fold)]
    for name, part in ((f'fold_{fold}_train.csv', train),
                       (f'fold_{fold}_test.csv', test)):
      path = out_dir / name
      part.to_csv(path, index=False)
      written.append(path)
  return written


def write_folds_manifest(path: Path, seed: int, folds: int, family: str,
                         source: Path, files: list[Path]) -> Path:
  '''Writes the manifest of a fold set.

  Args:
    path: The path of the manifest.
    seed: The seed of the split.
    folds: The number of folds.
    family: The fold family.
    source: The competition CSV the split was made from.
    files: The generated CSV files.

  Returns:
    The path the manifest was written to.
  '''
  document = {
      'seed': int(seed),
      'folds': int(folds),
      'family': family,
      'source': {'path': str(source), 'sha256': sha256_file(source)},
      'files': {file.name: {'path': str(file), 'sha256': sha256_file(file)}
                for file in sorted(files)},
  }
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
  parser.add_argument('--out-dir', type=Path, required=True,
                      help='The directory the fold family is written to.')
  parser.add_argument('--folds', type=int, default=4,
                      help='The number of folds.')
  parser.add_argument('--seed', type=int, default=42,
                      help='The seed of the split.')
  parser.add_argument('--family', choices=FAMILIES, default='final_sed',
                      help='The fold family, which names the output '
                           'subdirectory.')
  return parser


def main(argv: Optional[list[str]] = None) -> int:
  '''Writes the fold CSVs.

  Args:
    argv: The command line arguments, or ``None`` for the real ones.

  Returns:
    The exit code.
  '''
  args = build_parser().parse_args(argv)
  frame = pd.read_csv(args.train_csv)
  if CODE_COLUMN not in frame.columns:
    raise SystemExit(f'ERROR: {args.train_csv} has no {CODE_COLUMN!r} '
                     f'column; it has {list(frame.columns)}.')
  assignment = assign_folds(frame, args.folds, args.seed)
  out_dir = Path(args.out_dir) / args.family
  files = write_folds(frame, assignment, out_dir, args.folds)
  manifest = write_folds_manifest(out_dir / 'folds_manifest.json', args.seed,
                                  args.folds, args.family, args.train_csv,
                                  files)
  print(f'Wrote {len(files)} fold CSVs to {out_dir} and the manifest to '
        f'{manifest}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
