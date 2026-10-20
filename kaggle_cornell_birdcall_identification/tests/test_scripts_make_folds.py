#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the fold generator.

The original's fold CSVs are gone, so this generator produces a new,
reproducible split.  The three properties that make it useful are
checked here: the same seed gives byte-identical files, every species is
in every fold, and the split keys on the primary label only.
'''

import json
from pathlib import Path

import pandas as pd
import pytest

from birdcall.provenance import sha256_file
import birdcall.scripts.make_folds as folds_script

NUM_SPECIES = 6
ROWS_PER_SPECIES = 8


def _competition_csv(tmp_path: Path, secondary: bool = True) -> Path:
  '''Builds a synthetic competition CSV.

  Args:
    tmp_path: The pytest temporary directory.
    secondary: Whether the rows carry secondary labels.

  Returns:
    The path of the CSV.
  '''
  rows = []
  for index in range(NUM_SPECIES):
    for row in range(ROWS_PER_SPECIES):
      rows.append({
          'filename': f'REC{index}_{row}.mp3',
          'ebird_code': f'spe{index:03d}',
          # The competition stores the secondary labels as a list literal.
          'secondary_labels': ['American Crow'] if secondary else [],
      })
  path = tmp_path / 'train.csv'
  path.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(rows).to_csv(path, index=False)
  return path


def _run(tmp_path: Path, csv_path: Path, seed: int = 42, folds: int = 4,
         family: str = 'final_sed') -> Path:
  '''Runs the generator.

  Args:
    tmp_path: The pytest temporary directory.
    csv_path: The competition CSV.
    seed: The seed of the split.
    folds: The number of folds.
    family: The fold family.

  Returns:
    The directory the fold family was written to.
  '''
  out_dir = tmp_path / 'folds'
  assert folds_script.main(['--train-csv', str(csv_path),
                             '--out-dir', str(out_dir),
                             '--folds', str(folds),
                             '--seed', str(seed),
                             '--family', family]) == 0
  return out_dir / family


def test_the_same_seed_gives_byte_identical_csvs(tmp_path):
  '''A reproducible split is the whole point of the seed.'''
  csv_path = _competition_csv(tmp_path)
  first = _run(tmp_path / 'a', csv_path, seed=42)
  second = _run(tmp_path / 'b', csv_path, seed=42)
  names = sorted(path.name for path in first.glob('fold_*.csv'))
  assert names
  for name in names:
    assert (first / name).read_bytes() == (second / name).read_bytes()


def test_a_different_seed_gives_a_different_split(tmp_path):
  '''Two seeds are two splits.'''
  csv_path = _competition_csv(tmp_path)
  first = _run(tmp_path / 'a', csv_path, seed=42)
  second = _run(tmp_path / 'b', csv_path, seed=43)
  assert (first / 'fold_0_train.csv').read_bytes() != (
      second / 'fold_0_train.csv').read_bytes()


def test_every_fold_contains_every_species(tmp_path):
  '''The split is stratified, so no species is left out of a fold.'''
  csv_path = _competition_csv(tmp_path)
  out_dir = _run(tmp_path, csv_path, folds=4)
  for fold in range(4):
    frame = pd.read_csv(out_dir / f'fold_{fold}_test.csv')
    assert sorted(frame['ebird_code'].unique()) == [
        f'spe{index:03d}' for index in range(NUM_SPECIES)]


def test_the_train_and_test_folds_partition_the_data(tmp_path):
  '''Nothing is lost and nothing is counted twice.'''
  csv_path = _competition_csv(tmp_path)
  out_dir = _run(tmp_path, csv_path, folds=4)
  source = pd.read_csv(csv_path)
  for fold in range(4):
    train = pd.read_csv(out_dir / f'fold_{fold}_train.csv')
    test = pd.read_csv(out_dir / f'fold_{fold}_test.csv')
    assert len(train) + len(test) == len(source)
    assert set(train['filename']).isdisjoint(set(test['filename']))
    assert set(train['filename']) | set(test['filename']) == set(
        source['filename'])


def test_the_stratification_is_on_the_primary_label_only(tmp_path):
  '''The secondary labels do not take part in the split.'''
  with_secondary = _competition_csv(tmp_path / 'with', secondary=True)
  without_secondary = _competition_csv(tmp_path / 'without', secondary=False)
  first = _run(tmp_path / 'a', with_secondary, seed=42)
  second = _run(tmp_path / 'b', without_secondary, seed=42)
  for fold in range(4):
    left = pd.read_csv(first / f'fold_{fold}_test.csv')
    right = pd.read_csv(second / f'fold_{fold}_test.csv')
    assert sorted(left['ebird_code']) == sorted(right['ebird_code'])
    assert sorted(left['filename']) == sorted(right['filename'])


def test_the_family_names_the_output_directory(tmp_path):
  '''The two fold families are two independent split families.'''
  csv_path = _competition_csv(tmp_path)
  assert _run(tmp_path, csv_path, family='final_sed').name == 'final_sed'
  assert _run(tmp_path, csv_path,
              family='final_sed_5_fold').name == 'final_sed_5_fold'


def test_the_manifest_records_the_seed_and_the_hashes(tmp_path):
  '''The split can be re-derived from the manifest alone.'''
  csv_path = _competition_csv(tmp_path)
  out_dir = _run(tmp_path, csv_path, seed=7, folds=4)
  manifest = json.loads(
      (out_dir / 'folds_manifest.json').read_text(encoding='utf-8'))
  assert manifest['seed'] == 7
  assert manifest['folds'] == 4
  assert manifest['family'] == 'final_sed'
  assert manifest['source']['sha256'] == sha256_file(csv_path)
  assert len(manifest['files']) == 8
  for name, entry in manifest['files'].items():
    assert entry['sha256'] == sha256_file(out_dir / name)


def test_the_five_fold_family_writes_five_folds(tmp_path):
  '''The number of folds is a parameter, not a constant.'''
  csv_path = _competition_csv(tmp_path)
  out_dir = _run(tmp_path, csv_path, folds=5, family='final_sed_5_fold')
  for fold in range(5):
    assert (out_dir / f'fold_{fold}_train.csv').exists()
    assert (out_dir / f'fold_{fold}_test.csv').exists()


def test_a_csv_without_the_code_column_is_refused(tmp_path):
  '''A CSV that is not the competition's is a clear error.'''
  path = tmp_path / 'wrong.csv'
  pd.DataFrame([{'filename': 'a.mp3'}]).to_csv(path, index=False)
  with pytest.raises(SystemExit, match='ebird_code'):
    folds_script.main(['--train-csv', str(path),
                       '--out-dir', str(tmp_path / 'out')])


def test_one_fold_is_refused():
  '''A split needs at least one fold.'''
  frame = pd.DataFrame([{'ebird_code': 'a'}])
  with pytest.raises(ValueError, match='fold'):
    folds_script.assign_folds(frame, 0, 42)
