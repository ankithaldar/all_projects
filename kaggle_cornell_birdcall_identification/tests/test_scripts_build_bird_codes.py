#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the label space generator.

The packaged ``bird_codes.json`` is the **recovered original** label
space, not a placeholder, so the generator refuses to overwrite it
unless it is told to.  The tests therefore write to a temporary
directory, and check the refusal against a copy rather than against the
package asset.
'''

import json
from pathlib import Path

import pandas as pd
import pytest

from birdcall.provenance import sha256_file
import birdcall.scripts.build_bird_codes as codes_script

NUM_SPECIES = 264


def _competition_csv(tmp_path: Path, num_species: int = NUM_SPECIES) -> Path:
  '''Builds a synthetic competition CSV.

  The columns are the competition's own: the file name, the primary
  species code and the secondary labels.

  Args:
    tmp_path: The pytest temporary directory.
    num_species: The number of species in the CSV.

  Returns:
    The path of the CSV.
  '''
  rows = []
  for index in range(num_species):
    code = f'spe{index:03d}'
    rows.append({'filename': f'REC{index:05d}.mp3',
                 'ebird_code': code,
                 'secondary_labels': '[]'})
  path = tmp_path / 'train.csv'
  path.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(rows).to_csv(path, index=False)
  return path


def test_the_default_output_is_the_packaged_label_space():
  '''The generator writes the package asset unless it is told otherwise.'''
  assert codes_script.DEFAULT_OUTPUT.exists()
  assert codes_script.DEFAULT_OUTPUT.name == 'bird_codes.json'
  assert codes_script.DEFAULT_OUTPUT.parent.name == 'config_params'


def test_it_refuses_to_overwrite_an_existing_file(tmp_path):
  '''The recovered original label space is protected.'''
  csv_path = _competition_csv(tmp_path)
  output = tmp_path / 'bird_codes.json'
  output.write_text('{"codes": {}}', encoding='utf-8')
  with pytest.raises(SystemExit) as error:
    codes_script.main(['--train-csv', str(csv_path), '--output', str(output)])
  assert '--force' in str(error.value)
  assert output.read_text(encoding='utf-8') == '{"codes": {}}'


def test_it_refuses_to_overwrite_the_packaged_label_space(tmp_path):
  '''The packaged file is not replaced without an explicit instruction.'''
  csv_path = _competition_csv(tmp_path)
  before = codes_script.DEFAULT_OUTPUT.read_bytes()
  with pytest.raises(SystemExit) as error:
    codes_script.main(['--train-csv', str(csv_path)])
  assert '--force' in str(error.value)
  assert codes_script.DEFAULT_OUTPUT.read_bytes() == before


def test_with_force_it_writes_the_contiguous_label_space(tmp_path):
  '''The regenerated label space has 264 codes with contiguous indices.'''
  csv_path = _competition_csv(tmp_path)
  output = tmp_path / 'bird_codes.json'
  assert codes_script.main(['--train-csv', str(csv_path), '--output',
                             str(output), '--force']) == 0
  document = json.loads(output.read_text(encoding='utf-8'))
  assert len(document['codes']) == NUM_SPECIES
  assert document['num_classes'] == NUM_SPECIES
  assert sorted(document['codes'].values()) == list(range(NUM_SPECIES))
  # The codes are sorted, so the index order is the sorted order.
  assert list(document['codes']) == sorted(document['codes'])
  assert document['codes']['spe000'] == 0


def test_the_source_names_the_csv_and_its_digest(tmp_path):
  '''The label space can be traced back to the file it came from.'''
  csv_path = _competition_csv(tmp_path)
  output = tmp_path / 'bird_codes.json'
  codes_script.main(['--train-csv', str(csv_path), '--output', str(output)])
  document = json.loads(output.read_text(encoding='utf-8'))
  assert 'train.csv' in document['source']
  assert sha256_file(csv_path) in document['source']


def test_the_codes_are_the_ones_of_the_csv(tmp_path):
  '''Only the species of the CSV are in the label space.'''
  csv_path = _competition_csv(tmp_path, num_species=7)
  output = tmp_path / 'bird_codes.json'
  codes_script.main(['--train-csv', str(csv_path), '--output', str(output)])
  document = json.loads(output.read_text(encoding='utf-8'))
  assert len(document['codes']) == 7
  assert 'spe000' in document['codes']


def test_a_csv_without_the_code_column_is_refused(tmp_path):
  '''A CSV that is not the competition's is a clear error.'''
  path = tmp_path / 'wrong.csv'
  pd.DataFrame([{'filename': 'a.mp3'}]).to_csv(path, index=False)
  with pytest.raises(ValueError, match='ebird_code'):
    codes_script.read_codes(path)


def test_a_csv_that_is_not_there_is_refused(tmp_path):
  '''A missing competition CSV is an error.'''
  with pytest.raises(FileNotFoundError):
    codes_script.read_codes(tmp_path / 'nope.csv')
