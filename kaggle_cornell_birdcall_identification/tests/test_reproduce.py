#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the replay command.

The command has two jobs, and the order matters: verify the inputs, and
only then re-run the experiment.  An input that no longer hashes the
same is a hard error, and the command refuses to run; a package version
that moved is worth saying out loud but is not a reason to refuse.
'''

import json
import sys
from pathlib import Path

import pytest

from birdcall.config_params.parameters import Parameters
from birdcall.provenance import build_manifest
from birdcall.provenance import sha256_file
from birdcall.provenance import write_manifest
from birdcall.reproduce import check_inputs
from birdcall.reproduce import check_packages
from birdcall.reproduce import load_manifest
from birdcall.reproduce import main


def _dataset(root: str) -> dict:
  '''Builds the keys of one dataset parameter group.

  Args:
    root: The name of the split.

  Returns:
    A mapping of the required dataset keys.
  '''
  return {
      'root_dir': Path('/data'),
      'csv_dir': Path(f'/folds/{root}.csv'),
      'period': 30,
      'isTraining': root == 'train',
      'num_test_samples': 1,
  }


def _parameters() -> Parameters:
  '''Builds a valid configuration.

  Returns:
    The validated configuration.
  '''
  return Parameters(train_ds_params=_dataset('train'),
                     valid_ds_params=_dataset('valid'),
                     test_ds_params=_dataset('test'))


def _marker_command(marker: Path) -> list[str]:
  '''Builds a command that leaves a marker file behind.

  Args:
    marker: The path of the marker file.

  Returns:
    The command, as a list of arguments.
  '''
  return [sys.executable, '-c',
          'import sys; from pathlib import Path; '
          'Path(sys.argv[1]).write_text(sys.argv[2])',
          str(marker), 'ran']


def _manifest(tmp_path, marker: Path | None = None):
  '''Writes a manifest with one input and, optionally, one command.

  Args:
    tmp_path: The pytest temporary directory.
    marker: The path of the marker file the command would leave.

  Returns:
    The path of the manifest.
  '''
  fold = tmp_path / 'fold_0_train.csv'
  fold.write_text('filename,ebird_code\n', encoding='utf-8')
  extra = {}
  if marker is not None:
    extra['command'] = _marker_command(marker)
  manifest = build_manifest(_parameters(),
                            inputs={'fold_0_train.csv': fold},
                            extra=extra or None)
  return write_manifest(manifest, tmp_path / 'manifest.json')


def test_a_manifest_that_matches_reports_nothing_wrong(tmp_path, capsys):
  '''A run whose inputs are unchanged passes the check.'''
  path = _manifest(tmp_path)
  assert main([str(path)]) == 0
  captured = capsys.readouterr()
  assert 'ERROR' not in captured.err
  assert 'ERROR' not in captured.out


def test_a_changed_input_is_reported_by_name(tmp_path, capsys):
  '''The audio changed underneath the run, so the run is not the same.'''
  path = _manifest(tmp_path)
  (tmp_path / 'fold_0_train.csv').write_text('filename,ebird_code\nx\n',
                                             encoding='utf-8')
  assert main([str(path)]) == 1
  captured = capsys.readouterr()
  assert 'fold_0_train.csv' in captured.err
  assert 'ERROR' in captured.err


def test_a_changed_input_refuses_to_run_the_command(tmp_path):
  '''The command is only replayed when the checks pass.'''
  marker = tmp_path / 'ran.txt'
  path = _manifest(tmp_path, marker)
  (tmp_path / 'fold_0_train.csv').write_text('filename,ebird_code\nx\n',
                                             encoding='utf-8')
  assert main([str(path), '--run']) == 1
  assert not marker.exists()


def test_check_only_does_not_run_the_command(tmp_path):
  '''The check is a check, not a replay.'''
  marker = tmp_path / 'ran.txt'
  path = _manifest(tmp_path, marker)
  assert main([str(path), '--check-only']) == 0
  assert not marker.exists()


def test_run_replays_the_recorded_command(tmp_path):
  '''The manifest is an executable description of the experiment.'''
  marker = tmp_path / 'ran.txt'
  path = _manifest(tmp_path, marker)
  assert main([str(path), '--run']) == 0
  assert marker.read_text(encoding='utf-8') == 'ran'


def test_a_missing_input_is_reported_by_name(tmp_path, capsys):
  '''An input that is gone is a mismatch too.'''
  path = _manifest(tmp_path)
  (tmp_path / 'fold_0_train.csv').unlink()
  assert main([str(path)]) == 1
  captured = capsys.readouterr()
  assert 'fold_0_train.csv' in captured.err
  assert 'missing' in captured.err


def test_the_check_names_every_mismatch(tmp_path, capsys):
  '''Every mismatching input is reported, not just the first.'''
  first = tmp_path / 'first.csv'
  first.write_text('a\n', encoding='utf-8')
  second = tmp_path / 'second.csv'
  second.write_text('b\n', encoding='utf-8')
  manifest = build_manifest(_parameters(),
                            inputs={'first.csv': first, 'second.csv': second})
  first.write_text('changed\n', encoding='utf-8')
  second.write_text('changed\n', encoding='utf-8')
  problems = check_inputs(manifest)
  assert len(problems) == 2
  assert any('first.csv' in problem for problem in problems)
  assert any('second.csv' in problem for problem in problems)


def test_the_check_compares_the_recorded_digest(tmp_path):
  '''The check is a digest comparison, not a mtime comparison.'''
  path = tmp_path / 'data.csv'
  path.write_text('a\n', encoding='utf-8')
  manifest = build_manifest(_parameters(), inputs={'data.csv': path})
  assert check_inputs(manifest) == []
  path.write_text('b\n', encoding='utf-8')
  problems = check_inputs(manifest)
  assert len(problems) == 1
  assert sha256_file(path) in problems[0]


def test_a_package_difference_is_reported_but_not_fatal(tmp_path, capsys):
  '''A patch bump is not a different experiment, so it does not stop the run.'''
  path = _manifest(tmp_path)
  document = json.loads(path.read_text(encoding='utf-8'))
  document['packages']['numpy'] = '0.0.0-not-installed'
  path.write_text(json.dumps(document), encoding='utf-8')
  assert main([str(path)]) == 0
  captured = capsys.readouterr()
  assert 'numpy' in captured.err
  assert 'WARNING' in captured.err
  assert 'ERROR' not in captured.err


def test_a_package_that_is_gone_is_reported(tmp_path, capsys):
  '''A package that was recorded but is not installed is worth saying.'''
  path = _manifest(tmp_path)
  document = json.loads(path.read_text(encoding='utf-8'))
  document['packages']['wandb'] = '0.16.0'
  path.write_text(json.dumps(document), encoding='utf-8')
  differences = check_packages(json.loads(path.read_text(encoding='utf-8')))
  assert any('wandb' in difference for difference in differences)


def test_a_manifest_that_is_not_a_mapping_is_refused(tmp_path):
  '''A manifest is a mapping, and nothing else.'''
  path = tmp_path / 'manifest.json'
  path.write_text('[1, 2, 3]', encoding='utf-8')
  with pytest.raises(ValueError, match='mapping'):
    load_manifest(path)


def test_a_manifest_that_is_not_there_is_refused(tmp_path):
  '''A missing manifest is an error, not an empty check.'''
  with pytest.raises(FileNotFoundError):
    load_manifest(tmp_path / 'nope.json')


def test_check_only_and_run_cannot_be_combined(tmp_path):
  '''The two modes are exclusive; asking for both is a usage error.'''
  path = _manifest(tmp_path)
  with pytest.raises(SystemExit):
    main([str(path), '--check-only', '--run'])
