#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the run manifest.

The manifest is the difference between "a run happened" and "a run can be
repeated", so the properties that matter are checked here: the digests
are the real ones, the dirty git tree is visible, a package that is not
installed is absent rather than invented, and every value in the
document can be serialised.
'''

import json
from pathlib import Path

import numpy as np
import pytest

from birdcall.config_params.parameters import Parameters
from birdcall.provenance import EMPTY_DIGEST
from birdcall.provenance import PACKAGE_NAMES
from birdcall.provenance import build_manifest
from birdcall.provenance import git_state
from birdcall.provenance import package_versions
from birdcall.provenance import sha256_bytes
from birdcall.provenance import sha256_file
from birdcall.provenance import write_manifest
from birdcall.provenance import write_run_manifest

HELLO_DIGEST = ('2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e730433629'
                '38b9824')


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


def _parameters(**overrides) -> Parameters:
  '''Builds a valid configuration.

  Args:
    **overrides: The values that differ from the default recipe.

  Returns:
    The validated configuration.
  '''
  values = {
      'train_ds_params': _dataset('train'),
      'valid_ds_params': _dataset('valid'),
      'test_ds_params': _dataset('test'),
  }
  values.update(overrides)
  return Parameters(**values)


def test_the_digest_of_a_known_file_is_the_known_digest(tmp_path):
  '''The digest is the real SHA-256 of the bytes on disk.'''
  path = tmp_path / 'hello.txt'
  path.write_bytes(b'hello')
  assert sha256_file(path) == HELLO_DIGEST
  assert sha256_bytes(b'hello') == HELLO_DIGEST


def test_the_digest_of_an_empty_file_is_the_empty_digest(tmp_path):
  '''The empty file hashes to the well known empty digest.'''
  path = tmp_path / 'empty.bin'
  path.write_bytes(b'')
  assert sha256_file(path) == EMPTY_DIGEST
  assert sha256_bytes(b'') == EMPTY_DIGEST
  assert len(EMPTY_DIGEST) == 64


def test_a_large_file_is_hashed_in_chunks(tmp_path):
  '''A file bigger than the chunk size still hashes correctly.'''
  path = tmp_path / 'big.bin'
  data = b'birdcall' * 200000
  path.write_bytes(data)
  assert sha256_file(path) == sha256_bytes(data)


def test_a_missing_file_is_an_error(tmp_path):
  '''A digest of a file that is not there is not a digest.'''
  with pytest.raises(FileNotFoundError):
    sha256_file(tmp_path / 'nope.bin')


def test_the_git_state_of_a_directory_that_is_not_a_repository(tmp_path,
                                                              monkeypatch):
  '''A directory outside a repository has no git state, and does not raise.'''
  monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
  state = git_state(tmp_path)
  assert state == {'commit': None, 'branch': None, 'dirty': None}


def test_the_git_state_of_a_repository_carries_the_commit_and_the_branch():
  '''A repository is described by its commit, its branch and its diff.'''
  state = git_state(Path.cwd())
  assert set(state) == {'commit', 'branch', 'dirty'}
  assert state['commit'] is None or len(state['commit']) == 40
  assert state['dirty'] is None or len(state['dirty']) == 64


def test_the_package_versions_carry_the_python_version():
  '''The interpreter is part of the environment.'''
  versions = package_versions()
  assert 'python' in versions
  assert versions['python'].count('.') == 2


def test_the_package_versions_carry_the_installed_packages():
  '''Every package that matters is recorded with its version.'''
  versions = package_versions()
  for name in PACKAGE_NAMES:
    if name in versions:
      assert isinstance(versions[name], str)
      assert versions[name]
  assert 'numpy' in versions
  assert 'pandas' in versions


def test_a_package_that_is_not_installed_is_absent(monkeypatch):
  '''A package that is not there is not recorded with a made up version.'''
  from importlib import metadata as importlib_metadata

  def missing(name: str) -> str:
    raise importlib_metadata.PackageNotFoundError(name)

  monkeypatch.setattr('birdcall.provenance.metadata.version', missing)
  versions = package_versions()
  assert 'python' in versions
  assert not set(PACKAGE_NAMES) & set(versions)


def test_the_manifest_is_json_serialisable(tmp_path):
  '''Every value in the document can be written as JSON.'''
  config = tmp_path / 'experiment.yaml'
  config.write_text('name: test\n', encoding='utf-8')
  inputs = {'fold_0_train.csv': tmp_path / 'fold_0_train.csv'}
  inputs['fold_0_train.csv'].write_text('filename,ebird_code\n',
                                        encoding='utf-8')
  outputs = {'submission.csv': tmp_path / 'submission.csv'}
  outputs['submission.csv'].write_text('row_id,labels\n', encoding='utf-8')
  manifest = build_manifest(_parameters(),
                            config_path=config,
                            inputs=inputs,
                            outputs=outputs,
                            extra={'tracker': {'run_id': 'abc123',
                                               'url': 'https://wandb/run/abc'}})
  assert json.dumps(manifest)
  assert json.loads(json.dumps(manifest)) == manifest


def test_the_manifest_holds_the_resolved_configuration(tmp_path):
  '''The configuration is recorded flat, with the nested blocks.'''
  manifest = build_manifest(_parameters(lr=0.005))
  config = manifest['config']
  assert config['lr'] == 0.005
  assert config['model_config']['classes_num'] == 264
  assert config['criterion_params']['gamma'] == 0.0
  assert config['run_params']['max_epochs'] == 50
  assert config['checkpoint_params']['n_saved'] == 5
  # A path is a string in JSON.
  assert isinstance(config['checkpoint_params']['save_dir'], str)


def test_the_manifest_holds_the_seeds(tmp_path):
  '''Every seed is part of the record of the run.'''
  manifest = build_manifest(_parameters(seed=7, mixup_seed=99,
                                        deterministic=False))
  assert manifest['seeds'] == {'seed': 7, 'mixup_seed': 99,
                               'deterministic': False}


def test_the_manifest_holds_the_config_file_and_its_hash(tmp_path):
  '''The configuration file is named and hashed.'''
  config = tmp_path / 'experiment.yaml'
  config.write_text('name: test\n', encoding='utf-8')
  manifest = build_manifest(_parameters(), config_path=config)
  assert manifest['config_file']['path'] == str(config)
  assert manifest['config_file']['sha256'] == sha256_file(config)


def test_the_manifest_without_a_config_file_records_none(tmp_path):
  '''A run without a configuration file says so.'''
  manifest = build_manifest(_parameters())
  assert manifest['config_file'] is None


def test_the_manifest_holds_the_git_state_and_the_packages():
  '''The environment the run happened in is recorded.'''
  manifest = build_manifest(_parameters())
  assert set(manifest['git']) == {'commit', 'branch', 'dirty'}
  assert manifest['packages']['python']
  assert manifest['packages']['numpy']


def test_the_manifest_holds_the_input_and_output_hashes(tmp_path):
  '''The inputs and the outputs are hashed, not just named.'''
  fold = tmp_path / 'fold_0_train.csv'
  fold.write_text('filename,ebird_code\n', encoding='utf-8')
  submission = tmp_path / 'submission.csv'
  submission.write_text('row_id,labels\n', encoding='utf-8')
  manifest = build_manifest(_parameters(),
                            inputs={'fold_0_train.csv': fold},
                            outputs={'submission.csv': submission})
  assert manifest['inputs']['fold_0_train.csv'] == {
      'path': str(fold),
      'sha256': sha256_file(fold),
      'missing': False,
  }
  assert manifest['outputs']['submission.csv']['sha256'] == sha256_file(
      submission)


def test_the_manifest_holds_the_tracker_record():
  '''The tracker is a sink, but the manifest links to it.'''
  manifest = build_manifest(_parameters(),
                            extra={'tracker': {'run_id': 'abc123',
                                               'url': 'https://wandb/run/abc'}})
  assert manifest['tracker']['run_id'] == 'abc123'
  assert manifest['tracker']['url'] == 'https://wandb/run/abc'


def test_a_manifest_without_a_tracker_records_none():
  '''No tracker is a fact, not a gap.'''
  assert build_manifest(_parameters())['tracker'] is None


def test_the_manifest_converts_numpy_scalars():
  '''A numpy scalar in the extra payload is a plain number in the JSON.'''
  manifest = build_manifest(_parameters(),
                            extra={'value': np.float32(1.5),
                                   'count': np.int64(3)})
  assert manifest['value'] == pytest.approx(1.5)
  assert manifest['count'] == 3
  assert json.dumps(manifest)


def test_the_manifest_carries_the_timestamp_in_utc():
  '''The timestamp is ISO 8601 and in UTC.'''
  manifest = build_manifest(_parameters())
  assert manifest['timestamp'].endswith('+00:00')
  assert 'T' in manifest['timestamp']


def test_the_manifest_records_the_command_it_can_be_replayed_with():
  '''The command is what makes the manifest executable.'''
  manifest = build_manifest(_parameters(),
                            extra={'command': ['python', '-m',
                                               'birdcall.train_pl']})
  assert manifest['command'] == ['python', '-m', 'birdcall.train_pl']


def test_a_written_manifest_reads_back_equal(tmp_path):
  '''The written document is the document that was handed in.'''
  manifest = build_manifest(_parameters())
  path = write_manifest(manifest, tmp_path / 'nested' / 'manifest.json')
  assert path.exists()
  assert json.loads(path.read_text(encoding='utf-8')) == manifest


def test_a_written_manifest_is_indented(tmp_path):
  '''The manifest is meant to be read by a human as well.'''
  manifest = build_manifest(_parameters())
  path = write_manifest(manifest, tmp_path / 'manifest.json')
  assert '\n  ' in path.read_text(encoding='utf-8')

def test_write_run_manifest_writes_next_to_the_checkpoints(tmp_path):
  '''The run manifest lands beside the checkpoints and records the inputs.

  Args:
    tmp_path: The temporary directory of the test.
  '''
  save_dir = tmp_path / 'saved_models'
  parameters = _parameters()
  parameters.checkpoint_params.save_dir = save_dir
  path = write_run_manifest(parameters, config_path=None, mode='train')
  assert path == save_dir / 'manifest.json'
  manifest = json.loads(path.read_text(encoding='utf-8'))
  assert manifest['mode'] == 'train'
  assert manifest['seeds']['seed'] == parameters.seed
  assert manifest['config']['seed'] == parameters.seed
  assert 'label_space' in manifest['inputs']
  assert 'mel_matrix' in manifest['inputs']
  assert manifest['inputs']['label_space']['sha256']
  # A fold CSV that does not exist is reported as missing, not fatal.
  assert manifest['inputs']['train_csv']['missing'] is True
  assert manifest['inputs']['train_csv']['sha256'] is None
