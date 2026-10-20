#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the Keras entry point.

The command line is the original's, and the configuration is read from
either an experiment YAML file or a dotted module path.
'''

import argparse
from pathlib import Path

import pytest

from birdcall.config_params.loader import load_parameters
from birdcall.helpers.tf_seed import seed_everything
from birdcall.train_tf import build_callbacks
from birdcall.train_tf import build_logger
from birdcall.train_tf import build_parser
from birdcall.train_tf import load_config
from birdcall.train_tf import main
from birdcall.train_tf import run

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'src' / 'birdcall' / \
    'config_params' / 'experiments'


def test_the_parser_takes_the_original_arguments() -> None:
  '''The five arguments of the entry point are there.'''
  parser = build_parser()
  args = parser.parse_args(['--config', 'a.yaml', '--backend', 'tf',
                             '--mode', 'train', '--checkpoint', 'c.weights',
                             '--output', 'out'])
  assert args.config == 'a.yaml'
  assert args.backend == 'tf'
  assert args.mode == 'train'
  assert args.checkpoint == 'c.weights'
  assert args.output == 'out'


def test_the_config_is_required() -> None:
  '''The experiment is not optional.'''
  with pytest.raises(SystemExit):
    build_parser().parse_args([])


def test_an_unknown_backend_is_refused() -> None:
  '''The backend name is checked.'''
  with pytest.raises(SystemExit):
    build_parser().parse_args(['--config', 'a.yaml', '--backend', 'jax'])


def test_an_unknown_mode_is_refused() -> None:
  '''The mode name is checked.'''
  with pytest.raises(SystemExit):
    build_parser().parse_args(['--config', 'a.yaml', '--mode', 'predict'])


def test_a_yaml_experiment_is_loaded() -> None:
  '''The experiment YAML files are the configuration.'''
  path = CONFIG_DIR / 'final_sed' / 'final_sed_dense121_mix_fold0.yaml'
  config = load_config(str(path))
  assert config.model_name == 'sed_dense121att'
  assert config.criterion_name in (
      'focal_loss_standard', 'sed_removed_focal_loss',
      'sed_scaled_pos_neg_focal_loss', 'sed_scaled_pos_neg_focal_loss_augd',
      'bce_controlled_standard_loss')
  assert config.model_config.classes_num == 264


def test_a_dotted_module_path_is_loaded(tmp_path, monkeypatch) -> None:
  '''The original's dotted module path still works.'''
  module = tmp_path / 'fake_experiment.py'
  module.write_text('class Parameters:\n  name = "fake"\n',
                    encoding='utf-8')
  monkeypatch.syspath_prepend(str(tmp_path))
  config = load_config('fake_experiment')
  assert config.name == 'fake'


def test_a_module_without_parameters_is_refused() -> None:
  '''A module without a Parameters class is an error.'''
  with pytest.raises(ValueError, match='Parameters'):
    load_config('birdcall.config_params.loader')


def test_a_missing_file_is_refused() -> None:
  '''A configuration file that does not exist is an error.'''
  with pytest.raises((FileNotFoundError, ValueError)):
    load_config('/nonexistent/experiment.yaml')


def test_the_logger_follows_the_configuration() -> None:
  '''The tracker is the one the configuration names.'''
  config = load_config(str(CONFIG_DIR / 'final_sed' /
                           'final_sed_dense121_mix_fold0.yaml'))
  tracker = build_logger(config)
  assert tracker is not None
  assert tracker.log_every == config.logger_params.log_every


def test_the_callback_list_is_built() -> None:
  '''The callback list holds the tracker, the checkpoint and the hook.'''
  config = load_config(str(CONFIG_DIR / 'final_sed' /
                           'final_sed_dense121_mix_fold0.yaml'))
  callbacks = build_callbacks(config, Path('/tmp/birdcall-test'))
  assert len(callbacks) == 3
  names = [type(callback).__name__ for callback in callbacks]
  assert 'KeepBestNCallback' in names
  assert 'DiscordCallback' in names


def test_the_checkpoint_callback_is_ranked_by_the_tracked_metric() -> None:
  '''The monitor is the tracked metric of the configuration.'''
  config = load_config(str(CONFIG_DIR / 'final_sed' /
                           'final_sed_dense121_mix_fold0.yaml'))
  callbacks = build_callbacks(config, Path('/tmp/birdcall-test'))
  keeper = [c for c in callbacks
            if type(c).__name__ == 'KeepBestNCallback'][0]
  assert keeper.monitor == config.track_metric
  assert keeper.n_saved == config.checkpoint_params.n_saved


def test_the_entry_point_does_nothing_for_the_other_backend() -> None:
  '''The torch backend is not run by this entry point.'''
  path = CONFIG_DIR / 'final_sed' / 'final_sed_dense121_mix_fold0.yaml'
  assert main(['--config', str(path), '--backend', 'pl']) == 0


def test_the_seeding_is_the_shared_one() -> None:
  '''The run seeds every global generator from the configuration.'''
  seed = seed_everything(42, deterministic=False)
  assert isinstance(seed, int)
  assert seed == seed_everything(42, deterministic=False)


def write_wav(path, seconds: float = 1.0, seed: int = 1) -> None:
  '''Writes a synthetic wav file.

  Args:
    path: The path of the file.
    seconds: The length of the file, in seconds.
    seed: The seed of the generator.
  '''
  import numpy as np
  import soundfile as sf

  path.parent.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(seed)
  samples = rng.standard_normal(int(32000 * seconds)).astype(np.float32)
  sf.write(path, samples, 32000)


@pytest.mark.slow
def test_a_whole_epoch_runs(tmp_path) -> None:
  '''One epoch of the whole pipeline runs, and writes a checkpoint.'''
  import pandas as pd

  root = tmp_path / 'audio'
  rows = []
  for index, code in enumerate(['aldfly', 'ameavo']):
    write_wav(root / code / f'clip{index}.wav', seconds=1.0, seed=index)
    rows.append({'filename': f'clip{index}.mp3', 'ebird_code': code,
                 'secondary_labels': '[]'})
  csv = tmp_path / 'fold.csv'
  pd.DataFrame(rows).to_csv(csv, index=False)
  experiment = tmp_path / 'experiment.yaml'
  experiment.write_text(
      'modules:\n'
      '- model\n'
      '- augmentation\n'
      '- loss\n'
      '- optimizer\n'
      '- scheduler\n'
      '- data\n'
      '- metrics\n'
      '- logging\n'
      '- checkpoint\n'
      '- runtime\n'
      'overrides:\n'
      '  logger_name: print\n'
      '  train_bs: 2\n'
      '  valid_bs: 2\n'
      '  eval_every: 1\n'
      '  run_params:\n'
      '    max_epochs: 1\n'
      '  checkpoint_params:\n'
      f'    save_dir: {tmp_path / "out"}\n'
      f'  train_ds_params:\n'
      f'    root_dir: {root}\n'
      f'    csv_dir: {csv}\n'
      '    period: 1\n'
      f'  valid_ds_params:\n'
      f'    root_dir: {root}\n'
      f'    csv_dir: {csv}\n'
      '    period: 1\n'
      f'  test_ds_params:\n'
      f'    root_dir: {root}\n'
      f'    csv_dir: {csv}\n'
      '    period: 1\n',
      encoding='utf-8')
  config = load_config(str(experiment))
  assert run(config, mode='train', output_dir=tmp_path / 'out') == 0
  checkpoints = list((tmp_path / 'out').glob('*.keras'))
  assert len(checkpoints) == 1
