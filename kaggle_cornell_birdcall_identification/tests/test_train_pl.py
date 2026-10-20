#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the entry point.

The configuration is resolved, the backend selection is read and the
command line is parsed.  No run is started: the data of the shipped
experiments is not in this environment.
'''

from pathlib import Path

import pytest

from birdcall.config_params.backend import BackendSelection
from birdcall.config_params.loader import backend_selection_path
from birdcall.train_pl import build_callbacks
from birdcall.train_pl import build_logger
from birdcall.train_pl import main
from birdcall.train_pl import parse_args
from birdcall.train_pl import resolve_backend
from birdcall.train_pl import resolve_configuration

EXAMPLE_CONFIG = (Path(__file__).resolve().parent.parent / 'src' / 'birdcall' /
                  'config_params' / 'experiments' / 'example_config' /
                  'example_config.yaml')


def test_the_command_line_is_parsed():
  '''Every flag of the entry point is parsed.'''
  args = parse_args(['--config', 'run.yaml',
                     '--backend', 'pl',
                     '--mode', 'infer',
                     '--checkpoint', 'weights.ckpt',
                     '--output', 'artefacts'])
  assert args.config == 'run.yaml'
  assert args.backend == 'pl'
  assert args.mode == 'infer'
  assert args.checkpoint == 'weights.ckpt'
  assert args.output == 'artefacts'


def test_the_mode_defaults_to_training():
  '''A run trains unless it is told otherwise.'''
  assert parse_args(['--config', 'run.yaml']).mode == 'train'


def test_an_unknown_mode_is_refused():
  '''The mode is one of the three the entry point knows.'''
  with pytest.raises(SystemExit):
    parse_args(['--config', 'run.yaml', '--mode', 'predict'])


def test_the_configuration_is_resolved():
  '''The experiment file becomes a validated configuration.'''
  parameters = resolve_configuration(EXAMPLE_CONFIG)
  assert parameters.name == 'example_config'
  assert parameters.model_name == 'sed_dense121att'
  assert parameters.criterion_name == 'sed_scaled_pos_neg_focal_loss'
  assert parameters.track_metric == 'f1score_clip'
  assert parameters.run_params.max_epochs == 50


def test_an_unknown_configuration_is_a_clear_error():
  '''A configuration that does not exist says so.'''
  with pytest.raises(FileNotFoundError, match='Configuration file not found'):
    resolve_configuration('/tmp/not_a_configuration.yaml')


def test_the_packaged_backend_selection_is_read():
  '''The packaged file says which backends run.'''
  selection = resolve_backend()
  assert isinstance(selection, BackendSelection)
  assert selection.backend == 'both'
  assert selection.is_enabled('pl') is True
  assert selection.is_enabled('tf') is True


def test_the_backend_flag_overrides_the_packaged_file():
  '''The command line wins over the packaged selection.'''
  selection = resolve_backend('tf')
  assert selection.backend == 'tf'
  assert selection.is_enabled('pl') is False
  assert selection.is_enabled('tf') is True


def test_the_packaged_selection_file_exists():
  '''The packaged selection is where the loader says it is.'''
  assert backend_selection_path().exists()


def test_a_run_of_another_backend_does_nothing():
  '''The entry point steps aside when its backend is not selected.'''
  assert main(['--config', str(EXAMPLE_CONFIG), '--backend', 'tf']) == 0


def test_an_unknown_configuration_stops_the_run():
  '''The configuration is resolved before anything is built.'''
  with pytest.raises(FileNotFoundError):
    main(['--config', '/tmp/not_a_configuration.yaml',
          '--backend', 'pl'])


def test_the_vote_mode_hands_over_to_the_vote(monkeypatch, tmp_path):
  '''The vote is run by its own entry point, with the three flags it
  knows.'''
  import birdcall.run_vote

  called = {}

  def fake_main(argv):
    '''Records the arguments of the vote.

    Args:
      argv: The arguments the entry point built.

    Returns:
      The exit code of the vote.
    '''
    called['argv'] = argv
    return 0

  monkeypatch.setattr(birdcall.run_vote, 'main', fake_main)
  code = main(['--config', 'ensemble.yaml',
               '--mode', 'vote',
               '--output', str(tmp_path)])
  assert code == 0
  assert called['argv'] == [
      '--ensemble', 'ensemble.yaml',
      '--scores', str(tmp_path),
      '--output', str(tmp_path / 'submission.csv'),
  ]


def test_the_logger_follows_the_configuration():
  '''The logger of the run is the one the configuration names.'''
  parameters = resolve_configuration(EXAMPLE_CONFIG)
  assert type(build_logger(parameters)).__name__ == 'WandbLoggerAdapter'
  parameters.logger_name = 'print'
  assert type(build_logger(parameters)).__name__ == 'PrintLoggerAdapter'


def test_the_callbacks_of_a_run_are_built():
  '''The checkpoint callback is one of the callbacks of the run.'''
  parameters = resolve_configuration(EXAMPLE_CONFIG)
  callbacks = build_callbacks(parameters)
  names = [type(callback).__name__ for callback in callbacks]
  assert 'ModelCheckpoint' in names
  assert 'LogEveryN' in names
  assert 'DiscordCallback' in names
