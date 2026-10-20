#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The Lightning trainer, the checkpoint callback and the resume helpers.

This module is the replacement of the original's nine ordering-sensitive
Ignite registrations.  Every knob of the original's control plane is
mapped here, and the two that Lightning decouples are decided explicitly:

* the original's ``eval_every`` was **two** knobs in one field: the
  validation frequency *and* the checkpoint frequency, because the
  checkpoint saver was attached to the evaluator's ``COMPLETED`` event.
  Lightning has one knob for the validation and one for the checkpoint,
  so the checkpoint is written after every validation, which is what the
  original did;
* the original's checkpoint was written every ``eval_every`` epochs.  The
  port keeps that cadence.

The monitor name is the one the module actually logs, ``valid_`` plus the
tracked metric.  A monitor that is never logged is the Lightning
equivalent of the original's ``score_function`` ``KeyError``: the
checkpoint callback simply never fires, and the run trains for fifty
epochs without a checkpoint.
'''

import logging
from pathlib import Path
from typing import Any

import lightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from birdcall.config_params.parameters import Parameters

LOGGER = logging.getLogger(__name__)

VALID_PREFIX = 'valid_'
DIST_KEYS = ('strategy', 'devices', 'accelerator')
'''The ``dist_params`` keys the trainer understands.'''


def track_metric_name(track_metric: str) -> str:
  '''Returns the name the tracked metric is logged under.

  Args:
    track_metric: The name of the tracked metric, for example
      ``f1score_clip``.

  Returns:
    The name of the logged metric, which is the bare name with the
    ``valid_`` prefix.
  '''
  if track_metric.startswith(VALID_PREFIX):
    return track_metric
  return f'{VALID_PREFIX}{track_metric}'


def dist_kwargs(dist_params: dict[str, Any]) -> dict[str, Any]:
  '''Maps the distributed parameters onto the trainer's keywords.

  Args:
    dist_params: The ``dist_params`` of the configuration.

  Returns:
    The trainer keywords, empty when the run is single process.

  Raises:
    ValueError: If a key is not one the trainer understands.
  '''
  unknown = sorted(set(dist_params) - set(DIST_KEYS))
  if unknown:
    raise ValueError(
        f'Unknown dist_params keys {unknown}. Valid keys: '
        f'{list(DIST_KEYS)}.')
  return {key: value for key, value in dist_params.items()
          if value is not None}


def build_model_checkpoint(parameters: Parameters) -> ModelCheckpoint:
  '''Builds the checkpoint callback of a run.

  Args:
    parameters: The validated configuration.

  Returns:
    The callback.  It keeps ``n_saved`` checkpoints, ranked by the
    tracked metric, and one ``last`` checkpoint.  Note that Lightning's
    ``save_top_k=0`` writes no ranked checkpoint at all, and
    ``save_top_k=-1`` keeps every one of them.
  '''
  mode = 'max' if parameters.metric_factor >= 0 else 'min'
  return ModelCheckpoint(
      dirpath=str(parameters.checkpoint_params.save_dir),
      filename=parameters.checkpoint_params.prefix_name,
      monitor=track_metric_name(parameters.track_metric),
      mode=mode,
      save_top_k=parameters.checkpoint_params.n_saved,
      save_last=True)


def build_trainer(parameters: Parameters, callbacks=None, logger=None,
                  **overrides) -> lightning.Trainer:
  '''Builds the trainer of a run.

  Args:
    parameters: The validated configuration.
    callbacks: The callbacks of the run, or ``None``.
    logger: The logger of the run, or ``None`` for no logger at all.
    **overrides: Trainer keywords that win over the configuration.

  Returns:
    The trainer.
  '''
  kwargs: dict[str, Any] = {
      'max_epochs': parameters.run_params.max_epochs,
      'accumulate_grad_batches': parameters.accumulation_steps,
      'gradient_clip_val': parameters.gradient_clip_val,
      'check_val_every_n_epoch': parameters.eval_every,
      'enable_progress_bar': parameters.add_pbar,
      'num_sanity_val_steps': 0,
      'default_root_dir': str(parameters.checkpoint_params.save_dir),
      'callbacks': callbacks,
      'logger': logger,
  }
  kwargs.update(dist_kwargs(parameters.dist_params))
  kwargs.update(overrides)
  return lightning.Trainer(**kwargs)


def load_weights_only(module: torch.nn.Module, checkpoint_path) -> None:
  '''Restores only the weights of a checkpoint.

  This is the original's ``load_model_only=True`` mode, which every
  shipped configuration selects: the ``model`` entry of the checkpoint is
  loaded and the optimiser is left at its initial state.

  Args:
    module: The module the weights are loaded into.
    checkpoint_path: The path of the checkpoint file.

  Raises:
    FileNotFoundError: If the checkpoint does not exist.
    KeyError: If the checkpoint holds no weights.
  '''
  path = Path(checkpoint_path)
  if not path.exists():
    raise FileNotFoundError(f'Checkpoint not found: {path}')
  # The original's checkpoint holds the optimiser state next to the
  # weights, so the safe unpickler cannot read it.
  checkpoint = torch.load(path, map_location='cpu', weights_only=False)
  if not isinstance(checkpoint, dict):
    raise KeyError(f'Checkpoint {path} is not a dictionary.')
  state = checkpoint.get('model')
  if state is None:
    # A checkpoint Lightning wrote itself keeps the weights under
    # 'state_dict'.
    state = checkpoint.get('state_dict')
  if state is None:
    raise KeyError(
        f'Checkpoint {path} holds no "model" entry, so there are no '
        f'weights to load. Its keys are {sorted(checkpoint)}.')
  module.load_state_dict(state)
  LOGGER.info('Restored the weights of %s.', path)


def resume_checkpoint_path(parameters: Parameters) -> Path | None:
  '''Returns the checkpoint a run resumes from.

  Args:
    parameters: The validated configuration.

  Returns:
    The path of the checkpoint file, or ``None`` when the run does not
    resume.  A directory holds the ``last`` checkpoint of the previous
    run, and the most recent ranked one otherwise.
  '''
  directory = parameters.checkpoint_dir
  if directory is None:
    return None
  path = Path(directory)
  if path.is_file():
    return path
  if not path.is_dir():
    return None
  last = path / 'last.ckpt'
  if last.is_file():
    return last
  candidates = sorted(path.glob('*.ckpt'))
  if not candidates:
    return None
  return max(candidates, key=lambda item: item.stat().st_mtime)
