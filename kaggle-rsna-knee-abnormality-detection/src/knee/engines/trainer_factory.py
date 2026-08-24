#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Trainer factory: assembles a fully-configured Lightning Trainer.

Single place where the declarative ``train.trainer`` /
``train.callbacks`` / ``train.loggers`` ComponentSpecs become a live
Trainer. Fold-specific concerns (checkpoint directory, run naming,
fold tags) are injected here so experiment YAMLs stay fold-agnostic.

Design notes:
- Open/Closed: new callbacks, loggers, or precisions are YAML changes;
  only logger-specific runtime fields need an adapter entry below.
- NeptuneLogger entries whose credentials did not resolve (empty
  NEPTUNE_API_TOKEN) are silently dropped, keeping local runs
  zero-config while kernels with secrets get full tracking.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import lightning.pytorch as pl

from knee.config_params.loader import instantiate
from knee.config_params.schema import PathsConfig, TrainConfig


def _inject_logger_fields(
  spec_target: str, params: dict, run_name: str, fold: int, paths: PathsConfig
) -> dict:
  """Add runtime-only fields a specific logger implementation requires.

  Args:
      spec_target: Dotted class path of the logger spec.
      params: Mutable copy of YAML-declared logger params.
      run_name: Experiment name from the config.
      fold: Fold id used for versioning/tagging.
      paths: Path settings anchoring local log directories.

  Returns:
      Updated params dict ready for instantiation.
  """
  if spec_target.endswith('CSVLogger'):
    params.setdefault('save_dir', str(Path(paths.output_dir)))
    params.setdefault('name', run_name)
    params.setdefault('version', f'fold{fold}')
  elif spec_target.endswith('NeptuneLogger'):
    params.setdefault('name', run_name)
    tags = list(params.get('tags', []))
    params['tags'] = sorted({*tags, f'fold{fold}', run_name})
  elif spec_target.endswith('WandbLogger'):
    # Deterministic per-fold run id + resume='allow' => re-running the same
    # experiment (or resuming from a checkpoint) CONTINUES the same W&B run
    # instead of spawning a new one. log_model='all' mirrors every saved
    # checkpoint into a W&B 'model' artifact so crashes can resume from cloud.
    params.setdefault('name', run_name)
    params.setdefault('id', deterministic_run_id(run_name, fold))
    params.setdefault('resume', 'allow')
    params.setdefault('log_model', 'all')
    tags = list(params.get('tags', []))
    params['tags'] = sorted({*tags, f'fold{fold}'})
  return params


def deterministic_run_id(run_name: str, fold: int) -> str:
  """Stable per-fold W&B run identifier shared by logger and artifacts.

  Args:
      run_name: Experiment name from config.
      fold: Zero-based fold id.

  Returns:
      Deterministic id string, e.g. ``student_2p5d-fold3``.
  """
  return f'{run_name}-fold{fold}'


def _neptune_configured(params: dict) -> bool:
  """Check whether Neptune credentials resolved to usable values.

  Args:
      params: Logger params after env interpolation.

  Returns:
      True when both api_key and project are non-empty strings.
  """
  return bool(params.get('api_key')) and bool(params.get('project'))


def _wandb_configured(params: dict) -> bool:
  """Check whether W&B should activate.

  Args:
      params: Logger params after env interpolation.

  Returns:
      True when an api_key resolved, or when offline mode is requested.
  """
  return bool(params.get('api_key')) or params.get('mode') == 'offline'


def find_latest_checkpoint(
  ckpt_dir: str | Path, fold: int | None = None
) -> Path | None:
  """Locate the newest checkpoint for a fold, preferring ``last.ckpt``.

  ``last.ckpt`` (exact trainer state: optimizer, scheduler, callback
  state such as the EMA shadow) wins when present; otherwise the most
  recently modified matching checkpoint is returned.

  Args:
      ckpt_dir: Directory scanned for checkpoints.
      fold: Restrict to this fold's prefix; ``None`` scans all files.

  Returns:
      Path of the best resume candidate, or ``None`` when nothing exists.
  """
  directory = Path(ckpt_dir)
  if not directory.exists():
    return None
  last = directory / 'last.ckpt'
  if last.exists():
    return last
  pattern = f'fold{fold}-*.ckpt' if fold is not None else '*.ckpt'
  candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
  return candidates[-1] if candidates else None


def build_trainer(
  train_cfg: TrainConfig, paths: PathsConfig, fold: int, run_name: str
) -> pl.Trainer:
  """Build a Lightning Trainer for one cross-validation fold.

  Callbacks are instantiated from the config list; a
  ``ModelCheckpoint`` entry automatically receives a fold-scoped
  ``dirpath`` (``<checkpoints>/fold<N>/``, isolating ``last.ckpt`` per
  fold so interrupted kernels resume the RIGHT fold) and filename
  prefix unless it defines its own. Loggers become a Lightning logger
  *list*; unconfigured NeptuneLogger specs are dropped rather than
  raising.

  Args:
      train_cfg: Validated train section containing trainer, callback
          and logger ComponentSpecs.
      paths: Path settings; ``output_dir`` anchors all artifacts.
      fold: Zero-based fold id used for artifact scoping.
      run_name: Experiment name forwarded to every logger.

  Returns:
      A Trainer ready for ``.fit()`` / ``.predict()``.
  """
  callbacks: list[pl.Callback] = []
  ckpt_dir = Path(paths.output_dir) / paths.checkpoints_dir
  fold_ckpt_dir = ckpt_dir / f'fold{fold}'
  fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
  for spec in train_cfg.callbacks:
    params = dict(spec.params)
    if spec.target.endswith('ModelCheckpoint'):
      params.setdefault('dirpath', str(fold_ckpt_dir))
      params['filename'] = f'fold{fold}-' + str(
        params.get('filename', '{epoch}')
      )
    callbacks.append(instantiate(spec, **params))

  loggers: list = []
  for spec in train_cfg.loggers:
    params = _inject_logger_fields(
      spec.target, dict(spec.params), run_name, fold, paths
    )
    if spec.target.endswith('NeptuneLogger') and not _neptune_configured(
      params
    ):
      continue  # no token in env/.env/Kaggle Secrets: skip silently
    loggers.append(instantiate(spec, **params))

  trainer_params = dict(train_cfg.trainer.params)
  if train_cfg.epochs is not None:
    trainer_params['max_epochs'] = int(train_cfg.epochs)
  # Kaggle kernels die hard at the 12 h wall; Lightning's max_time stops
  # cleanly, running checkpoint/early-stop callbacks one last time. An
  # explicit YAML max_time always wins over the shorthand budget.
  if (
    train_cfg.time_budget_hours is not None and 'max_time' not in trainer_params
  ):
    trainer_params['max_time'] = timedelta(
      hours=float(train_cfg.time_budget_hours)
    )
  trainer_params.update(
    default_root_dir=str(paths.output_dir),
    callbacks=callbacks,
    logger=loggers or None,
  )
  return instantiate(train_cfg.trainer, **trainer_params)
