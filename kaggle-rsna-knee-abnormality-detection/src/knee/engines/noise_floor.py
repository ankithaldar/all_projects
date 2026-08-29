#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Noise-floor harness (BLUEPRINT 11.0-1).

Trains the SAME config at several seeds on a fixed fold list and measures
the spread of the final-epoch OOF macro-AUC. The resulting
``mean + 2 * std`` gate is the keep/drop threshold every backlog
experiment must beat before being adopted.

The sweep is session-safe: progress lives in ``noise_floor_state.json``
(pushed alongside the summary through the artifact-sync protocol), each
seed trains in an ISOLATED checkpoint/oof directory and pushes to its own
Kaggle dataset so the main experiment's checkpoints are never touched,
and an interrupted run resumes through the standard last.ckpt protocol.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import time

import numpy as np
import pandas as pd

from knee.helpers.utils import get_logger
from knee.metrics.auc import MultilabelAUC

_LOGGER = get_logger(__name__)

STATE_NAME = 'noise_floor_state.json'
SUMMARY_NAME = 'noise_floor_summary.csv'
PROB_SUFFIX = '_prob'


def run_key(seed: int, fold: int) -> str:
  """Canonical identifier for one (seed, fold) training run.

  Args:
      seed: Experiment seed of the run.
      fold: Cross-validation fold of the run.

  Returns:
      String key such as ``seed42_fold0``.
  """
  return f'seed{int(seed)}_fold{int(fold)}'


def load_state(path: str) -> dict:
  """Read sweep state; a missing or corrupt file means a fresh sweep.

  Args:
      path: Location of ``noise_floor_state.json``.

  Returns:
      State dictionary with a ``completed`` mapping keyed by ``run_key``.
  """
  if not os.path.exists(path):
    return {'completed': {}}
  try:
    with open(path, encoding='utf-8') as handle:
      state = json.load(handle)
    if not isinstance(state.get('completed'), dict):
      raise ValueError('completed section is not a mapping')
    return state
  except (OSError, ValueError, json.JSONDecodeError) as exc:
    _LOGGER.warning(
      'unreadable noise-floor state %s (%s); restarting', path, exc
    )
    return {'completed': {}}


def save_state(state: dict, path: str) -> None:
  """Persist sweep state atomically enough for ephemeral kernels.

  Args:
      state: State dictionary to write.
      path: Destination file path.
  """
  state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
  os.makedirs(os.path.dirname(path), exist_ok=True)
  temporary = f'{path}.tmp'
  with open(temporary, 'w', encoding='utf-8') as handle:
    json.dump(state, handle, indent=2, sort_keys=True)
  os.replace(temporary, path)


def plan_runs(
  state: dict, seeds: list[int], folds: list[int]
) -> list[tuple[int, int]]:
  """List the (seed, fold) runs still missing from the state.

  Args:
      state: Loaded sweep state with completed runs.
      seeds: Seed values requested by config (order defines priority).
      folds: Fold values requested by config.

  Returns:
      Ordered (seed, fold) tuples not yet completed.
  """
  completed = state.get('completed', {})
  return [
    (int(seed), int(fold))
    for seed in seeds
    for fold in folds
    if run_key(seed, fold) not in completed
  ]


def collect_run_result(oof_csv: str, target_columns: list[str]) -> dict:
  """Score a finished run's final-epoch OOF predictions.

  The training module rewrites ``oof_fold{k}.csv`` on every validation
  epoch, so the file on disk after the ``done`` marker holds the LAST
  epoch's predictions - exactly the quantity the noise floor needs.

  Args:
      oof_csv: Path to the run's ``oof_fold{k}.csv``.
      target_columns: Canonical 12-target order.

  Returns:
      Dictionary with ``macro_auc`` plus per-target AUCs under ``per_class``.

  Raises:
      FileNotFoundError: When the OOF file is absent.
  """
  if not os.path.exists(oof_csv):
    raise FileNotFoundError(oof_csv)
  frame = pd.read_csv(oof_csv)
  probs = frame[[f'{c}{PROB_SUFFIX}' for c in target_columns]].to_numpy()
  targets = frame[target_columns].to_numpy()
  metric = MultilabelAUC(target_columns)
  metric.update(probs, targets, frame['StudyInstanceUID'].astype(str).tolist())
  summary = metric.summary()
  per_class = {
    name[len('auc/') :]: value
    for name, value in summary.items()
    if name != 'auc/macro'
  }
  return {'macro_auc': summary['auc/macro'], 'per_class': per_class}


def summarize(completed: list[dict]) -> dict:
  """Aggregate finished runs into the noise-floor gate statistics.

  Args:
      completed: Entries carrying ``macro_auc`` floats.

  Returns:
      Dictionary with ``n``, ``mean``, ``std`` (ddof=1) and ``gate``
      (``mean + 2 * std``); std is 0.0 for fewer than two runs and the
      whole summary is empty when nothing completed yet.
  """
  values = [
    float(entry['macro_auc'])
    for entry in completed
    if np.isfinite(entry.get('macro_auc', float('nan')))
  ]
  if not values:
    return {
      'n': 0,
      'mean': float('nan'),
      'std': float('nan'),
      'gate': float('nan'),
    }
  array = np.asarray(values, dtype=np.float64)
  std = float(array.std(ddof=1)) if array.size > 1 else 0.0
  mean = float(array.mean())
  return {
    'n': int(array.size),
    'mean': mean,
    'std': std,
    'gate': mean + 2.0 * std,
  }


def write_summary_csv(completed: list[dict], path: str) -> None:
  """Write one row per completed run plus the aggregate statistics.

  Args:
      completed: State entries with seed/fold/macro_auc fields.
      path: Destination CSV path.
  """
  os.makedirs(os.path.dirname(path), exist_ok=True)
  stats = summarize(completed)
  columns = ['seed', 'fold', 'macro_auc', 'run_key']
  with open(path, 'w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    for entry in completed:
      macro = float(entry['macro_auc'])
      writer.writerow(
        {
          'seed': entry['seed'],
          'fold': entry['fold'],
          'macro_auc': f'{macro:.6f}',
          'run_key': entry.get(
            'run_key', run_key(entry['seed'], entry['fold'])
          ),
        }
      )
    count = stats['n']
    std = stats['std']
    gate = stats['gate']
    aggregate_key = f'n={count} std={std:.6f} gate={gate:.6f}'
    mean = stats['mean']
    writer.writerow(
      {
        'seed': 'aggregate',
        'fold': '',
        'macro_auc': f'{mean:.6f}',
        'run_key': aggregate_key,
      }
    )


def format_discord(stats: dict, seeds: list[int], folds: list[int]) -> str:
  """Render the noise-floor headline for Discord.

  Args:
      stats: Output of :func:`summarize`.
      seeds: Requested seeds (for the sweep scope line).
      folds: Requested folds.

  Returns:
      Human-readable one-liner with the keep/drop gate.
  """
  count = stats['n']
  mean = stats['mean']
  std = stats['std']
  gate = stats['gate']
  return (
    f'noise floor: n={count} '
    f'(seeds={list(seeds)}, folds={list(folds)}) '
    f'macro-AUC mean {mean:.4f} +/- {std:.4f} | '
    f'keep/drop gate (mean + 2*std) = {gate:.4f}'
  )


def config_for_run(
  config: dict,
  seed: int,
  fold: int,
  remaining_budget_h: float | None = None,
) -> dict:
  """Derive the isolated per-run configuration (deep copy; input untouched).

  Patches applied:

  * ``experiment.seed`` -> the sweep seed; ``experiment.name`` gains a
    ``_seed<k>`` suffix so CSV/W&B/Discord streams never collide.
  * ``paths.checkpoint_dir`` / ``paths.oof_dir`` gain a ``seed<k>/``
    segment - seeds must never resume each other's checkpoints.
  * ``resume.checkpoint_dataset_slug`` -> the per-seed slug
    ``<noise_floor.dataset_slug>-s<seed>`` so every seed owns an
    isolated Kaggle dataset and the main experiment's checkpoint slug
    stays pristine.
  * ``run.folds`` -> ``[fold]``.
  * ``session_time_budget_h`` shrinks to the remaining kernel budget so
    the per-run TimeBudgetCallback honours the WHOLE sweep session, not
    just the current fit.

  Args:
      config: Composed experiment configuration.
      seed: Seed for this run.
      fold: Fold for this run.
      remaining_budget_h: Remaining sweep-session hours; None keeps the
          configured budget unchanged.

  Returns:
      Patched deep copy of the configuration.
  """
  patched = copy.deepcopy(config)
  nf_cfg = config.get('noise_floor', {})
  base_name = config['experiment']['name']
  patched['experiment']['seed'] = int(seed)
  patched['experiment']['name'] = f'{base_name}_{run_key(seed, fold)}'
  for path_key in ('checkpoint_dir', 'oof_dir'):
    patched['paths'][path_key] = os.path.join(
      config['paths'][path_key], f'seed{int(seed)}'
    )
  slug = str(nf_cfg.get('dataset_slug', ''))
  if slug:
    patched['resume']['checkpoint_dataset_slug'] = f'{slug}-s{int(seed)}'
  patched['run']['folds'] = [int(fold)]
  if remaining_budget_h is not None:
    patched['session_time_budget_h'] = max(0.25, float(remaining_budget_h))
  return patched
