#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the noise-floor harness (BLUEPRINT 11.0-1).

Covers the pure planning/scoring/aggregation layer plus the config
isolation guarantees; training itself is exercised by the selftest and
the Kaggle sweep stage, not here.
"""

# pytest fixtures + white-box state access mirror the other suites.
# pylint: disable=redefined-outer-name,protected-access

import csv
import json
import math
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from knee.engines.noise_floor import (
  STATE_NAME,
  SUMMARY_NAME,
  collect_run_result,
  config_for_run,
  format_discord,
  load_state,
  plan_runs,
  run_key,
  save_state,
  state_dir,
  summarize,
  write_summary_csv,
)

TARGETS = ['acl', 'meniscus', 'oa']


def _base_config(tmp_path):
  """Minimal composed-config stand-in for patching tests.

  Args:
      tmp_path: pytest temporary directory.

  Returns:
      Configuration dictionary shaped like the real composed experiment.
  """
  return {
    'experiment': {'name': 'mvp', 'seed': 42},
    'paths': {
      'checkpoint_dir': str(tmp_path / 'checkpoints'),
      'oof_dir': str(tmp_path / 'oof'),
    },
    'resume': {
      'enabled': True,
      'checkpoint_dataset_slug': 'rsna-knee-mvp-ckpt',
    },
    'run': {'folds': [0, 1, 2, 3, 4]},
    'session_time_budget_h': 11.5,
    'noise_floor': {
      'seeds': [42, 43, 44],
      'folds': [0],
      'dataset_slug': 'rsna-knee-mvp-nf',
      'budget_floor_h': 0.5,
    },
  }


def _write_oof(path, seed_shift):
  """Create an OOF CSV with deterministic, separable scores.

  Args:
      path: Destination ``oof_fold0.csv`` path.
      seed_shift: Offset added to class-0 probabilities so different
          "seeds" produce different (but valid) AUCs.
  """
  os.makedirs(os.path.dirname(path), exist_ok=True)
  rows = 40
  probs = np.zeros((rows, len(TARGETS)))
  targets = np.zeros((rows, len(TARGETS)))
  for row in range(rows):
    # Class 0: perfectly separable, shifted by seed_shift.
    probs[row, 0] = 0.05 + 0.9 * (row % 2) + seed_shift * 0.01
    targets[row, 0] = row % 2
    # Class 1: noisy-but-computable; seed_shift swaps two predictions so
    # different "seeds" yield genuinely different AUCs (AUC is invariant
    # to monotone shifts, hence the non-monotone perturbation).
    probs[row, 1] = 0.3 + 0.4 * ((row * 7 + 3) % 5) / 4.0
    targets[row, 1] = 1 if (row * 7 + 3) % 5 > 2 else 0
    # Class 2: single-outcome (degenerate) - must be excluded.
    probs[row, 2] = 0.5
    targets[row, 2] = 1
  if seed_shift != 0:
    probs[0, 1], probs[1, 1] = probs[1, 1], probs[0, 1]
  frame_rows = [
    {'StudyInstanceUID': f'uid_{i}', 'fold': 0} for i in range(rows)
  ]
  for col, name in enumerate(TARGETS):
    for i in range(rows):
      frame_rows[i][f'{name}_prob'] = float(probs[i, col])
      frame_rows[i][name] = int(targets[i, col])
  pd.DataFrame(frame_rows).to_csv(path, index=False)
  return probs, targets


class TestRunKey:
  """Key formatting contract."""

  def test_format(self):
    assert run_key(42, 0) == 'seed42_fold0'


class TestStateDir:
  """Sweep state staging directory."""

  def test_dedicated_dir_under_artifact_root(self, tmp_path):
    config = {'paths': {'artifact_dir': str(tmp_path)}}
    directory = state_dir(config)
    assert directory == os.path.join(str(tmp_path), 'noise_floor')
    assert directory != str(tmp_path)


class TestState:
  """State load/save round trip and corruption tolerance."""

  def test_missing_file_is_fresh(self, tmp_path):
    state = load_state(str(tmp_path / 'absent.json'))
    assert state == {'completed': {}}

  def test_corrupt_file_is_fresh(self, tmp_path):
    bad = tmp_path / STATE_NAME
    bad.write_text('{not json', encoding='utf-8')
    assert load_state(str(bad)) == {'completed': {}}

  def test_wrong_schema_is_fresh(self, tmp_path):
    bad = tmp_path / STATE_NAME
    bad.write_text('{"completed": []}', encoding='utf-8')
    assert load_state(str(bad)) == {'completed': {}}

  def test_round_trip(self, tmp_path):
    path = str(tmp_path / STATE_NAME)
    state = {'completed': {'seed1_fold0': {'macro_auc': 0.9}}}
    save_state(state, path)
    assert load_state(path)['completed']['seed1_fold0'] == {'macro_auc': 0.9}
    assert 'updated_utc' in load_state(path)


class TestPlanRuns:
  """Remaining-run planning."""

  def test_full_product_in_order(self):
    runs = plan_runs({'completed': {}}, [42, 43], [0, 1])
    assert runs == [(42, 0), (42, 1), (43, 0), (43, 1)]

  def test_completed_are_skipped(self):
    state = {
      'completed': {
        'seed42_fold0': {'macro_auc': 0.8},
        'seed43_fold0': {'macro_auc': 0.9},
      }
    }
    assert plan_runs(state, [42, 43, 44], [0]) == [(44, 0)]

  def test_all_done_yields_empty(self):
    state = {'completed': {'seed42_fold0': {'macro_auc': 0.8}}}
    assert plan_runs(state, [42], [0]) == []


class TestCollectRunResult:
  """OOF scoring matches a direct sklearn computation."""

  def test_scores_final_epoch_table(self, tmp_path):
    oof = str(tmp_path / 'oof' / 'oof_fold0.csv')
    probs, targets = _write_oof(oof, seed_shift=0)
    result = collect_run_result(oof, TARGETS)
    expected_class0 = roc_auc_score(targets[:, 0], probs[:, 0])
    assert result['per_class']['acl'] == pytest.approx(expected_class0)
    # Degenerate class (single outcome) excluded from the macro mean.
    assert math.isnan(result['per_class']['oa'])
    assert result['macro_auc'] == pytest.approx(
      (result['per_class']['acl'] + result['per_class']['meniscus']) / 2
    )

  def test_seed_shift_changes_score(self, tmp_path):
    first = str(tmp_path / 'a' / 'oof_fold0.csv')
    second = str(tmp_path / 'b' / 'oof_fold0.csv')
    _write_oof(first, seed_shift=0)
    _write_oof(second, seed_shift=-3)
    scores = [
      collect_run_result(path, TARGETS)['macro_auc'] for path in (first, second)
    ]
    assert scores[0] != scores[1]

  def test_missing_oof_raises(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      collect_run_result(str(tmp_path / 'gone.csv'), TARGETS)


class TestSummarize:
  """Gate statistics math."""

  def test_known_values(self):
    entries = [
      {'macro_auc': 0.80},
      {'macro_auc': 0.82},
      {'macro_auc': 0.84},
    ]
    stats = summarize(entries)
    assert stats['n'] == 3
    assert stats['mean'] == pytest.approx(0.82)
    # ddof=1 sample std of [0.80, 0.82, 0.84] is 0.02.
    assert stats['std'] == pytest.approx(0.02)
    assert stats['gate'] == pytest.approx(0.82 + 2 * 0.02)

  def test_single_run_has_zero_std(self):
    stats = summarize([{'macro_auc': 0.7}])
    assert stats['std'] == 0.0
    assert stats['gate'] == pytest.approx(0.7)

  def test_empty_is_nan(self):
    stats = summarize([])
    assert stats['n'] == 0
    assert math.isnan(stats['mean'])


class TestSummaryCsv:
  """Summary CSV layout."""

  def test_rows_and_aggregate_line(self, tmp_path):
    path = str(tmp_path / SUMMARY_NAME)
    entries = [
      {'seed': 42, 'fold': 0, 'macro_auc': 0.8},
      {'seed': 43, 'fold': 0, 'macro_auc': 0.84},
    ]
    write_summary_csv(entries, path)
    with open(path, encoding='utf-8') as handle:
      rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]['seed'] == '42'
    assert rows[0]['macro_auc'] == '0.800000'
    stats = summarize(entries)
    count = stats['n']
    std = stats['std']
    gate = stats['gate']
    expected_key = f'n={count} std={std:.6f} gate={gate:.6f}'
    aggregate = rows[-1]
    assert aggregate['seed'] == 'aggregate'
    assert aggregate['run_key'] == expected_key


class TestFormatDiscord:
  """Discord headline rendering."""

  def test_contains_gate(self):
    stats = {'n': 3, 'mean': 0.82, 'std': 0.01, 'gate': 0.84}
    line = format_discord(stats, [42, 43, 44], [0])
    assert 'n=3' in line
    assert '0.8200' in line
    assert 'gate' in line and '0.8400' in line


class TestConfigForRun:
  """Per-run isolation patches."""

  def test_patches_are_applied(self, tmp_path):
    config = _base_config(tmp_path)
    patched = config_for_run(config, 43, 0, remaining_budget_h=6.0)
    assert patched['experiment']['seed'] == 43
    assert patched['experiment']['name'] == 'mvp_seed43_fold0'
    assert patched['paths']['checkpoint_dir'].endswith(
      os.path.join('checkpoints', 'seed43')
    )
    # OOF lives inside the pushed fold dir so it travels with the
    # checkpoint dataset (session-local oof dirs would be lost).
    assert patched['paths']['oof_dir'].endswith(
      os.path.join('checkpoints', 'seed43', 'fold0')
    )
    assert patched['resume']['checkpoint_dataset_slug'] == (
      'rsna-knee-mvp-nf-s43'
    )
    assert patched['run']['folds'] == [0]
    assert patched['session_time_budget_h'] == pytest.approx(6.0)

  def test_input_untouched(self, tmp_path):
    config = _base_config(tmp_path)
    original = json.dumps(config, sort_keys=True)
    config_for_run(config, 43, 0, remaining_budget_h=6.0)
    assert json.dumps(config, sort_keys=True) == original

  def test_budget_floor_clamps(self, tmp_path):
    config = _base_config(tmp_path)
    patched = config_for_run(config, 44, 0, remaining_budget_h=0.01)
    assert patched['session_time_budget_h'] == pytest.approx(0.25)

  def test_budget_none_keeps_configured(self, tmp_path):
    config = _base_config(tmp_path)
    patched = config_for_run(config, 44, 0, remaining_budget_h=None)
    assert patched['session_time_budget_h'] == pytest.approx(11.5)

  def test_empty_slug_keeps_main(self, tmp_path):
    config = _base_config(tmp_path)
    config['noise_floor']['dataset_slug'] = ''
    patched = config_for_run(config, 44, 0)
    assert patched['resume']['checkpoint_dataset_slug'] == (
      'rsna-knee-mvp-ckpt'
    )
