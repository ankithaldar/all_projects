#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke tests for the config system and core contracts.

Deliberately dependency-light (no torch/timm imports) so the suite runs
anywhere, including CI containers without GPU stacks installed.
"""

from __future__ import annotations

import pydantic
import pytest

from knee.config_params.loader import ComponentSpec, instantiate
from knee.config_params.schema import TARGETS, ExperimentConfig
from knee.loggers.discord_notifier import DiscordNotifier


def _minimal_config_dict() -> dict:
  """Build the smallest valid ExperimentConfig payload.

  Returns:
      Dict satisfying every required field of the root schema.
  """
  return {
    'name': 'smoke',
    'model': {
      'encoder': {'target': 'collections.OrderedDict', 'params': {}},
      'aggregator': {'target': 'collections.OrderedDict', 'params': {}},
      'metadata_encoder': {'target': 'collections.OrderedDict', 'params': {}},
    },
    'loss': {'criterion': {'target': 'collections.OrderedDict', 'params': {}}},
    'optimizer': {
      'optimizer': {'target': 'collections.OrderedDict', 'params': {'lr': 1e-3}}
    },
  }


class TestTargetsContract:
  """The submission column order must never drift."""

  def test_twelve_unique_targets(self):
    assert len(TARGETS) == 12
    assert len(set(TARGETS)) == 12

  def test_expected_names(self):
    assert TARGETS[0] == 'ACL'
    assert TARGETS[-1] == 'Fracture'


class TestExperimentConfig:
  """Root-schema validation behaviour."""

  def test_minimal_config_validates(self):
    cfg = ExperimentConfig.model_validate(_minimal_config_dict())
    assert cfg.name == 'smoke'
    assert cfg.train.n_folds == 5

  def test_unknown_key_rejected(self):
    bad = _minimal_config_dict()
    bad['typo_section'] = {}
    with pytest.raises(ValueError):
      ExperimentConfig.model_validate(bad)


class TestInstantiate:
  """ComponentSpec factory resolution semantics."""

  def test_resolves_dotted_path(self):
    spec = ComponentSpec(target='collections.OrderedDict', params={'b': 2})
    obj = instantiate(spec, a=1)
    assert obj['a'] == 1 and obj['b'] == 2

  def test_kwargs_override_params(self):
    spec = ComponentSpec(target='collections.OrderedDict', params={'x': 1})
    assert instantiate(spec, x=9)['x'] == 9

  def test_rejects_bare_name(self):
    with pytest.raises((ImportError, pydantic.ValidationError)):
      instantiate(ComponentSpec(target='OrderedDict'))


class TestKaggleBudgetFields:
  """Streaming/LRU + time-budget knobs added for Kaggle's limits."""

  def test_datamodule_defaults_streaming_safe(self):
    cfg = ExperimentConfig.model_validate(_minimal_config_dict())
    assert cfg.datamodule.lru_max_volumes >= 1
    assert cfg.datamodule.lru_max_gb >= 1

  def test_datamodule_defaults_fit_kaggle_ram(self):
    """Host-RAM envelope: batch bytes x in-flight must stay bounded.

    batch ~ bs x max_series x in_chans x image_size^2 x 4 B; loader
    holds ~(workers x prefetch + 1) batches plus model state. Kaggle
    GPU pods have ~13-15 GB shared with any DDP ranks.
    """
    cfg = ExperimentConfig.model_validate(_minimal_config_dict())
    dm = cfg.datamodule
    batch_mb = (
      dm.batch_size
      * dm.max_series_per_study
      * cfg.data.in_chans
      * cfg.data.image_size**2
      * 4
      / 1024**2
    )
    in_flight = dm.num_workers * dm.prefetch_factor + 2  # +main+staging
    total_gb = batch_mb * in_flight / 1024 + dm.num_workers * dm.lru_max_gb
    assert total_gb < 8, f'defaults would use ~{total_gb:.1f} GB host RAM'
    # Single-device default: DDP doubles every host-side pipeline.
    assert cfg.train.trainer.params.get('devices') == 1

  def test_time_budget_rejects_nonpositive(self):
    payload = _minimal_config_dict()
    payload['train'] = {'time_budget_hours': 0}
    with pytest.raises(ValueError):
      ExperimentConfig.model_validate(payload)

  def test_time_budget_optional(self):
    cfg = ExperimentConfig.model_validate(_minimal_config_dict())
    assert cfg.train.time_budget_hours is None


class TestDiscordNotifierDisabledByDefault:
  def test_no_webhook_is_noop(self):
    notifier = DiscordNotifier(webhook_url='')
    assert notifier.enabled is False
    assert notifier.send('hello') is False
