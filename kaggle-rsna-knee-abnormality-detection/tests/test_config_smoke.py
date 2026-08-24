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


class TestDiscordNotifierDisabledByDefault:
  def test_no_webhook_is_noop(self):
    notifier = DiscordNotifier(webhook_url='')
    assert notifier.enabled is False
    assert notifier.send('hello') is False
