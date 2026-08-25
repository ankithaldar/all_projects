#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Guard against Lightning hook-signature drift.

Lightning calls hooks with keyword arguments that changed across major
versions (configure_gradient_clipping grew gradient_clip_algorithm in
2.x). An override with a narrower signature crashes every optimizer
step at runtime, so this test asserts each overridden hook accepts at
least the base class's parameters -- no model instantiation required.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from lightning.pytorch import LightningModule

from knee.engines.study_lit_module import KneeStudyLitModule

_OVERRIDDEN = (
  'configure_gradient_clipping',
  'training_step',
  'validation_step',
  'on_before_optimizer_step',
)


def _param_names(func) -> set[str]:
  """Names of positional/keyword parameters of a function.

  Args:
      func: Callable to introspect.

  Returns:
      Set including 'self' for method parity comparisons.
  """
  return {
    name
    for name, param in inspect.signature(func).parameters.items()
    if param.kind
    not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
  }


@pytest.mark.parametrize('hook', _OVERRIDDEN)
class TestHookSignatures:
  def test_override_accepts_base_parameters(self, hook: str):
    ours = inspect.signature(getattr(KneeStudyLitModule, hook))
    base = inspect.signature(
      getattr(LightningModule, hook)
      if hasattr(LightningModule, hook)
      else _dummy_hook
    )
    missing = {
      name
      for name, param in base.parameters.items()
      if param.kind
      not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    } - set(ours.parameters)
    assert not missing, f'{hook} is missing base kwargs: {missing}'


def _dummy_hook(*args, **kwargs):  # pragma: no cover - placeholder only
  """Signature source when the base class lacks a hook.

  Args:
      *args: Ignored.
      **kwargs: Ignored.
  """


class TestGradClipForwarding:
  """No custom strategy -> Trainer values must reach clip_gradients."""

  def test_super_forwarding_receives_algorithm(self, monkeypatch):
    module = KneeStudyLitModule.__new__(KneeStudyLitModule)
    module.grad_clip = None
    captured = {}

    def fake_base(
      self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
      captured['val'] = gradient_clip_val
      captured['algo'] = gradient_clip_algorithm

    monkeypatch.setattr(
      LightningModule,
      'configure_gradient_clipping',
      fake_base,
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    KneeStudyLitModule.configure_gradient_clipping(
      module,
      optimizer,
      gradient_clip_val=10.0,
      gradient_clip_algorithm='norm',
    )
    assert captured == {'val': 10.0, 'algo': 'norm'}
