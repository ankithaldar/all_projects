#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for the optimization-enhancement toolkit.

Torch-dependent modules are skipped gracefully when torch is absent so
the suite stays runnable in minimal CI containers.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip('torch')

from knee.losses.curriculum_weights import (  # noqa: E402  # pylint: disable=wrong-import-position
  ConfidenceRampCurriculum,
  DynamicLossWeighter,
)
from knee.optimizers.gradients import (  # noqa: E402  # pylint: disable=wrong-import-position
  AdaptiveClipping,
  GradientNoiseInjector,
  NormClipping,
  PercentileClipping,
)
from knee.optimizers.lookahead import (  # noqa: E402  # pylint: disable=wrong-import-position
  Lookahead,
)


def _linear_with_grad(value: float = 1e3):
  """Build a one-param linear module carrying a known gradient.

  Args:
      value: Scalar assigned to the parameter gradient.

  Returns:
      Module whose single weight grad equals ``value`` everywhere.
  """
  layer = torch.nn.Linear(2, 1)
  layer.weight.grad = torch.full_like(layer.weight, value)
  layer.bias.grad = torch.full_like(layer.bias, value)
  return layer


class TestClippingStrategies:
  """Gradient clipping strategy contracts."""

  def test_norm_clipping_caps_global_norm(self):
    clipper = NormClipping(max_norm=10.0)
    layer = _linear_with_grad(1000.0)
    metrics = clipper.clip(layer.parameters())
    norm_after = torch.norm(
      torch.stack(
        [p.grad.norm(2) for p in layer.parameters() if p.grad is not None]
      )
    )
    assert metrics['grad_norm'] > 10.0
    assert float(norm_after) <= 10.0 + 1e-4

  def test_percentile_clipping_damps_spikes(self):

    clipper = PercentileClipping(percentile=90.0, history_size=50)
    for v in [1.0] * 30:
      clipper.clip(_linear_with_grad(v).parameters())
    spike = _linear_with_grad(500.0)
    metrics = clipper.clip(spike.parameters())
    assert metrics['scale'] < 1.0

  def test_adaptive_clipping_warmup_then_adapts(self):
    clipper = AdaptiveClipping(multiplier=1.5, warmup_steps=3)
    for _ in range(3):
      clipper.clip(_linear_with_grad(2.0).parameters())
    metrics = clipper.clip(_linear_with_grad(1000.0).parameters())
    assert metrics['scale'] < 1.0


class TestGradientNoise:
  """Gradient noise injection behaviour."""

  def test_injection_perturbs_gradients(self):
    layer = _linear_with_grad(1.0)
    before = layer.weight.grad.clone()
    GradientNoiseInjector(eta=0.5).inject(layer.parameters())
    assert not torch.equal(before, layer.weight.grad)

  def test_invalid_eta_raises(self):
    with pytest.raises(ValueError):
      GradientNoiseInjector(eta=0.0)


class TestLookahead:
  """Lookahead slow-weight sync semantics."""

  def test_slow_weight_sync_every_k(self):
    param = torch.nn.Parameter(torch.zeros(()))
    inner = torch.optim.SGD([param], lr=1.0)
    opt = Lookahead(inner, k=2, alpha=0.5)
    for step in range(1, 5):
      param.grad = torch.ones(())
      opt.step()
      if step == 2:
        # fast=2 -> slow = 0*0.5 + 2*0.5 = 1; fast pulled back to slow
        assert param.item() == pytest.approx(1.0)

  def test_param_groups_shared_with_inner(self):
    inner = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    opt = Lookahead(inner, k=2)
    for group in opt.param_groups:
      group['lr'] = 0.9
    assert inner.param_groups[0]['lr'] == 0.9


class TestCurriculumAndDWA:
  """Curriculum floors and DWA weighting."""

  def test_confidence_ramp_schedule(self):
    curriculum = ConfidenceRampCurriculum(
      start_floor=0.9, hold_epochs=2, ramp_epochs=4
    )
    assert curriculum.weight_floor(0) == pytest.approx(0.9)
    assert curriculum.weight_floor(2) == pytest.approx(0.675)
    assert curriculum.weight_floor(6) == pytest.approx(0.0)
    assert curriculum.weight_floor(99) == pytest.approx(0.0)

  def test_dwa_weights_normalized(self):
    weighter = DynamicLossWeighter(initial_weights=(1.0, 1.0))
    w_sup, w_dis = weighter.update((0.5, 0.25))
    assert w_sup + w_dis == pytest.approx(2.0)
