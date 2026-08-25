#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shape-contract tests for the composite study model.

Regression: additive metadata conditioning used to require the YAML
metadata out_dim to equal the encoder feature width (1280 vs 64 crash
on the first validation batch). A learned projection now bridges them.
"""

from __future__ import annotations

import pytest
import torch

from knee.layers.aggregators import (
  AttentionMILAggregator,
  TransformerAggregator,
)
from knee.layers.metadata_encoder import MetadataEncoder
from knee.models.study_model import KneeStudyModel


class _TinyEncoder(torch.nn.Module):
  """Stand-in for a per-series encoder with a fixed output width."""

  def __init__(self, feat_dim: int = 8) -> None:
    super().__init__()
    self.feat_dim = feat_dim
    self.proj = torch.nn.Linear(3 * 16 * 16, feat_dim)

  def forward(self, flat: torch.Tensor) -> torch.Tensor:
    return self.proj(flat.flatten(1))


def _build(aggregator, meta_out: int = 64) -> KneeStudyModel:
  """Assemble the model exactly as the factory would from YAML.

  Args:
      aggregator: Real aggregator instance carrying feat_dim.
      meta_out: Metadata embedding width (base.yaml uses 64).

  Returns:
      Ready-to-call KneeStudyModel on CPU.
  """
  return KneeStudyModel(
    encoder=_TinyEncoder(feat_dim=getattr(aggregator, 'feat_dim', 8)),
    aggregator=aggregator,
    metadata_encoder=MetadataEncoder(out_dim=meta_out),
    n_targets=12,
  )


def _batch(studies: int = 2, series: int = 4):
  """Synthetic batch honoring the datamodule contract.

  Args:
      studies: Batch size B.
      series: Series slots S.

  Returns:
      (images, meta, mask) tensors.
  """
  images = torch.randn(studies, series, 3, 16, 16)
  meta = torch.randn(studies, series, 5)
  mask = torch.tensor([[True] * (series - 1) + [False]] * studies)
  return images, meta, mask


@pytest.mark.parametrize(
  'aggregator',
  [
    AttentionMILAggregator(feat_dim=1280, hidden_dim=32),
    TransformerAggregator(
      feat_dim=1280, hidden_dim=32, num_layers=1, num_heads=4
    ),
  ],
  ids=['attention-mil', 'transformer'],
)
class TestKneeStudyModelShapes:
  def test_metadata_width_need_not_match_encoder(self, aggregator):
    """The original crash: meta out 64 added to encoder feats 1280."""
    model = _build(aggregator, meta_out=64)
    images, meta, mask = _batch()
    logits = model(images, meta, mask)
    assert tuple(logits.shape) == (2, 12)

  def test_masked_series_do_not_contribute(self, aggregator):
    torch.manual_seed(0)
    model = _build(aggregator).eval()
    images, meta, _ = _batch()
    studies, series = images.shape[:2]
    full_mask = torch.ones(studies, series, dtype=torch.bool)
    partial_mask = full_mask.clone()
    partial_mask[:, -1] = False
    with torch.no_grad():
      full = model(images, meta, full_mask)
      partial = model(images, meta, partial_mask)
    assert not torch.allclose(full, partial)


class TestAggregatorFeatDim:
  def test_aggregators_expose_feat_dim(self):
    assert AttentionMILAggregator(feat_dim=768).feat_dim == 768
    assert TransformerAggregator(feat_dim=2048).feat_dim == 2048

  def test_missing_feat_dim_raises_clear_error(self):
    class _Broken(torch.nn.Module):
      out_dim = 8

      def forward(self, feats, mask):  # pragma: no cover - unreachable
        raise AssertionError

    with pytest.raises(ValueError, match='feat_dim'):
      KneeStudyModel(
        encoder=torch.nn.Identity(),
        aggregator=_Broken(),
        metadata_encoder=MetadataEncoder(out_dim=6),
        n_targets=12,
      )
