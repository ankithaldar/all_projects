#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shape/contract tests for KneeNet with a small offline backbone."""

import pytest
import torch

from knee.layers.pooling import AttentionPool2d, StudyAggregator
from knee.models.knee_net import KneeNet

torch.manual_seed(0)


@pytest.fixture(scope='module')
def model() -> KneeNet:
  """Build a tiny KneeNet without pretrained weights, in eval mode."""
  net = KneeNet(
      backbone_name='resnet18',
      img_size=64,
      n_slices=4,
      n_series_tokens_max=3,
      n_targets=12,
      metadata_dim=12,
      slice_pool=AttentionPool2d(embed_dim=512, num_heads=4),
      study_aggregator=StudyAggregator(embed_dim=512, num_heads=4, ff_mult=2),
      trunk_hidden=128,
      pretrained=False,
  )
  net.eval()
  return net


def make_batch(batch: int = 2) -> dict:
  """Create a collated batch matching the dataset contract.

  Args:
      batch: Number of studies.

  Returns:
      Dictionary of tensors per KneeNet.forward expectations.
  """
  return {
      'slices': torch.randn(batch * 3 * 4, 3, 64, 64),   # 3 series x 4 slices
      'slice_counts': torch.tensor([[4, 4, 4], [4, 0, 0]]),
      'series_meta': torch.randn(batch, 3, 3),
      'metadata': torch.randn(batch, 12),
  }


class TestKneeNet:
  """Forward and grouping contracts."""

  def test_output_shape(self, model):
    logits = model(make_batch())
    assert logits.shape == (2, 12)

  def test_masked_series_ignored(self, model):
    batch = make_batch()
    with torch.no_grad():
      logits_full = model(batch)
      # Garbage in padded slots of the second study must not leak.
      polluted = {k: v for k, v in batch.items()}
      polluted['series_meta'] = batch['series_meta'].clone()
      polluted['series_meta'][1, 1:, :] = -99.0
      with torch.no_grad():
        logits_again = model(polluted)
    assert torch.allclose(logits_full[0], logits_again[0])
    assert logits_full.shape == logits_again.shape

  def test_parameter_groups_split(self, model):
    groups = model.parameter_groups(0.1)
    head_ids = {id(p) for p in model.head.parameters()}
    backbone_ids = {id(p) for p in model.series_encoder.backbone.parameters()}
    assert head_ids.issubset({id(p) for p in groups[0]['params']})
    assert backbone_ids.issubset({id(p) for p in groups[1]['params']})


class TestStudyAggregator:
  """Padding-mask behavior of the aggregator."""

  def test_padding_does_not_leak(self):
    aggregator = StudyAggregator(embed_dim=16, num_heads=2).eval()
    tokens = torch.randn(1, 3, 16)
    mask = torch.tensor([[True, False, False]])
    with torch.no_grad():
      out_real = aggregator(tokens, mask)
      tokens_padded = tokens.clone()
      tokens_padded[:, 1:, :] = 99.0  # garbage in padded slots
      out_garbage = aggregator(tokens_padded, mask)
    assert torch.allclose(out_real, out_garbage, atol=1e-5)
