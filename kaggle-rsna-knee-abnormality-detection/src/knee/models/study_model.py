#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Composite study model: series encoder + metadata fusion
+ MIL aggregator + head.

Assembled entirely from config ComponentSpecs by the caller (script or
Lightning module); this class owns only the forward composition.

Metadata is used twice by design:
1. injected into every series feature before aggregation (per-series routing),
2. mean-pooled over valid series and concatenated to the study embedding
   (global protocol context).
"""

from __future__ import annotations

import torch
from torch import nn

from knee.layers.aggregators import (
  AttentionMILAggregator,
  TransformerAggregator,
)
from knee.layers.metadata_encoder import MetadataEncoder
from knee.models.encoders import Monai3DEncoder, TimmSeriesEncoder

__all__ = ['KneeStudyModel', 'N_TARGETS']

N_TARGETS = 12


class KneeStudyModel(nn.Module):
  """Composite study classifier over multiple MRI series.

  Args:
      encoder: Per-series feature extractor (2.5D timm or MONAI 3D).
      aggregator: Masked MIL pooling over series embeddings.
      metadata_encoder: Acquisition-metadata MLP.
      n_targets: Number of sigmoid outputs (12 findings).
      head_dropout: Dropout before the final linear head.
  """

  def __init__(
    self,
    encoder: TimmSeriesEncoder | Monai3DEncoder,
    aggregator: AttentionMILAggregator | TransformerAggregator,
    metadata_encoder: MetadataEncoder,
    n_targets: int = N_TARGETS,
    head_dropout: float = 0.1,
  ) -> None:
    super().__init__()
    self.encoder = encoder
    self.aggregator = aggregator
    self.metadata_encoder = metadata_encoder
    # Additive per-series conditioning requires both branches to live in
    # the aggregator's input space; project the (small) metadata
    # embedding up to the encoder feature dim instead of forcing YAMLs
    # to duplicate encoder-specific widths in out_dim.
    feat_dim = int(getattr(self.aggregator, 'feat_dim', 0))
    if feat_dim <= 0:
      raise ValueError(
        'aggregator must expose feat_dim matching its encoder output; '
        'got '
        f'{type(self.aggregator).__name__} without a positive feat_dim'
      )
    self.meta_proj = nn.Linear(int(metadata_encoder.out_dim), feat_dim)
    head_dim = aggregator.out_dim + metadata_encoder.out_dim
    self.head = nn.Sequential(
      nn.Dropout(head_dropout), nn.Linear(head_dim, n_targets)
    )

  def forward(
    self,
    images: torch.Tensor,  # (B, S, C|1[, D], H, W)
    meta: torch.Tensor,  # (B, S, 5)
    series_mask: torch.Tensor | None = None,  # (B, S) bool; True = valid series
  ) -> torch.Tensor:
    """Returns logits (B, n_targets); apply sigmoid for probabilities."""
    b, s = images.shape[:2]
    flat = images.reshape(b * s, *images.shape[2:])
    feats = self.encoder(flat).reshape(b, s, -1)  # (B,S,D_enc)

    if series_mask is None:
      series_mask = images.new_ones(b, s, dtype=torch.bool)

    meta_emb = self.metadata_encoder(meta)  # (B,S,M)
    fused = feats + self.meta_proj(meta_emb)  # (B,S,D_enc), conditioned
    study_emb = self.aggregator(fused, series_mask)  # (B,H)

    mask_f = series_mask.unsqueeze(-1).to(meta_emb.dtype)
    pooled_meta = (meta_emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
    return self.head(torch.cat([study_emb, pooled_meta], dim=-1))
