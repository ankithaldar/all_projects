#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Series-level encoder: shared 2D backbone + temporal attention pooling."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from knee.layers.pooling import AttentionPool2d
from knee.models.backbones import TimmBackbone


class SeriesEncoder(nn.Module):
  """Encode every series of a study batch into one token per series.

  All slices from all series/studies are flattened into a single backbone
  pass (throughput-critical on T4s), then regrouped by caller-provided
  counts and pooled per series with :class:`AttentionPool2d`.
  """

  def __init__(
    self,
    backbone: TimmBackbone,
    slice_pool: AttentionPool2d,
    series_meta_dim: int = 3,
  ) -> None:
    """Compose the encoder.

    Args:
        backbone: Configured timm feature extractor.
        slice_pool: Pooling head collapsing slice embeddings.
        series_meta_dim: Width of the per-series feature vector
            (plane index + fluid-sensitive + fat-suppression).
    """
    super().__init__()
    self.backbone = backbone
    embed_dim = backbone.embed_dim
    if slice_pool.attention.embed_dim != embed_dim:
      raise ValueError(
        f'slice_pool.embed_dim={slice_pool.attention.embed_dim} '
        f'must match backbone.embed_dim={embed_dim}'
      )
    self.slice_pool = slice_pool
    self.series_projection = nn.Linear(series_meta_dim, embed_dim)
    self.norm = nn.LayerNorm(embed_dim)

  def forward(
    self,
    slices: Tensor,
    slice_counts: Tensor,
    series_meta: Tensor,
  ) -> tuple[Tensor, Tensor]:
    """Encode flattened slices into grouped series tokens.

    Args:
        slices: ``(total_slices, channels, height, width)`` float tensor;
            padded slots must be zeros.
        slice_counts: ``(batch, max_series)`` real-slice counts per series.
        series_meta: ``(batch, max_series, series_meta_dim)`` features;
            padded entries must be zeros.

    Returns:
        Tuple of ``(series_tokens (batch, max_series, embed_dim),
        series_mask (batch, max_series))`` where mask marks real series.
    """
    batch, max_series = slice_counts.shape
    flat_embeddings = self.backbone(slices)
    tokens = flat_embeddings.new_zeros(
      (batch * max_series, flat_embeddings.shape[-1])
    )
    offsets = torch.cumsum(slice_counts.reshape(-1), dim=0).tolist()
    start = 0
    pooled_rows = []
    for row_end in offsets:
      count = row_end - start
      if count > 0:
        group = flat_embeddings[start:row_end].unsqueeze(0)
        pooled_rows.append(self.slice_pool(group))
      else:
        pooled_rows.append(None)
      start = row_end
    for idx, pooled in enumerate(pooled_rows):
      if pooled is not None:
        tokens[idx] = pooled[0]
    tokens = tokens.view(batch, max_series, -1)
    meta_embedding = self.series_projection(series_meta.to(tokens.dtype))
    tokens = self.norm(tokens + meta_embedding)
    mask = slice_counts > 0
    return tokens, mask
