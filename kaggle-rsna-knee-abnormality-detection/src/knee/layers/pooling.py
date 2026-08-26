#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pooling and aggregation layers for the hierarchical knee network.

Hierarchy implemented here:

* :class:`AttentionPool2d` collapses variable-length slice embeddings of one
  series into a single series token.
* :class:`StudyAggregator` lets a learnable study query cross-attend over
  series tokens, tolerating missing series through key-padding masks.
* :class:`FiLM` injects tabular metadata as feature-wise affine conditioning.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AttentionPool2d(nn.Module):
    """Query-conditioned attention pooling over a sequence of embeddings."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        """Configure the pooling head.

        Args:
            embed_dim: Feature dimension of input embeddings.
            num_heads: Attention heads; ``embed_dim`` must be divisible.
            dropout: Dropout applied inside attention.
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """Pool a batch of sequences.

        Args:
            tokens: ``(batch, seq_len, embed_dim)`` embeddings.

        Returns:
            ``(batch, embed_dim)`` pooled representations.
        """
        queries = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(queries, tokens, tokens, need_weights=False)
        return self.norm(pooled.squeeze(1))


class StudyAggregator(nn.Module):
    """Single-layer cross-attention aggregator producing a study token.

    A learned query attends over available series tokens; absent series are
    masked out so variable protocol composition never leaks padding signal.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Configure the aggregator.

        Args:
            embed_dim: Series-token dimension.
            num_heads: Cross-attention heads.
            ff_mult: Hidden expansion factor of the feed-forward block.
            dropout: Dropout used in attention and feed-forward blocks.
        """
        super().__init__()
        self.study_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden = embed_dim * ff_mult
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, series_tokens: Tensor, series_mask: Tensor) -> Tensor:
        """Aggregate series tokens into study representations.

        Args:
            series_tokens: ``(batch, max_series, embed_dim)`` tokens.
                Padded entries must be zeros.
            series_mask: ``(batch, max_series)`` boolean mask, True marking
                real (non-padded) series.

        Returns:
            ``(batch, embed_dim)`` study-level embeddings.
        """
        queries = self.study_query.expand(series_tokens.shape[0], -1, -1)
        key_padding = ~series_mask.bool()
        attended, _ = self.cross_attention(
            queries,
            series_tokens,
            series_tokens,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        hidden = self.norm_attn(series_tokens.mean(dim=1, keepdim=True) * 0.0 + attended)
        hidden = self.norm_ffn(hidden + self.feed_forward(hidden))
        return hidden.squeeze(1)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on tabular metadata."""

    def __init__(self, metadata_dim: int, embed_dim: int) -> None:
        """Configure the conditioning projection.

        Args:
            metadata_dim: Width of the metadata feature vector.
            embed_dim: Width of the embeddings being modulated.
        """
        super().__init__()
        self.projection = nn.Linear(metadata_dim, embed_dim * 2)

    def forward(self, embeddings: Tensor, metadata: Tensor) -> Tensor:
        """Apply scale/shift modulation.

        Args:
            embeddings: ``(batch, embed_dim)`` features.
            metadata: ``(batch, metadata_dim)`` conditioning vector.

        Returns:
            Modulated ``(batch, embed_dim)`` features.
        """
        gamma, beta = self.projection(metadata).chunk(2, dim=-1)
        return embeddings * (1.0 + torch.tanh(gamma)) + beta
