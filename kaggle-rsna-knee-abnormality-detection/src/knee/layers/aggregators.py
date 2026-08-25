#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Study-level aggregation of per-series features (multiple-instance learning).

Contract: (B, S, D) series features + (B, S) validity mask
-> (B, H) study embedding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AttentionMILAggregator(nn.Module):
  """Gated attention MIL (Ilse et al., 2018) -- strong, cheap, robust
  to variable numbers of series per study via masking."""

  def __init__(
    self, feat_dim: int = 1280, hidden_dim: int = 512, dropout: float = 0.1
  ) -> None:
    super().__init__()
    self.feat_dim = int(feat_dim)
    self.out_dim = hidden_dim
    self.project = nn.Sequential(
      nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
    )
    self.attention_v = nn.Linear(hidden_dim, hidden_dim)
    self.attention_u = nn.Linear(hidden_dim, hidden_dim)
    self.attention_w = nn.Linear(hidden_dim, 1)
    self.out_dim = hidden_dim
    self.dropout = nn.Dropout(dropout)

  def forward(
    self, series_feats: torch.Tensor, mask: torch.Tensor
  ) -> torch.Tensor:
    h = self.project(series_feats)  # (B,S,H)
    scores = self.attention_w(
      torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h))
    ).squeeze(-1)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=1)  # (B,S)
    pooled = torch.bmm(weights.unsqueeze(1), self.dropout(h)).squeeze(1)
    return F.gelu(pooled)


class TransformerAggregator(nn.Module):
  """Learned [CLS] token transformer over series embeddings; uses
  key_padding_mask so missing/localizer series are ignored."""

  def __init__(
    self,
    feat_dim: int = 1280,
    hidden_dim: int = 512,
    num_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    self.feat_dim = int(feat_dim)
    self.out_dim = hidden_dim
    self.input_proj = nn.Sequential(
      nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim)
    )
    self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=hidden_dim,
      nhead=num_heads,
      dim_feedforward=hidden_dim * 4,
      dropout=dropout,
      activation='gelu',
      batch_first=True,
      norm_first=True,
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.norm = nn.LayerNorm(hidden_dim)
    self.out_dim = hidden_dim
    nn.init.trunc_normal_(self.cls_token, std=0.02)

  def forward(
    self, series_feats: torch.Tensor, mask: torch.Tensor
  ) -> torch.Tensor:
    tokens = self.input_proj(series_feats)
    cls = self.cls_token.expand(tokens.size(0), -1, -1)
    tokens = torch.cat([cls, tokens], dim=1)
    pad_mask = ~torch.cat(
      [torch.ones_like(mask[:, :1]), mask], dim=1
    )  # CLS always valid
    encoded = self.encoder(tokens, src_key_padding_mask=pad_mask)
    return self.norm(encoded[:, 0])
