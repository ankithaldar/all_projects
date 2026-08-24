#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Series-metadata conditioning layer.

Fluid sensitivity / fat suppression / anatomical plane are available at
TEST time (unlike reports), making them safe inputs that let one model
specialise its behaviour per sequence type.
Input vector per series:
[fluid_sensitive, fat_suppression, onehot(plane)] = 5 dims.
"""

from __future__ import annotations

import torch
from torch import nn

from knee.activations.factory import get_activation

N_META_FEATURES = 5


class MetadataEncoder(nn.Module):
  """MLP embedding of per-series acquisition metadata.

  Args:
      out_dim: Embedding dimensionality appended to series features.
      in_dim: Metadata vector width (5 by default).
      activation: Activation name resolved via the shared factory.
  """

  def __init__(
    self,
    out_dim: int = 64,
    in_dim: int = N_META_FEATURES,
    activation: str = 'silu',
  ) -> None:
    """Assemble Linear -> LayerNorm -> activation -> Linear.

    Args:
        out_dim: Output embedding size.
        in_dim: Input metadata width.
        activation: Factory name for the nonlinearity.

    Raises:
        ValueError: If ``activation`` is not registered in the factory.
    """
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, out_dim),
      nn.LayerNorm(out_dim),
      get_activation(activation),
      nn.Linear(out_dim, out_dim),
    )
    self.out_dim = out_dim

  def forward(self, meta: torch.Tensor) -> torch.Tensor:
    """Embed acquisition metadata.

    Args:
        meta: ``(B, S, in_dim)`` or ``(B, in_dim)`` tensor.

    Returns:
        Tensor with the last dimension replaced by ``out_dim``.
    """
    return self.net(meta)


def build_meta_vector(
  fluid_sensitive: int | bool, fat_suppression: int | bool, plane: str
) -> list[float]:
  """Build the canonical 5-dim metadata vector for one series.

  Shared by dataset and inference so train/test layouts cannot diverge.

  Args:
      fluid_sensitive: Fluid_Sensitive flag from series metadata.
      fat_suppression: Fat_Suppression flag from series metadata.
      plane: One of 'Sagittal', 'Coronal', 'Axial'.

  Returns:
      ``[fluid, fat, sagittal_onehot, coronal_onehot, axial_onehot]``.

  Raises:
      ValueError: If ``plane`` is not a known anatomical plane.
  """
  planes = ['Sagittal', 'Coronal', 'Axial']
  if plane not in planes:
    raise ValueError(f"unknown plane '{plane}'")
  return [
    float(bool(fluid_sensitive)),
    float(bool(fat_suppression)),
    *[float(plane == p) for p in planes],
  ]
