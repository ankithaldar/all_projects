#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GPU-resident differentiable augmentation applied inside the module.

Classic DiffAugment ops (Zhao et al., 2020) reimplemented in pure torch
so gradients flow and no extra dependency is needed. Operates on the
already-normalized ``(B, S, C, H, W)`` stacks; spatial ops remain valid
because they never flip left-right.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiffAugment(nn.Module):
  """Chain of probabilistic differentiable augmentation ops.

  Args:
      ops: Subset of ``('color', 'translation', 'cutout')``.
      probability: Per-op application probability.
      cutout_px: Cutout patch size (square).

  Raises:
      ValueError: If an unknown op name is supplied.
  """

  _VALID = frozenset({'color', 'translation', 'cutout'})

  def __init__(
    self,
    ops: list[str] | None = None,
    probability: float = 0.5,
    cutout_px: int = 32,
  ) -> None:
    """Validate op names and store strengths.

    Args:
        ops: Op subset; defaults to all three.
        probability: Application probability per op.
        cutout_px: Edge length of cutout squares.
    """
    super().__init__()
    ops = ops if ops is not None else ['color', 'translation', 'cutout']
    unknown = set(ops) - self._VALID
    if unknown:
      raise ValueError(f'unknown diff-augment ops: {sorted(unknown)}')
    self.ops = list(ops)
    self.probability = probability
    self.cutout_px = cutout_px

  @staticmethod
  def _color(x: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    """Per-channel brightness/bias jitter.

    Args:
        x: Normalized image tensor ``(N, C, H, W)``.
        strength: Std of gain and bias perturbations.

    Returns:
        Perturbed tensor of identical shape.
    """
    n, c = x.shape[0], x.shape[1]
    gain = 1.0 + torch.randn(n, c, 1, 1, device=x.device) * strength
    bias = torch.randn(n, c, 1, 1, device=x.device) * strength * 0.5
    return x * gain + bias

  @staticmethod
  def _translation(
    x: torch.Tensor, max_fraction: float = 0.0625
  ) -> torch.Tensor:
    """Random sub-pixel-exact shift via grid sampling (border replicate).

    Args:
        x: Image tensor ``(N, C, H, W)``.
        max_fraction: Maximum shift as fraction of image size.

    Returns:
        Translated tensor.
    """
    n = x.shape[0]
    max_shift = int(x.shape[-1] * max_fraction)
    tx = (torch.rand(n, device=x.device) * 2 - 1) * max_shift / x.shape[-1]
    ty = (torch.rand(n, device=x.device) * 2 - 1) * max_shift / x.shape[-2]
    grid_x, grid_y = torch.meshgrid(
      torch.linspace(-1, 1, x.shape[-1]),
      torch.linspace(-1, 1, x.shape[-2]),
      indexing='xy',
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(n, 1, 1, 1)
    grid[..., 0] += tx.view(n, 1, 1)
    grid[..., 1] += ty.view(n, 1, 1)
    return F.grid_sample(
      x, grid.to(x.dtype), padding_mode='reflection', align_corners=False
    )

  def _cutout(self, x: torch.Tensor) -> torch.Tensor:
    """Zero random square patches per sample.

    Args:
        x: Image tensor ``(N, C, H, W)``.

    Returns:
        Tensor with cutouts applied in-place-safe manner.
    """
    n, _, h, w = x.shape
    cy = torch.randint(h, (n,))
    cx = torch.randint(w, (n,))
    half = self.cutout_px // 2
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    mask_y = (ys - cy.view(n, 1, 1)).abs() <= half
    mask_x = (xs - cx.view(n, 1, 1)).abs() <= half
    keep = ~(mask_y & mask_x).unsqueeze(1)
    return x * keep

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    """Apply the configured ops to a batched stack.

    Args:
        images: ``(B, S, C, H, W)`` tensor.

    Returns:
        Augmented tensor of identical shape.
    """
    b, s = images.shape[:2]
    flat = images.reshape(b * s, *images.shape[2:])
    for op in self.ops:
      if torch.rand(()) > self.probability:
        continue
      if op == 'color':
        flat = self._color(flat)
      elif op == 'translation':
        flat = self._translation(flat)
      elif op == 'cutout':
        flat = self._cutout(flat)
    return flat.reshape(b, s, *flat.shape[1:])
