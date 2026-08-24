#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Series encoders: 2.5D channel-stacked timm backbones and true-3D MONAI nets.

Contract (Interface Segregation): ``forward`` maps
  2.5D: (N, C, H, W) -> (N, feat_dim)      C = stacked adjacent slices
  3D:   (N, 1, D, H, W) -> (N, feat_dim)   D = sampled depth
and every encoder exposes ``feat_dim`` so aggregators/head can be sized
without hard-coding.
"""

from __future__ import annotations

import torch
from torch import nn


class TimmSeriesEncoder(nn.Module):
  """ImageNet-pretrained 2D backbone treating the slice stack as channels.

  Pretrained weights transfer dramatically better than from-scratch 3D,
  which is why this is the primary encoder for the competition.
  """

  def __init__(
    self,
    model_name: str = 'tf_efficientnetv2_s.in21k_ft_in1k',
    pretrained: bool = True,
    in_chans: int = 32,
    drop_path_rate: float = 0.2,
    global_pool: str = 'avg',
  ) -> None:
    super().__init__()
    import timm  # pylint: disable=import-outside-toplevel

    self.backbone = timm.create_model(
      model_name,
      pretrained=pretrained,
      in_chans=in_chans,
      drop_path_rate=drop_path_rate,
      global_pool=global_pool,
      num_classes=0,  # feature extractor: forward returns pooled features
    )
    self.feat_dim = self.backbone.num_features

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5:  # tolerate (N, C, H, W) only; squeeze accidental depth axis
      raise ValueError(
        f'TimmSeriesEncoder expects (N,C,H,W), got {tuple(x.shape)}'
      )
    return self.backbone(x)


class Monai3DEncoder(nn.Module):
  """True-volumetric encoder via MONAI (spatial_dims=3). Diversity member:
  sees inter-slice continuity the 2.5D stack cannot."""

  def __init__(
    self,
    model_name: str = 'seresnext50',
    pretrained: bool = False,
    spatial_dims: int = 3,
    in_channels: int = 1,
    dropout_prob: float = 0.1,
  ) -> None:
    del pretrained
    super().__init__()
    try:
      # pylint: disable=import-outside-toplevel
      from monai.networks.nets import get_network
    except ImportError as exc:  # pragma: no cover - env without monai
      raise ImportError('MONAI3DEncoder requires `pip install monai`') from exc
    kwargs = dict(
      spatial_dims=spatial_dims,
      in_channels=in_channels,
      dropout_prob=dropout_prob,
    )
    try:
      self.backbone = get_network(  # type: ignore[arg-type]
        model_name, **kwargs
      )
    except TypeError:  # some MONAI nets lack dropout_prob
      kwargs.pop('dropout_prob')
      self.backbone = get_network(  # type: ignore[arg-type]
        model_name, **kwargs
      )
    self.pool = nn.AdaptiveAvgPool3d(1)
    feat_dim = getattr(self.backbone, 'feat_dim', None)
    self.feat_dim = (
      int(feat_dim)
      if feat_dim is not None
      else self._infer_dim(in_channels=in_channels)
    )

  def _infer_dim(self, in_channels: int) -> int:
    device = next(self.backbone.parameters()).device
    was_training = self.backbone.training
    self.backbone.eval()
    with torch.no_grad():
      out = self.backbone(torch.zeros(1, in_channels, 8, 32, 32, device=device))
    if was_training:
      self.backbone.train()
    return int(out.flatten(1).shape[1])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
      raise ValueError(
        f'Monai3DEncoder expects (N,C,D,H,W), got {tuple(x.shape)}'
      )
    return self.backbone(x).flatten(1)
