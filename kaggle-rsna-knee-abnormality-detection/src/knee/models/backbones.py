#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""timm-backed image encoder with offline checkpoint support."""

from __future__ import annotations

import timm
import torch
from torch import Tensor, nn


class TimmBackbone(nn.Module):
  """Feature-extractor wrapper around a timm classification model.

  The wrapper always runs the model through ``forward_features`` style
  global pooling (``num_classes=0``), returning a flat embedding vector.
  Offline checkpoints (Kaggle datasets) are loaded from ``pretrained_cfg.file``
  when provided; otherwise timm's cached hub weights apply.
  """

  def __init__(
    self,
    backbone_name: str,
    img_size: int,
    drop_path_rate: float = 0.0,
    pretrained: bool = True,
    pretrained_cfg: dict | None = None,
    grad_checkpointing: bool = False,
    chunk_size: int = 0,
  ) -> None:
    """Create and optionally warm-load the backbone.

    Args:
        backbone_name: timm model name, e.g. ``tf_efficientnetv2_s``.
        img_size: Square input resolution the model will see.
        drop_path_rate: Stochastic-depth rate forwarded to timm.
        pretrained: Request timm-pretrained weights (ignored when a
            local checkpoint file is supplied).
        pretrained_cfg: Optional mapping with ``file`` pointing at a
            state-dict checkpoint bundled offline.
        grad_checkpointing: Enable timm activation checkpointing; trades
            ~25% compute for a large reduction in retained activations
            (essential when a whole study batch is flattened into one
            forward on 16 GB GPUs).
        chunk_size: When > 0, run the backbone in slices-chunks of this
            size instead of one giant pass, bounding peak transient
            buffers. Autograd keeps the graph intact across chunks.

    Raises:
        RuntimeError: When an explicit checkpoint file fails to load.
    """
    super().__init__()
    checkpoint_file = (pretrained_cfg or {}).get('file')
    self.chunk_size = max(0, int(chunk_size))
    self.model = self._create_model(
      backbone_name,
      img_size,
      drop_path_rate,
      pretrained=pretrained and not checkpoint_file,
    )
    if grad_checkpointing:
      setter = getattr(self.model, 'set_grad_checkpointing', None)
      if setter is None:
        raise RuntimeError(
          f'timm model {backbone_name!r} does not expose '
          'set_grad_checkpointing; disable grad_checkpointing'
        )
      setter(True)
    if checkpoint_file:
      state = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
      state = state.get('state_dict', state)
      cleaned = {k.removeprefix('model.'): v for k, v in state.items()}
      missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
      if missing:
        raise RuntimeError(f'Checkpoint missing keys: {missing[:5]}')
      del unexpected  # head/cls tokens may legitimately differ

  @staticmethod
  def _create_model(
    backbone_name: str,
    img_size: int,
    drop_path_rate: float,
    pretrained: bool,
  ) -> nn.Module:
    """Instantiate the underlying timm module.

    Args:
        backbone_name: timm model name.
        img_size: Input resolution hint for models supporting it.
        drop_path_rate: Stochastic depth rate.
        pretrained: Whether timm should download/use cached weights.

    Returns:
        Feature extractor nn.Module producing ``(batch, embed_dim)``.
    """
    try:
      return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=0,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
      )
    except TypeError:
      # Backbone ignores img_size/drop_path kwargs (e.g. legacy nets).
      return timm.create_model(
        backbone_name, pretrained=pretrained, num_classes=0
      )

  @property
  def embed_dim(self) -> int:
    """Return the output feature dimension of the wrapped model.

    Returns:
        Integer feature width used by downstream pooling layers.
    """
    return int(self.model.num_features)

  def forward(self, images: Tensor) -> Tensor:
    """Extract embeddings for a batch of slice images.

    Args:
        images: ``(batch, channels, height, width)`` float tensor.

    Returns:
        ``(batch, embed_dim)`` embedding matrix.
    """
    if self.chunk_size <= 0 or images.shape[0] <= self.chunk_size:
      return self.model(images)
    chunks = [self.model(part) for part in torch.split(images, self.chunk_size)]
    return torch.cat(chunks, dim=0)
