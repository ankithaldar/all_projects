#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 6 engine: multi-checkpoint TTA inference over study batches.

Loads every fold checkpoint of a :class:`KneeStudyLitModule`, averages
sigmoid probabilities across checkpoints and test-time-augmentation
views (vertical flip, slice-axis reversal, multi-scale), and returns a
per-study probability frame ready for :func:`save_submission`.

TTA safety follows the BLUEPRINT: no left/right flips ever -- medial vs
lateral compartments are target-defining.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from knee.config_params.schema import TARGETS, TTAConfig
from knee.engines.study_lit_module import KneeStudyLitModule


def _tta_views(images: torch.Tensor, tta_config: TTAConfig) -> list:
  """Enumerate augmented views of one image batch.

  Args:
      images: Batch ``(B, S, C, H, W)``.
      tta_config: Enabled view set plus optional extra scales.

  Returns:
      List of equally shaped tensors (identity first).
  """
  views = []
  if tta_config.vertical_flip:
    views.append(torch.flip(images, dims=[-1]))
  if tta_config.slice_reversal:
    views.append(torch.flip(images, dims=[1]))
  height = images.shape[-1]
  for scale in tta_config.multi_scale:
    if int(scale) != int(height):
      resized = F.interpolate(
        images.flatten(0, 1),
        size=(int(scale), int(scale)),
        mode='bilinear',
        align_corners=False,
      )
      shape = (*images.shape[:3], int(scale), int(scale))
      views.append(resized.reshape(shape))
  views.append(images)
  return views


def _load_module(checkpoint: str | Path) -> KneeStudyLitModule:
  """Restore a trained student module onto the best available device.

  Args:
      checkpoint: Path to a Lightning ``.ckpt`` file.

  Returns:
      Module in eval mode on CUDA when available, else CPU.
  """
  module = KneeStudyLitModule.load_from_checkpoint(
    str(checkpoint), map_location='cpu'
  )
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  return module.to(device).eval()


@torch.no_grad()
def run_predict(
  checkpoints: Iterable[str | Path],
  datamodule,
  tta_config: TTAConfig,
) -> pd.DataFrame:
  """Predict every study once per checkpoint/view and pool the results.

  Args:
      checkpoints: Fold checkpoint paths; predictions are averaged.
      datamodule: Object exposing ``predict_dataloader()`` (the fitted
          KneeDataModule or a validation-loader adapter).
      tta_config: View configuration from the experiment YAML.

  Returns:
      Frame with columns ``[StudyInstanceUID, *TARGETS]`` holding the
      mean probability per study.
  """
  loader = datamodule.predict_dataloader()
  accumulator: dict[str, list[np.ndarray]] = {}
  for checkpoint in checkpoints:
    module = _load_module(checkpoint)
    for batch in loader:
      images = batch['images'].to(module.device)
      meta = batch['meta'].to(module.device)
      mask = batch['series_mask'].to(module.device)
      view_probs = [
        module.predict_logits(view, meta, mask).cpu().numpy()
        for view in _tta_views(images, tta_config)
      ]
      pooled = np.mean(view_probs, axis=0)
      for uid, prob in zip(batch['StudyInstanceUID'], pooled, strict=False):
        accumulator.setdefault(str(uid), []).append(prob)

  frame = pd.DataFrame(
    {
      'StudyInstanceUID': list(accumulator),
      **{
        target: [np.mean(runs)[c] for runs in accumulator.values()]
        for c, target in enumerate(TARGETS)
      },
    }
  ).clip(subset=list(TARGETS), lower=0.0, upper=1.0)
  return frame
