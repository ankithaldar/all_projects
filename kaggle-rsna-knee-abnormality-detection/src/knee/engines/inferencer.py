#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Eager fp16 inference and submission assembly."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from knee.engines.assembly import (
  TARGET_COLUMNS,
  build_datamodule,
  build_datasets,
  build_model,
)
from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

CLIP_EPS = 1.0e-4


def load_fold_model(config: dict, checkpoint_path: str) -> torch.nn.Module:
  """Build a model and load one fold's Lightning checkpoint into it.

  Args:
      config: Composed experiment configuration.
      checkpoint_path: Path to a Lightning ``last.ckpt`` file.

  Returns:
      KneeNet in eval mode on the current device.

  Raises:
      FileNotFoundError: If the checkpoint file is missing.
  """
  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f'Missing checkpoint: {checkpoint_path}')
  model = build_model(config)
  state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
  raw_state = state.get('state_dict', state)
  cleaned = {k.removeprefix('model.'): v for k, v in raw_state.items()}
  model.load_state_dict(cleaned, strict=True)
  model.eval()
  return model


@torch.no_grad()
def predict_studies(
  config: dict,
  index_df,
  study_ids: list[str],
  checkpoint_paths: list[str],
) -> pd.DataFrame:
  """Predict probabilities for studies, averaging across fold checkpoints.

  Args:
      config: Composed experiment configuration.
      index_df: Header-scan index (sop lists exploded).
      study_ids: Test studies to score.
      checkpoint_paths: Fold checkpoints to ensemble.

  Returns:
      Frame indexed by StudyInstanceUID with 12 probability columns.
  """
  _, valid_dataset = build_datasets(
    config,
    index_df,
    labels_df=None,
    valid_study_ids=study_ids,
    train_study_ids=None,
  )
  loader = build_datamodule(config, None, valid_dataset).val_dataloader()
  models = [load_fold_model(config, path) for path in checkpoint_paths]
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  for model in models:
    model.to(device)
  fp16 = bool(config['infer']['fp16']) and device == 'cuda'
  uids: list[str] = []
  outputs: list[np.ndarray] = []
  for batch in loader:
    batch_tensors = {
      key: value.to(device, non_blocking=True)
      for key, value in batch.items()
      if hasattr(value, 'to')
    }
    with torch.autocast(device_type=device.split(':')[0], enabled=fp16):
      fold_probs = [torch.sigmoid(model(batch_tensors)) for model in models]
    mean_probs = torch.stack(fold_probs).mean(dim=0).float().cpu().numpy()
    outputs.append(mean_probs)
    uids.extend(batch['study_uid'])
  return pd.DataFrame(
    np.concatenate(outputs, axis=0),
    columns=[f'{column}_prob' for column in TARGET_COLUMNS],
    index=pd.Index(uids, name='StudyInstanceUID'),
  )


def write_submission(
  predictions: pd.DataFrame,
  submission_path: str,
  expected_uids: set[str],
) -> None:
  """Validate schema and persist the competition submission csv.

  Hard asserts guard against silent crashes producing an invalid file:
  exact header order, row count, and UID set equality are all enforced.

  Args:
      predictions: Frame from :func:`predict_studies`.
      submission_path: Destination csv path.
      expected_uids: Exact UID set required by the test csv.

  Raises:
      AssertionError: On any schema violation.
  """
  frame = predictions.reindex(sorted(expected_uids))
  assert not frame.isna().any().any(), (
    'Missing predictions for some test studies'
  )
  clipped = frame.clip(lower=CLIP_EPS, upper=1.0 - CLIP_EPS)
  assert list(clipped.columns) == [f'{c}_prob' for c in TARGET_COLUMNS]
  clipped.columns = TARGET_COLUMNS
  clipped.to_csv(submission_path, index_label='StudyInstanceUID')
  _LOGGER.info(
    'Submission written: %s (%d rows)', submission_path, len(clipped)
  )
