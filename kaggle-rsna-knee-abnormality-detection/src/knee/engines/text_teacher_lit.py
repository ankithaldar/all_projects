#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier-2 weak supervision: multilingual text teacher (reports -> probs).

XLM-R encodes radiology reports in any of the competition's ~10
languages; 12 sigmoid heads predict per-finding probabilities. Long
reports are handled at inference by stride-sliding windows with
max-pooling over window probabilities (BLUEPRINT section 3, Tier 2).

The module is intentionally self-contained: it carries its own pydantic
schema (``configs/labeling/text_teacher.yaml``) because the input
modality differs entirely from the image-experiment configs.
"""

from __future__ import annotations

from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import Dataset

from knee.config_params.schema import TARGETS


class _TeacherModelCfg(BaseModel):
  """Backbone + tokenization settings."""

  model_config = ConfigDict(extra='forbid')

  backbone: str = 'xlm-roberta-base'
  n_targets: int = Field(default=12, ge=1)
  max_length: int = Field(default=512, ge=32)
  stride: int = Field(default=128, ge=0)
  pooling: Literal['mean', 'cls', 'max'] = 'mean'


class _TeacherDataCfg(BaseModel):
  """File locations for teacher training artifacts."""

  model_config = ConfigDict(extra='forbid')

  train_csv: str
  folds_csv: str = ''
  output_dir: str = '.'


class _TeacherTrainCfg(BaseModel):
  """Optimization schedule for the teacher."""

  model_config = ConfigDict(extra='forbid')

  n_folds_to_train: int = Field(default=5, ge=1)
  epochs: int = Field(default=3, ge=1)
  batch_size: int = Field(default=16, ge=1)
  grad_accum: int = Field(default=1, ge=1)
  lr: float = Field(default=2e-5, gt=0)
  weight_decay: float = Field(default=0.01, ge=0)
  warmup_ratio: float = Field(default=0.1, ge=0, le=1)
  amp: str = 'bf16'
  early_stopping_patience: int = Field(default=2, ge=1)
  time_budget_hours: float | None = Field(
    default=None,
    gt=0,
    description=(
      'Wall-clock cap for the whole fold loop (Kaggle 12 h kernel '
      'limit); each Trainer gets the remaining time as max_time.'
    ),
  )


class _TeacherFusionCfg(BaseModel):
  """How rule seeds and teacher probabilities fuse downstream."""

  model_config = ConfigDict(extra='forbid')

  rule_confidence_floor: float = Field(default=0.9, ge=0, le=1)
  teacher_weight: float = Field(default=1.0, ge=0)
  min_positive_prob: float = Field(default=0.35, ge=0, le=1)
  output_parquet: str = 'weak_labels.parquet'


class TextTeacherConfig(BaseModel):
  """Root schema validated from ``configs/labeling/text_teacher.yaml``."""

  model_config = ConfigDict(extra='forbid')

  run_name: str = 'text_teacher'
  seed: int = 42
  model: _TeacherModelCfg = Field(default_factory=_TeacherModelCfg)
  data: _TeacherDataCfg
  train: _TeacherTrainCfg = Field(default_factory=_TeacherTrainCfg)
  fusion: _TeacherFusionCfg = Field(default_factory=_TeacherFusionCfg)


class ReportDataset(Dataset):
  """Tokenized report texts for teacher training/prediction."""

  def __init__(self, texts, tokenizer, max_length: int) -> None:
    """Store raw texts; tokenization happens lazily per item.

    Args:
        texts: Iterable of report strings.
        tokenizer: HF tokenizer for the teacher backbone.
        max_length: Truncation length per report (training mode).
    """
    self.texts = list(texts)
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self) -> int:
    """Number of reports.

    Returns:
        Dataset length.
    """
    return len(self.texts)

  def __getitem__(self, index: int) -> dict:
    """Tokenize one truncated report.

    Args:
        index: Report position.

    Returns:
        Dict with input_ids / attention_mask long tensors.
    """
    encoded = self.tokenizer(
      str(self.texts[index]),
      truncation=True,
      max_length=self.max_length,
      padding='max_length',
    )
    return {
      key: torch.tensor(encoded[key], dtype=torch.long)
      for key in ('input_ids', 'attention_mask')
    }


class TextTeacherLitModule(pl.LightningModule):
  """Lightning wrapper around a pooled multilingual encoder."""

  def __init__(
    self,
    backbone: str,
    lr: float,
    weight_decay: float,
    n_targets: int = 12,
    pooling: str = 'mean',
    warmup_ratio: float = 0.1,
  ) -> None:
    """Build encoder + linear head; heavy imports stay lazy.

    Args:
        backbone: HF model id (multilingual encoder).
        lr: Peak learning rate for AdamW.
        weight_decay: AdamW weight decay.
        n_targets: Number of sigmoid outputs.
        pooling: Token pooling strategy before the head.
        warmup_ratio: Fraction of steps in LR warmup.

    Raises:
        ImportError: When ``transformers`` is not installed.
    """
    super().__init__()
    # pylint: disable=import-outside-toplevel
    from transformers import AutoModel

    self.save_hyperparameters()
    self.encoder = AutoModel.from_pretrained(backbone)
    hidden = self.encoder.config.hidden_size
    self.head = torch.nn.Linear(hidden, n_targets)
    self.lr = float(lr)
    self.weight_decay = float(weight_decay)
    self.pooling = pooling
    self.warmup_ratio = float(warmup_ratio)
    self.criterion = torch.nn.BCEWithLogitsLoss()
    self._val_probs: list[np.ndarray] = []
    self._val_targets: list[np.ndarray] = []

  def _pool(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
  ) -> torch.Tensor:
    """Reduce token states to one sentence embedding.

    Args:
        hidden_states: ``(B, T, H)`` encoder output.
        attention_mask: ``(B, T)`` with 1 on real tokens.

    Returns:
        ``(B, H)`` sentence embeddings.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    if self.pooling == 'cls':
      return hidden_states[:, 0]
    if self.pooling == 'max':
      negatives = torch.finfo(hidden_states.dtype).min
      return hidden_states.masked_fill(mask == 0, negatives).amax(dim=1)
    return (hidden_states * mask).sum(1) / mask.sum(1).clamp_min(1e-6)

  def forward(
    self, input_ids: torch.Tensor, attention_mask: torch.Tensor
  ) -> torch.Tensor:
    """Predict per-finding logits.

    Args:
        input_ids: ``(B, T)`` token ids.
        attention_mask: ``(B, T)`` padding mask.

    Returns:
        Raw logits ``(B, n_targets)``.
    """
    out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    return self.head(self._pool(out.last_hidden_state, attention_mask))

  def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
    """Standard multi-label BCE step.

    Args:
        batch: Tokenized batch plus ``targets`` matrix.
        batch_idx: Index within the epoch (unused).

    Returns:
        Scalar loss.
    """
    del batch_idx
    logits = self(batch['input_ids'], batch['attention_mask'])
    loss = self.criterion(logits, batch['targets'].float())
    self.log('train_loss', loss, prog_bar=True, sync_dist=True)
    return loss

  def validation_step(self, batch: dict, batch_idx: int) -> None:
    """Accumulate validation probabilities for pooled macro-AUC.

    Args:
        batch: Tokenized batch plus ``targets`` matrix.
        batch_idx: Index within the loop (unused).
    """
    del batch_idx
    logits = self(batch['input_ids'], batch['attention_mask'])
    self._val_probs.append(torch.sigmoid(logits).cpu().numpy())
    self._val_targets.append(batch['targets'].cpu().numpy())

  def on_validation_epoch_end(self) -> None:
    """Log pooled macro-AUC across defined classes."""
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
      roc_auc_score,
    )

    if not self._val_probs:
      return
    probs = np.concatenate(self._val_probs)
    targets = np.concatenate(self._val_targets)
    aucs = []
    for c in range(probs.shape[1]):
      col = targets[:, c]
      if col.size and np.unique(col > 0.5).size > 1:
        aucs.append(roc_auc_score(col > 0.5, probs[:, c]))
    if aucs:
      self.log('val_macro_auc', float(np.mean(aucs)), prog_bar=True)
    self._val_probs.clear()
    self._val_targets.clear()

  def configure_optimizers(self):
    """AdamW over grouped params + linear warmup/decay schedule.

    Returns:
        Dict with optimizer and lr_scheduler for Lightning.
    """
    # pylint: disable=import-outside-toplevel
    from transformers import get_linear_schedule_with_warmup

    no_decay = ('bias', 'LayerNorm.weight')
    groups = [
      {
        'params': [
          p
          for n, p in self.named_parameters()
          if not any(nd in n for nd in no_decay)
        ],
        'weight_decay': self.weight_decay,
      },
      {
        'params': [
          p
          for n, p in self.named_parameters()
          if any(nd in n for nd in no_decay)
        ],
        'weight_decay': 0.0,
      },
    ]
    optimizer = torch.optim.AdamW(groups, lr=self.lr)
    total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=int(total_steps * self.warmup_ratio),
      num_training_steps=total_steps,
    )
    return {
      'optimizer': optimizer,
      'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
    }


@torch.no_grad()
def predict_probs(
  module: TextTeacherLitModule,
  tokenizer,
  texts,
  max_length: int,
  stride: int,
  batch_size: int = 16,
) -> np.ndarray:
  """Sliding-window inference with max-pooling over window probs.

  Args:
      module: Trained :class:`TextTeacherLitModule` (any device).
      tokenizer: Matching HF tokenizer.
      texts: Report strings, aligned with ``StudyInstanceUID`` upstream.
      max_length: Window length in tokens.
      stride: Overlap between consecutive windows.
      batch_size: Windows forwarded per step.

  Returns:
      Float32 probability matrix ``(len(texts), n_targets)``.
  """
  device = next(module.parameters()).device
  module.eval()
  step = max(max_length - int(stride), 1)
  windows: list[tuple[int, list[int]]] = []
  for text_index, text in enumerate(texts):
    ids = tokenizer(str(text), truncation=False, add_special_tokens=True)[
      'input_ids'
    ]
    starts = range(0, max(len(ids) - max_length, 0) + 1, step)
    chunks = [ids[start : start + max_length] for start in starts] or [
      ids[:max_length]
    ]
    for chunk in chunks:
      windows.append((text_index, chunk))
  scores = np.zeros((len(texts), module.hparams['n_targets']), np.float32)
  seen = np.zeros(len(texts), bool)
  for start in range(0, len(windows), batch_size):
    chunk_batch = windows[start : start + batch_size]
    encoded = tokenizer.pad(
      [{'input_ids': chunk} for _, chunk in chunk_batch],
      padding=True,
      return_tensors='pt',
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    logits = module(**encoded)
    probs = torch.sigmoid(logits).float().cpu().numpy()
    for (owner, _), prob in zip(chunk_batch, probs, strict=False):
      scores[owner] = np.maximum(scores[owner], prob)
      seen[owner] = True
  scores[~seen] = 0.5  # empty/unreadable reports: uninformative prior
  return scores


def oof_frame_from_matrix(uids, matrix: np.ndarray) -> pd.DataFrame:
  """Package an OOF probability matrix as a parquet-ready frame.

  Args:
      uids: StudyInstanceUID sequence aligned with ``matrix`` rows.
      matrix: Probability matrix ``(N, 12)``.

  Returns:
      Frame with StudyInstanceUID + TARGET columns.
  """
  frame = pd.DataFrame({'StudyInstanceUID': list(uids)})
  for column, target in enumerate(TARGETS):
    frame[target] = matrix[:, column]
  return frame
