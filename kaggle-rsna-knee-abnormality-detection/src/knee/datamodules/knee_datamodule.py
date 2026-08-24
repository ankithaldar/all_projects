#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fold-scoped LightningDataModule over cached study volumes.

Contract with :class:`knee.engines.study_lit_module.KneeStudyLitModule`
-- every batch is a dict::

    images        (B, S, C, H, W) float32  S=series slots, C=in_chans
    meta          (B, S, 5)       float32  [fluid, fat, plane onehot]
    series_mask   (B, S)          bool     True where a real series sits
    y_hard        (B, 12)         float32  supervised targets
    y_soft        (B, 12)         float32  distillation targets
    sample_weight (B,)            float32  label-trust weight
    is_gold       (B,)            bool     gold vs weak provenance
    StudyInstanceUID              list[str]

Label policy follows ``train.label_source``: 'gold' keeps only fully
labeled studies; 'weak'/'mixed' fuse gold labels with the weak-label
parquet produced by ``build_weak_labels.py`` (its per-class source
columns decide provenance).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from knee.augmentations.stack_transforms import build_transform
from knee.config_params.loader import instantiate
from knee.config_params.schema import (
  TARGETS,
  AugmentConfig,
  DataConfig,
  DataModuleConfig,
  PathsConfig,
  SamplerConfig,
  TrainConfig,
)
from knee.datasets.volume_builder import decode_series_volume
from knee.helpers.seeding import worker_init_fn
from knee.layers.metadata_encoder import build_meta_vector

_META_ALIASES = {
  'fluid': ['Fluid_Sensitive', 'FluidSensitive', 'fluid_sensitive'],
  'fat': ['Fat_Suppression', 'FatSuppression', 'fat_suppression'],
  'plane': ['Anatomical_Plane', 'Plane', 'anatomical_plane'],
}


def _first_column(table: pd.DataFrame, aliases: list[str]) -> str | None:
  """Return the first matching metadata column name or None.

  Args:
      table: Candidate series table.
      aliases: Acceptable column spellings, most canonical first.

  Returns:
      Column name present in ``table``, else None.
  """
  for alias in aliases:
    if alias in table.columns:
      return alias
  return None


def _series_priority(row: pd.Series) -> tuple[int, int]:
  """Sort key favouring sagittal fluid-sensitive series first.

  Args:
      row: Annotated series-table row (``_fluid``/``_plane`` present).

  Returns:
        Tuple key; smaller sorts earlier.
  """
  plane = str(row.get('_plane', '') or '').lower()
  fluid = bool(float(row.get('_fluid', 0) or 0))
  return (0 if 'sagittal' in plane else 1, -int(fluid))


class KneeStudyDataset(Dataset):
  """One item per study: padded series stacks + fused label dict."""

  def __init__(
    self,
    frame: pd.DataFrame,
    series_table: pd.DataFrame,
    data_root: str,
    transform,
    max_series: int,
  ) -> None:
    """Precompute per-study series layouts once.

    Args:
        frame: Study-level frame with fused labels.
        series_table: Manifest or CSV describing every series.
        data_root: Dataset root for DICOM fallback decoding.
        transform: Stack callable mapping (C,H,W) uint8 -> float array.
        max_series: Series slots per study (padding target).
    """
    self.frame = frame.reset_index(drop=True)
    self.transform = transform
    self.max_series = max_series
    self.data_root = str(data_root)
    self.series_by_study: dict[str, list[dict]] = {}
    fluid_col = _first_column(series_table, _META_ALIASES['fluid'])
    fat_col = _first_column(series_table, _META_ALIASES['fat'])
    plane_col = _first_column(series_table, _META_ALIASES['plane'])
    annotated = series_table.copy()
    annotated['_fluid'] = (
      pd.to_numeric(annotated[fluid_col], errors='coerce').fillna(0)
      if fluid_col else 0.0
    )
    annotated['_fat'] = (
      pd.to_numeric(annotated[fat_col], errors='coerce').fillna(0)
      if fat_col else 0.0
    )
    annotated['_plane'] = annotated[plane_col] if plane_col else ''
    for uid, group in annotated.groupby('StudyInstanceUID'):
      ranked = group.assign(
        _p=[_series_priority(r) for _, r in group.iterrows()]
      ).sort_values('_p')
      self.series_by_study[str(uid)] = ranked.head(max_series).to_dict(
        'records'
      )

  def __len__(self) -> int:
    """Number of studies.

    Returns:
        Dataset length.
    """
    return len(self.frame)

  def _load_volume(self, row: dict) -> np.ndarray | None:
    """Fetch one series volume from cache or DICOM fallback.

    Args:
        row: Series-table record with ids and cache_path.

    Returns:
        uint8 ``(C, H, W)`` array, or None when undecodable.
    """
    cache_path = str(row.get('cache_path', '') or '')
    if cache_path and Path(cache_path).exists():
      with np.load(cache_path) as payload:
        return payload['volume']
    try:
      base = Path(self.data_root) / 'train_series'
      study = str(row.get('StudyInstanceUID', '') or '')
      series = str(row.get('SeriesInstanceUID', '') or '')
      candidates = [base / study / series, base / series]
      directory = next((c for c in candidates if c.is_dir()), None)
      if directory is None:
        return None
      return decode_series_volume(
        directory,
        image_size=int(self._image_size),
        num_slices=int(self._num_slices),
      )
    except Exception:  # pylint: disable=broad-exception-caught
      return None

  def __getitem__(self, index: int) -> dict:
    """Assemble one study item.

    Args:
        index: Row index into the study frame.

    Returns:
        Batch-ready dict of tensors plus the study UID string.
    """
    row = self.frame.iloc[index]
    uid = str(row['StudyInstanceUID'])
    series_rows = self.series_by_study.get(uid, [])
    volumes = [self._load_volume(source) for source in series_rows]
    usable = [
      (source, volume)
      for source, volume in zip(series_rows, volumes, strict=False)
      if volume is not None
    ]
    channels, height, width = (
      usable[0][1].shape if usable else (
        int(self._num_slices), int(self._image_size), int(self._image_size),
      )
    )
    images = np.zeros(
      (self.max_series, channels, height, width), dtype=np.float32
    )
    mask = np.zeros(self.max_series, dtype=bool)
    meta = np.zeros((self.max_series, 5), dtype=np.float32)
    for slot, (source, volume) in enumerate(usable):
      transformed = np.asarray(self.transform(volume), dtype=np.float32)
      images[slot] = transformed
      mask[slot] = True
      meta[slot] = build_meta_vector(
        source.get('_fluid', 0),
        source.get('_fat', 0),
        str(source.get('_plane', '') or 'Sagittal'),
      )
    return {
      'images': torch.from_numpy(images),
      'meta': torch.from_numpy(meta),
      'series_mask': torch.from_numpy(mask),
      'y_hard': torch.from_numpy(np.asarray(row['_hard'], np.float32)),
      'y_soft': torch.from_numpy(np.asarray(row['_soft'], np.float32)),
      'sample_weight': torch.tensor(float(row['_weight'])),
      'is_gold': torch.tensor(bool(row['_is_gold'])),
      'StudyInstanceUID': uid,
    }


class KneeDataModule(LightningDataModule):
  """Builds fold-scoped train/val/predict loaders from config objects."""

  def __init__(
    self,
    paths: PathsConfig,
    data_cfg: DataConfig,
    dm_cfg: DataModuleConfig,
    augment: AugmentConfig,
    train_cfg: TrainConfig,
    sampler_cfg: SamplerConfig,
    fold: int,
    test_studies_csv: str | None = None,
  ) -> None:
    """Store configs; nothing touches the filesystem before ``setup``.

    Args:
        paths: Data/cache/artifact locations.
        data_cfg: Volume shaping parameters.
        dm_cfg: Loader wiring (batch size, workers, series cap).
        augment: Declarative augmentation specs.
        train_cfg: Label policy + fold scheme.
        sampler_cfg: Optional balanced-sampler spec.
        fold: Held-out fold id; ignored for predict-only use.
        test_studies_csv: Explicit test CSV overriding paths.test_csv.
    """
    super().__init__()
    self.paths = paths
    self.data_cfg = data_cfg
    self.dm_cfg = dm_cfg
    self.augment = augment
    self.train_cfg = train_cfg
    self.sampler_cfg = sampler_cfg
    self.fold = fold
    self.test_studies_csv = test_studies_csv
    self.batch_size = int(dm_cfg.batch_size)
    self._image_size = int(data_cfg.image_size)
    self._num_slices = int(data_cfg.num_slices)
    self.train_ds: KneeStudyDataset | None = None
    self.val_ds: KneeStudyDataset | None = None
    self.predict_ds: KneeStudyDataset | None = None

  # ------------------------------------------------------------------ #
  def set_batch_size(self, batch_size: int) -> None:
    """Runtime batch-size hook consumed by AdaptiveBatchSizeCallback.

    Args:
        batch_size: New studies-per-step value.
    """
    self.dm_cfg.batch_size = int(batch_size)
    self.batch_size = int(batch_size)

  def _series_table(self, split: str) -> pd.DataFrame:
    """Load the series manifest (cache manifest preferred over CSV).

    Args:
        split: 'train' or 'test' selects which descriptor table to use.

    Returns:
        Series table covering every study of the split; when no cache
        manifest or CSV exists, the table is synthesized by walking the
        DICOM tree (see ``volume_builder.synthesize_series_table``).
    """
    cache_dir = self.paths.volumes_cache
    manifest = (
      Path(cache_dir) / 'volumes_manifest.parquet' if cache_dir else None
    )
    if manifest and manifest.exists():
      return pd.read_parquet(manifest)
    csv_path = Path(self.paths.data_root) / (
      self.paths.test_series_csv if split == 'test'
      else self.paths.train_series_csv
    )
    if csv_path.exists():
      return pd.read_csv(csv_path)
    from knee.datasets.volume_builder import synthesize_series_table

    return synthesize_series_table(self.paths.data_root, split)

  def _fused_frame(self) -> pd.DataFrame:
    """Merge folds CSV with gold labels and the weak-label parquet."""
    folds = pd.read_csv(self.paths.folds_csv)
    present = [t for t in TARGETS if t in folds.columns]
    is_gold = (
      folds[present].notna().all(axis=1).to_numpy()
      if present else np.zeros(len(folds), bool)
    )
    gold_values = (
      folds[present].fillna(0.0).to_numpy(np.float32) if present
      else np.zeros((len(folds), len(TARGETS)), np.float32)
    )
    soft = np.full((len(folds), len(TARGETS)), 0.5, np.float32)
    if present:
      soft[:, : len(present)] = gold_values
    weight = np.ones(len(folds), np.float32)
    if self.paths.weak_labels_parquet:
      weak = pd.read_parquet(self.paths.weak_labels_parquet)
      prob_cols = ['StudyInstanceUID', *TARGETS]
      weak_probs_frame = weak[
        [c for c in prob_cols if c in weak.columns]
      ].rename(columns={t: f'_w_{t}' for t in TARGETS})
      folds = folds.merge(weak_probs_frame, on='StudyInstanceUID',
                          how='left')
      weak_probs = folds[[f'_w_{t}' for t in TARGETS]].to_numpy(
        np.float32
      )
      have_weak = np.isfinite(weak_probs).all(axis=1)
      soft = np.where(have_weak[:, None], weak_probs, soft)
      if 'weight' in folds.columns:
        weight = np.where(
          have_weak, folds['weight'].fillna(1.0).to_numpy(np.float32),
          weight,
        ).astype(np.float32)
      folds = folds.drop(columns=[f'_w_{t}' for t in TARGETS])
    hard = np.where(is_gold[:, None], gold_values, (soft >= 0.5))
    frame = pd.DataFrame({'StudyInstanceUID': folds['StudyInstanceUID']})
    frame['fold'] = (
      folds['fold'].to_numpy() if 'fold' in folds.columns
      else np.full(len(folds), -1)
    )
    frame['_hard'] = list(hard.astype(np.float32))
    frame['_soft'] = list(soft.astype(np.float32))
    frame['_weight'] = weight
    frame['_is_gold'] = is_gold
    return frame

  def _apply_label_policy(self, frame: pd.DataFrame) -> pd.DataFrame:
    """Filter/adjust the merged frame according to ``label_source``.

    Args:
        frame: Merged study frame.

    Returns:
        Policy-conforming frame.
    """
    policy = self.train_cfg.label_source
    if policy == 'gold':
      return frame[frame['_is_gold']].reset_index(drop=True)
    if policy == 'weak':
      frame = frame.copy()
      frame['_is_gold'] = False
    return frame.reset_index(drop=True)

  def _predict_frame(self) -> pd.DataFrame:
    """Zero-label frame for the inference kernel."""
    csv_path = self.test_studies_csv or str(
      Path(self.paths.data_root) / self.paths.test_csv
    )
    uids = pd.read_csv(csv_path)['StudyInstanceUID'].astype(str)
    zeros = np.zeros((len(uids), len(TARGETS)), np.float32)
    half = np.full((len(uids), len(TARGETS)), 0.5, np.float32)
    return pd.DataFrame(
      {
        'StudyInstanceUID': uids,
        '_hard': list(zeros),
        '_soft': list(half),
        '_weight': np.zeros(len(uids), np.float32),
        '_is_gold': np.zeros(len(uids), bool),
      }
    )

  def setup(self, stage: str) -> None:
    """Materialize datasets for a Lightning stage.

    Args:
        stage: 'fit' | 'validate' | 'predict'.
    """
    valid_tf = build_transform(self.augment, 'valid', self.data_cfg.in_chans)
    if stage == 'predict':
      self.predict_ds = KneeStudyDataset(
        self._predict_frame(), self._series_table('test'),
        self.paths.data_root, valid_tf, self.dm_cfg.max_series_per_study,
      )
      return
    frame = self._apply_label_policy(self._fused_frame())
    table = self._series_table('train')
    self.val_ds = KneeStudyDataset(
      frame[frame['fold'] == self.fold], table, self.paths.data_root,
      valid_tf, self.dm_cfg.max_series_per_study,
    )
    if stage == 'fit':
      self.train_ds = KneeStudyDataset(
        frame[frame['fold'] != self.fold], table, self.paths.data_root,
        build_transform(self.augment, 'train', self.data_cfg.in_chans),
        self.dm_cfg.max_series_per_study,
      )

  def _loader(self, dataset: KneeStudyDataset, shuffle: bool):
    """Shared DataLoader construction.

    Args:
        dataset: Materialized dataset.
        shuffle: Whether to shuffle (ignored when a sampler is set).

    Returns:
        Configured DataLoader.
    """
    sampler = None
    if shuffle and self.sampler_cfg.sampler is not None:
      weights = instantiate(self.sampler_cfg.sampler)(
        np.stack(list(dataset.frame['_hard']))
      )
      sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
      )
      shuffle = False
    return DataLoader(
      dataset,
      batch_size=self.batch_size,
      shuffle=shuffle,
      sampler=sampler,
      num_workers=self.dm_cfg.num_workers,
      pin_memory=self.dm_cfg.pin_memory,
      worker_init_fn=worker_init_fn,
    )

  def train_dataloader(self) -> DataLoader:
    """Training loader with optional balanced sampling.

    Returns:
        Configured DataLoader.
    """
    return self._loader(self.train_ds, shuffle=True)

  def val_dataloader(self) -> DataLoader:
    """Validation loader for the held-out fold.

    Returns:
        Configured DataLoader.
    """
    return self._loader(self.val_ds, shuffle=False)

  def predict_dataloader(self) -> DataLoader:
    """Inference loader over the test study list.

    Returns:
        Configured DataLoader.
    """
    return self._loader(self.predict_ds, shuffle=False)
