#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Study-level dataset and collation for hierarchical 2.5D training.

One item = one study: selected series are decoded to fixed-length slice
stacks, flattened into a single tensor, and paired with per-series features,
study metadata, and a masked 12-target label vector (``-1`` = unknown).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from albumentations.core.composition import BaseCompose
from torch.utils.data import Dataset

from knee.datasets.series_dataset import SeriesReader

SERIES_META_DIM = 3  # plane_idx + fluid_sensitive + fat_suppression


def select_series(
  series_rows: pd.DataFrame, selection_cfg: dict
) -> pd.DataFrame:
  """Choose the most diagnostically valuable series for one study.

  Priority entries from configuration match rows by plane / fluid-sensitive /
  fat-suppression flags; unmatched series fill remaining slots in original
  order so studies with sparse protocols stay usable.

  Args:
      series_rows: All index rows of one study.
      selection_cfg: Mapping with ``max_series_per_study`` and ``priority``
          (list of {plane, fluid_sensitive, fat_suppression} dicts).

  Returns:
      Selected subset of ``series_rows`` in priority order.
  """
  limit = int(selection_cfg['max_series_per_study'])
  chosen_indices: list[int] = []
  remaining = list(range(len(series_rows)))
  for rule in selection_cfg['priority']:
    if len(chosen_indices) >= limit:
      break
    for idx in list(remaining):
      row = series_rows.iloc[idx]
      if (
        row['plane'] == rule['plane']
        and int(row['fluid_sensitive']) == int(rule['fluid_sensitive'])
        and int(row['fat_suppression']) == int(rule['fat_suppression'])
        and idx not in chosen_indices
      ):
        chosen_indices.append(idx)
        remaining.remove(idx)
        if len(chosen_indices) >= limit:
          break
  for idx in remaining:
    if len(chosen_indices) >= limit:
      break
    chosen_indices.append(idx)
  return series_rows.iloc[chosen_indices]


def build_metadata(
  study_series: pd.DataFrame,
  sex_value: str,
  metadata_cfg: dict,
  plane_order: list[str],
  sex_order: list[str],
) -> np.ndarray:
  """Assemble the fixed-width study metadata vector.

  Layout (must equal ``model.metadata_dim``):
  plane one-hot(3) + sex one-hot(3) + fs/fl any-flags(2) +
  log1p(#series)(1) + log1p(#slices)(1) + mean spacing(1) + geometry flag(1).

  Args:
      study_series: Index rows of the study (all or selected subset).
      sex_value: PatientSex string from the index ('M'/'F'/'O'/other).
      metadata_cfg: ``metadata_features`` section from data.yaml.
      plane_order: Canonical plane ordering used for the one-hot.
      sex_order: Canonical sex values used for the one-hot.

  Returns:
      Float32 vector of width implied by the enabled feature switches.
  """
  first_row = study_series.iloc[0]
  plane_onehot = [float(first_row['plane'] == name) for name in plane_order]
  sex_onehot = (
    [float(str(sex_value).upper() == s) for s in sex_order]
    if metadata_cfg.get('use_sex', False)
    else []
  )
  flags = (
    [
      float(study_series['fluid_sensitive'].max()),
      float(study_series['fat_suppression'].max()),
    ]
    if len(study_series)
    else [0.0, 0.0]
  )
  counts = []
  if metadata_cfg.get('use_log_counts', True):
    counts.append(math.log1p(float(len(study_series))))
    counts.append(math.log1p(float(study_series['n_slices'].sum())))
  spacing = []
  if metadata_cfg.get('use_spacing_stats', True):
    column = study_series.get('pixel_spacing')
    spacing.append(
      float(column.mean())
      if column is not None and column.notna().any()
      else -1.0
    )
    has_geometry = (
      float(study_series['has_geometry'].astype(bool).any())
      if 'has_geometry' in study_series.columns
      else 0.0
    )
    spacing.append(has_geometry)
  return np.asarray(
    plane_onehot + sex_onehot + flags + counts + spacing, dtype=np.float32
  )


class StudyDataset(Dataset):
  """Yield model-ready tensors for one study per index."""

  def __init__(
    self,
    index_df: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    study_ids: list[str],
    reader: SeriesReader,
    augmentations: BaseCompose | None,
    img_size: int,
    n_slices: int,
    n_series_tokens_max: int,
    series_selection: dict,
    metadata_features: dict,
    normalize_output: dict,
    target_columns: list[str],
    dicom_root_override: str | None = None,
  ) -> None:
    """Store immutable references to shared state.

    Args:
        index_df: Header-scan index (one row per series).
        labels_df: Optional labels frame aligned on StudyInstanceUID;
            None marks inference usage (labels returned as unknown).
        study_ids: Studies exposed by this split.
        reader: Configured SeriesReader performing decode/normalize.
        augmentations: Albumentations pipeline applied per slice.
        img_size: Square resize target appended before user transforms.
        n_slices: Contract mirrored from data.yaml for shape checks.
        n_series_tokens_max: Padded series capacity at collate time.
        series_selection: Priority policy consumed by select_series.
        metadata_features: Feature switches consumed by build_metadata.
        normalize_output: {'mean': [...], 'std': [...]} applied post-augment.
        target_columns: Canonical 12-target column order in labels_df.
        dicom_root_override: Alternate DICOM root (test mount differs).
    """
    self.index_df = index_df
    self.labels_df = labels_df
    self.study_ids = study_ids
    self.reader = reader
    self.augmentations = augmentations
    self.img_size = img_size
    self.n_slices = n_slices
    self.n_series_tokens_max = n_series_tokens_max
    self.series_selection = series_selection
    self.metadata_features = metadata_features
    self.normalize_output = normalize_output
    self.target_columns = target_columns
    self.dicom_root = dicom_root_override or reader.dicom_root
    self._groups = dict(tuple(index_df.groupby('study')))
    self._label_lookup = (
      labels_df.set_index('StudyInstanceUID') if labels_df is not None else None
    )

  def __len__(self) -> int:
    """Return the number of studies in this split.

    Returns:
        Integer dataset size.
    """
    return len(self.study_ids)

  def _load_series_stack(self, row: pd.Series) -> np.ndarray:
    """Decode one series into its fixed-shape uint8 stack.

    Args:
        row: Single index row with ordered SOP list.

    Returns:
        ``(n_slices, H, W)`` uint8 array.

    Raises:
        ValueError: If decoded depth disagrees with configured n_slices.
    """
    record = {
      'study': row['study'],
      'series': row['series'],
      'sop_uids': row['sop_uids'],
      'rows': row.get('rows', 0),
      'cols': row.get('cols', 0),
    }
    original_root = self.reader.dicom_root
    if self.dicom_root != original_root:
      self.reader.dicom_root = self.dicom_root
    try:
      stack = self.reader.read(record)
    finally:
      self.reader.dicom_root = original_root
    if stack.shape[0] != self.n_slices:
      series_id = row['series']
      raise ValueError(
        f'Series {series_id} produced depth {stack.shape[0]} '
        f'!= configured n_slices {self.n_slices}'
      )
    return stack

  def _apply_transform(self, image: np.ndarray) -> torch.Tensor:
    """Run the full per-slice pipeline (resize/augment/normalize/tensor).

    Args:
        image: uint8 2D array.

    Returns:
        ``(3, img_size, img_size)`` float32 tensor produced by the
        composed pipeline (ToTensorV2 output).
    """
    return self.augmentations(image=image)['image']

  def __getitem__(self, idx: int) -> dict[str, Any]:
    """Materialize one study.

    Args:
        idx: Positional index within this split's study list.

    Returns:
        Dictionary with keys ``slices`` ``(n_sel * n_slices, H, W)``,
        ``slice_counts`` list padded to capacity, ``series_meta``
        ``(capacity, SERIES_META_DIM)``, ``metadata``, ``label``,
        and ``study_uid``.
    """
    study_uid = self.study_ids[idx]
    group: pd.DataFrame = self._groups[study_uid].sort_values('series')
    selected = select_series(
      group.reset_index(drop=True), self.series_selection
    )
    stacks = [self._load_series_stack(row) for _, row in selected.iterrows()]
    transformed = [
      self._apply_transform(stack[pos])
      for stack in stacks
      for pos in range(stack.shape[0])
    ]
    slices_tensor = torch.stack(transformed)  # (N, 3, S, S)

    counts = [self.n_slices] * len(selected)
    pad = self.n_series_tokens_max - len(counts)
    counts += [0] * pad
    meta_rows = np.zeros(
      (self.n_series_tokens_max, SERIES_META_DIM), dtype=np.float32
    )
    for pos, (_, row) in enumerate(selected.iterrows()):
      plane_idx = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}.get(
        row.get('plane'), 0
      )
      meta_rows[pos] = (
        float(plane_idx),
        float(row.get('fluid_sensitive', 0)),
        float(row.get('fat_suppression', 0)),
      )
    label_vector = np.full((len(self.target_columns),), -1.0, dtype=np.float32)
    if self._label_lookup is not None and study_uid in self._label_lookup.index:
      row = self._label_lookup.loc[study_uid]
      label_vector = np.asarray(
        [row[column] for column in self.target_columns], dtype=np.float32
      )
    metadata = build_metadata(
      selected,
      str(group['sex'].iloc[0]) if 'sex' in group.columns else 'Unknown',
      self.metadata_features,
      plane_order=['Sagittal', 'Coronal', 'Axial'],
      sex_order=['M', 'F', 'O'],
    )
    return {
      'slices': slices_tensor,
      'slice_counts': torch.tensor(counts, dtype=torch.long),
      'series_meta': torch.from_numpy(meta_rows),
      'metadata': torch.from_numpy(metadata),
      'label': torch.from_numpy(label_vector),
      'study_uid': study_uid,
    }


def collate_studies(batch: list[dict[str, Any]]) -> dict[str, Any]:
  """Stack per-study dictionaries into KneeNet's flat batch contract.

  Slice tensors are concatenated across the batch; consumers re-split them
  via ``slice_counts`` cumulative offsets inside SeriesEncoder.

  Args:
      batch: List of StudyDataset items.

  Returns:
      Dictionary matching ``KneeNet.forward`` expectations plus UIDs.
  """
  flat_slices = torch.cat([item['slices'] for item in batch], dim=0)
  return {
    'slices': flat_slices,
    'slice_counts': torch.stack([item['slice_counts'] for item in batch]),
    'series_meta': torch.stack([item['series_meta'] for item in batch]),
    'metadata': torch.stack([item['metadata'] for item in batch]),
    'label': torch.stack([item['label'] for item in batch]),
    'study_uid': [item['study_uid'] for item in batch],
  }
