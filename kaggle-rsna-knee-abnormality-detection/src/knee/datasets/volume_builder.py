#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 1 engine: decode every DICOM series once into an npz cache.

Kaggle kernels are I/O-bound on DICOM; decoding days become hours when
each series is read exactly once and stored as a compressed uint8 array
(BLUEPRINT section 2). Per-series processing:

1. sort slices by ``ImagePositionPatient . slice_normal`` when position
   tags exist, else by ``InstanceNumber``, else SOPInstanceUID,
2. percentile-window intensities (p1-p99 over non-air voxels) to uint8,
3. resize in-plane to ``image_size``,
4. uniformly resample to exactly ``num_slices`` slices,
5. save ``<cache-dir>/<SeriesInstanceUID>.npz`` with key ``volume``.

The returned manifest carries study/series ids, cache paths, raw slice
counts and any metadata columns from the input CSV for downstream use.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from knee.config_params.schema import DataConfig
from knee.helpers.logging_utils import get_logger

_VOLUME_KEY = 'volume'


def _series_dir(data_root: str, series_uid: str, study_uid: str) -> Path:
  """Resolve the DICOM directory of one series under the data root.

  Supports both nested ``<root>/train_series/<study>/<series>`` and flat
  ``<root>/train_series/<series>`` layouts.

  Args:
      data_root: Competition dataset root.
      series_uid: SeriesInstanceUID.
      study_uid: StudyInstanceUID (may be empty in flat layouts).

  Returns:
      Existing directory path.

  Raises:
      FileNotFoundError: If no layout matches.
  """
  root = Path(data_root)
  candidates = []
  for base in (root / 'train_series', root):
    if study_uid:
      candidates.append(base / str(study_uid) / str(series_uid))
    candidates.append(base / str(series_uid))
  for candidate in candidates:
    if candidate.is_dir():
      return candidate
  raise FileNotFoundError(
    f'no DICOM dir for series {series_uid} under {data_root}'
  )


def _sort_paths(paths: list) -> list:
  """Order slices anatomically with graceful tag fallbacks.

  Args:
      paths: Paths to single-frame DICOM files of one series.

  Returns:
      Paths ordered along the slice axis.
  """
  try:
    # pylint: disable=import-outside-toplevel
    import pydicom

    datasets = [pydicom.dcmread(str(p), stop_before_pixels=True) for p in paths]
  except Exception:  # pylint: disable=broad-exception-caught
    return sorted(paths, key=lambda p: p.name)
  positions = [
    getattr(ds, 'ImagePositionPatient', None) for ds in datasets
  ]
  if all(p is not None for p in positions) and len(datasets) > 1:
    delta = np.array(positions[1], dtype=float) - np.array(
      positions[0], dtype=float
    )
    norm = np.linalg.norm(delta)
    if norm > 1e-6:
      normal = delta / norm
      keys = [
        float(np.dot(np.array(p, dtype=float), normal)) for p in positions
      ]
      return [p for _, p in sorted(zip(keys, paths, strict=False))]
  numbers = [getattr(ds, 'InstanceNumber', None) for ds in datasets]
  if all(n is not None for n in numbers):
    order = sorted(range(len(numbers)), key=lambda i: int(numbers[i]))
    return [paths[i] for i in order]
  uids = [str(getattr(ds, 'SOPInstanceUID', p.name)) for ds, p in
          zip(datasets, paths, strict=False)]
  return [p for _, p in sorted(zip(uids, paths, strict=False))]


def decode_series_volume(
  directory: Path | str, image_size: int, num_slices: int,
  percentile_clip: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
  """Decode one DICOM series into a fixed-shape uint8 volume.

  Args:
      directory: Directory holding the series' DICOM files.
      image_size: In-plane resize target (square).
      num_slices: Output depth after uniform resampling.
      percentile_clip: Windowing percentiles over non-zero voxels.

  Returns:
      uint8 array ``(num_slices, image_size, image_size)``.

  Raises:
      ValueError: If the series contains no readable slices.
  """
  # pylint: disable=import-outside-toplevel
  import cv2
  import pydicom

  paths = _sort_paths(sorted(Path(directory).glob('*.dcm')))
  slices = []
  lo, hi = percentile_clip
  for path in paths:
    try:
      arr = pydicom.dcmread(str(path)).pixel_array.astype(np.float32)
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    slices.append(arr)
  if not slices:
    raise ValueError(f'no readable DICOM slices in {directory}')
  stack = [s for s in slices if s.size]
  reference_shape = stack[0].shape
  stack = [s for s in stack if s.shape == reference_shape]
  pooled = np.concatenate([s.ravel() for s in stack])
  nonzero = pooled[pooled > 0]
  source = nonzero if nonzero.size >= max(1024, pooled.size // 100) else pooled
  p_lo, p_hi = np.percentile(source, [lo, hi])
  span = max(float(p_hi - p_lo), 1e-6)
  resized = []
  for arr in stack:
    clipped = np.clip((arr - float(p_lo)) / span, 0.0, 1.0)
    plane = (clipped * 255.0).astype(np.uint8)
    if plane.shape != (image_size, image_size):
      plane = cv2.resize(
        plane, (image_size, image_size), interpolation=cv2.INTER_AREA
      )
    resized.append(plane)
  indices = np.linspace(0, len(resized) - 1, num_slices).round().astype(int)
  return np.stack([resized[i] for i in indices])


def _process_one(job: dict) -> dict:
  """Worker entry point decoding a single series.

  Args:
      job: Dict with series/study ids, data_root and DataConfig fields.

  Returns:
      Manifest row dict including status and cache_path ('' on failure).
  """
  try:
    directory = _series_dir(
      job['data_root'], job['SeriesInstanceUID'], job['StudyInstanceUID']
    )
    volume = decode_series_volume(
      directory,
      job['image_size'],
      job['num_slices'],
      tuple(job['percentile_clip']),
    )
    out = Path(job['cache_dir']) / f"{job['SeriesInstanceUID']}.npz"
    np.savez_compressed(out, **{_VOLUME_KEY: volume})
    row = {k: v for k, v in job.items() if k not in {
      'data_root', 'cache_dir', 'image_size', 'num_slices',
      'percentile_clip', 'min_series_slices',
    }}
    row.update(
      n_slices=int(volume.shape[0]), cache_path=str(out), status='ok'
    )
    return row
  except Exception as exc:  # pylint: disable=broad-exception-caught
    row = {k: v for k, v in job.items() if k not in {
      'data_root', 'cache_dir', 'image_size', 'num_slices',
      'percentile_clip', 'min_series_slices',
    }}
    row.update(cache_path='', status=f'failed: {exc}')
    return row


def prepare_all(
  series_csv: str,
  data_root: str,
  data_cfg: DataConfig,
  cache_dir: str,
  workers: int = 4,
) -> pd.DataFrame:
  """Decode every listed series into the cache and tabulate a manifest.

  Args:
      series_csv: Per-series CSV (SeriesInstanceUID required; optional
          StudyInstanceUID plus acquisition-metadata columns).
      data_root: Dataset root containing the DICOM tree.
      data_cfg: Volume shaping parameters (size/depth/windowing).
      cache_dir: Output directory for npz files.
      workers: Parallel decode processes.

  Returns:
      One manifest row per input series with columns
      ``[SeriesInstanceUID, StudyInstanceUID, ..., n_slices,
      cache_path, status]``, preserving extra CSV columns.
  """
  log = get_logger('volume_builder')
  table = pd.read_csv(series_csv, dtype=str)
  if 'SeriesInstanceUID' not in table.columns:
    raise ValueError(f'{series_csv} lacks a SeriesInstanceUID column')
  if 'StudyInstanceUID' not in table.columns:
    table['StudyInstanceUID'] = ''
  cache = Path(cache_dir)
  cache.mkdir(parents=True, exist_ok=True)

  jobs = []
  for record in table.to_dict('records'):
    jobs.append(
      {
        **record,
        'data_root': str(data_root),
        'cache_dir': str(cache),
        'image_size': int(data_cfg.image_size),
        'num_slices': int(data_cfg.num_slices),
        'percentile_clip': list(data_cfg.percentile_clip),
        'min_series_slices': int(data_cfg.min_series_slices),
      }
    )

  rows: list[dict] = []
  if workers > 1:
    with ProcessPoolExecutor(max_workers=workers) as pool:
      futures = [pool.submit(_process_one, job) for job in jobs]
      for done, future in enumerate(as_completed(futures), start=1):
        rows.append(future.result())
        if done % 50 == 0 or done == len(jobs):
          log.info('decoded %d/%d series', done, len(jobs))
  else:
    for index, job in enumerate(jobs, start=1):
      rows.append(_process_one(job))
      if index % 50 == 0 or index == len(jobs):
        log.info('decoded %d/%d series', index, len(jobs))

  manifest = pd.DataFrame(rows)
  failures = manifest[manifest['status'] != 'ok']
  if len(failures):
    log.warning('%d series failed to decode', len(failures))
  return manifest.reset_index(drop=True)
