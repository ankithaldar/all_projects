#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optional Kernel 1 engine: batch-decode DICOM series into npz shards.

Kaggle kernels are I/O-bound on DICOM; decoding once into compressed
uint8 volumes (BLUEPRINT section 2) turns days into hours -- but the
full tree (~570 GB) cannot fit Kaggle's 30 GB of scratch, so this
pipeline is *bounded* and *optional*:

* existing ``.npz`` outputs are skipped, so interrupted kernels resume;
* decoding stops early when free disk drops below a floor
  (:func:`knee.datasets.volume_store.free_disk_bytes`);
* ``shard``/``num_shards`` slice the series table across kernels,
  each publishing its shard as a separate versioned dataset.

Training/inference do not require this cache: they stream-decode through
:class:`knee.datasets.volume_store.StreamingVolumeStore`. Per-series
processing (ordering, windowing, resize, depth resampling) lives in the
shared decoder :func:`volume_store.decode_series_volume`.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from knee.config_params.schema import DataConfig
from knee.datasets.volume_store import (
  decode_series_volume,
  free_disk_bytes,
  series_dir,
  shard_indices,
)
from knee.helpers.logging_utils import get_logger

_VOLUME_KEY = 'volume'

__all__ = [
  'decode_series_volume',
  'prepare_all',
  'synthesize_series_table',
]


def _process_one(job: dict) -> dict:
  """Worker entry point decoding a single series.

  Existing npz outputs are never recomputed, so an interrupted kernel
  (Kaggle's 12 h wall) resumes where it stopped.

  Args:
      job: Dict with series/study ids, data_root and DataConfig fields.

  Returns:
      Manifest row dict including status and cache_path ('' on failure).
  """
  row = {
    k: v
    for k, v in job.items()
    if k
    not in {
      'data_root',
      'cache_dir',
      'image_size',
      'num_slices',
      'percentile_clip',
      'min_series_slices',
    }
  }
  out = Path(job['cache_dir']) / f'{job["SeriesInstanceUID"]}.npz'
  if out.exists():
    # NaN slice count = unknown; downstream localizer filters keep it.
    row.update(n_slices=float('nan'), cache_path=str(out), status='cached')
    return row
  try:
    directory = series_dir(
      job['data_root'], job['SeriesInstanceUID'], job['StudyInstanceUID']
    )
    volume = decode_series_volume(
      directory,
      job['image_size'],
      job['num_slices'],
      tuple(job['percentile_clip']),
    )
    np.savez_compressed(out, **{_VOLUME_KEY: volume})
    row.update(n_slices=int(volume.shape[0]), cache_path=str(out), status='ok')
    return row
  except Exception as exc:  # pylint: disable=broad-exception-caught
    row.update(cache_path='', status=f'failed: {exc}')
    return row


def synthesize_series_table(
  data_root: str, split: str = 'train'
) -> pd.DataFrame:
  """Discover series by walking the DICOM tree when no CSV exists.

  Supports nested ``<split>_series/<study>/<series>`` layouts (both ids
  recorded) and flat ``<split>_series/<series>`` layouts (empty study).

  Args:
      data_root: Competition dataset root.
      split: 'train' or 'test'; selects which directory to walk.

  Returns:
      Frame with StudyInstanceUID/SeriesInstanceUID string columns.
  """
  base = Path(data_root) / f'{split}_series'
  base = base if base.is_dir() else Path(data_root)
  rows = []
  for current, dirs, files in os.walk(base):
    if not any(name.lower().endswith('.dcm') for name in files):
      continue
    parts = Path(current).relative_to(base).parts
    rows.append(
      {
        'SeriesInstanceUID': parts[-1],
        'StudyInstanceUID': parts[-2] if len(parts) > 1 else '',
      }
    )
    dirs[:] = []  # a series dir holds only slices; do not descend
  return pd.DataFrame(rows, dtype=str)


def prepare_all(
  series_csv: str,
  data_root: str,
  data_cfg: DataConfig,
  cache_dir: str,
  workers: int = 4,
  shard: int = 0,
  num_shards: int = 1,
  min_free_gb: float = 2.0,
) -> pd.DataFrame:
  """Decode listed series into a partial npz shard and tabulate it.

  Args:
      series_csv: Per-series CSV (SeriesInstanceUID required; optional
          StudyInstanceUID plus acquisition-metadata columns).
      data_root: Dataset root containing the DICOM tree.
      data_cfg: Volume shaping parameters (size/depth/windowing).
      cache_dir: Output directory for npz files.
      workers: Parallel decode processes.
      shard: Zero-based shard id for multi-kernel caching.
      num_shards: Total number of shards (1 = process everything).
      min_free_gb: Abort new decodes below this much remaining space.

  Returns:
      One manifest row per input series with columns
      ``[SeriesInstanceUID, StudyInstanceUID, ..., n_slices,
      cache_path, status]``, preserving extra CSV columns.
  """
  log = get_logger('volume_builder')
  csv_path = Path(series_csv)
  if csv_path.exists():
    table = pd.read_csv(series_csv, dtype=str)
  else:
    log.warning(
      '%s not found; synthesizing series table by walking %s',
      series_csv,
      data_root,
    )
    table = synthesize_series_table(str(data_root), 'train')
  if 'SeriesInstanceUID' not in table.columns:
    raise ValueError(f'{series_csv} lacks a SeriesInstanceUID column')
  if 'StudyInstanceUID' not in table.columns:
    table['StudyInstanceUID'] = ''
  cache = Path(cache_dir)
  cache.mkdir(parents=True, exist_ok=True)

  records = table.to_dict('records')
  keep = set(shard_indices(len(records), shard, num_shards))
  jobs = []
  for index, record in enumerate(records):
    if index not in keep:
      continue
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
  floor_bytes = int(min_free_gb * (1 << 30))

  def _disk_ok() -> bool:
    """Check writable-disk headroom before spending decode time."""
    return free_disk_bytes(cache) >= floor_bytes

  if workers > 1:
    # Bounded in-flight batches bound the disk overshoot between free
    # space checks while still keeping `workers` processes saturated.
    batch_size = max(workers * 2, 4)
    with ProcessPoolExecutor(max_workers=workers) as pool:
      for start in range(0, len(jobs), batch_size):
        if not _disk_ok():
          log.warning(
            'free disk below %.1f GB; stopping cache build early '
            '(partial shard kept; re-run to continue)',
            min_free_gb,
          )
          break
        batch = jobs[start : start + batch_size]
        futures = [pool.submit(_process_one, job) for job in batch]
        for future in as_completed(futures):
          rows.append(future.result())
        log.info('decoded %d/%d series', len(rows), len(jobs))
  else:
    for index, job in enumerate(jobs, start=1):
      if not _disk_ok():
        log.warning(
          'free disk below %.1f GB; stopping cache build early at '
          '%d/%d series (re-run to continue)',
          min_free_gb,
          index - 1,
          len(jobs),
        )
        break
      rows.append(_process_one(job))
      if index % 50 == 0 or index == len(jobs):
        log.info('decoded %d/%d series', index, len(jobs))

  manifest = pd.DataFrame(rows)
  failures = manifest[manifest['status'].str.startswith('failed')]
  if len(failures):
    log.warning('%d series failed to decode', len(failures))
  return manifest.reset_index(drop=True)
