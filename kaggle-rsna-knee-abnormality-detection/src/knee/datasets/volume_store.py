#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Disk-free volume IO primitives: decode, stream-cache, disk/shard utils.

Kaggle constraint this module solves (BLUEPRINT section 9): the raw
competition data is ~570 GB while ``/kaggle/working`` offers only 30 GB,
so a complete ``volumes_cache`` of npz files cannot exist. Volumes are
decoded on demand straight from the read-only DICOM mount and kept in a
bounded LRU RAM cache (:class:`StreamingVolumeStore`) so repeated
sampling -- common under balanced oversampling -- never re-decodes.

An existing npz cache (e.g. a partial shard minted by
``scripts/prepare_volumes.py`` and mounted from ``/kaggle/input``) is
still honoured when present: shards are treated as read-only fast paths
and nothing is ever written to them.

Design notes:
* SRP: this module owns *how single volumes are read* (DICOM decode,
  slice ordering, windowing) plus cache/disk/shard policy;
  ``volume_builder`` owns the optional batch-caching pipeline; datasets
  own study assembly.
* DIP: callers depend on the small :class:`StreamingVolumeStore`
  surface, so alternative stores drop in without touching datasets.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from knee.config_params.schema import DataConfig
from knee.helpers.logging_utils import get_logger


def free_disk_bytes(path: str | Path) -> int:
  """Report free space on the filesystem holding ``path``.

  Args:
      path: Any path on the filesystem to inspect.

  Returns:
      Free bytes; 0 when the path cannot be stat'd.
  """
  try:
    return shutil.disk_usage(os.path.abspath(path)).free
  except OSError:
    return 0


def shard_indices(total: int, shard: int, num_shards: int) -> list[int]:
  """Deterministically partition ``range(total)`` across kernels.

  Round-robin assignment keeps every shard equally sized and stable
  across runs, which multi-kernel caching requires.

  Args:
      total: Number of records to partition.
      shard: Zero-based shard id.
      num_shards: Total number of shards (>= 1).

  Returns:
      Ascending list of indices belonging to ``shard``.

  Raises:
      ValueError: If ``shard`` lies outside ``[0, num_shards)``.
  """
  if not 0 <= shard < max(num_shards, 1):
    raise ValueError(f'shard {shard} out of range [0, {num_shards})')
  return list(range(shard, total, max(num_shards, 1)))


def series_dir(
  data_root: str, series_uid: str, study_uid: str, split: str = 'train'
) -> Path:
  """Resolve the DICOM directory of one series under the data root.

  Supports both nested ``<root>/<split>_series/<study>/<series>`` and
  flat ``<root>/<split>_series/<series>`` layouts.

  Args:
      data_root: Competition dataset root.
      series_uid: SeriesInstanceUID.
      study_uid: StudyInstanceUID (may be empty in flat layouts).
      split: 'train' or 'test'; names the series subdirectory.

  Returns:
      Existing directory path.

  Raises:
      FileNotFoundError: If no layout matches.
  """
  root = Path(data_root)
  candidates = []
  for base in (root / f'{split}_series', root):
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
  positions = [getattr(ds, 'ImagePositionPatient', None) for ds in datasets]
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
  uids = [
    str(getattr(ds, 'SOPInstanceUID', p.name))
    for ds, p in zip(datasets, paths, strict=False)
  ]
  return [p for _, p in sorted(zip(uids, paths, strict=False))]


def decode_series_volume(
  directory: Path | str,
  image_size: int,
  num_slices: int,
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


class LruVolumeCache:
  """Size-bounded LRU cache for decoded uint8 volumes.

  Thread-safe via a lock; eviction is by recency with byte accounting
  so worker RAM stays bounded regardless of series shapes.

  Attributes:
      max_items: Hard cap on cached volumes.
      max_bytes: Soft cap on summed ``ndarray.nbytes``.
  """

  def __init__(self, max_items: int = 256, max_bytes: int = 4 << 30) -> None:
    """Create an empty bounded cache.

    Args:
        max_items: Maximum number of volumes retained.
        max_bytes: Maximum retained payload bytes (default 4 GiB).
    """
    self.max_items = int(max_items)
    self.max_bytes = int(max_bytes)
    self._entries: OrderedDict[str, np.ndarray] = OrderedDict()
    self._bytes = 0
    self._lock = threading.Lock()

  def __len__(self) -> int:
    """Number of cached volumes.

    Returns:
        Current entry count.
    """
    with self._lock:
      return len(self._entries)

  @property
  def nbytes(self) -> int:
    """Total payload bytes currently retained.

    Returns:
        Sum of cached array sizes in bytes.
    """
    with self._lock:
      return self._bytes

  def get(self, key: str) -> np.ndarray | None:
    """Return the cached volume for ``key``, refreshing recency.

    Args:
        key: SeriesInstanceUID cache key.

    Returns:
        Cached uint8 volume or None on miss.
    """
    with self._lock:
      if key not in self._entries:
        return None
      self._entries.move_to_end(key)
      return self._entries[key]

  def put(self, key: str, volume: np.ndarray) -> None:
    """Insert a volume, evicting least-recently-used entries as needed.

    Args:
        key: SeriesInstanceUID cache key.
        volume: Decoded uint8 ``(C, H, W)`` volume to retain.
    """
    oversized = volume.nbytes > self.max_bytes
    with self._lock:
      if key in self._entries:
        self._bytes -= self._entries.pop(key).nbytes
      if oversized:
        return  # never let one giant volume evict everything
      self._entries[key] = volume
      self._bytes += volume.nbytes
      while len(self._entries) > self.max_items or (
        self._bytes > self.max_bytes and len(self._entries) > 1
      ):
        _, evicted = self._entries.popitem(last=False)
        self._bytes -= evicted.nbytes


class StreamingVolumeStore:
  """Decode-on-demand series volumes behind an LRU + optional npz shards.

  Resolution order per series:
  1. in-memory LRU (per-process, so DataLoader workers are independent),
  2. ``<cache_dir>/<SeriesInstanceUID>.npz`` for each dir of
     ``cache_dirs`` (multiple shard datasets may be mounted),
  3. live DICOM decode from the competition mount.

  Failures (missing directory, unreadable DICOMs) are memoized and
  return ``None`` so one broken series cannot crash an epoch.

  Attributes:
      data_cfg: Volume shaping parameters (size/depth/windowing).
      data_root: Competition dataset root holding the DICOM tree.
      split: 'train' or 'test'; selects which subdirectory layout applies.
      cache_dirs: Optional read-only npz shard dirs.
  """

  def __init__(
    self,
    data_root: str,
    data_cfg: DataConfig,
    split: str = 'train',
    cache_dir: str | None = None,
    max_items: int = 256,
    max_bytes: int = 4 << 30,
  ) -> None:
    """Store configuration only; nothing is read until first lookup.

    Args:
        data_root: Dataset root containing ``<split>_series`` dirs.
        data_cfg: Volume shaping parameters for the decoder.
        split: Which split's DICOM tree to resolve against.
        cache_dir: Optional npz cache location(s); multiple shards are
            separated by ``os.pathsep`` ('' or None disables).
        max_items: LRU capacity in volumes.
        max_bytes: LRU capacity in approximate bytes.
    """
    self.data_cfg = data_cfg
    self.data_root = str(data_root)
    self.split = split
    self.cache_dirs: tuple[str, ...] = tuple(
      d for d in str(cache_dir or '').split(os.pathsep) if d
    )
    self._cache = LruVolumeCache(max_items=max_items, max_bytes=max_bytes)
    self._failed: set[str] = set()
    # Decode telemetry: surfaced as periodic INFO summaries so streamed
    # kernel logs show the CPU-side pipeline working before any GPU use.
    self._decode_count = 0
    self._decode_seconds = 0.0
    self._log_every = 10
    self._slow_seconds = 5.0
    self._log = get_logger('volume_store')

  def _read_npz(self, series_uid: str) -> np.ndarray | None:
    """Load one volume from the optional disk-cache shards.

    Args:
        series_uid: SeriesInstanceUID file stem.

    Returns:
        uint8 volume or None when absent/unreadable everywhere.
    """
    for cache_root in self.cache_dirs:
      path = Path(cache_root) / f'{series_uid}.npz'
      if not path.exists():
        continue
      try:
        with np.load(path) as payload:
          return payload['volume']
      except Exception as exc:  # pylint: disable=broad-exception-caught
        self._log.warning('npz %s unreadable (%s); skipping', path, exc)
    return None

  def _decode(self, row: dict) -> np.ndarray | None:
    """Decode one series straight from its DICOM directory.

    Args:
        row: Series record with Study/SeriesInstanceUID.

    Returns:
        uint8 ``(C, H, W)`` volume, or None when undecodable.
    """
    study = str(row.get('StudyInstanceUID', '') or '')
    series = str(row.get('SeriesInstanceUID', '') or '')
    if series in self._failed:
      return None
    started = time.perf_counter()
    try:
      directory = series_dir(self.data_root, series, study, split=self.split)
      n_files = len(list(directory.glob('*.dcm')))
      volume = decode_series_volume(
        directory,
        image_size=int(self.data_cfg.image_size),
        num_slices=int(self.data_cfg.num_slices),
        percentile_clip=tuple(self.data_cfg.percentile_clip),
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Memoize failures: broken series would otherwise re-pay the full
      # directory scan + decode attempt on every epoch/sample.
      self._failed.add(series)
      self._log.warning(
        'series %s undecodable (%s); substituting zeros', series, exc
      )
      return None
    elapsed = time.perf_counter() - started
    self._decode_count += 1
    self._decode_seconds += elapsed
    if elapsed >= self._slow_seconds:
      self._log.warning(
        '[pid %d] SLOW decode series %s: %d dcm files in %.1fs',
        os.getpid(),
        series,
        n_files,
        elapsed,
      )
    elif self._decode_count % self._log_every == 0:
      avg = self._decode_seconds / self._decode_count
      self._log.info(
        '[pid %d] decoded %d series so far (%.1fs total, %.2fs avg/series)',
        os.getpid(),
        self._decode_count,
        self._decode_seconds,
        avg,
      )
    return volume

  def get(self, row: dict) -> np.ndarray | None:
    """Fetch one series volume through LRU -> npz shards -> DICOM.

    Args:
        row: Series-table record carrying at least SeriesInstanceUID
            and optionally StudyInstanceUID/cache_path.

    Returns:
        uint8 ``(C, H, W)`` volume or None when unavailable.
    """
    series = str(row.get('SeriesInstanceUID', '') or '')
    hit = self._cache.get(series)
    if hit is not None:
      return hit
    explicit = str(row.get('cache_path', '') or '')
    volume = None
    if explicit and Path(explicit).exists():
      try:
        with np.load(explicit) as payload:
          volume = payload['volume']
      except Exception:  # pylint: disable=broad-exception-caught
        volume = None
    if volume is None:
      volume = self._read_npz(series)
    if volume is None:
      volume = self._decode(row)
    if volume is not None:
      self._cache.put(series, volume)
    return volume
