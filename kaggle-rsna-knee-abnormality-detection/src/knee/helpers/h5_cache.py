#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HDF5 volume cache: one-time decode of every indexed series.

Decoding 819k DICOM frames on the fly starves both T4s; this module turns
the per-series pipeline (decode every slice -> percentile normalize ->
autocrop -> resize -> uint8) into sharded HDF5 files written once and read
forever:

* :class:`ShardWriter` rolls ``volume_shard_NNN.h5`` files under an
  uncompressed byte cap, each dataset keyed by SeriesInstanceUID with
  shape ``(n_slices_i, img_size, img_size)`` and per-slice chunks so
  training reads only its sampled rows.
* :func:`decode_series_volume` is the pure worker used by
  ``main.py build-cache``; a series is cached only when *every* frame
  decodes, so zero-fallback frames can never fossilize into storage.
* :class:`H5SeriesReader` implements the exact ``read(record)`` contract
  of :class:`knee.datasets.series_dataset.SeriesReader`, sampling
  ``n_slices`` rows via h5py fancy indexing. Cache misses fall back to
  live decoding, so training never hard-fails on an incomplete cache.
* The manifest (``cache_manifest.parquet``) maps series to shards and
  merges across multiple mounted roots via :func:`load_manifest`.

Parallelism mirrors helpers.header_scan: a ProcessPoolExecutor over
series records carrying their own configuration, avoiding pickling
closures or per-worker initializers.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor

import cv2
import h5py
import numpy as np
import pandas as pd

from knee.helpers import intensity
from knee.helpers.dicom_io import DecoderRegistry
from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

MANIFEST_NAME = 'cache_manifest.parquet'
SHARD_SUFFIX = '.h5'
DEFAULT_SHARD_GIB = 10
DEFAULT_GZIP_LEVEL = 4
GIB = 1024**3

_SHARD_RE = re.compile(r'volume_shard_(\d+)\.h5$')


def _shard_index(name: str) -> int | None:
  """Extract numeric ordinal from a shard filename.

  Args:
      name: Basename such as ``volume_shard_007.h5``.

  Returns:
      Ordinal, or None for non-shard names.
  """
  match = _SHARD_RE.search(name)
  return int(match.group(1)) if match else None


# --------------------------------------------------------------------------- #
# Writer                                                                       #
# --------------------------------------------------------------------------- #


class ShardWriter:
  """Rolling HDF5 shard writer capped by uncompressed byte accounting."""

  def __init__(
    self,
    cache_dir: str,
    img_size: int,
    shard_bytes_cap: int = DEFAULT_SHARD_GIB * GIB,
    gzip_level: int = DEFAULT_GZIP_LEVEL,
  ) -> None:
    """Prepare the shard directory and resume state.

    Args:
        cache_dir: Destination directory, created when absent.
        img_size: Square edge every stored slice was resized to.
        shard_bytes_cap: Uncompressed bytes tolerated per shard file
            before rolling to the next one.
        gzip_level: h5py deflate level applied per chunk.
    """
    os.makedirs(cache_dir, exist_ok=True)
    self.cache_dir = cache_dir
    self.img_size = img_size
    self.shard_bytes_cap = int(shard_bytes_cap)
    self.gzip_level = gzip_level
    self._handle: h5py.File | None = None
    self.shard_name = ''
    self._bytes_in_shard = 0
    ordinals = sorted((_shard_index(n) or -1 for n in os.listdir(cache_dir)))
    self._next_index = ordinals[-1] + 1 if ordinals else 0
    self._session_uids: set[str] = set()
    self.series_written = 0

  def _uncompressed_size(self, name: str) -> int:
    """Exact uncompressed payload of a shard file.

    Args:
        name: Shard basename inside cache_dir.

    Returns:
        Sum of stored dataset nbytes; 0 when unreadable.
    """
    try:
      with h5py.File(os.path.join(self.cache_dir, name), 'r') as probe:
        return int(sum(ds.nbytes for ds in probe.values()))
    except OSError:
      return 0

  def existing_uids(self) -> set[str]:
    """Union of series keys already on disk across every shard.

    Returns:
        Set of SeriesInstanceUIDs present in any readable shard.
    """
    found: set[str] = set()
    for name in sorted(os.listdir(self.cache_dir)):
      if not name.endswith(SHARD_SUFFIX):
        continue
      try:
        with h5py.File(os.path.join(self.cache_dir, name), 'r') as handle:
          found.update(handle.keys())
      except OSError as exc:
        _LOGGER.warning('Skipping unreadable shard %s (%s)', name, exc)
    return found

  def _current_path(self) -> str:
    """Absolute path of the shard currently being written.

    Returns:
        Shard filename derived from the next ordinal.
    """
    return os.path.join(
      self.cache_dir, f'volume_shard_{self._next_index:03d}.h5'
    )

  def _roll(self, incoming_bytes: int) -> None:
    """Open the active shard, rolling over when capacity demands it.

    First call of a resumed session re-attaches to the newest existing
    shard when it still has room, using exact uncompressed accounting.

    Args:
        incoming_bytes: Uncompressed size about to be appended.
    """
    if self._handle is not None and (
      self._bytes_in_shard + incoming_bytes <= self.shard_bytes_cap
    ):
      return
    if self._handle is not None:
      self._handle.close()
      self._next_index += 1

    newest = f'volume_shard_{self._next_index - 1:03d}.h5'
    newest_path = os.path.join(self.cache_dir, newest)
    if self.shard_name == '' and os.path.exists(newest_path):
      used = self._uncompressed_size(newest)
      if 0 < used + incoming_bytes <= self.shard_bytes_cap:
        # Resume into the partial newest shard.
        self.shard_name = newest
        self._bytes_in_shard = used
      else:
        while os.path.exists(self._current_path()):
          self._next_index += 1
        self.shard_name = os.path.basename(self._current_path())
        self._bytes_in_shard = 0
    elif not self.shard_name:
      while os.path.exists(self._current_path()):
        self._next_index += 1
      self.shard_name = os.path.basename(self._current_path())
      self._bytes_in_shard = 0
    self._handle = h5py.File(os.path.join(self.cache_dir, self.shard_name), 'a')

  def add_series(self, series_uid: str, study_uid: str, stack: np.ndarray):
    """Append one full decoded volume, rolling shards as required.

    Args:
        series_uid: Primary key of the stored dataset.
        study_uid: Recorded as dataset attribute for traceability.
        stack: ``(n_slices, img_size, img_size)`` uint8 array.

    Returns:
        Shard filename the series landed in.

    Raises:
        ValueError: When dtype/shape contradict writer configuration.
    """
    if stack.dtype != np.uint8 or stack.ndim != 3:
      raise ValueError(
        f'expected (n,{self.img_size},{self.img_size}) uint8; '
        f'got {stack.shape} {stack.dtype}'
      )
    if series_uid in self._session_uids:
      raise ValueError(
        f'duplicate series key within one cache build: {series_uid!r} '
        '(index rows must be unique; check for shared dir names across '
        'study roots)'
      )
    incoming = int(stack.nbytes)
    self._roll(incoming)
    # Chunks never exceed actual spatial extents (synthetic/small volumes
    # may be smaller than the configured img_size).
    height = min(int(stack.shape[1]), self.img_size)
    width = min(int(stack.shape[2]), self.img_size)
    dataset = self._handle.create_dataset(
      series_uid,
      data=stack,
      chunks=(1, height, width),
      compression='gzip',
      compression_opts=self.gzip_level,
    )
    dataset.attrs['study_uid'] = str(study_uid)
    self._session_uids.add(series_uid)
    self._bytes_in_shard += incoming
    self.series_written += 1
    return self.shard_name

  def write_manifest(self, extra_columns: pd.DataFrame | None = None) -> str:
    """Persist the uid -> shard/n_slices mapping beside the shards.

    Source of truth is the files themselves, so manifests regenerate
    identically after resumed runs.

    Args:
        extra_columns: Optional frame keyed on SeriesInstanceUID merged
            in (e.g. push-time slug assignment).

    Returns:
        Manifest path.
    """
    mapping = collect_shard_map(self.cache_dir)
    frame = pd.DataFrame(
      {
        'SeriesInstanceUID': [uid for uid, _ in mapping.items()],
        'shard_file': [shard for _, (shard, _) in mapping.items()],
        'n_slices': [n for _, (_, n) in mapping.items()],
      }
    )
    if extra_columns is not None and len(extra_columns):
      frame = frame.merge(extra_columns, on='SeriesInstanceUID', how='left')
    path = os.path.join(self.cache_dir, MANIFEST_NAME)
    frame.to_parquet(path, index=False)
    return path

  def close(self) -> None:
    """Flush/close the currently open shard."""
    if self._handle is not None:
      self._handle.close()
      self._handle = None

  def __enter__(self) -> 'ShardWriter':
    return self

  def __exit__(self, *exc_info) -> None:
    self.close()


def collect_shard_map(cache_dir: str) -> dict[str, tuple[str, int]]:
  """Build uid -> (shard basename, n_slices) straight from files.

  Args:
      cache_dir: Directory containing volume_shard_*.h5 files.

  Returns:
      Mapping covering every series present in readable shards.
  """
  mapping: dict[str, tuple[str, int]] = {}
  for name in sorted(os.listdir(cache_dir)):
    if not name.endswith(SHARD_SUFFIX):
      continue
    try:
      with h5py.File(os.path.join(cache_dir, name), 'r') as handle:
        for key, dataset in handle.items():
          if key in mapping:
            _LOGGER.error(
              'duplicate series key %s in %s (already from %s); '
              'manifest keeps the later occurrence',
              key,
              name,
              mapping[key][0],
            )
          mapping[key] = (name, int(dataset.shape[0]))
    except OSError as exc:
      _LOGGER.warning('Unreadable shard %s during scan (%s)', name, exc)
  return mapping


# --------------------------------------------------------------------------- #
# Decode worker                                                                #
# --------------------------------------------------------------------------- #


def _record_with_defaults(record: dict) -> dict:
  """Fill reader-relevant defaults onto an index row copy.

  Args:
      record: Index row (possibly missing override fields).

  Returns:
      New dict guaranteed to carry img, decoder_order, percentiles,
      margin and dicom_root keys consumed by the worker.
  """
  payload = dict(record)
  payload.setdefault('img', 384)
  payload.setdefault('decoder_order', ['native', 'gdcm', 'pylibjpeg'])
  payload.setdefault('percentiles', [0.005, 0.995])
  payload.setdefault('margin', 0.05)
  payload.setdefault('dicom_root', '')
  return payload


def decode_series_volume(record: dict) -> tuple[str, np.ndarray] | None:
  """Decode EVERY slice of one series into the cached pixel format.

  Pure function over the task dict so pool transport stays picklable; a
  single failed frame disqualifies the whole series from caching because
  zero-fallback frames are train-time robustness, never permanent data.

  Args:
      record: Index row plus defaults from :func:`_record_with_defaults`.

  Returns:
      ``(series_uid, (S, img, img) uint8)`` on success, else None.
  """
  task = _record_with_defaults(record)
  registry = DecoderRegistry(list(task['decoder_order']))
  series_dir = os.path.join(
    task['dicom_root'], str(task['study']), str(task['series'])
  )
  sop_uids: list[str] = list(task['sop_uids'])
  if not sop_uids:
    return None
  frames: list[np.ndarray] = []
  for sop in sop_uids:
    pixels, info = registry.read_slice(os.path.join(series_dir, f'{sop}.dcm'))
    if pixels is None:
      _LOGGER.debug(
        'cache skip %s/%s (%s)', task['study'], task['series'], info['errors']
      )
      return None
    frames.append(pixels.astype(np.float32))
  stack = np.stack(frames, axis=0)
  del frames
  normalized = intensity.normalize_percentile(stack, tuple(task['percentiles']))
  center = normalized[len(normalized) // 2]
  _, (y0, y1, x0, x1) = intensity.autocrop(center, float(task['margin']))
  cropped = normalized[:, y0:y1, x0:x1]
  del stack, normalized
  out_width = int(task['img'])
  volume = np.empty((cropped.shape[0], out_width, out_width), dtype=np.uint8)
  for i, plane in enumerate(cropped):
    resized = cv2.resize(
      plane,
      (out_width, out_width),
      interpolation=cv2.INTER_LINEAR,
    )
    volume[i] = intensity.to_uint8(resized)
  return str(task['series']), volume


# --------------------------------------------------------------------------- #
# Pool runner                                                                  #
# --------------------------------------------------------------------------- #


def progress_state(
  writer: ShardWriter,
  processed: int,
  total_series: int,
  cached: int,
  skipped: int,
  files_done: int,
  total_files: int | None = None,
) -> dict:
  """Snapshot current cache-build statistics for reporting.

  Args:
      writer: Active writer (provides shard GiB + completed series).
      processed: Series handled so far.
      total_series: Expected series count.
      cached: Series successfully stored.
      skipped: Series rejected (failed frames).
      files_done: DICOM frames decoded into the cache.
      total_files: Optional grand-total frame estimate for ratios.

  Returns:
      Flat stats mapping safe for JSON/markdown formatting.
  """
  return {
    'files_done': files_done,
    'total_files': total_files,
    'series_processed': processed,
    'total_series': total_series,
    'series_cached': cached,
    'series_skipped': skipped,
    'shards_gib': round(
      sum(
        os.path.getsize(os.path.join(writer.cache_dir, n))
        for n in os.listdir(writer.cache_dir)
        if n.endswith(SHARD_SUFFIX)
      )
      / GIB,
      2,
    ),
  }


def format_progress(state: dict, elapsed_s: float) -> str:
  """Render a one-line human/progress message from stats.

  Args:
      state: Mapping produced by :func:`progress_state`.
      elapsed_s: Seconds since the run started (drives ETA).

  Returns:
      Message body WITHOUT any experiment/fold prefix.
  """
  done = state['files_done']
  prefix = f'{done:,} files'
  total = state.get('total_files')
  if total:
    pct = 100.0 * done / max(1, total)
    eta_h = (total - done) / max(done, 1) * elapsed_s / 3600
    eta = f', ETA ~{eta_h:.1f}h' if elapsed_s > 0 and done > 0 else ''
    prefix = f'{prefix}/{total:,} ({pct:.1f}%{eta})'
  cached_n = state['series_cached']
  skipped_n = state['series_skipped']
  shards_gib = state['shards_gib']
  return (
    f'{prefix} | series {cached_n} cached, '
    f'{skipped_n} skipped | shards {shards_gib} GiB'
  )


def run_pool_tasks(
  tasks: list[dict],
  workers: int,
  writer: ShardWriter,
  log_every: int = 50,
  on_progress=None,
  files_every: int = 0,
  total_files: int | None = None,
) -> tuple[int, int]:
  """Fan decode tasks to processes, streaming results into shards.

  Args:
      tasks: Index rows enriched with ``dicom_root`` (other reader
          overrides optional via _record_with_defaults).
      workers: Process count; <=1 executes inline (tests/CI).
      writer: Destination :class:`ShardWriter`.
      log_every: Progress log cadence in completed series.
      on_progress: Optional callable receiving a :func:`progress_state`
          dict every time ``files_every`` more files complete.
      files_every: File-count cadence for on_progress; 0 disables.
      total_files: Forwarded to progress_state for ratio/ETA display.

  Returns:
      Tuple ``(cached, skipped)`` counts.

  Raises:
      RuntimeError: When tasks existed yet every one failed to decode.
  """
  cached = skipped = 0
  if workers > 1:
    executor = ProcessPoolExecutor(max_workers=workers)
    results = executor.map(decode_series_volume, tasks, chunksize=2)
  else:
    executor = None
    results = map(decode_series_volume, tasks)

  processed = 0
  files_done = 0
  next_file_mark = int(files_every) if files_every and files_every > 0 else None
  try:
    for result in results:
      processed += 1
      if result is None:
        skipped += 1
      else:
        uid, volume = result
        writer.add_series(uid, uid, volume)
        cached += 1
        files_done += int(volume.shape[0])
        if (
          next_file_mark is not None
          and on_progress is not None
          and files_done >= next_file_mark
        ):
          on_progress(
            progress_state(
              writer,
              processed=processed,
              total_series=len(tasks),
              cached=cached,
              skipped=skipped,
              files_done=files_done,
              total_files=total_files,
            )
          )
          while next_file_mark <= files_done:
            next_file_mark += max(1, int(files_every))
      if processed % log_every == 0:
        gb = (
          sum(
            os.path.getsize(os.path.join(writer.cache_dir, n))
            for n in os.listdir(writer.cache_dir)
            if n.endswith(SHARD_SUFFIX)
          )
          / GIB
        )
        _LOGGER.info(
          'cache progress %d/%d (cached=%d skipped=%d shards=%.1f GiB)',
          processed,
          len(tasks),
          cached,
          skipped,
          gb,
        )
  finally:
    if executor is not None:
      executor.shutdown()

  if tasks and cached == 0:
    raise RuntimeError(f'volume cache produced nothing from {len(tasks)}')
  return cached, skipped


# --------------------------------------------------------------------------- #
# Manifest / reader                                                            #
# --------------------------------------------------------------------------- #


def find_cache_roots(config: dict) -> list[str]:
  """Resolve directories that may host shards+manifest.

  KNEE_HDF5_CACHE_DIRS (colon-separated) overrides configuration so a
  Kaggle session can point at multiple mounted datasets without edits.

  Args:
      config: Composed experiment configuration.

  Returns:
      Existing directory paths, possibly empty.
  """
  raw = os.environ.get('KNEE_HDF5_CACHE_DIRS')
  candidates = (
    [p for p in raw.split(':') if p]
    if raw
    else [config['paths'].get('volume_cache_dir', '')]
  )
  return [p for p in candidates if p and os.path.isdir(p)]


def load_manifest(roots: list[str]) -> pd.DataFrame | None:
  """Merge manifests across roots; later roots win duplicate-UID ties.

  Args:
      roots: Candidate cache directories.

  Returns:
      Merged manifest with a ``_root`` column, or None when absent.
  """
  frames: list[pd.DataFrame] = []
  for order, root in enumerate(roots):
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.exists(path):
      continue
    frame = pd.read_parquet(path)
    frame['_root'] = root
    frame['_order'] = order
    frames.append(frame)
  if not frames:
    return None
  merged = pd.concat(frames, ignore_index=True)
  merged = merged.sort_values('_order').drop_duplicates(
    'SeriesInstanceUID', keep='last'
  )
  merged = merged.drop(columns=['_order']).reset_index(drop=True)
  _LOGGER.info(
    'HDF5 cache engaged: %d series, %d root(s)', len(merged), len(frames)
  )
  return merged


class H5SeriesReader:
  """Serve cached volumes with transparent live-decode fallback."""

  def __init__(
    self, base_reader, manifest: pd.DataFrame, n_slices: int
  ) -> None:
    """Bind reader state.

    Args:
        base_reader: Live SeriesReader handling UID misses/corruption.
        manifest: Output of :func:`load_manifest`.
        n_slices: Evenly spaced sample count returned by read().
    """
    self.base = base_reader
    self.n_slices = n_slices
    self._locations = {
      str(row['SeriesInstanceUID']): (
        str(row['_root']),
        str(row['shard_file']),
      )
      for _, row in manifest.iterrows()
    }
    self._handles: dict[str, h5py.File] = {}

  def _open_dataset(self, root: str, shard: str):
    """Memoized shard opening rooted per manifest entry.

    Args:
        root: Directory hosting this shard.
        shard: Shard basename.

    Returns:
        Open h5py.File or None when unusable.
    """
    path = os.path.join(root, shard)
    if path in self._handles:
      return self._handles[path]
    try:
      handle = h5py.File(path, 'r')
    except OSError as exc:
      _LOGGER.warning('Cache shard unusable %s (%s)', path, exc)
      self._handles[path] = None
      return None
    self._handles[path] = handle
    return handle

  def read(self, record: dict) -> np.ndarray:
    """Return sampled uint8 stack matching SeriesReader's contract.

    Args:
        record: Index-row mapping with study/series/sop_uids.

    Returns:
        ``(n_slices, H, W)`` uint8 array.
    """
    uid = str(record.get('series') or record.get('SeriesInstanceUID'))
    location = self._locations.get(uid)
    if location is not None:
      handle = self._open_dataset(*location)
      if handle is not None and uid in handle:
        dataset = handle[uid]
        indices = np.unique(
          np.linspace(0, dataset.shape[0] - 1, self.n_slices, dtype=int)
        )
        return np.ascontiguousarray(dataset[indices])
    return self.base.read(record)


__all__ = [
  'H5SeriesReader',
  'MANIFEST_NAME',
  'ShardWriter',
  'collect_shard_map',
  'decode_series_volume',
  'find_cache_roots',
  'load_manifest',
  'run_pool_tasks',
]
