#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for disk-free streaming volume IO under Kaggle constraints.

Covers the bounded LRU cache, multi-kernel shard partitioning, the
disk-budget helper and the StreamingVolumeStore fallback ladder
(LRU -> npz shards -> live DICOM decode -> memoized failure).
Deliberately dependency-light: no torch/pydicom/cv2 required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee.config_params.schema import DataConfig
from knee.datasets.volume_store import (
  LruVolumeCache,
  StreamingVolumeStore,
  free_disk_bytes,
  shard_indices,
)


def _volume(seed: int = 0, depth: int = 4, size: int = 8) -> np.ndarray:
  """Deterministic uint8 stand-in for a decoded series volume.

  Args:
      seed: Fill value selector (keeps arrays distinguishable).
      depth: Number of slices.
      size: Square in-plane resolution.

  Returns:
      uint8 array ``(depth, size, size)`` filled with ``seed + 1``.
  """
  return np.full((depth, size, size), seed + 1, dtype=np.uint8)


class TestLruVolumeCache:
  """Eviction must respect both item count and byte ceilings."""

  def test_evicts_least_recently_used_by_count(self):
    cache = LruVolumeCache(max_items=2, max_bytes=1 << 30)
    for i in range(3):
      cache.put(f's{i}', _volume(seed=i))
    assert len(cache) == 2
    assert cache.get('s0') is None
    assert cache.get('s2') is not None

  def test_get_refreshes_recency(self):
    cache = LruVolumeCache(max_items=2, max_bytes=1 << 30)
    cache.put('a', _volume(0))
    cache.put('b', _volume(1))
    assert cache.get('a') is not None  # 'b' becomes the LRU victim
    cache.put('c', _volume(2))
    assert cache.get('a') is not None
    assert cache.get('b') is None

  def test_byte_ceiling_eviction(self):
    tiny = _volume(size=16)  # 16*16*4 = 1024 bytes
    cache = LruVolumeCache(max_items=100, max_bytes=2048)
    for i in range(4):
      cache.put(f'v{i}', tiny)
    assert cache.nbytes <= 2048
    assert len(cache) <= 2

  def test_reput_replaces_byte_accounting(self):
    cache = LruVolumeCache(max_items=10, max_bytes=1 << 30)
    big = _volume(size=64)  # 16384 bytes
    cache.put('k', big)
    cache.put('k', big)
    assert cache.nbytes == big.nbytes
    assert len(cache) == 1

  def test_oversized_volume_is_not_cached(self):
    cache = LruVolumeCache(max_items=10, max_bytes=512)
    cache.put('keep', _volume())  # 4*8*8 = 256 bytes fits
    cache.put('huge', _volume(size=256))  # 256 KiB exceeds ceiling
    assert cache.get('huge') is None
    assert cache.get('keep') is not None


class TestShardIndices:
  """Multi-kernel shard partitions must be exact and stable."""

  def test_partition_covers_every_index_once(self):
    total, shards = 103, 8
    seen: list[int] = []
    for shard in range(shards):
      part = shard_indices(total, shard, shards)
      assert part == sorted(part)
      seen.extend(part)
    assert sorted(seen) == list(range(total))

  def test_balanced_sizes(self):
    parts = [shard_indices(101, s, 4) for s in range(4)]
    sizes = {len(p) for p in parts}
    assert max(sizes) - min(sizes) <= 1

  def test_invalid_shard_raises(self):
    with pytest.raises(ValueError):
      shard_indices(10, 5, 5)


class TestFreeDiskBytes:
  def test_reports_positive_free_space(self, tmp_path: Path):
    assert free_disk_bytes(tmp_path) > 0

  def test_missing_path_returns_zero(self):
    assert free_disk_bytes('/nonexistent/no/such/dir') >= 0


class TestStreamingVolumeStore:
  """Fallback ladder: LRU -> mounted npz shards -> live DICOM decode."""

  def _store(
    self, tmp_path: Path, cache_dir: str | None
  ) -> StreamingVolumeStore:
    return StreamingVolumeStore(
      data_root=str(tmp_path),
      data_cfg=DataConfig(image_size=32, num_slices=8),
      split='train',
      cache_dir=cache_dir,
      max_items=4,
      max_bytes=1 << 28,
    )

  def test_npz_shard_fast_path(self, tmp_path: Path, monkeypatch):
    shard_dir = tmp_path / 'shard0'
    shard_dir.mkdir()
    np.savez_compressed(shard_dir / 's1.npz', volume=_volume(seed=7))
    calls = []

    def _fail_decode(*args, **kwargs):  # pragma: no cover - guard
      calls.append(args)
      raise AssertionError('decode must not run when npz exists')

    monkeypatch.setattr(
      'knee.datasets.volume_store.decode_series_volume', _fail_decode
    )
    store = self._store(tmp_path, str(shard_dir))
    out = store.get({'SeriesInstanceUID': 's1'})
    assert out is not None and int(out[0, 0, 0]) == 8

  def test_multiple_shard_dirs_via_pathsep(self, tmp_path: Path):
    first, second = tmp_path / 'a', tmp_path / 'b'
    second.mkdir()
    np.savez_compressed(second / 's9.npz', volume=_volume(seed=3))
    store = self._store(tmp_path, os_pathsep_join(first, second))
    out = store.get({'SeriesInstanceUID': 's9'})
    assert out is not None and int(out[0, 0, 0]) == 4

  def test_fallback_decode_and_lru_hit(self, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
      'knee.datasets.volume_store.series_dir',
      lambda *args, **kwargs: tmp_path,
    )
    counter = {'n': 0}

    def _fake_decode(
      directory, image_size, num_slices, percentile_clip=(1.0, 99.0)
    ):
      counter['n'] += 1
      return _volume(seed=1)

    monkeypatch.setattr(
      'knee.datasets.volume_store.decode_series_volume', _fake_decode
    )
    store = self._store(tmp_path, None)
    row = {'StudyInstanceUID': 'st', 'SeriesInstanceUID': 'sx'}
    first = store.get(row)
    second = store.get(row)
    assert first is not None and second is first
    assert counter['n'] == 1  # decoded once, then served from LRU

  def test_failure_is_memoized(self, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
      'knee.datasets.volume_store.series_dir',
      lambda *args, **kwargs: tmp_path,
    )
    counter = {'n': 0}

    def _boom(*args, **kwargs):
      counter['n'] += 1
      raise FileNotFoundError('no dicoms')

    monkeypatch.setattr(
      'knee.datasets.volume_store.decode_series_volume', _boom
    )
    store = self._store(tmp_path, None)
    row = {'SeriesInstanceUID': 'broken'}
    assert store.get(row) is None
    assert store.get(row) is None
    assert counter['n'] == 1


class TestLocalizerFilter:
  """Short localizer series are dropped only when counts are known."""

  def test_drops_short_series_keeps_unknown(self):
    pytest.importorskip('torch')
    from knee.datamodules.knee_datamodule import _drop_localizers

    table = pd.DataFrame(
      {
        'SeriesInstanceUID': ['a', 'b', 'c'],
        'n_slices': [3, 12, np.nan],
      }
    )
    kept = _drop_localizers(table, min_slices=5)
    assert list(kept['SeriesInstanceUID']) == ['b', 'c']


def os_pathsep_join(*paths: Path) -> str:
  """Join candidate shard dirs with the platform path separator.

  Args:
      paths: Directories to join.

  Returns:
      ``os.pathsep``-joined string (empty parts dropped).
  """
  import os

  return os.pathsep.join(str(p) for p in paths if p.is_dir())
