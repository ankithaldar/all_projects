#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the HDF5 volume cache (writer, reader swap, manifest merge).

Synthetic volumes stand in for DICOM output; a tiny fake index row
provides records. DICOM-root fallback behavior is exercised through the
H5SeriesReader contract only.
"""

# Local fixtures shadow module names; conditional imports isolate the
# monkeypatch target; assert-on-empty-list reads clearer for state.
# pylint: disable=redefined-outer-name,unused-argument,import-outside-toplevel,use-implicit-booleaness-not-comparison,protected-access

import json
import os

import h5py
import numpy as np
import pandas as pd
import pytest

from knee.helpers.h5_cache import (
  GIB,
  MANIFEST_NAME,
  H5SeriesReader,
  ShardWriter,
  decode_series_volume,
  find_cache_roots,
  format_progress,
  load_manifest,
  run_pool_tasks,
)


def _volume(seed: int, img: int = 16) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return rng.integers(0, 256, size=(4, img, img), dtype=np.uint8)


class FakeReader:
  """Live-reader stub recording calls for fallback assertions."""

  def __init__(self) -> None:
    self.calls = []

  def read(self, record):
    self.calls.append(record['series'])
    return np.zeros((3, 8, 8), dtype=np.uint8)


@pytest.fixture()
def writer(tmp_path):
  return ShardWriter(
    str(tmp_path / 'cache'),
    img_size=16,
    shard_bytes_cap=64 * 1024,
    gzip_level=1,
  )


def test_shard_rollover_and_manifest_roundtrip(writer, tmp_path):
  """Volumes exceeding the cap land in sequential shards + manifest."""
  uids = []
  for i in range(6):
    uid = f'uid-{i}'
    shard = writer.add_series(uid, f'study-{i}', _volume(i))
    assert shard.startswith('volume_shard_')
    uids.append(uid)
  manifest_path = writer.write_manifest()
  assert manifest_path.endswith(MANIFEST_NAME)

  frame = pd.read_parquet(manifest_path)
  # Each synthetic volume is 4*16*16 = 1 KiB, cap 64 KiB -> one shard.
  assert frame['SeriesInstanceUID'].tolist() == uids
  assert frame['n_slices'].tolist() == [4] * 6


def test_resume_existing_uids_after_reopen(writer):
  existing = {'uid-0'}
  writer.add_series('uid-0', 'study-0', _volume(0))
  assert writer.existing_uids() == existing

  # A fresh writer instance must see the same resume state.
  reopened = ShardWriter(
    writer.cache_dir, img_size=16, shard_bytes_cap=2 * GIB, gzip_level=0
  )
  assert existing <= reopened.existing_uids()


class TestDecodeWorker:
  """decode_series_volume rejects partial decodes by returning None."""

  class FakeRegistry:
    def __init__(self, fail_on):
      self.fail_on = fail_on
      self.order_seen = None

    def read_slice(self, path):
      if any(f in path for f in getattr(self, 'fail_on', ())):
        return None, {'errors': 'synthetic'}
      return np.full((8, 8), 128.0), {}

  def test_full_series_volume_shape(self, monkeypatch, tmp_path):
    from knee.helpers import h5_cache

    registry = self.FakeRegistry(fail_on=())
    monkeypatch.setattr(h5_cache, 'DecoderRegistry', lambda order: registry)
    record = {
      'study': 'st',
      'series': 'sr',
      'sop_uids': ['a', 'b'],
      'dicom_root': str(tmp_path),
    }
    result = decode_series_volume({**record})
    assert result is not None
    uid, vol = result
    assert uid == 'sr'
    assert vol.shape == (2, 384, 384)
    assert vol.dtype == np.uint8

  def test_single_failed_frame_skips_whole_series(self, monkeypatch, tmp_path):
    from knee.helpers import h5_cache

    registry = self.FakeRegistry(fail_on=('b.dcm',))
    monkeypatch.setattr(h5_cache, 'DecoderRegistry', lambda order: registry)
    for path in ('a.dcm', 'b.dcm'):
      (tmp_path / 'st' / 'sr').mkdir(parents=True, exist_ok=True)
      (tmp_path / 'st' / 'sr' / path).touch()
    record = {
      'study': 'st',
      'series': 'sr',
      'sop_uids': ['a', 'b'],
      'dicom_root': str(tmp_path),
    }
    assert decode_series_volume(record) is None

  def test_run_pool_inline_streams_into_writer(self, monkeypatch, tmp_path):
    import knee.helpers.h5_cache as h5c

    monkeypatch.setattr(
      h5c,
      'decode_series_volume',
      lambda rec: (rec['series'], _volume(int(rec['series']))),
    )
    writer = ShardWriter(
      str(tmp_path / 'pool'), img_size=16, shard_bytes_cap=GIB, gzip_level=0
    )
    try:
      tasks = [{'dicom_root': 'x', 'series': str(i)} for i in range(4)]
      cached, skipped = run_pool_tasks(tasks, workers=1, writer=writer)
      assert (cached, skipped) == (4, 0)
      manifest = pd.read_parquet(writer.write_manifest())
      assert len(manifest) == 4
    finally:
      writer.close()


def test_reader_serves_cached_uid_and_falls_back_on_miss(tmp_path):
  writer = ShardWriter(
    str(tmp_path), img_size=16, shard_bytes_cap=GIB, gzip_level=0
  )
  stored = _volume(7)
  writer.add_series('hit-uid', 'study-a', stored)
  writer.write_manifest()
  manifest = load_manifest([str(tmp_path)])
  assert manifest is not None and len(manifest) == 1

  base = FakeReader()
  reader = H5SeriesReader(base, manifest, n_slices=3)
  got = reader.read({'series': 'hit-uid', 'study': 'x'})
  assert got.shape == (3, 16, 16)
  # Values must come from storage, not fallback zeros.
  assert got.max() > 0
  assert base.calls == []

  miss = reader.read({'series': 'nope', 'study': 'y'})
  assert miss.shape == (3, 8, 8)
  assert base.calls == ['nope']


def test_manifest_merges_across_roots_preferring_later(tmp_path):
  root_a, root_b = tmp_path / 'a', tmp_path / 'b'
  root_a.mkdir()
  root_b.mkdir()
  wa = ShardWriter(str(root_a), img_size=8, gzip_level=0)
  wb = ShardWriter(str(root_b), img_size=8, gzip_level=0)
  wa.add_series('shared', 's1', _volume(0, img=8))
  wa.write_manifest()
  wb.add_series('shared', 's2', _volume(1, img=8))  # duplicate uid
  wb.add_series('only-b', 's3', _volume(2, img=8))
  wb.write_manifest()

  merged = load_manifest([str(root_a), str(root_b)])
  rows = merged.set_index('SeriesInstanceUID')
  assert set(rows.index) == {'shared', 'only-b'}
  # Later root wins on duplicates.
  assert rows.loc['shared', '_root'] == str(root_b)


def test_find_cache_roots_env_override(tmp_path, monkeypatch):
  root = tmp_path / 'mounted'
  root.mkdir()
  config = {'paths': {'volume_cache_dir': str(tmp_path / 'absent')}}
  assert find_cache_roots(config) == []

  monkeypatch.setenv('KNEE_HDF5_CACHE_DIRS', f'{root}:/nonexistent:{tmp_path}')
  assert find_cache_roots(config) == [str(root), str(tmp_path)]


def test_bad_stack_dtype_raises(writer):
  with pytest.raises(ValueError):
    writer.add_series('u', 's', _volume(0).astype(np.float32))


if __name__ == '__main__':
  pytest.main([__file__])


class TestDiscordProgress:
  """File-count heartbeat cadence for the caching stage."""

  def test_boundaries_fire_every_files_every(self, tmp_path, monkeypatch):
    import knee.helpers.h5_cache as h5c

    # Two fake series of 6,000 frames each -> cross 10k exactly once.
    def big_volume(rec):
      vol = np.full((6000, 4, 4), int(rec['series']) + 1, dtype=np.uint8)
      return rec['series'], vol

    monkeypatch.setattr(h5c, 'decode_series_volume', big_volume)
    writer = ShardWriter(
      str(tmp_path / 'hb'), img_size=16, shard_bytes_cap=GIB, gzip_level=0
    )
    events = []
    try:
      tasks = [{'dicom_root': 'x', 'series': str(i)} for i in range(2)]
      run_pool_tasks(
        tasks,
        workers=1,
        writer=writer,
        on_progress=events.append,
        files_every=10_000,
      )
      assert len(events) == 1
      assert events[0]['files_done'] == 12_000
      assert events[0]['series_cached'] == 2
    finally:
      writer.close()

  def test_disabled_when_zero(self, tmp_path, monkeypatch):
    import knee.helpers.h5_cache as h5c

    monkeypatch.setattr(
      h5c,
      'decode_series_volume',
      lambda rec: (rec['series'], _volume(int(rec['series']))),
    )
    writer = ShardWriter(
      str(tmp_path / 'off'), img_size=16, shard_bytes_cap=GIB, gzip_level=0
    )
    events = []
    try:
      run_pool_tasks(
        [{'dicom_root': 'x', 'series': '0'}],
        workers=1,
        writer=writer,
        on_progress=events.append,
      )
      assert not events
    finally:
      writer.close()

  def test_format_progress_percent_and_eta(self):
    state = {
      'files_done': 20_000,
      'total_files': 800_000,
      'series_cached': 400,
      'series_skipped': 3,
      'shards_gib': 3.14,
    }
    text = format_progress(state, elapsed_s=3600)
    assert '20,000 files/800,000' in text
    assert '(2.5%' in text and 'ETA ~' in text
    assert 'skipped' in text and '3.14 GiB' in text


class TestShardMarkers:
  """Completion sentinels drive resume/pipelined-push semantics."""

  def test_completed_shards_legacy_fallback(self, tmp_path):
    import knee.helpers.h5_cache as hc

    w = hc.ShardWriter(str(tmp_path), img_size=8, gzip_level=0)
    w.add_series('u1', 's1', _volume(0, img=8))  # no close -> unmarked
    assert hc.completed_shards(str(tmp_path)) == ['volume_shard_000.h5']
    w.close()  # graceful close stamps the sentinel
    listing = sorted(os.listdir(tmp_path))
    assert 'volume_shard_000.h5.complete' in listing

  def test_drop_unfinished_keeps_marked_only(self, tmp_path):
    import knee.helpers.h5_cache as hc

    w = hc.ShardWriter(str(tmp_path), img_size=8, gzip_level=0)
    w.add_series('done', 's', _volume(1, img=8))
    w.close()
    # Simulate a crashed partial tail WITHOUT a sentinel.
    with h5py.File(tmp_path / 'volume_shard_001.h5', 'a') as f:
      f.create_dataset('ghost', data=_volume(2, img=8))
    removed = hc.drop_unfinished_shards(str(tmp_path))
    assert removed == ['volume_shard_001.h5']
    resumed = hc.ShardWriter(str(tmp_path), img_size=8, gzip_level=0)
    # 'ghost' is gone; 'done' must still count toward existing state.
    assert resumed.existing_uids() == {'done'}

  def test_callback_fires_per_roll_with_move_consumer(self, tmp_path):
    import knee.helpers.h5_cache as hc

    moved: list[str] = []
    out = tmp_path / 'staging'
    out.mkdir()

    def consume(path):
      dest = out / os.path.basename(path)
      os.rename(path, dest)  # pipelined consumer relocates instantly
      moved.append(dest.name)

    vol_bytes = 3 * 4 * 3  # one (3,4,3) uint8 volume
    w = hc.ShardWriter(
      str(tmp_path / 'build'),
      img_size=4,
      shard_bytes_cap=vol_bytes,  # exactly one volume per shard
      gzip_level=0,
      on_shard_complete=consume,
    )
    for i in range(3):
      w.add_series(f'u{i}', 's', _volume(i, img=4)[:, :3])
    w.close()
    # REGRESSION: ordinals must ADVANCE under a moving consumer; ordinal
    # reuse made every shard push into the SAME dataset slug, surfacing
    # as endless Kaggle version bumps that overwrote prior content.
    assert moved == [
      'volume_shard_000.h5',
      'volume_shard_001.h5',
      'volume_shard_002.h5',
    ]
    assert not (tmp_path / 'build' / 'volume_shard_000.h5').exists()


def _shard_ordinals_from_slugs(slugs):
  return [int(s.rsplit('-', 1)[1]) for s in slugs]


class TestShardSlugNaming:
  """<base>-NNN slug derivation matches helpers conventions."""

  def test_base_and_ordinal(self):
    base = 'ah2002-rsna-knee-abnormality-detection-cache'
    name = 'volume_shard_007.h5'
    ordinal = int(name.rsplit('_', maxsplit=1)[-1].split('.')[0])
    assert f'{base}-{ordinal:03d}' == (base + '-007')
    assert _shard_ordinals_from_slugs([base + '-004']) == [4]


def test_push_version_inplace_applies_credentials(tmp_path):
  """Regression: qualification used to crash before creds were applied."""
  from knee.helpers.kaggle_io import KaggleDatasetClient

  calls = {'applied': 0}

  class StubResolver:
    def __init__(self):
      self.username_key = self.token_key = ''

    def apply(self):  # simulates env/Kaggle-secrets materialization
      calls['applied'] += 1
      os.environ['KAGGLE_USERNAME'] = 'ah2002'
      os.environ['KAGGLE_API_TOKEN'] = 'tok'

    def _lookup(self, name):
      return os.environ.get(name)

  client = KaggleDatasetClient(
    StubResolver(),
    runner=lambda *a, **k: type(
      'R', (), {'returncode': 0, 'stderr': '', 'stdout': ''}
    ),
  )
  folder = tmp_path / 'payload'
  folder.mkdir()
  monkey_patch_env_clear = {'KAGGLE_USERNAME': None, 'KAGGLE_API_TOKEN': None}
  saved = {k: os.environ.pop(k, None) for k in monkey_patch_env_clear}
  try:
    client.push_version_inplace('user/ds', str(folder))
  finally:
    for key, val in saved.items():
      if val is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = val
  # At least the explicit pre-metadata apply ran, and qualification
  # succeeded where it previously crashed with 'Cannot qualify slug'.
  assert calls['applied'] >= 1
  meta = json.loads((folder / 'dataset-metadata.json').read_text())
  assert meta['id'] == 'user/ds'


class TestArtifactRestore:
  """Attached-dataset copy beats the slow kaggle-CLI rehydration path."""

  def test_restores_missing_files_from_mount_roots(self, tmp_path, monkeypatch):
    from main import restore_artifacts_from_mounts

    mount = (
      tmp_path / 'input' / 'datasets' / 'haldarankit' / 'rsna-knee-mvp-index'
    )
    mount.mkdir(parents=True)
    (mount / 'index.parquet').write_text('PARQUET_BYTES')
    (mount / 'folds.csv').write_text('study,fold\n')

    artifact_dir = tmp_path / 'working' / 'artifacts'
    config = {'paths': {'artifact_dir': str(artifact_dir)}}
    unrelated = str(tmp_path / 'input' / 'unrelated')
    monkeypatch.setenv('KNEE_INPUT_ROOTS', f'{mount}:{unrelated}')
    restored = restore_artifacts_from_mounts(config)
    assert sorted(restored) == ['folds.csv', 'index.parquet']
    assert (artifact_dir / 'index.parquet').read_text() == 'PARQUET_BYTES'
    # Idempotent: second call restores nothing.
    assert restore_artifacts_from_mounts(config) == []

  def test_no_roots_returns_empty(self, tmp_path, monkeypatch):
    from main import restore_artifacts_from_mounts

    monkeypatch.setenv('KNEE_INPUT_ROOTS', str(tmp_path / 'absent'))
    config = {'paths': {'artifact_dir': str(tmp_path / 'a')}}
    assert restore_artifacts_from_mounts(config) == []


class TestCrossSessionResume:
  """Reruns must consult PUSHED datasets, not just the local build dir."""

  def test_start_ordinal_continues_remote_sequence(self, tmp_path):
    import knee.helpers.h5_cache as hc

    # Emulate session restart: local dir fresh, floor from remote state.
    w = hc.ShardWriter(
      str(tmp_path / 'build'),
      img_size=8,
      gzip_level=0,
      start_ordinal=5,
    )
    w.add_series('u', 's', _volume(0, img=8))
    w.close()
    assert (tmp_path / 'build' / 'volume_shard_005.h5').exists()

  def test_remote_cache_state_parses_fragments(self, tmp_path, monkeypatch):
    import main as m

    frag = (
      tmp_path / 'input' / 'ah2002-rsna-knee-abnormality-detection-cache-000'
    )
    frag.mkdir(parents=True)
    pd.DataFrame(
      {
        'SeriesInstanceUID': ['a', 'b'],
        'shard_file': ['volume_shard_000.h5'] * 2,
        'n_slices': [4, 4],
      }
    ).to_parquet(frag / 'cache_manifest.parquet', index=False)
    frag2 = (
      tmp_path / 'input' / 'ah2002-rsna-knee-abnormality-detection-cache-003'
    )
    frag2.mkdir(parents=True)
    pd.DataFrame(
      {
        'SeriesInstanceUID': ['c'],
        'shard_file': ['volume_shard_003.h5'],
        'n_slices': [7],
      }
    ).to_parquet(frag2 / 'cache_manifest.parquet', index=False)

    monkeypatch.setenv('KNEE_INPUT_ROOTS', f'{frag}:{frag2}')
    uids, max_ord = m.remote_cache_state(
      {'paths': {'artifact_dir': str(tmp_path / 'art')}}
    )
    assert uids == {'a', 'b', 'c'}
    assert max_ord == 3

  def test_generation_mismatch_is_loud(self, tmp_path, caplog):
    import knee.helpers.h5_cache as hc

    root_a, root_b = tmp_path / 'a', tmp_path / 'b'
    root_a.mkdir()
    root_b.mkdir()
    (root_a / hc.CACHE_META_NAME).write_text(json.dumps({'img_size': 384}))
    (root_b / hc.CACHE_META_NAME).write_text(json.dumps({'img_size': 512}))
    for root, uids in ((root_a, {'x'}), (root_b, {'y'})):
      w = hc.ShardWriter(str(root), img_size=8, gzip_level=0)
      for uid in uids:
        w.add_series(uid, 's', _volume(0, img=8))
      w.close()
      w.write_manifest()

    with caplog.at_level('ERROR'):
      merged = hc.load_manifest([str(root_a), str(root_b)])
    assert merged is not None and len(merged) == 2
    assert 'generation mismatch' in caplog.text
    assert 'img_size=512' in caplog.text


class TestMountDiscovery:
  """Attached cache datasets are auto-discovered; artifact sets are not."""

  def _make_mount(self, base, slug, uids, ordinal):
    root = base / slug
    root.mkdir(parents=True)
    w = ShardWriter(str(root), img_size=8, gzip_level=0)
    for uid in uids:
      w.add_series(uid, 's', _volume(ordinal, img=8))
    w.close()
    w.write_manifest()
    return root

  def test_twelve_mounts_discovered_in_order(self, tmp_path, monkeypatch):
    import knee.helpers.h5_cache as hc

    base = tmp_path / 'input' / 'datasets' / 'haldarankit'
    for n in range(12):
      self._make_mount(
        base,
        f'ah2002-rsna-knee-abnormality-detection-cache-{n:03d}',
        [f'uid-{n}'],
        n,
      )
    # Artifact dataset WITHOUT a manifest fragment must NOT qualify.
    (base / 'rsna-knee-mvp-index').mkdir()
    (base / 'rsna-knee-mvp-index' / 'index.parquet').write_text('x')

    config = {'paths': {'volume_cache_dir': str(tmp_path / 'build')}}
    (tmp_path / 'build').mkdir()  # stages create this before resolving
    monkeypatch.setenv('KNEE_INPUT_ROOTS', str(base))
    roots = hc.find_cache_roots(config)
    # 12 discovered mounts + the local build dir appended last
    # (tie-break priority for the newest generation of fresh shards).
    assert len(roots) == 13
    assert all(
      'ah2002-rsna-knee-abnormality-detection-cache-' in r for r in roots[:12]
    )
    # Local build dir appended last (tie-break priority on fresh shards).
    assert roots[-1] == str(tmp_path / 'build')

  def test_env_override_disables_discovery(self, tmp_path, monkeypatch):
    import knee.helpers.h5_cache as hc

    base = tmp_path / 'input'
    self._make_mount(base, 'cache-000', ['u0'], 0)
    only = tmp_path / 'explicit'
    only.mkdir()
    monkeypatch.setenv('KNEE_INPUT_ROOTS', str(base))
    monkeypatch.setenv('KNEE_HDF5_CACHE_DIRS', str(only))
    roots = hc.find_cache_roots(
      {'paths': {'volume_cache_dir': str(tmp_path / 'b')}}
    )
    assert roots == [str(only)]

  def test_copy_mounts_to_working(self, tmp_path, monkeypatch):
    import knee.helpers.h5_cache as hc

    base = tmp_path / 'input'
    self._make_mount(base, 'cache-000', ['u0'], 0)
    artifact_dir = tmp_path / 'working' / 'artifacts'
    config = {
      'paths': {
        'volume_cache_dir': str(tmp_path / 'b'),
        'artifact_dir': str(artifact_dir),
      },
      'volume_cache': {'copy_mounts_to_working': True},
    }
    (tmp_path / 'b').mkdir()
    monkeypatch.setenv('KNEE_INPUT_ROOTS', str(base))
    roots = hc.find_cache_roots(config)
    # Mount replaced by its copy; local dir appended last as usual.
    assert len(roots) == 2
    copied = artifact_dir / 'cache_roots' / 'cache-000'
    assert roots[0] == str(copied)
    assert copied.is_dir() and (copied / 'cache_manifest.parquet').exists()
    assert roots[-1] == str(tmp_path / 'b')
    # Idempotent second pass reuses the copy.
    again = hc.find_cache_roots(config)
    assert again == roots
