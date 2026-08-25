#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Incrementally fill volume-cache shards and publish them over time.

Runs INSIDE a kernel time budget and repeats the same cycle until every
series is cached:

  decode remaining series -> open shard fills up -> on ANY of
  {shard full, push interval, LOW DISK} the shard is published as a
  small versioned dataset (`<base>-vol00`, `-vol01`, ...), its local
  npz copies are DELETED to free scratch, and decoding continues.

Progress is recorded in ``train_folds.csv`` (column ``vol_shard``) of
the ORIGINAL dataset -- one line of truth for what is cached. Training
kernels attach every ``<base>-vol*`` dataset they want; the dispatcher
auto-discovers them and feeds the colon-joined list to
``paths.volumes_cache``. The streaming store prefers those npz files
and falls back to live DICOM decode on any miss, so partial coverage is
always safe.

Slug note: Kaggle slugs allow [a-z0-9-] only -- requested names like
``..._00`` are normalized to ``...-00`` by the shared sanitizer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knee.config_params.schema import DataConfig  # noqa: E402
from knee.datasets.volume_builder import (  # noqa: E402
  series_dir,
  synthesize_series_table,
)
from knee.datasets.volume_store import (  # noqa: E402
  decode_series_volume,
  free_disk_bytes,
)
from knee.helpers.logging_utils import get_logger  # noqa: E402
from publish_dataset import (  # noqa: E402  # pylint: disable=wrong-import-position
  push,
  resolve_credentials,
  sanitize_slug,
  write_metadata,
)

_VOL_SHARD_COL = 'vol_shard'
_MANIFEST = 'volumes_manifest.parquet'


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with data/folds paths, base dataset name, budgets.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-root', required=True)
  parser.add_argument(
    '--train-folds',
    required=True,
    help='train_folds.csv in $WORK; gains the vol_shard column.',
  )
  parser.add_argument(
    '--work',
    default='/kaggle/working',
    help='Scratch holding volumes_cache/<shard>/ folders.',
  )
  parser.add_argument(
    '--base-name',
    default='ah2022-rsna-knee-abnormality-detection',
    help='Shard datasets are named <base>-vol00, -vol01, ...',
  )
  parser.add_argument(
    '--series-csv',
    default='train_series.csv',
    help='Per-series table under data-root.',
  )
  parser.add_argument(
    '--minutes',
    type=float,
    default=480.0,
    help='Decode+publish wall-clock budget for this invocation.',
  )
  parser.add_argument(
    '--push-every-minutes',
    type=float,
    default=20.0,
    help='Publish the open shard this often.',
  )
  parser.add_argument(
    '--shard-size',
    type=int,
    default=1500,
    help='Close + publish a shard once it holds this many series.',
  )
  parser.add_argument(
    '--min-free-gb',
    type=float,
    default=6.0,
    help='Pause decoding below this much free scratch.',
  )
  parser.add_argument(
    '--image-size',
    type=int,
    default=384,
  )
  parser.add_argument(
    '--num-slices',
    type=int,
    default=32,
  )
  return parser.parse_args()


def load_series_table(data_root: str, series_csv: str) -> pd.DataFrame:
  """Series table with Study/SeriesInstanceUID columns.

  Args:
      data_root: Competition dataset root.
      series_csv: CSV filename under data-root (fallback: walk tree).

  Returns:
      Frame with SeriesInstanceUID (+StudyInstanceUID when known).
  """
  csv_path = Path(data_root) / series_csv
  if csv_path.exists():
    table = pd.read_csv(csv_path, dtype=str)
    if 'SeriesInstanceUID' not in table.columns:
      raise ValueError(f'{csv_path} lacks SeriesInstanceUID')
    if 'StudyInstanceUID' not in table.columns:
      table['StudyInstanceUID'] = ''
    return table[['SeriesInstanceUID', 'StudyInstanceUID']]
  return synthesize_series_table(data_root, 'train')


def load_folds(path: Path) -> pd.DataFrame:
  """Read train_folds.csv, adding the vol_shard column when absent.

  Args:
      path: Folds CSV location.

  Returns:
      Frame guaranteed to carry vol_shard (empty strings).
  """
  folds = pd.read_csv(path)
  if _VOL_SHARD_COL not in folds.columns:
    folds[_VOL_SHARD_COL] = ''
  return folds


def plan_work(
  table: pd.DataFrame, folds: pd.DataFrame, work_dir: Path
) -> pd.DataFrame:
  """Series still needing decode, in deterministic order.

  Done means: annotated with a shard OR already present as npz in the
  local cache (covers crashes between push and annotation).

  Args:
      table: All known series.
      folds: Folds frame possibly carrying vol_shard annotations.
      work_dir: Local cache root scanned for stray completed files.

  Returns:
      Subset of ``table`` ordered by SeriesInstanceUID.
  """
  done = set(
    folds.loc[folds[_VOL_SHARD_COL].fillna('') != '', 'SeriesInstanceUID']
  )
  cache_root = work_dir / 'volumes_cache'
  if cache_root.is_dir():
    done |= {p.stem for p in cache_root.rglob('*.npz')}
  remaining = table[~table['SeriesInstanceUID'].isin(done)]
  return remaining.sort_values('SeriesInstanceUID').reset_index(drop=True)


def next_shard_index(folds: pd.DataFrame, base_slug: str) -> int:
  """One past the highest published shard number in the annotations.

  Args:
      folds: Folds frame with vol_shard annotations.
      base_slug: Sanitized base name; shards look like f'{base}-{nn}'.

  Returns:
      Zero-based index for the OPEN shard.
  """
  used = folds[_VOL_SHARD_COL].dropna().astype(str)
  highest = -1
  prefix = f'{base_slug}-vol'
  for value in used:
    if value.startswith(prefix):
      try:
        highest = max(highest, int(value[len(prefix) :]))
      except ValueError:
        continue
  return highest + 1


def annotate(
  folds: pd.DataFrame, uids: list[str], shard_name: str
) -> pd.DataFrame:
  """Stamp the shard name onto every decoded study-series row.

  Args:
      folds: Folds frame (mutated copy returned).
      uids: SeriesInstanceUIDs that now live in ``shard_name``.
      shard_name: Published dataset slug.

  Returns:
      Updated folds frame.
  """
  out = folds.copy()
  out.loc[out['SeriesInstanceUID'].isin(uids), _VOL_SHARD_COL] = shard_name
  return out


def main() -> None:
  """Decode within budget, publishing closed shards periodically."""

  args = parse_args()
  log = get_logger('cache_volumes')
  username, _ = resolve_credentials()
  base_slug = sanitize_slug(args.base_name)

  folds_path = Path(args.train_folds)
  folds = load_folds(folds_path)
  table = load_series_table(args.data_root, args.series_csv)
  todo = plan_work(table, folds, Path(args.work))
  log.info(
    '%d/%d series already cached; %d to go',
    len(table) - len(todo),
    len(table),
    len(todo),
  )
  if todo.empty:
    log.info('nothing to do')
    return

  cfg = DataConfig(image_size=args.image_size, num_slices=args.num_slices)
  cache_root = Path(args.work) / 'volumes_cache'
  shard_idx = next_shard_index(folds, base_slug)
  shard_dir = cache_root / f'{base_slug}-vol{shard_idx:02d}'
  shard_name = f'{base_slug}-vol{shard_idx:02d}'
  shard_dir.mkdir(parents=True, exist_ok=True)
  manifest_rows: list[dict] = []
  shard_uids: list[str] = []

  started = time.monotonic()
  last_push = started
  floor_bytes = int(args.min_free_gb * (1 << 30))

  def close_shard() -> None:
    """Publish the open shard and record it in train_folds.csv."""
    nonlocal shard_idx, shard_dir, shard_name
    nonlocal manifest_rows, shard_uids, folds
    if not shard_uids:
      return
    pd.DataFrame(manifest_rows).to_parquet(shard_dir / _MANIFEST)
    write_metadata(shard_dir, username, shard_name)
    push(shard_dir, username, shard_name, f'{len(shard_uids)} volumes', 'skip')
    folds = annotate(folds, shard_uids, shard_name)
    folds.to_csv(folds_path, index=False)
    log.info(
      'published %s (%d volumes); folds updated', shard_name, len(shard_uids)
    )
    for npz in shard_dir.glob('*.npz'):
      npz.unlink()  # freed: the dataset now owns these bytes
    shard_idx += 1
    shard_name = f'{base_slug}-vol{shard_idx:02d}'
    shard_dir = cache_root / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows, shard_uids = [], []

  for position, row in enumerate(todo.itertuples(), start=1):
    elapsed_min = (time.monotonic() - started) / 60
    if elapsed_min >= args.minutes:
      log.info(
        'time budget %.0f min reached at %d/%d series',
        args.minutes,
        position - 1,
        len(todo),
      )
      break
    if free_disk_bytes(cache_root) < floor_bytes:
      # Disk pressure is a PUBLISH trigger, not a stop: push the open
      # shard, delete its local npz copies, keep decoding. Abort only
      # if freeing did not help (some foreign disk consumer).
      log.warning(
        'free disk under %.1f GB -> auto-publishing open shard '
        '(%d volumes) and deleting local copies',
        args.min_free_gb,
        len(shard_uids),
      )
      close_shard()
      last_push = time.monotonic()
      if free_disk_bytes(cache_root) < floor_bytes:
        log.error(
          'scratch still below %.1f GB after publish; aborting session '
          '(re-run later to continue where this left off)',
          args.min_free_gb,
        )
        break
      continue
    series_uid = row.SeriesInstanceUID
    try:
      directory = series_dir(
        args.data_root, series_uid, str(row.StudyInstanceUID)
      )
      volume = decode_series_volume(
        directory,
        image_size=int(cfg.image_size),
        num_slices=int(cfg.num_slices),
        percentile_clip=tuple(cfg.percentile_clip),
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log.warning('skip %s (%s)', series_uid, exc)
      continue
    out_path = shard_dir / f'{series_uid}.npz'
    np_savez(out_path, volume)
    manifest_rows.append({'SeriesInstanceUID': series_uid, 'status': 'ok'})
    shard_uids.append(series_uid)
    if len(shard_uids) >= args.shard_size or (
      (time.monotonic() - last_push) / 60 >= args.push_every_minutes
      and shard_uids
    ):
      close_shard()
      last_push = time.monotonic()

  close_shard()
  remaining_after = plan_work(table, load_folds(folds_path), Path(args.work))
  log.info(
    'session done: %d series newly cached, %d remain (re-run '
    "'cache-volumes' to continue)",
    len(todo) - len(remaining_after),
    len(remaining_after),
  )


def np_savez(path: Path, volume) -> None:  # noqa: ANN001
  """Persist one decoded volume as compressed npz.

  Args:
      path: Target ``<uid>.npz`` path.
      volume: uint8 ``(C, H, W)`` array.
  """
  import numpy as np  # noqa: PLC0415  # keep numpy lazy like decoders

  np.savez_compressed(path, **{'volume': volume})


if __name__ == '__main__':
  main()
