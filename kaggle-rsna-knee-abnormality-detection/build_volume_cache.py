#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone DICOM -> sharded HDF5 volume cache converter (offline).

Runs WITHOUT experiment configuration: point it at one or more DICOM
roots and optionally existing index parquets. Produces the exact shard +
manifest layout consumed by helpers.h5_cache.H5SeriesReader at train
time, so you can prebuild the cache on any machine (e.g. a CPU-only
session or local workstation) and later mount/push it.

Examples:
    # Using a prepared index parquet:
    python build_volume_cache.py --dicom-root /data/train_series \
        --index artifacts/index.parquet --out /tmp/volume_cache

    # Header-scan when no index exists yet:
    python build_volume_cache.py --dicom-root /data/train_series \
        --dicom-root /data/test_series --out /tmp/volume_cache

    # With Discord progress heartbeats (secret resolved via env first):
    DISCORD_WEBHOOK_URL=... python build_volume_cache.py ... \
        --discord-every 10000

Resume-safe: reruns skip every SeriesInstanceUID already present in the
output shards.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _bootstrap() -> None:
  """Make knee.* importable when launched from a bare checkout."""
  src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
  if os.path.isdir(src) and src not in sys.path:
    sys.path.insert(0, src)


_bootstrap()
# Imports MUST follow _bootstrap(): knee.* resolves through the repo's
# src/ only after sys.path injection above. Notifier imports stay inside
# main() because they are strictly optional (no webhook -> disabled).
# pylint: disable=wrong-import-position,import-outside-toplevel

import pandas as pd  # noqa: E402

from knee.helpers.h5_cache import (  # noqa: E402
  GIB,
  ShardWriter,
  format_progress,
  load_manifest,
  run_pool_tasks,
)
from knee.helpers.header_scan import build_index, explode_sop_uids  # noqa: E402
from knee.helpers.utils import get_logger  # noqa: E402

_LOGGER = get_logger('build_volume_cache')


def _parser() -> argparse.ArgumentParser:
  """Build CLI for the offline conversion.

  Returns:
      Configured ArgumentParser.
  """
  parser = argparse.ArgumentParser(
    description='Convert DICOM series into sharded HDF5 volumes'
  )
  parser.add_argument(
    '--dicom-root',
    action='append',
    default=[],
    help='DICOM root with <study>/<series>/*.dcm (repeatable)',
  )
  parser.add_argument(
    '--index',
    action='append',
    default=[],
    help='Index parquet aligned to a --dicom-root; repeatable, order '
    'must match --dicom-root occurrences',
  )
  parser.add_argument('--out', required=True, help='Output cache directory')
  parser.add_argument('--img-size', type=int, default=384)
  parser.add_argument('--shard-gib', type=int, default=10)
  parser.add_argument('--gzip-level', type=int, default=4)
  parser.add_argument('--workers', type=int, default=4)
  parser.add_argument(
    '--scan-workers',
    type=int,
    default=None,
    help='Worker count for header scans (defaults to --workers)',
  )
  parser.add_argument(
    '--files-every',
    type=int,
    default=10_000,
    help='Log/Discord heartbeat cadence in files; 0 off',
  )
  parser.add_argument(
    '--discord-secret',
    default='DISCORD_WEBHOOK_URL',
    help='Secret name resolving to a webhook URL; only '
    'used when the value is non-empty',
  )
  return parser.parse_args()


def _collect_tasks(args: argparse.Namespace) -> list[dict]:
  """Assemble decode tasks from indexes or by scanning roots.

  Args:
      args: Parsed CLI arguments.

  Returns:
      Index rows enriched with dicom_root.
  """
  tasks: list[dict] = []
  scan_workers = args.scan_workers or args.workers
  for pos, root in enumerate(args.dicom_root):
    index_path = args.index[pos] if pos < len(args.index) else None
    if index_path and os.path.exists(index_path):
      frame = pd.read_parquet(index_path)
      _LOGGER.info('%s: using index %s (%d rows)', root, index_path, len(frame))
    elif index_path:
      raise FileNotFoundError(f'--index given but missing: {index_path}')
    else:
      _LOGGER.info('%s: no index provided; scanning headers...', root)
      frame = build_index(root, workers=scan_workers)
    enriched = explode_sop_uids(frame)
    rows = enriched.to_dict('records')
    for row in rows:
      row['dicom_root'] = root
      row['img'] = args.img_size
      row['decoder_order'] = ['native', 'gdcm', 'pylibjpeg']
      row['percentiles'] = [0.005, 0.995]
      row['margin'] = 0.05
    tasks.extend(rows)
  return tasks


def main() -> None:  # noqa: C901
  """Run the offline conversion end-to-end."""
  args = _parser()
  if not args.dicom_root:
    _LOGGER.error('At least one --dicom-root is required')
    sys.exit(2)
  os.makedirs(args.out, exist_ok=True)

  tasks = _collect_tasks(args)
  total_files = sum(len(list(t['sop_uids'])) for t in tasks)
  started_at = time.time()

  notifier = None
  secret_name = args.discord_secret
  if secret_name:
    try:
      from knee.helpers.secrets import get_secret
      from knee.loggers.discord_logger import DiscordNotifier

      url = get_secret(secret_name)
      if url:
        notifier = DiscordNotifier(webhook_url=url, enabled=True)
      else:
        _LOGGER.warning(
          'Discord secret %r empty; progress disabled', secret_name
        )
    except Exception as exc:  # pylint: disable=broad-except
      _LOGGER.warning(
        'Notifier unavailable (%s); continuing without discord updates', exc
      )

  def say(text: str) -> None:
    """Best-effort console + optional Discord output.

    Args:
        text: Message body.
    """
    print(f'[cache] {text}', flush=True)
    if notifier is not None:
      notifier.notify(f'**[volume-cache]** {text}')

  def report(state: dict) -> None:
    """Forward pool progress events.

    Args:
        state: Stats mapping from helpers.h5_cache.progress_state.
    """
    say(format_progress(state, time.time() - started_at))

  writer = ShardWriter(
    args.out,
    img_size=args.img_size,
    shard_bytes_cap=args.shard_gib * GIB,
    gzip_level=args.gzip_level,
  )
  existing = writer.existing_uids()
  todo = [t for t in tasks if str(t['series']) not in existing]
  say(
    f'started: {len(todo):,}/{len(tasks):,} series to decode '
    f'({total_files:,} files), out={args.out}'
  )
  try:
    cached, skipped = run_pool_tasks(
      todo,
      args.workers,
      writer,
      # Console heartbeats always; Discord piggybacks via say().
      on_progress=report,
      files_every=args.files_every,
      total_files=sum(len(list(t['sop_uids'])) for t in todo) or None,
    )
    manifest_path = writer.write_manifest()
  finally:
    writer.close()

  manifest = load_manifest([args.out])
  shards = sorted(n for n in os.listdir(args.out) if n.endswith('.h5'))
  gib = sum(os.path.getsize(os.path.join(args.out, n)) for n in shards) / GIB
  minutes = (time.time() - started_at) / 60
  summary = (
    f'HDF5 cache complete in {minutes:.0f} min | cached={cached} '
    f'skipped={skipped} | {len(shards)} shard file(s) totalling '
    f'{gib:.1f} GiB under {args.out}\nmanifest: {manifest_path} '
    f'({len(manifest) if manifest is not None else 0} series)'
  )
  say(summary)


if __name__ == '__main__':
  main()
