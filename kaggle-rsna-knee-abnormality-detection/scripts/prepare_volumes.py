#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optional Kernel 1: decode a *bounded* npz shard of DICOM volumes.

Kaggle gives ~30 GB of scratch against ~570 GB of raw data, so a full
``volumes_cache`` cannot exist. This script therefore supports:

* **Sharding** -- ``--shard i --num-shards n`` decodes only slice ``i``
  of the series table; each kernel publishes its shard as its own
  versioned Kaggle dataset.
* **Disk guard** -- decoding stops before free space drops below
  ``--min-free-gb``, so the kernel never dies mid-write.
* **Resume** -- existing ``.npz`` outputs are never recomputed, so
  re-running after the 12 h wall continues where it stopped.

Training and inference do NOT require this cache: the datamodule
stream-decodes DICOMs directly (see ``knee.datasets.volume_store``).
Caching merely accelerates later kernels when shards are mounted.

Usage (Kaggle or local):
    python scripts/prepare_volumes.py \
        --data-root /kaggle/input/competitions/rsna-knee-abnormality-detection \
        --cache-dir /kaggle/working/volumes_cache \
        --shard 0 --num-shards 8 --min-free-gb 4 --workers 4

Outputs:
    <cache-dir>/<SeriesInstanceUID>.npz + volumes_manifest.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from knee.config_params.schema import DataConfig
from knee.datasets.volume_builder import prepare_all
from knee.helpers.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with data_root, series_csv, cache_dir, workers, shard,
      num_shards and min_free_gb.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-root', required=True)
  parser.add_argument('--series-csv', default='train_series.csv')
  parser.add_argument('--cache-dir', default='volumes_cache')
  parser.add_argument('--image-size', type=int, default=384)
  parser.add_argument('--num-slices', type=int, default=32)
  parser.add_argument('--workers', type=int, default=4)
  parser.add_argument(
    '--shard',
    type=int,
    default=0,
    help='Zero-based shard id when splitting work across kernels.',
  )
  parser.add_argument(
    '--num-shards',
    type=int,
    default=1,
    help='Total shard count; 1 processes the whole table.',
  )
  parser.add_argument(
    '--min-free-gb',
    type=float,
    default=4.0,
    help='Stop decoding once free disk drops below this many GB.',
  )
  return parser.parse_args()


def main() -> None:
  """Build one resumable, disk-bounded volume-cache shard."""

  args = parse_args()
  log = get_logger('prepare_volumes')
  if not 0 <= args.shard < max(args.num_shards, 1):
    raise SystemExit(f'--shard {args.shard} outside [0, {args.num_shards})')
  data_cfg = DataConfig(
    image_size=args.image_size,
    num_slices=args.num_slices,
    in_chans=args.num_slices,
  )
  manifest = prepare_all(
    series_csv=str(Path(args.data_root) / args.series_csv),
    data_root=args.data_root,
    data_cfg=data_cfg,
    cache_dir=args.cache_dir,
    workers=args.workers,
    shard=args.shard,
    num_shards=args.num_shards,
    min_free_gb=args.min_free_gb,
  )
  out = Path(args.cache_dir) / 'volumes_manifest.parquet'
  manifest.to_parquet(out)
  cached = int((manifest['status'] != 'failed').sum())
  log.info(
    'manifest -> %s (%d/%d series usable in shard %d/%d)',
    out,
    cached,
    len(manifest),
    args.shard,
    args.num_shards,
  )


if __name__ == '__main__':
  main()
