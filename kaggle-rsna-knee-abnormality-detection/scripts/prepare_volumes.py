#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 1: decode every DICOM series once into a compressed npz cache.

Usage (Kaggle or local):
    python scripts/prepare_volumes.py \
        --data-root /kaggle/input/rsna-knee-abnormality-detection \
        --cache-dir /kaggle/working/volumes_cache \
        --workers 4
Outputs:
    <cache-dir>/<SeriesInstanceUID>.npz  +  volumes_manifest.parquet
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
      Namespace with data_root, series_csv, cache_dir, workers.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-root', required=True)
  parser.add_argument('--series-csv', default='train_series.csv')
  parser.add_argument('--cache-dir', default='volumes_cache')
  parser.add_argument('--image-size', type=int, default=384)
  parser.add_argument('--num-slices', type=int, default=32)
  parser.add_argument('--workers', type=int, default=4)
  return parser.parse_args()


def main() -> None:
  """Build the full volume cache and manifest."""

  args = parse_args()
  log = get_logger('prepare_volumes')
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
  )
  out = Path(args.cache_dir) / 'volumes_manifest.parquet'
  manifest.to_parquet(out)
  log.info('manifest -> %s (%d series)', out, len(manifest))


if __name__ == '__main__':
  main()
