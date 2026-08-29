#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI entrypoint orchestrating the RSNA knee MVP pipeline.

Subcommands map one-to-one onto BLUEPRINT phases:

* ``build-index``: header-only DICOM scan -> index.parquet
* ``build-labels``: rule-based pseudo-labels -> labels_pseudo.csv
* ``build-folds``: grouped stratified folds -> folds.csv
* ``train``: resume-aware fold training with session budget + pushes
* ``infer``: fold-ensemble prediction -> submission.csv

Example:
    python main.py train \\
      --experiment configs/experiments/mvp_efnv2s_384_k24_5f.yaml \\
      --fold 0 --override data.n_slices=16
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import sys
import time

import pandas as pd
import pytorch_lightning as pl

from knee.callbacks.session import PeriodicPushCallback, TimeBudgetCallback
from knee.config_params.loader import dump_config, instantiate
from knee.engines.assembly import (
  build_datamodule,
  build_datasets,
  build_model,
  compose_experiment,
  find_resume_checkpoint,
  fold_done,
)
from knee.engines.inferencer import predict_studies, write_submission
from knee.engines.noise_floor import (
  STATE_NAME,
  SUMMARY_NAME,
  collect_run_result,
  config_for_run,
  format_discord,
  load_state,
  plan_runs,
  run_key,
  save_state,
  state_dir,
  summarize,
  write_summary_csv,
)
from knee.engines.selftest import render_summary, run_selftest
from knee.engines.train_module import KneeModule
from knee.helpers.folds import make_folds, resolve_group_column
from knee.helpers.h5_cache import (
  CACHE_META_NAME,
  GIB,
  MANIFEST_NAME,
  ShardWriter,
  collect_shard_map,
  drop_unfinished_shards,
  find_cache_roots,
  format_progress,
  mount_roots,
  run_pool_tasks,
)
from knee.helpers.header_scan import build_index, explode_sop_uids
from knee.helpers.kaggle_io import (
  ArtifactSync,
  CredentialResolver,
  KaggleDatasetClient,
)
from knee.helpers.logging_setup import setup_logging
from knee.helpers.nlp_labeling import RuleBasedLabeler, build_pseudo_labels
from knee.helpers.utils import get_logger
from knee.loggers.csv_logger import build_csv_logger
from knee.loggers.discord_logger import (
  DiscordCallback,
  notifier_from_config,
)
from knee.loggers.progress import ProgressLogCallback
from knee.loggers.wandb_logger import build_wandb_logger

_LOGGER = get_logger('main')


def _parser() -> argparse.ArgumentParser:
  """Build the CLI parser.

  Returns:
      ArgumentParser with subcommands and shared flags.
  """
  parser = argparse.ArgumentParser(description='RSNA knee MVP pipeline')
  sub = parser.add_subparsers(dest='command', required=True)

  def add_common(target):
    """Attach shared flags to a subparser.

    Args:
        target: Subparser being configured.
    """
    target.add_argument('--experiment', required=True)
    target.add_argument('--override', nargs='*', default=[])

  for name in [
    'build-index',
    'build-labels',
    'build-folds',
    'build-cache',
    'selftest',
    'train',
    'infer',
    'sweep',
  ]:
    add_common(sub.add_parser(name))
  sub.choices['train'].add_argument('--fold', type=int, default=None)
  return parser


def _client(config: dict) -> KaggleDatasetClient | None:
  """Build the Kaggle client when resume is enabled.

  Args:
      config: Composed experiment configuration.

  Returns:
      Client instance or None when resume is disabled.
  """
  if not config['resume']['enabled']:
    return None
  secrets = config['kaggle_secrets']
  resume = config['resume']
  return KaggleDatasetClient(
    CredentialResolver(secrets['username_key'], secrets['token_key']),
    retries=int(resume.get('cli_retries', 3)),
    backoff_seconds=float(resume.get('cli_backoff_seconds', 2.0)),
  )


ARTIFACT_BASENAMES = (
  'index.parquet',
  'index_test.parquet',
  'labels_pseudo.csv',
  'folds.csv',
)


def _mount_roots() -> list[str]:
  """Directories that may contain attached-dataset artifact copies.

  Shared discovery lives in helpers.h5_cache.mount_roots (same
  KNEE_INPUT_ROOTS override), keeping one mount convention everywhere.

  Returns:
      Existing directory paths.
  """
  return mount_roots()


def remote_cache_state(config: dict) -> tuple[set[str], int]:
  """Series already cached in PUSHED cache datasets + highest ordinal.

  Fragment manifests travel inside every pushed per-shard dataset, so
  attached mounts fully describe remote coverage without any CLI calls.
  Shard ordinals come from stored shard_file names (volume_shard_NNN),
  independent of mount naming, letting new sessions continue the
  sequence instead of colliding into version bumps of old shards.

  Args:
      config: Composed experiment configuration.

  Returns:
      Tuple (remote SeriesInstanceUIDs, max remote shard ordinal; -1 if
      no fragments found).
  """
  roots = _mount_roots()
  local_root = config.get('paths', {}).get('volume_cache_dir', '')
  if local_root and local_root not in roots:
    roots.append(local_root)
  uids: set[str] = set()
  max_ordinal = -1
  for root in roots:
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.exists(path):
      continue
    try:
      frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
      _LOGGER.warning('skipping unreadable fragment %s (%s)', path, exc)
      continue
    uids.update(frame['SeriesInstanceUID'].astype(str))
    for shard_name in frame['shard_file'].astype(str):
      match = re.search(r'volume_shard_(\d+)\.h5$', shard_name)
      if match:
        max_ordinal = max(max_ordinal, int(match.group(1)))
  if uids:
    _LOGGER.info(
      'remote cache coverage: %d series across shards up to ordinal %d',
      len(uids),
      max_ordinal,
    )
  return uids, max_ordinal


def restore_artifacts_from_mounts(config: dict) -> list[str]:
  """Copy tracked stage artifacts out of attached datasets when missing.

  Kaggle wipes /kaggle/working between sessions; stages that need
  index/labels/folds either rehydrate via the kaggle CLI pull or, when
  the backing datasets are already ATTACHED, simply copy the files -
  instant and network-free. Files stay on the mount untouched.

  Args:
      config: Composed experiment configuration.

  Returns:
      Basenames that were restored by this call.
  """
  artifact_dir = config['paths']['artifact_dir']
  os.makedirs(artifact_dir, exist_ok=True)
  restored: list[str] = []
  roots = _mount_roots()
  for name in ARTIFACT_BASENAMES:
    target = os.path.join(artifact_dir, name)
    if os.path.exists(target):
      continue
    for root in roots:
      candidate = os.path.join(root, name)
      if os.path.exists(candidate):
        shutil.copy2(candidate, target)
        restored.append(name)
        _LOGGER.info('restored %s from attached dataset %s', name, root)
        break
  return restored


def _artifact_sync(
  config: dict, client: KaggleDatasetClient | None
) -> ArtifactSync:
  """Build the sync helper for the small data-stage artifacts.

  Args:
      config: Composed experiment configuration.
      client: Shared Kaggle client (may be None when resume disabled).

  Returns:
      ArtifactSync tracking index/labels/folds outputs.
  """
  paths = config['paths']
  return ArtifactSync(
    client=client,
    slug=config['resume']['index_dataset_slug'],
    local_dir=paths['artifact_dir'],
    file_names=[
      os.path.basename(paths['index_parquet']),
      os.path.basename(paths['labels_csv']),
      os.path.basename(paths['folds_csv']),
    ],
  )


def cmd_build_index(config: dict) -> None:
  """Scan DICOM headers and persist the merged series index.

  Args:
      config: Composed experiment configuration.
  """
  frame = build_index(
    config['paths']['train_dicom_dir'],
    workers=int(config['data'].get('scan_workers', 4)),
    pool_chunksize=int(config['data'].get('scan_chunksize', 16)),
  )
  series_meta = pd.read_csv(config['paths']['train_series_csv'])
  merged = frame.merge(
    series_meta.rename(
      columns={
        'SeriesInstanceUID': 'series',
        'StudyInstanceUID': 'study',
        'Anatomical_Plane': 'plane',
      }
    )[['series', 'plane', 'Fluid_Sensitive', 'Fat_Suppression']],
    on='series',
    how='left',
  ).rename(
    columns={
      'Fluid_Sensitive': 'fluid_sensitive',
      'Fat_Suppression': 'fat_suppression',
    }
  )
  out_path = config['paths']['index_parquet']
  merged.to_parquet(out_path, index=False)
  _LOGGER.info('Index written: %s (%d series)', out_path, len(merged))
  _artifact_sync(config, _client(config)).push()


def cmd_build_labels(config: dict) -> None:
  """Derive rule-based pseudo-labels from reports.

  Args:
      config: Composed experiment configuration.
  """
  _artifact_sync(config, _client(config)).pull_if_missing()
  train_df = pd.read_csv(config['paths']['train_csv'])
  labeled = build_pseudo_labels(
    train_df,
    study_column='StudyInstanceUID',
    target_columns=list(config['data']['target_columns']),
    labeler=RuleBasedLabeler(),
  )
  out_path = config['paths']['labels_csv']
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  labeled.to_csv(out_path, index=False)
  _LOGGER.info('Labels written: %s (%d rows)', out_path, len(labeled))
  _artifact_sync(config, _client(config)).push()


def cmd_build_folds(config: dict) -> None:
  """Assign CV folds and persist the mapping.

  Args:
      config: Composed experiment configuration.
  """
  _artifact_sync(config, _client(config)).pull_if_missing()
  labels = pd.read_csv(config['paths']['labels_csv'])
  splitter = instantiate(config['folds'])
  stratify = config.get('stratify', {})
  fold_series = make_folds(
    labels,
    splitter,
    rare_targets=stratify.get('rare_targets', []),
    anchor_targets=stratify.get('anchor_targets', []),
  )
  group_column, _ = resolve_group_column(labels)
  out_frame = pd.DataFrame(
    {
      group_column: labels[group_column].values,
      'fold': fold_series.values,
    }
  )
  out_path = config['paths']['folds_csv']
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  out_frame.to_csv(out_path, index=False)
  _LOGGER.info('Folds written: %s (groups=%s)', out_path, group_column)
  _artifact_sync(config, _client(config)).push()


def cmd_build_cache(config: dict) -> None:
  """Decode every indexed series into sharded HDF5 volume files.

  Two push layouts selected by ``volume_cache.split_mode``:

  * ``shard`` (pipelined): each finished shard becomes its own Kaggle
    dataset ``<base>-NNN`` pushed IMMEDIATELY from the completion
    callback - peak disk stays ~2x one shard, fitting /kaggle/working
    quotas on the mandated build path. Every pushed dataset carries a
    manifest fragment; train sessions merge fragments across ALL
    attached datasets via helpers.h5_cache.load_manifest.
  * ``group``: shards accumulate locally, are grouped under
    KNEE_CACHE_SLUG_GIB_CAP GiB per dataset (-a/-b suffixes), and push
    once after decoding finishes.

  Resume-safe: UIDs present in COMPLETED (markered) shards skip; a
  killed session's partial tail shard is dropped so its series
  re-decode fresh. Rejected series (any failed frame) never fossilize.

  Args:
      config: Composed experiment configuration.
  """
  cache_cfg = config.get('volume_cache', {})
  workers = int(
    cache_cfg.get('scan_workers', config['data'].get('scan_workers', 4))
  )
  shard_cap = int(cache_cfg.get('shard_gib_cap', 10)) * GIB
  pool_chunksize = int(cache_cfg.get('pool_chunksize', 2))
  cache_log_every = int(cache_cfg.get('log_every_series', 50))
  gzip_level = int(cache_cfg.get('gzip_level', 4))
  split_mode = str(cache_cfg.get('split_mode', 'group'))
  cache_dir = config['paths']['volume_cache_dir']
  os.makedirs(cache_dir, exist_ok=True)

  tasks: list[dict] = []
  data_cfg = config['data']
  # Rehydrate artifacts BEFORE any read: attached datasets copy instantly
  # (user mandate), kaggle-CLI pull covers unattached fallbacks.
  restored = restore_artifacts_from_mounts(config)
  if restored:
    _LOGGER.info(
      'restored stage artifacts from attached datasets: %s', restored
    )
  for root_key, index_key in (
    ('train_dicom_dir', 'index_parquet'),
    ('test_dicom_dir', 'test_index_parquet'),
  ):
    index_path = config['paths'].get(index_key)
    if not index_path or not os.path.exists(index_path):
      continue
    frame = explode_sop_uids(pd.read_parquet(index_path))
    rows = frame.to_dict('records')
    for row in rows:
      row['dicom_root'] = config['paths'][root_key]
      row['img'] = int(data_cfg['img_size'])
      row['decoder_order'] = list(data_cfg['decode_backend_order'])
      row['percentiles'] = list(data_cfg['normalize_percentiles'])
      row['margin'] = float(data_cfg['autocrop_margin'])
      row['bg_threshold'] = float(
        data_cfg.get('autocrop_background_threshold', 0.02)
      )
      row['interpolation'] = int(data_cfg.get('resize_interpolation', 1))
      row['fallback_shape'] = tuple(data_cfg.get('fallback_shape', (512, 512)))
    tasks.extend(rows)
  if not tasks:
    # Last-resort rehydration via kaggle CLI (datasets not attached).
    _artifact_sync(config, _client(config)).pull_if_missing()
    for root_key, index_key in (
      ('train_dicom_dir', 'index_parquet'),
      ('test_dicom_dir', 'test_index_parquet'),
    ):
      index_path = config['paths'].get(index_key)
      if not (index_path and os.path.exists(index_path)):
        continue
      frame = explode_sop_uids(pd.read_parquet(index_path))
      rows = frame.to_dict('records')
      for row in rows:
        row['dicom_root'] = config['paths'][root_key]
        row['img'] = int(data_cfg['img_size'])
        row['decoder_order'] = list(data_cfg['decode_backend_order'])
        row['percentiles'] = list(data_cfg['normalize_percentiles'])
        row['margin'] = float(data_cfg['autocrop_margin'])
      tasks.extend(rows)
  if not tasks:
    artifact_dir = config['paths']['artifact_dir']
    raise RuntimeError(
      'No index parquet found to build the volume cache; attach '
      'rsna-knee-mvp-index (or run --stage index) so '
      f'{artifact_dir} can be populated'
    )
  base_slug = str(config['resume'].get('cache_dataset_slug') or '')
  client = _client(config)
  pushed: list[tuple[str, int, float]] = []
  staged_slugs: set[str] = set()
  if split_mode == 'shard':
    drop_unfinished_shards(cache_dir)
  # Cross-session resume: pushed datasets already carry coverage; their
  # fragment manifests (via attached mounts) prevent re-decoding AND the
  # ordinal floor prevents -NNN slug collisions/version churn.
  remote_uids, remote_max_ordinal = remote_cache_state(config)

  # Discord lifecycle plumbing; notifier_from_config logs LOUDLY when a
  # webhook cannot be resolved instead of degrading silently.
  notifier = notifier_from_config(config)
  experiment_name = config['experiment']['name']
  files_every = int(cache_cfg.get('discord_files_every', 10_000))
  started_at = time.time()

  def say(text: str) -> None:
    """Console-first heartbeat + optional Discord POST.

    Args:
        text: Message body without experiment prefix.
    """
    if notifier.enabled:
      notifier.notify(f'**[{experiment_name}]** cache: {text}')

  def push_shard_now(shard_path: str) -> None:
    """Pipelined consumer: relocate one finished shard into its dataset.

    Args:
        shard_path: Just-closed shard file (callback thread == main).
    """
    if client is None or not base_slug:
      return
    name = os.path.basename(shard_path)
    ordinal = int(name.rsplit('_', maxsplit=1)[-1].split('.')[0])
    slug = f'{base_slug}-{ordinal:03d}'
    if slug in staged_slugs:
      raise RuntimeError(
        f'shard ordinal regression: {slug} already staged this session '
        '(writer reused an ordinal; shard data would overwrite a '
        'previously pushed dataset version)'
      )
    staged_slugs.add(slug)
    staging = os.path.join(
      os.path.dirname(cache_dir.rstrip('/')), f'push_{slug}'
    )
    os.makedirs(staging, exist_ok=True)
    shutil.move(shard_path, os.path.join(staging, name))
    marker_src = f'{shard_path}.complete'
    if os.path.exists(marker_src):
      shutil.move(marker_src, os.path.join(staging, name) + '.complete')
    rows_for_shard = collect_shard_map(staging)
    fragment = pd.DataFrame(
      {
        'SeriesInstanceUID': [u for u, _ in rows_for_shard.items()],
        'shard_file': [v[0] for v in rows_for_shard.values()],
        'n_slices': [v[1] for v in rows_for_shard.values()],
      }
    ).sort_values('SeriesInstanceUID')
    fragment.to_parquet(os.path.join(staging, MANIFEST_NAME), index=False)
    # Generation stamp: reader-side load_manifest() detects mixed
    # preprocessing generations across mounts via this file.
    with open(
      os.path.join(staging, CACHE_META_NAME), 'w', encoding='utf-8'
    ) as meta_handle:
      json.dump(
        {
          'img_size': int(data_cfg['img_size']),
          'split_mode': split_mode,
          'shard_gib_cap_gib': int(cache_cfg.get('shard_gib_cap', 10)),
          'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        },
        meta_handle,
        indent=2,
      )
    gib = os.path.getsize(os.path.join(staging, name)) / GIB
    say(f'pushing dataset `{slug}` ({name}, {gib:.2f} GiB)...')
    client.push_version_inplace(slug, staging)
    _LOGGER.info('Pushed %s -> dataset %s', name, slug)
    pushed.append((slug, 1, gib))

  def report(state: dict) -> None:
    """Forward pool progress events to the notifier.

    Args:
        state: helpers.h5_cache.progress_state mapping.
    """
    elapsed = time.time() - started_at
    say(format_progress(state, elapsed))

  writer = ShardWriter(
    cache_dir,
    img_size=int(data_cfg['img_size']),
    shard_bytes_cap=shard_cap,
    gzip_level=gzip_level,
    on_shard_complete=(push_shard_now if split_mode == 'shard' else None),
    start_ordinal=(
      remote_max_ordinal + 1
      if split_mode == 'shard' and remote_max_ordinal >= 0
      else None
    ),
  )
  # Cross-session resume: remote fragments union local shards, so a
  # fresh kernel skips everything already pushed instead of re-decoding.
  existing = writer.existing_uids() | remote_uids
  todo = [t for t in tasks if str(t['series']) not in existing]
  total_files = sum(len(list(t['sop_uids'])) for t in todo) or None
  _LOGGER.info(
    'Resume state: %d cached already, %d to decode (%s mode)',
    len(existing),
    len(todo),
    split_mode,
  )
  img_edge = int(data_cfg['img_size'])
  say(
    f'started building HDF5 cache - {len(todo):,} series / '
    f'{(total_files or 0):,} DICOM files, workers={workers}, '
    f'img_size={img_edge}, split={split_mode}'
  )
  try:
    cached, skipped = run_pool_tasks(
      todo,
      workers,
      writer,
      on_progress=(report if notifier.enabled else None),
      files_every=files_every,
      total_files=total_files,
      pool_chunksize=pool_chunksize,
      log_every=cache_log_every,
    )
  finally:
    writer.close()  # fires callback for any data-bearing tail shard
  _LOGGER.info('Decoded %d series; skipped %d this pass', cached, skipped)

  union_map: dict[str, tuple[str, int]] = {}
  for push_root in sorted(
    glob.glob(os.path.dirname(cache_dir.rstrip('/')) + '/push_*')
  ):
    union_map.update(collect_shard_map(push_root))
  union_map.update(collect_shard_map(cache_dir))

  if split_mode == 'group':
    max_slug_gib = float(
      os.environ.get(
        'KNEE_CACHE_SLUG_GIB_CAP',
        str(config['resume'].get('push_slug_gib_cap', 95)),
      )
    )
    local = [n for n in sorted(os.listdir(cache_dir)) if n.endswith('.h5')]
    groups: list[list[str]] = [[]]
    running = 0.0
    for name in local:
      size_gib = os.path.getsize(os.path.join(cache_dir, name)) / GIB
      if running + size_gib > max_slug_gib and groups[-1]:
        groups.append([])
        running = 0.0
      groups[-1].append(name)
      running += size_gib
    if local:
      local_manifest_frame = pd.read_parquet(writer.write_manifest())
      for group_no, shard_names in enumerate(groups):
        suffix = '' if group_no == 0 else f'-{chr(97 + group_no)}'
        slug = f'{base_slug}{suffix}'
        staging = os.path.join(
          os.path.dirname(cache_dir.rstrip('/')), f'push_{slug}'
        )
        os.makedirs(staging, exist_ok=True)
        for name in shard_names:
          shutil.move(
            os.path.join(cache_dir, name), os.path.join(staging, name)
          )
        subset = local_manifest_frame[
          local_manifest_frame['shard_file'].isin(shard_names)
        ].copy()
        if subset.empty:
          raise RuntimeError(f'manifest lost coverage for {shard_names[:3]}')
        subset.to_parquet(os.path.join(staging, MANIFEST_NAME), index=False)
        group_gib = (
          sum(os.path.getsize(os.path.join(staging, n)) for n in shard_names)
          / GIB
        )
        say(
          f'pushing dataset `{slug}` ({len(shard_names)} shards, '
          f'{group_gib:.1f} GiB)... upload may take a while'
        )
        if client is None or not base_slug:
          _LOGGER.warning('no remote client/slugs; shards staged only')
          continue
        client.push_version_inplace(slug, staging)
        _LOGGER.info('Pushed %d shard(s) to %s', len(shard_names), slug)
        pushed.append((slug, len(shard_names), group_gib))

  # Operator manifest spanning pushed fragments + any local leftovers.
  union_frame = pd.DataFrame(
    {
      'SeriesInstanceUID': list(union_map.keys()),
      'shard_file': [v[0] for v in union_map.values()],
      'n_slices': [v[1] for v in union_map.values()],
    }
  ).sort_values('SeriesInstanceUID')
  union_frame.to_parquet(os.path.join(cache_dir, MANIFEST_NAME), index=False)

  if client is None or not base_slug:
    say(
      f'HDF5 cache built LOCAL-only (remote push unavailable) at '
      f'{cache_dir}; {cached:,} cached / {skipped:,} skipped'
    )
    return

  local_gb = (
    sum(
      os.path.getsize(os.path.join(cache_dir, n))
      for n in os.listdir(cache_dir)
      if n.endswith('.h5')
    )
    / GIB
  )
  names_line = ', '.join(
    f'`{slug}` ({n} {shard_word}, {gb:.2f} GiB)'
    for slug, n, shard_word, gb in [
      (slug, n, 'shard' if n == 1 else 'shards', gb) for slug, n, gb in pushed
    ]
  )
  mounts = ',\n'.join(f'/kaggle/input/{slug}' for slug, *_ in pushed)
  elapsed_min = (time.time() - started_at) / 60.0
  say(
    f'HDF5 cache complete in {elapsed_min:.0f} min. Datasets pushed:\n'
    f'{names_line}\n'
    f'coverage: {len(union_map):,} series ({cached:,} decoded this pass, '
    f'{skipped:,} rejected).\n'
    f'Train-time setup: attach ALL listed datasets AND set\n'
    f'KNEE_HDF5_CACHE_DIRS={mounts}\n'
    f'(local leftovers: {local_gb:.2f} GiB under {cache_dir})'
  )


def cmd_selftest(config: dict) -> None:
  """Preflight the training pipeline; exit 1 on any failed check.

  Checks run in isolation and never abort each other; the summary goes
  to the kernel log AND Discord (when enabled) so failures surface
  without scrolling.

  Args:
      config: Composed experiment configuration.

  Raises:
      SystemExit: Exit code 1 when at least one check failed.
  """
  results = run_selftest(config)
  summary = render_summary(results)
  print('\n=== SELFTEST ===\n' + summary + '\n================')
  notifier = notifier_from_config(config)
  if notifier.enabled:
    exp_name = config['experiment']['name']
    notifier.notify(f'**[{exp_name}]** {summary}')
  if any(not ok for _, ok, _ in results):
    sys.exit(1)


def cmd_train(config: dict, fold_id: int | None) -> None:
  """Run resume-aware fold training within one kernel session.

  Args:
      config: Composed experiment configuration.
      fold_id: Optional single-fold override of run.folds.
  """
  client = _client(config)
  checkpoint_dir = config['paths']['checkpoint_dir']
  if client is not None:
    client.pull_latest(
      config['resume']['checkpoint_dataset_slug'], checkpoint_dir
    )
  _artifact_sync(config, client).pull_if_missing()
  folds = [fold_id] if fold_id is not None else list(config['run']['folds'])
  # Visibility: which pixel source is live + whether the local mirror
  # is engaged (FUSE random reads were a measured step-time killer).

  roots = find_cache_roots(config)
  mirrored = [r for r in roots if '/kaggle/tmp/cache_roots' in r]
  _LOGGER.info(
    'pixel source: %d cache root(s), %d on local scratch; roots=%s',
    len(roots),
    len(mirrored),
    roots,
  )
  if not roots:
    _LOGGER.warning('training on LIVE DICOM decode - cache not mounted!')

  index_df = explode_sop_uids(pd.read_parquet(config['paths']['index_parquet']))
  labels = pd.read_csv(config['paths']['labels_csv'])
  folds_df = pd.read_csv(config['paths']['folds_csv'])
  labels = labels.merge(
    folds_df[['StudyInstanceUID', 'fold']], on='StudyInstanceUID'
  )
  label_lookup = labels.set_index('StudyInstanceUID')

  trainer_cfg = {
    key: value for key, value in config['trainer']['init_params'].items()
  }
  trainer_cfg.setdefault(
    'gradient_clip_val', config['optimizer'].get('gradient_clip_val')
  )
  total_epochs = int(trainer_cfg.get('max_epochs', 1))

  for current_fold in folds:
    if fold_done(checkpoint_dir, current_fold):
      _LOGGER.info('Fold %d already done; skipping', current_fold)
      continue
    valid_ids = label_lookup.index[
      label_lookup['fold'] == current_fold
    ].tolist()
    train_ids = label_lookup.index[
      label_lookup['fold'] != current_fold
    ].tolist()
    train_ds, valid_ds = build_datasets(
      config, index_df, labels, valid_ids, train_ids
    )
    import math  # pylint: disable=import-outside-toplevel

    batch = int(
      config['datamodule']['init_params'].get('batch_size', 1)
    )
    devices = int(
      config['trainer']['init_params'].get('devices', 1) or 1
    )
    steps_hint = math.ceil(
      len(train_ds) / max(1, batch * devices)
    )
    module = KneeModule(
      model=build_model(config),
      criterion=instantiate(config['loss']),
      optimizer_cfg=config['optimizer'],
      scheduler_cfg=config['optimizer'].get('scheduler'),
      warmup_epochs=int(config['optimizer'].get('warmup_epochs', 0)),
      backbone_lr_scale=float(config['optimizer']['backbone_lr_scale']),
      total_epochs=total_epochs,
      target_columns=list(config['data']['target_columns']),
      oof_dir=config['paths']['oof_dir'],
      fold_id=current_fold,
      steps_per_epoch_hint=steps_hint,
    )
    progress_cfg = config.get('logging', {}).get('progress', {})
    train_callbacks = []
    if progress_cfg.get('enabled', True):
      train_callbacks.append(
        ProgressLogCallback(
          log_every_n_steps=int(trainer_cfg.get('log_every_n_steps', 25)),
          log_gpu_mem=bool(progress_cfg.get('gpu_mem', True)),
          log_host_ram=bool(progress_cfg.get('log_host_ram', True)),
        )
      )
      from pytorch_lightning.callbacks import (  # pylint: disable=import-outside-toplevel
        TQDMProgressBar,
      )

      train_callbacks.append(
        TQDMProgressBar(refresh_rate=int(progress_cfg.get('refresh_rate', 25)))
      )
    callbacks = train_callbacks + [
      TimeBudgetCallback(
        session_time_budget_h=float(config['session_time_budget_h']),
        time_margin_min=float(config.get('time_margin_min', 30.0)),
      ),
      PeriodicPushCallback(
        checkpoint_dir=checkpoint_dir,
        fold_id=current_fold,
        push_every_n_epochs=int(config['checkpoint_every_n_epochs']),
        client=client,
        push_slug=(
          config['resume']['checkpoint_dataset_slug'] if client else None
        ),
      ),
    ]
    discord_notifier = notifier_from_config(config)
    if discord_notifier.enabled:
      callbacks.append(
        DiscordCallback(
          notifier=discord_notifier,
          experiment_name=config['experiment']['name'],
          fold_id=current_fold,
          step_interval=int(
            config.get('integrations', {})
            .get('discord', {})
            .get('every_n_steps', 50)
          ),
          first_step_ping=bool(
            config.get('integrations', {})
            .get('discord', {})
            .get('first_step_ping', True)
          ),
        )
      )
    loggers = [
      build_csv_logger(
        config['experiment']['output_dir'],
        config['experiment']['name'],
        current_fold,
      )
    ]
    wandb_logger = build_wandb_logger(config, current_fold)
    if wandb_logger is not None:
      loggers.append(wandb_logger)
    trainer = pl.Trainer(
      callbacks=callbacks,
      logger=loggers,
      **trainer_cfg,
    )
    resume_path = find_resume_checkpoint(checkpoint_dir, current_fold)
    _LOGGER.info('Fold %d fit start (resume=%s)', current_fold, resume_path)
    trainer.fit(
      module,
      datamodule=build_datamodule(config, train_ds, valid_ds),
      ckpt_path=resume_path,
    )


def cmd_sweep(config: dict) -> None:
  """Run the noise-floor study (BLUEPRINT 11.0-1) seed by seed.

  Each remaining (seed, fold) pair trains through the standard
  resume-aware ``train`` path in an ISOLATED directory + dataset slug,
  so a sweep can share a session with nothing else and can be stopped
  at any epoch without losing work. Finished runs are scored from their
  final-epoch OOF file; the state + summary persist through the
  artifact-sync protocol and the aggregate gate (mean + 2*std) is
  announced on Discord at the end of every session.

  Args:
      config: Composed experiment configuration.
  """
  nf_cfg = config.get('noise_floor', {})
  seeds = [int(seed) for seed in nf_cfg.get('seeds', [])]
  folds = [int(fold) for fold in nf_cfg.get('folds', [0])]
  if not seeds:
    raise RuntimeError('noise_floor.seeds is empty; nothing to sweep')
  client = _client(config)
  # State files live in a dedicated staging dir: ArtifactSync pushes the
  # WHOLE local_dir, and artifact_dir also holds index/labels/folds +
  # resolved yamls that must NOT bloat the noise-floor dataset.
  staging_dir = state_dir(config)
  os.makedirs(staging_dir, exist_ok=True)
  state_path = os.path.join(staging_dir, STATE_NAME)
  summary_path = os.path.join(staging_dir, SUMMARY_NAME)
  sync = ArtifactSync(
    client,
    str(nf_cfg.get('dataset_slug', '')),
    staging_dir,
    [STATE_NAME, SUMMARY_NAME],
  )
  sync.pull_if_missing()
  state = load_state(state_path)
  runs = plan_runs(state, seeds, folds)
  _LOGGER.info(
    'noise-floor sweep: %d/%d run(s) remaining (%s)',
    len(runs),
    len(seeds) * len(folds),
    ', '.join(run_key(seed, fold) for seed, fold in runs) or 'none',
  )
  budget_h = float(config['session_time_budget_h'])
  floor_h = float(nf_cfg.get('budget_floor_h', 0.5))
  started = time.time()
  notifier = notifier_from_config(config)
  experiment_name = config['experiment']['name']
  targets = list(config['data']['target_columns'])

  for seed, fold in runs:
    remaining_h = budget_h - (time.time() - started) / 3600.0
    if remaining_h <= floor_h:
      _LOGGER.warning(
        'session budget below floor (%.2f h left); stopping sweep - '
        'rerun `sweep` in a fresh session to continue',
        remaining_h,
      )
      break
    run_config = config_for_run(config, seed, fold, remaining_h)
    dump_config(
      run_config,
      os.path.join(
        staging_dir,
        f"resolved_{run_config['experiment']['name']}.yaml",
      ),
    )
    _LOGGER.info(
      'sweep run %s start (remaining budget %.2f h)',
      run_key(seed, fold),
      remaining_h,
    )
    cmd_train(run_config, fold)
    if not fold_done(run_config['paths']['checkpoint_dir'], fold):
      _LOGGER.warning(
        'run %s not finished (budget); it will resume next session',
        run_key(seed, fold),
      )
      break
    result = collect_run_result(
      os.path.join(
        run_config['paths']['oof_dir'], f'oof_fold{int(fold)}.csv'
      ),
      targets,
    )
    state['completed'][run_key(seed, fold)] = {
      'seed': seed,
      'fold': fold,
      'macro_auc': result['macro_auc'],
      'per_class': result['per_class'],
    }
    completed_entries = list(state['completed'].values())
    save_state(state, state_path)
    write_summary_csv(completed_entries, summary_path)
    sync.push()
    _LOGGER.info(
      'sweep run %s done: macro-AUC %.4f',
      run_key(seed, fold),
      result['macro_auc'],
    )
    if notifier.enabled:
      macro = result['macro_auc']
      notifier.notify(
        f'**[{experiment_name}]** sweep: {run_key(seed, fold)} '
        f'macro-AUC {macro:.4f}'
      )

  completed_entries = list(state['completed'].values())
  if not completed_entries:
    _LOGGER.info('sweep session ended with no completed runs yet')
    return
  stats = summarize(completed_entries)
  save_state(state, state_path)
  write_summary_csv(completed_entries, summary_path)
  sync.push()
  headline = format_discord(stats, seeds, folds)
  _LOGGER.info('%s', headline)
  if notifier.enabled:
    notifier.notify(f'**[{experiment_name}]** {headline}')


def _collect_fold_checkpoints(config: dict) -> list[str]:
  """List completed-fold checkpoints honoring infer.yaml's fold selection.

  Args:
      config: Composed experiment configuration.

  Returns:
      Checkpoint paths for folds carrying both ``done`` and ``last.ckpt``.
  """
  checkpoint_dir = config['paths']['checkpoint_dir']
  requested = config['infer']['folds']
  keep = {f'fold{int(f)}' for f in requested} if requested != 'all' else None
  paths = []
  for ckpt in sorted(
    glob.glob(os.path.join(checkpoint_dir, 'fold*', 'last.ckpt'))
  ):
    fold_name = os.path.basename(os.path.dirname(ckpt))
    done_marker = os.path.join(os.path.dirname(ckpt), 'done')
    if not os.path.exists(done_marker):
      continue
    if keep is not None and fold_name not in keep:
      continue
    paths.append(ckpt)
  return paths


def cmd_infer(config: dict) -> None:
  """Ensemble fold checkpoints and emit submission.csv.

  Args:
      config: Composed experiment configuration.
  """
  client = _client(config)
  checkpoint_dir = config['paths']['checkpoint_dir']
  if client is not None:
    client.pull_latest(
      config['resume']['checkpoint_dataset_slug'], checkpoint_dir
    )
  fold_paths = _collect_fold_checkpoints(config)
  assert fold_paths, 'No completed fold checkpoints found for inference'

  test_csv = pd.read_csv(config['paths']['test_csv'])
  test_index_path = config['paths']['index_parquet'].replace(
    '.parquet', '_test.parquet'
  )
  test_index = explode_sop_uids(pd.read_parquet(test_index_path))
  config['paths']['train_dicom_dir'] = config['paths']['test_dicom_dir']

  predictions = predict_studies(
    config,
    test_index,
    test_csv['StudyInstanceUID'].tolist(),
    fold_paths,
  )
  write_submission(
    predictions,
    config['infer']['submission_path'],
    expected_uids=set(test_csv['StudyInstanceUID']),
  )


def main() -> None:
  """Parse arguments and dispatch the selected command."""
  args = _parser().parse_args()
  config = compose_experiment(args.experiment, args.override or None)
  experiment_name = config['experiment']['name']
  log_cfg = config.get('logging', {})
  setup_logging(
    log_dir=config['paths'].get('log_dir', '/kaggle/working/logs'),
    stage=args.command,
    experiment_name=experiment_name,
    level=str(log_cfg.get('level', 'INFO')),
    capture_streams=bool(log_cfg.get('capture_streams', True)),
  )
  try:
    import subprocess  # pylint: disable=import-outside-toplevel

    unknown = 'unknown (not a checkout)'
    code_hash = subprocess.run(
      ['git', 'rev-parse', '--short', 'HEAD'],
      cwd=os.path.dirname(os.path.abspath(__file__)),
      capture_output=True,
      text=True,
      check=False,
    ).stdout.strip()
    print(f'code version: {code_hash or unknown}', flush=True)
  except OSError:
    print('code version: unknown', flush=True)
  dump_config(
    config,
    os.path.join(
      config['paths']['artifact_dir'],
      f'resolved_{experiment_name}.yaml',
    ),
  )
  handlers = {
    'build-index': lambda: cmd_build_index(config),
    'build-labels': lambda: cmd_build_labels(config),
    'build-folds': lambda: cmd_build_folds(config),
    'build-cache': lambda: cmd_build_cache(config),
    'selftest': lambda: cmd_selftest(config),
    'train': lambda: cmd_train(config, args.fold),
    'infer': lambda: cmd_infer(config),
    'sweep': lambda: cmd_sweep(config),
  }
  try:
    handlers[args.command]()
  except SystemExit:
    raise
  except BaseException:  # noqa: BLE001 - log then re-raise for exit code
    logging.getLogger('main').exception(
      'stage %s crashed; full traceback above', args.command
    )
    raise


if __name__ == '__main__':
  main()
