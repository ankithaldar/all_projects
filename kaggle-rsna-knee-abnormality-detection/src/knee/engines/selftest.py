#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pipeline selftest: exercise the training path end-to-end in minutes.

Purpose: fail FAST and LOUDLY before a 12 h GPU session burns on a
misconfiguration. Every previous production failure class is covered as
an isolated, non-fatal check:

1. artifacts      - index/labels/folds rehydrated (mount copy first,
                    kaggle-CLI fallback) and schema-loadable.
2. dicom_mount    - a sample of indexed series still resolves to real
                    files under the configured DICOM root (mount drift
                    detector).
3. cache_coverage - when a volume cache is mounted: manifest coverage
                    vs index + a real H5SeriesReader sample read; when
                    absent, reports live-decode mode (not a failure).
4. model_build    - the REAL experiment model instantiates (timm
                    backbone, embed-dim contracts, pooling wiring).
5. training_step  - two real optimizer steps + one validation batch
                    through pl.Trainer on a 4-study slice, checkpoint
                    written and non-empty. Uses the actual configured
                    model/loss/optimizer/scheduler and the cache-or-live
                    reader, so OOM/shape/decoder bugs surface here.
6. discord_status - informational only (notifier_from_config is loud).

Failures never abort later checks; run_selftest returns a summary of
(name, ok, detail) and cmd_selftest exits nonzero on any failure.
"""

from __future__ import annotations

import copy
import os
import tempfile
import time

import pandas as pd
import pytorch_lightning as pl

from knee.config_params.loader import instantiate
from knee.datasets.series_dataset import SeriesReader
from knee.engines.assembly import (
  TARGET_COLUMNS,
  build_datamodule,
  build_datasets,
  build_model,
)
from knee.engines.train_module import KneeModule
from knee.helpers.dicom_io import DecoderRegistry
from knee.helpers.h5_cache import (
  MANIFEST_NAME,
  H5SeriesReader,
  find_cache_roots,
  load_manifest,
)
from knee.helpers.header_scan import explode_sop_uids
from knee.helpers.utils import get_logger

_LOGGER = get_logger('selftest')

ARTIFACT_KEYS = (
  ('index_parquet', 'study'),
  ('labels_csv', 'StudyInstanceUID'),
  ('folds_csv', 'StudyInstanceUID'),
)
SAMPLE_SERIES = 8
SELFTEST_FOLD_ID = -1


def _cuda_available() -> bool:
  """GPU probe without requiring torch at call sites.

  Returns:
      True when at least one CUDA device is visible.
  """
  try:
    import torch  # pylint: disable=import-outside-toplevel

    return bool(torch.cuda.is_available())
  except (ImportError, OSError):
    return False


def _scoped_config(config: dict) -> dict:
  """Copy config with disposable checkpoint/OOF/output paths.

  The selftest must never write into real run artifacts; everything the
  trainer produces lands in a temp tree.

  Args:
      config: Composed experiment configuration.

  Returns:
      Deep copy safe to mutate for the trainer smoke run.
  """
  run_cfg = copy.deepcopy(config)
  tmp = tempfile.mkdtemp(prefix='knee_selftest_')
  run_cfg['paths']['checkpoint_dir'] = os.path.join(tmp, 'ckpts')
  run_cfg['paths']['oof_dir'] = os.path.join(tmp, 'oof')
  os.makedirs(run_cfg['paths']['checkpoint_dir'], exist_ok=True)
  os.makedirs(run_cfg['paths']['oof_dir'], exist_ok=True)
  return run_cfg


def check_artifacts(config: dict) -> tuple[bool, str]:
  """Rehydrate and schema-check index/labels/folds artifacts.

  Args:
      config: Composed experiment configuration.

  Returns:
      (ok, detail) with loaded row counts.
  """
  # Lazy import: main imports engines at module scope (cyclic); both
  # pragmas document the deliberate exception rather than silence.
  # pylint: disable-next=import-outside-toplevel,cyclic-import
  from main import restore_artifacts_from_mounts

  restored = restore_artifacts_from_mounts(config)
  index = explode_sop_uids(pd.read_parquet(config['paths']['index_parquet']))
  labels = pd.read_csv(config['paths']['labels_csv'])
  folds = pd.read_csv(config['paths']['folds_csv'])
  for frame, (path_key, key_col) in (
    (index, ARTIFACT_KEYS[0]),
    (labels, ARTIFACT_KEYS[1]),
    (folds, ARTIFACT_KEYS[2]),
  ):
    if frame.empty:
      return False, f'{path_key} is empty'
    if key_col not in frame.columns:
      return False, f'{path_key} missing key column {key_col!r}'
  missing = [c for c in TARGET_COLUMNS if c not in labels.columns]
  if missing:
    return False, f'labels missing target columns: {missing[:3]}'
  # Series-selection schema: priority rules and metadata features read
  # these columns from the index; their absence would crash training
  # hours in - exactly the failure class this stage exists to surface.
  missing_index = [
    c
    for c in (
      'series',
      'study',
      'sop_uids',
      'plane',
      'fluid_sensitive',
      'fat_suppression',
    )
    if c not in index.columns
  ]
  if missing_index:
    return False, f'index missing series-selection columns: {missing_index}'
  restored_display = restored or 'none needed'
  return True, (
    f'index={len(index):,} labels={len(labels):,} folds={len(folds):,}; '
    f'mount-restored={restored_display}'
  )


def load_index(config: dict) -> pd.DataFrame:
  """Exploded index frame or raised error, shared by later checks.

  Args:
      config: Composed experiment configuration.

  Returns:
      Index frame with sop_uids as lists.

  Raises:
      FileNotFoundError: When the parquet is absent.
  """
  path = config['paths']['index_parquet']
  if not os.path.exists(path):
    raise FileNotFoundError(path)
  return explode_sop_uids(pd.read_parquet(path))


def check_dicom_mount(config: dict, index: pd.DataFrame) -> tuple[bool, str]:
  """Verify a sample of indexed series still exist on the DICOM mount.

  Args:
      config: Composed experiment configuration.
      index: Exploded index frame.

  Returns:
      (ok, detail) naming the first missing probe file on failure.
  """
  root = config['paths']['train_dicom_dir']
  if not os.path.isdir(root):
    return False, f'DICOM root missing: {root}'
  stride = max(1, len(index) // SAMPLE_SERIES)
  checked = 0
  for _, row in index.iloc[::stride].head(SAMPLE_SERIES).iterrows():
    series_dir = os.path.join(root, str(row['study']), str(row['series']))
    sops = list(row['sop_uids'])
    for sop in (sops[0], sops[-1]):
      probe = os.path.join(series_dir, f'{sop}.dcm')
      if not os.path.exists(probe):
        return False, f'missing file {probe}'
    checked += 1
  return True, (
    f'{checked}/{SAMPLE_SERIES} sampled series resolved under {root}'
  )


def check_cache(config: dict, index: pd.DataFrame) -> tuple[bool, str]:
  """Report cache coverage and exercise one cached read when mounted.

  Args:
      config: Composed experiment configuration.
      index: Exploded index frame.

  Returns:
      (ok, detail); ok=True with 'live-decode mode' when no cache root
      is mounted (informational, not a failure).
  """
  roots = find_cache_roots(config)
  if not roots:
    return True, 'no cache root mounted; live DICOM decode will be used'
  manifest = load_manifest(roots)
  if manifest is None:
    return False, f'roots {roots} lack {MANIFEST_NAME}'
  coverage = len(manifest) / max(1, len(index))
  # Plain live reader as fallback: build_reader() would itself return an
  # H5SeriesReader when cache roots are mounted, double-wrapping.
  data = config['data']
  base_reader = SeriesReader(
    dicom_root=config['paths']['train_dicom_dir'],
    registry=DecoderRegistry(data['decode_backend_order']),
    n_slices=int(data['n_slices']),
    percentiles=tuple(data['normalize_percentiles']),
    autocrop_margin=float(data['autocrop_margin']),
  )
  reader = H5SeriesReader(
    base_reader=base_reader,
    manifest=manifest,
    n_slices=int(data['n_slices']),
  )
  hit_uid = str(manifest.iloc[0]['SeriesInstanceUID'])
  rows = index[index['series'].astype(str) == hit_uid]
  if rows.empty:
    return False, f'cached uid {hit_uid} absent from index'
  volume = reader.read(rows.iloc[0].to_dict())
  expected = int(config['data']['n_slices'])
  # Sampling returns min(n_slices, stored) rows - SeriesReader's own
  # contract for series shorter than the sampling window.
  if not 0 < volume.shape[0] <= expected:
    return False, f'cached read returned {volume.shape}, want <= {expected}'
  return True, (
    f'coverage {len(manifest):,}/{len(index):,} ({coverage:.0%}); '
    f'sample read {volume.shape} OK'
  )


def check_model_build(config: dict) -> tuple[bool, str]:
  """Instantiate the real experiment model.

  Args:
      config: Composed experiment configuration.

  Returns:
      (ok, detail) naming backbone and parameter count.
  """
  model = build_model(config)
  params_m = sum(p.numel() for p in model.parameters()) / 1e6
  init = config['model']['init_params']
  backbone = init['backbone_name']
  return True, f'{backbone}: {params_m:.1f}M params'


def check_training_step(config: dict) -> tuple[bool, str]:
  """Run two optimizer steps + one val batch + checkpoint roundtrip.

  Uses the REAL configured model/loss/optimizer/scheduler and the
  cache-or-live reader; writes only under the scoped temp paths.

  Args:
      config: Composed experiment configuration.

  Returns:
      (ok, detail) with device, step timing and checkpoint size.
  """
  run_cfg = _scoped_config(config)
  index = load_index(run_cfg)
  labels = pd.read_csv(run_cfg['paths']['labels_csv'])
  folds = pd.read_csv(run_cfg['paths']['folds_csv'])
  labels = labels.merge(
    folds[['StudyInstanceUID', 'fold']], on='StudyInstanceUID'
  )
  ids = labels['StudyInstanceUID'].astype(str).tolist()[:4]
  if len(ids) < 4:
    return False, f'only {len(ids)} labeled studies; need 4'
  train_ids, valid_ids = ids[:2], ids[2:]

  train_ds, valid_ds = build_datasets(
    run_cfg, index, labels, valid_ids, train_ids
  )
  device = 'gpu' if _cuda_available() else 'cpu'
  module = KneeModule(
    model=build_model(run_cfg),
    criterion=instantiate(run_cfg['loss']),
    optimizer_cfg=run_cfg['optimizer'],
    scheduler_cfg=run_cfg['optimizer'].get('scheduler'),
    warmup_epochs=int(run_cfg['optimizer'].get('warmup_epochs', 0)),
    backbone_lr_scale=float(run_cfg['optimizer']['backbone_lr_scale']),
    total_epochs=1,
    target_columns=TARGET_COLUMNS,
    oof_dir=run_cfg['paths']['oof_dir'],
    fold_id=SELFTEST_FOLD_ID,
  )
  started = time.time()
  # PL2.x writes nothing with enable_checkpointing=True alone; an
  # explicit ModelCheckpoint proves the save path end-to-end.
  checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=run_cfg['paths']['checkpoint_dir'],
    filename='selftest_last',
    save_top_k=1,
    save_on_train_epoch_end=True,
  )
  trainer = pl.Trainer(
    max_epochs=1,
    limit_train_batches=2,
    limit_val_batches=1,
    num_sanity_val_steps=0,
    accelerator=device,
    devices=1,
    precision=run_cfg['trainer']['init_params'].get('precision', '32-true'),
    logger=[],
    enable_checkpointing=True,
    callbacks=[checkpoint],
    deterministic=False,
    benchmark=False,
  )
  trainer.fit(module, datamodule=build_datamodule(run_cfg, train_ds, valid_ds))
  elapsed = time.time() - started
  steps = int(trainer.global_step)
  status = str(getattr(trainer.state, 'status', 'unknown'))
  if steps == 0:
    return False, (
      f'fit completed 0 optimizer steps (status={status}); '
      'train loader yielded nothing - check study id selection'
    )
  # Explicit save: the roundtrip under test is write-then-reload, not
  # PL's epoch-end bookkeeping (which proved version-conditional).
  manual_ckpt = os.path.join(
    run_cfg['paths']['checkpoint_dir'], 'selftest_manual.ckpt'
  )
  trainer.save_checkpoint(manual_ckpt)
  ckpt_dir = run_cfg['paths']['checkpoint_dir']
  if not os.path.exists(manual_ckpt):
    return False, (
      f'save_checkpoint wrote nothing after {steps} steps '
      f'(status={status}); dir={ckpt_dir}'
    )
  try:
    import torch  # pylint: disable=import-outside-toplevel

    torch.load(manual_ckpt, map_location='cpu', weights_only=False)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return False, f'checkpoint {manual_ckpt} not loadable: {exc}'
  size_mb = os.path.getsize(manual_ckpt) / (1024 * 1024)
  return True, (
    f'{steps} steps + 1 val on {device} in {elapsed:.0f}s '
    f'(status={status}); checkpoint {os.path.basename(manual_ckpt)} '
    f'({size_mb:.0f} MB) written and reloaded'
  )


def run_selftest(
  config: dict, with_training: bool = True
) -> list[tuple[str, bool, str]]:
  """Execute every check, collecting results instead of raising.

  Args:
      config: Composed experiment configuration.
      with_training: Set False for the fast artifact/mount/model subset
          (used when a caller wants plumbing checks without GPU work).

  Returns:
      List of (check_name, ok, detail) in execution order.
  """
  results: list[tuple[str, bool, str]] = []

  def run(name: str, fn) -> None:
    """Isolate one check; never let it abort the others.

    Args:
        name: Check label for the summary.
        fn: Zero-arg callable returning (ok, detail).
    """
    started = time.time()
    try:
      ok, detail = fn()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      ok, detail = False, f'{type(exc).__name__}: {exc}'
    results.append((name, ok, detail))
    mark = 'OK  ' if ok else 'FAIL'
    _LOGGER.info(
      '[%s] %-14s (%.1fs) %s', mark, name, time.time() - started, detail
    )

  run('artifacts', lambda: check_artifacts(config))
  index: pd.DataFrame | None = None
  try:
    index = load_index(config)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    _LOGGER.warning('index unavailable for mount/cache checks: %s', exc)
  if index is not None:
    run('dicom_mount', lambda: check_dicom_mount(config, index))
    run('cache_coverage', lambda: check_cache(config, index))
  run('model_build', lambda: check_model_build(config))
  if with_training:
    run('training_step', lambda: check_training_step(config))
  return results


def render_summary(results: list[tuple[str, bool, str]]) -> str:
  """Render the verdict block used for console and Discord alike.

  Args:
      results: Output of :func:`run_selftest`.

  Returns:
      Multi-line summary with per-check marks and overall verdict.
  """
  failed = [name for name, ok, _ in results if not ok]
  failed_line = ', '.join(failed)
  verdict = 'PASS' if not failed else f'FAIL ({failed_line})'
  lines = [f'selftest: {verdict}']
  for name, ok, detail in results:
    mark = 'PASS' if ok else 'FAIL'
    lines.append(f'  {mark} {name}: {detail}')
  return '\n'.join(lines)


__all__ = ['render_summary', 'run_selftest']
