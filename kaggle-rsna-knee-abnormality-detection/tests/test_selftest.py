#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the selftest preflight stage (pipeline failure detector).

Covers schema checks, mount probes, summary rendering, the quick
no-GPU subset, and a REAL 2-step training integration on synthetic
DICOMs with resnet18 (pretrained=False, no network).
"""

# Fixture dirs are hermetic; model stubs avoid timm downloads.
# pylint: disable=redefined-outer-name,protected-access

import os

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset

from knee.config_params.loader import load_experiment
from knee.engines import selftest as st
from knee.engines.assembly import TARGET_COLUMNS
from knee.helpers.header_scan import build_index, explode_sop_uids

BASE = '1.2.826.0.1.3680043.10.1.1'


def _write_synth_dicoms(root, studies=4, series_per=1, slices=8):
  """Minimal 16px MONOCHROME2 DICOMs under <study>/<series>/ dirs."""
  rng = np.random.default_rng(5)
  for s_i in range(studies):
    study_uid = f'{BASE}.1.{s_i}'
    for se_i in range(series_per):
      series_uid = f'{study_uid}.2.{se_i}'
      d = root / study_uid / series_uid
      d.mkdir(parents=True)
      for z in range(slices):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        ds = FileDataset(
            '', pydicom.Dataset(), preamble=b'\0' * 128, file_meta=meta
        )
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        ds.SOPInstanceUID = f'{series_uid}.3.{z}'
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.PatientID = f'PAT{s_i}'
        ds.PatientSex = 'M'
        ds.Modality = 'MR'
        ds.Rows = 16
        ds.Columns = 16
        px = (rng.integers(0, 4000, size=(16, 16)) + z * 50).astype(
            np.uint16
        )
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.RescaleSlope = '1.0'
        ds.RescaleIntercept = '-1024.0'
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, float(z)]
        ds.PixelSpacing = [1.0, 1.0]
        ds.InstanceNumber = z + 1
        ds.SliceThickness = 1.0
        ds.PixelData = px.tobytes()
        pydicom.dcmwrite(
            str(d / f'{ds.SOPInstanceUID}.dcm'),
            ds,
            enforce_file_format=True,
        )


@pytest.fixture()
def pipeline(tmp_path):
  """Synthetic DICOM tree + artifacts + experiment config."""
  dicom_root = tmp_path / 'dicom'
  _write_synth_dicoms(dicom_root)

  index = build_index(str(dicom_root), workers=1)
  # Mirror cmd_build_index's train_series.csv merge: the real artifact
  # always carries these (lowercase, post-rename) selection columns.
  index['plane'] = [
      'Sagittal' if i % 2 == 0 else 'Coronal' for i in range(len(index))
  ]
  index['fluid_sensitive'] = 1
  index['fat_suppression'] = 1

  artifact_dir = tmp_path / 'artifacts'
  artifact_dir.mkdir(parents=True)
  index.to_parquet(artifact_dir / 'index.parquet', index=False)

  studies = index['study'].astype(str).unique().tolist()
  labels = pd.DataFrame(
      {'StudyInstanceUID': studies,
       **{c: [0.0] * len(studies) for c in TARGET_COLUMNS}}
  )
  labels.to_csv(artifact_dir / 'labels_pseudo.csv', index=False)
  pd.DataFrame(
      {
          'StudyInstanceUID': studies,
          'fold': [i % 2 for i in range(len(studies))],
      }
  ).to_csv(artifact_dir / 'folds.csv', index=False)

  config = load_experiment('configs/experiments/smoke_ci.yaml')
  # load_experiment resolves ${paths.artifact_dir} interpolations at
  # LOAD time; derived keys must be recomputed after redirecting the
  # artifact root or they still point at /tmp/knee_smoke.
  config['paths'].update(
      artifact_dir=str(artifact_dir),
      index_parquet=str(artifact_dir / 'index.parquet'),
      labels_csv=str(artifact_dir / 'labels_pseudo.csv'),
      folds_csv=str(artifact_dir / 'folds.csv'),
      train_dicom_dir=str(dicom_root),
      volume_cache_dir=str(tmp_path / 'no_cache'),
  )
  config['paths']['checkpoint_dir'] = str(tmp_path / 'ckpts')
  config['paths']['oof_dir'] = str(tmp_path / 'oof')
  config['data']['n_slices'] = 8  # matches the synthetic series depth
  config['model']['init_params']['pretrained'] = False
  return config


def test_render_summary_pass_and_fail():
  results = [('a', True, 'fine'), ('b', False, 'boom: x')]
  text = st.render_summary(results)
  assert 'selftest: FAIL (b)' in text
  assert 'PASS a' in text and 'FAIL b: boom: x' in text
  ok_text = st.render_summary([('a', True, 'fine')])
  assert ok_text.splitlines()[0] == 'selftest: PASS'


def test_check_artifacts_schema_and_targets(pipeline):
  ok, detail = st.check_artifacts(pipeline)
  assert ok, detail
  assert 'labels=' in detail


def test_check_artifacts_missing_target_column(pipeline):
  labels = pd.read_csv(pipeline['paths']['labels_csv'])
  labels = labels.drop(columns=[TARGET_COLUMNS[0]])
  labels.to_csv(pipeline['paths']['labels_csv'], index=False)
  ok, detail = st.check_artifacts(pipeline)
  assert not ok
  assert TARGET_COLUMNS[0] in detail


def test_check_artifacts_missing_selection_columns(pipeline):
  index = pd.read_parquet(pipeline['paths']['index_parquet'])
  index = index.drop(columns=['fluid_sensitive'])
  index.to_parquet(pipeline['paths']['index_parquet'], index=False)
  ok, detail = st.check_artifacts(pipeline)
  assert not ok
  assert 'fluid_sensitive' in detail


def test_check_dicom_mount_probes_first_and_last(pipeline):
  index = explode_sop_uids(
      pd.read_parquet(pipeline['paths']['index_parquet'])
  )
  ok, detail = st.check_dicom_mount(pipeline, index)
  assert ok, detail
  # Remove one probe file -> failure names the exact path.
  row = index.iloc[0]
  probe = os.path.join(
      pipeline['paths']['train_dicom_dir'],
      str(row['study']),
      str(row['series']),
      f"{row['sop_uids'][-1]}.dcm",
  )
  os.remove(probe)
  ok, detail = st.check_dicom_mount(pipeline, index)
  assert not ok
  assert 'missing file' in detail


def test_check_cache_reports_live_mode_without_roots(pipeline):
  index = explode_sop_uids(
      pd.read_parquet(pipeline['paths']['index_parquet'])
  )
  ok, detail = st.check_cache(pipeline, index)
  assert ok, detail
  assert 'live DICOM decode' in detail


def test_check_cache_reads_mounted_manifest(pipeline, tmp_path):
  # Local import keeps the hc namespace adjacent to its usage.
  import knee.helpers.h5_cache as hc  # pylint: disable=import-outside-toplevel

  index = explode_sop_uids(
      pd.read_parquet(pipeline['paths']['index_parquet'])
  )
  cache_root = tmp_path / 'mounted_cache'
  writer = hc.ShardWriter(str(cache_root), img_size=8, gzip_level=0)
  uid = str(index.iloc[0]['series'])
  stored = np.zeros((int(index.iloc[0]['n_slices']), 8, 8), np.uint8)
  writer.add_series(uid, str(index.iloc[0]['study']), stored)
  writer.close()
  writer.write_manifest()
  pipeline['paths']['volume_cache_dir'] = str(cache_root)
  ok, detail = st.check_cache(pipeline, index)
  assert ok, detail
  assert 'coverage' in detail and 'sample read' in detail


def test_run_selftest_quick_subset_all_pass(pipeline, monkeypatch):
  monkeypatch.setattr(st, 'check_model_build', lambda cfg: (True, 'stub'))
  results = st.run_selftest(pipeline, with_training=False)
  names = [n for n, _, _ in results]
  assert names == [
      'artifacts', 'dicom_mount', 'cache_coverage', 'model_build'
  ]
  assert all(ok for _, ok, _ in results), results


def test_check_training_step_real_two_steps(pipeline):
  """REAL end-to-end: decode -> dataset -> model -> loss -> ckpt."""
  results = st.run_selftest(pipeline, with_training=True)
  by_name = {name: (ok, detail) for name, ok, detail in results}
  assert by_name['model_build'][0], by_name['model_build'][1]
  ok, detail = by_name['training_step']
  assert ok, detail
  assert 'checkpoint' in detail


if __name__ == '__main__':
  pytest.main([__file__])
