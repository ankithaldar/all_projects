#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Header-only DICOM scanning to produce the series index.

Notebook 02 / ``main.py build-index`` walk every series once, reading:

* per-slice minimal tags (SOPInstanceUID, InstanceNumber, IPP, IOP) for
  geometry-based ordering (EDA confirmed oblique acquisitions and missing
  IPP on some series), and
* series-level tags from the first slice (Rows/Cols/PixelSpacing/Sex/
  PatientID/MagneticFieldStrength/ScanningSequence).

The output ``index.parquet`` is the single source of truth consumed by
training and inference; DICOMs are never parsed again afterwards.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pydicom

from knee.helpers.geometry import SliceGeometry, order_slices
from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

_SERIES_COLUMNS = [
    'study',
    'series',
    'n_slices',
    'rows',
    'cols',
    'pixel_spacing',
    'slice_thickness',
    'sex',
    'patient_id',
    'magnetic_field',
    'scanning_sequence',
    'has_geometry',
]


def scan_one_series(series_dir: str) -> dict | None:
    """Scan a single series directory into one index record.

    Args:
        series_dir: Path shaped ``<root>/<study>/<series>``.

    Returns:
        Record matching ``_SERIES_COLUMNS`` plus ordered ``sop_uids``, or
        None when the directory is unreadable or empty.
    """
    parts = os.path.normpath(series_dir).split(os.sep)
    study_uid, series_uid = parts[-2], parts[-1]
    try:
        files = sorted(
            entry.name for entry in os.scandir(series_dir)
            if entry.name.endswith('.dcm')
        )
        if not files:
            return None
        geometries: list[SliceGeometry] = []
        first_meta: pydicom.dataset.Dataset | None = None
        for name in files:
            dataset = pydicom.dcmread(
                os.path.join(series_dir, name), stop_before_pixels=True
            )
            geometries.append(SliceGeometry.from_dataset(dataset))
            if first_meta is None:
                first_meta = dataset
        sop_uids = order_slices(geometries)
        spacing = getattr(first_meta, 'PixelSpacing', None)
        pixel_spacing = float(np.mean([float(x) for x in spacing])) if spacing else -1.0
        record = {
            'study': study_uid,
            'series': series_uid,
            'n_slices': len(sop_uids),
            'rows': int(getattr(first_meta, 'Rows', 0) or 0),
            'cols': int(getattr(first_meta, 'Columns', 0) or 0),
            'pixel_spacing': pixel_spacing,
            'slice_thickness': float(getattr(first_meta, 'SliceThickness', -1.0) or -1.0),
            'sex': str(getattr(first_meta, 'PatientSex', 'Unknown') or 'Unknown'),
            'patient_id': str(getattr(first_meta, 'PatientID', '') or ''),
            'magnetic_field': str(
                getattr(first_meta, 'MagneticFieldStrength', 'Unknown') or 'Unknown'
            ),
            'scanning_sequence': str(
                getattr(first_meta, 'ScanningSequence', 'Unknown') or 'Unknown'
            ),
            'has_geometry': all(g.position is not None for g in geometries),
            'sop_uids': sop_uids,
        }
        return record
    except Exception as exc:  # noqa: BLE001 - scan must survive bad files
        _LOGGER.warning('Failed scanning %s (%s: %s)', series_dir, type(exc).__name__, exc)
        return None


def build_index(dicom_root: str, workers: int) -> pd.DataFrame:
    """Scan every series under the root into an index DataFrame.

    Args:
        dicom_root: Root directory holding ``<study>/<series>/`` trees.
        workers: Parallel processes used for scanning.

    Returns:
        Frame with one row per successfully scanned series; ``sop_uids``
        stored as JSON strings so parquet round-trips safely.

    Raises:
        RuntimeError: If no series could be scanned at all.
    """
    series_dirs = []
    for study_entry in os.scandir(dicom_root):
        if not study_entry.is_dir():
            continue
        for series_entry in os.scandir(study_entry.path):
            if series_entry.is_dir():
                series_dirs.append(series_entry.path)
    _LOGGER.info('Scanning %d series with %d workers', len(series_dirs), workers)
    records: list[dict] = []
    executor_args = {} if workers <= 1 else {'max_workers': workers}
    with ProcessPoolExecutor(**executor_args) as pool:
        for result in pool.map(scan_one_series, series_dirs, chunksize=16):
            if result is not None:
                records.append(result)
    if not records:
        raise RuntimeError(f'No readable series found under {dicom_root}')
    frame = pd.DataFrame(records, columns=_SERIES_COLUMNS + ['sop_uids'])
    frame['sop_uids'] = frame['sop_uids'].apply(lambda uids: '|'.join(uids))
    return frame


def explode_sop_uids(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore ordered SOP lists after a parquet round-trip.

    Args:
        frame: Index frame whose ``sop_uids`` column holds pipe-joined UIDs.

    Returns:
        Copy of the frame with ``sop_uids`` as Python string lists.
    """
    copied = frame.copy()
    copied['sop_uids'] = copied['sop_uids'].astype(str).str.split('|')
    return copied
