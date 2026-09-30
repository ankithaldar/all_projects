#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared test fixtures."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from forecasting_ml_framework import constants
from forecasting_ml_framework.core.metadata import Metadata


@pytest.fixture(autouse=True)
def _reset_framework_logger() -> Any:
  """Restore the framework logger's level after each test.

  Several tests assert on the framework's own log records -- the operating point,
  the resolved feature list, the importance source, the degradation warnings --
  because the central observability finding is that the framework's
  diagnostics are the only place its decisions are explained. A fixture that
  quietly raised the level for "readable output" would silence exactly the
  records those tests exist to pin down, so the level is left alone and only the
  handler noise is trimmed.
  """
  import logging  # noqa: PLC0415

  logger = logging.getLogger('forecasting_ml_framework')
  saved_level, saved_propagate = logger.level, logger.propagate
  yield
  logger.setLevel(saved_level)
  logger.propagate = saved_propagate


@pytest.fixture
def project_root() -> Path:
  """Return the project root.

  Returns:
    The repository root.
  """
  return Path(__file__).resolve().parents[1]


@pytest.fixture
def conf_dir(project_root: Path) -> Path:
  """Return the base configuration directory.

  Args:
    project_root: The project root fixture.

  Returns:
    The base configuration directory.
  """
  return project_root / 'conf' / 'base'


@pytest.fixture
def scratch_conf(tmp_path: Path, conf_dir: Path) -> Path:
  """Copy the base configuration into a writable scratch directory.

  The configuration rewriter mutates documents on disk, so a test that exercises
  it must not touch the version-controlled ones.

  Args:
    tmp_path: pytest's temporary directory.
    conf_dir: The base configuration directory.

  Returns:
    The scratch configuration directory.
  """
  target = tmp_path / 'conf' / 'base'
  target.parent.mkdir(parents=True, exist_ok=True)
  shutil.copytree(conf_dir, target)
  return target


@pytest.fixture
def metadata() -> Metadata:
  """Return a small but complete contract.

  Returns:
    A ``Metadata`` instance covering every role namespace.
  """
  return Metadata(
    target_col='label',
    feature_cols=['usage_01', 'usage_02', 'tenure_days', 'plan_tier', 'region_code', 'is_new_line'],
    numerical_cols=['usage_01', 'usage_02', 'tenure_days'],
    categorical_cols=['plan_tier', 'region_code'],
    seq_cols=[],
    id_cols=['cust_id'],
    intermediate_file_path='/tmp/forecasting_ml_framework/routine_state/test',
  )


@pytest.fixture
def labelled_frame() -> pd.DataFrame:
  """Return a small labelled frame with a real class imbalance.

  Returns:
    A pandas ``DataFrame``.
  """
  rng = np.random.default_rng(42)
  size = 400
  frame = pd.DataFrame(
    {
      'cust_id': [f'c{i:04d}' for i in range(size)],
      'usage_01': rng.normal(10, 3, size).round(3),
      'usage_02': rng.normal(5, 2, size).round(3),
      'tenure_days': rng.integers(1, 900, size),
      'plan_tier': rng.choice(['A', 'B', 'C'], size),
      'region_code': rng.choice(['north', 'south'], size),
      'is_new_line': rng.integers(0, 2, size),
    }
  )
  # ~4% positive rate, which is realistic for churn and exercises the
  # precision guard on the rate calculation.
  frame['label'] = (rng.random(size) < 0.04).astype(int)
  return frame


@pytest.fixture
def minimal_trainer_kwargs() -> dict[str, Any]:
  """Return constructor kwargs for a fast, dependency-light trainer.

  Returns:
    A mapping of trainer keyword arguments.
  """
  return {
    'model_handle': 'sklearn.tree.DecisionTreeClassifier',
    'model_params': {'max_depth': 4, 'random_state': 42},
    'balance_class_weight': False,
    'eval_size': 0.25,
    'threshold_selection_method': 'mcc_curve',
    'prediction_col': 'prediction',
    'prediction_probability_col': 'probability',
  }


@pytest.fixture
def run_params() -> dict[str, Any]:
  """Return a resolved runtime-parameter mapping.

  Returns:
    A mapping shaped like a scheduler-supplied parameter set.
  """
  return {
    constants.MODEL_KEY: 'forecast_demo_retention',
    constants.RUN_DATE: '2026-09-29',
    'perf_base_date': '2026-09-30',
    'data_project': 'local-project',
    'external_project': 'local-project',
    'db_metrics_data': 'metrics',
    'db_staging_data': 'staging',
    'run_mode': 'prod',
    **constants.SLICE_TOKEN_MAP,
  }


@pytest.fixture
def tmp_object_store():
  """Yield a temporary directory usable as an object-store root.

  Yields:
    The directory path.
  """
  path = Path(tempfile.mkdtemp(prefix='forecasting-ml-'))
  try:
    yield path
  finally:
    shutil.rmtree(path, ignore_errors=True)
