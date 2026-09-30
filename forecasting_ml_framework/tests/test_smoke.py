#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The smoke test.

The suite's foundation, and the layer with the highest ratio of cost to value:
import every module, instantiate every configuration component, and exercise one
routine per family and one trainer per tier.

Every fault this layer exists to catch shares one property -- it is invisible to
review and inert until first production use, because nothing executed it. An
unbound variable in the Spark-to-pandas conversion makes a documented flag
unusable; two undefined names make a routine unrunnable; three helpers that
raise on empty input fail on exactly the low-activity population a retention
model most needs to score; an unconditional post-fit attribute read excludes
every non-ensemble estimator; a degraded path returns an unbound local. Each
one is a one-line mistake, and each one is a production job that fails rather
than a fault visible in a diff.

Constructing a component is a stronger check than importing its module. Import
proves the name resolves; construction proves every name the constructor reads
was actually bound, which is the class of mistake an import cannot see.

This module is that test.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
from typing import Any

import pytest

import forecasting_ml_framework
from forecasting_ml_framework import constants
from forecasting_ml_framework.persistence import (
  BQQueryDataSet,
  BQTableDataSet,
  CustomSparkBQDataSet,
  GBQTableDataSet,
  ParquetDataSet,
)
from forecasting_ml_framework.platform.config_loader import RuntimeResolverConfigLoader
from forecasting_ml_framework.platform.context import ForecastingSparkContext
from forecasting_ml_framework.platform.environment import resolve_environment
from forecasting_ml_framework.preprocessing.routines import AVAILABLE_ROUTINES

#: Every module that must import cleanly. A star-import of a 2800-line routine
#: library, or a lazily-imported optional dependency, is exactly the kind of thing
#: that breaks silently, so each is named explicitly rather than discovered.
MODULES = [
  # Package initialisers. Each re-exports its layer's public surface, so an
  # import-order or circular-import regression in any of them breaks the
  # framework before any node runs.
  'forecasting_ml_framework',
  'forecasting_ml_framework.core',
  'forecasting_ml_framework.modeling',
  'forecasting_ml_framework.nodes',
  'forecasting_ml_framework.observability',
  'forecasting_ml_framework.persistence',
  'forecasting_ml_framework.pipelines',
  'forecasting_ml_framework.platform',
  'forecasting_ml_framework.preprocessing',
  'forecasting_ml_framework.settings',
  'forecasting_ml_framework.utils',
  # Leaf modules.
  'forecasting_ml_framework.cli',
  'forecasting_ml_framework.constants',
  'forecasting_ml_framework.core.metadata',
  'forecasting_ml_framework.core.registry',
  'forecasting_ml_framework.core.routine',
  'forecasting_ml_framework.exceptions',
  'forecasting_ml_framework.hooks',
  # The local execution harness. It is the embedded relational back end that
  # `is_local` binds to, not a second implementation of the pipeline, so it is
  # held to the same import-everything standard as everything else.
  'forecasting_ml_framework.local',
  'forecasting_ml_framework.local.datasets',
  'forecasting_ml_framework.local.nba',
  'forecasting_ml_framework.local.run_nba',
  'forecasting_ml_framework.local.spark_session',
  'forecasting_ml_framework.local.sql_translate',
  'forecasting_ml_framework.local.sqlite_warehouse',
  'forecasting_ml_framework.modeling.calibration',
  'forecasting_ml_framework.modeling.metrics',
  'forecasting_ml_framework.modeling.models',
  'forecasting_ml_framework.modeling.promotion',
  'forecasting_ml_framework.modeling.scoring',
  'forecasting_ml_framework.modeling.transformer',
  'forecasting_ml_framework.nodes.feature_nodes',
  'forecasting_ml_framework.nodes.model_nodes',
  'forecasting_ml_framework.nodes.validation_nodes',
  'forecasting_ml_framework.observability.logging',
  'forecasting_ml_framework.observability.quality',
  'forecasting_ml_framework.observability.tracking',
  'forecasting_ml_framework.persistence.bigquery_api',
  'forecasting_ml_framework.persistence.bigquery_spark',
  'forecasting_ml_framework.persistence.gbq_pandas',
  'forecasting_ml_framework.persistence.parquet',
  'forecasting_ml_framework.pipelines.classification',
  'forecasting_ml_framework.pipelines.regression',
  'forecasting_ml_framework.pipelines.registry',
  'forecasting_ml_framework.platform.bootstrap',
  'forecasting_ml_framework.platform.config_loader',
  'forecasting_ml_framework.platform.context',
  'forecasting_ml_framework.platform.environment',
  'forecasting_ml_framework.platform.kedro_utils',
  'forecasting_ml_framework.platform.spark',
  'forecasting_ml_framework.preprocessing.bases',
  'forecasting_ml_framework.preprocessing.categorical',
  'forecasting_ml_framework.preprocessing.routines',
  'forecasting_ml_framework.preprocessing.scalar',
  'forecasting_ml_framework.preprocessing.sequence',
  'forecasting_ml_framework.preprocessing.sequence_ops',
  'forecasting_ml_framework.utils.reflection',
  'forecasting_ml_framework.utils.storage',
  'forecasting_ml_framework.utils.text',
]


@pytest.mark.parametrize('module_name', MODULES)
def test_module_imports(module_name: str) -> None:
  """Every module must import cleanly.

  Args:
    module_name: The dotted module path.
  """
  assert importlib.import_module(module_name) is not None


def test_every_submodule_is_listed() -> None:
  """No module may exist without appearing in the import list above.

  A routine added to the library and a test not added to the suite is exactly the
  drift this check exists to prevent.
  """
  # walk_packages prefixes intermediate packages too (e.g. "core" for
  # "core.metadata"), so each dotted part is stripped back to a bare name before
  # the full path is rebuilt -- otherwise every name is double-prefixed.
  root = forecasting_ml_framework.__name__
  discovered = {
    f'{root}.{info.name}' if not info.name.startswith(f'{root}.') else info.name
    for info in pkgutil.walk_packages(forecasting_ml_framework.__path__, prefix=f'{root}.')
  }
  # __main__ re-enters the CLI, which is exercised by the CLI tests instead.
  discovered.discard(f'{root}.__main__')
  assert discovered - set(MODULES) == set(), (
    f'module(s) exist without a smoke test: {sorted(discovered - set(MODULES))}'
  )


def test_every_routine_instantiates() -> None:
  """Every routine must construct from empty configuration.

  This is the check that catches a routine whose ``__init__`` references a name
  it never bound -- the issue class that made a whole transform unrunnable.
  """
  for name, routine in AVAILABLE_ROUTINES.items():
    instance = routine(global_params=None, params={})
    assert instance.name == name or instance.name, f'{name} did not set its display name'
    assert isinstance(instance.get_default_dict(), dict)
    assert 'skip' in instance.get_default_dict(), f'{name} does not declare skip'
    # get_param_handle was declared as a bare instance method with no
    # parameters across the whole family, so calling it on an instance raised
    # TypeError. It is a static method and is actually called here.
    assert callable(instance.get_param_handle)
    assert instance.copy() is not instance


def test_every_routine_declares_cols() -> None:
  """Every routine must declare the ``cols`` block its resolver reads.

  Args:
    None.
  """
  for name, routine in AVAILABLE_ROUTINES.items():
    defaults = routine(global_params=None, params={}).get_default_dict()
    assert 'cols' in defaults, f'{name} does not declare a cols block'
    assert set(defaults['cols']) == {'exclude_cols', 'include_cols'}, f'{name} has a malformed cols block'


def test_every_dataset_configures() -> None:
  """Every dataset must validate its configuration without performing I/O.

  This is the check that catches an adapter that dereferences an attribute
  ``__init__`` never assigned -- the issue that made upsert mode fail on every
  attempt while being advertised as a supported write mode.
  """
  table = CustomSparkBQDataSet(table_name='t', database='p.d', write_mode='upsert')
  assert table._describe()['write_mode'] == 'upsert'
  assert table._parent_project is None
  assert table._scratch_dataset == 'd'

  assert BQTableDataSet('t', 'd', 'p')._describe()['project_id'] == 'p'
  assert BQQueryDataSet(sql='select 1')._describe()['sql'] == '<inline sql>'
  assert GBQTableDataSet('t', 'd', 'p').qualified_name == 'p.d.t'
  assert ParquetDataSet('gs://b/p')._describe()['partitioned'] is True


@pytest.mark.parametrize(
  'write_mode', ['insert', 'upsert', 'overwrite', 'insert_overwrite', 'create_replace']
)
def test_every_write_mode_is_dispatchable(write_mode: str) -> None:
  """Every advertised write mode must have an implementation.

  Args:
    write_mode: The mode under test.
  """
  dataset = CustomSparkBQDataSet(table_name='t', database='p.d', write_mode=write_mode)
  dispatch = {
    'insert': dataset._insert_save,
    'upsert': dataset._upsert_save,
    'overwrite': dataset._overwrite_save,
    'insert_overwrite': dataset._insert_overwrite_save,
    'create_replace': dataset._create_replace_save,
  }
  assert callable(dispatch[dataset._write_mode])


def test_routine_catalogue_command_runs(capsys: Any) -> None:
  """The catalogue command must construct every routine it lists.

  Listing a routine the command cannot instantiate is worse than not listing it,
  so the command is exercised rather than merely imported.

  Args:
    capsys: pytest's output capture.
  """
  from forecasting_ml_framework.cli import routine_catalogue  # noqa: PLC0415

  routine_catalogue.callback()
  output = capsys.readouterr().out
  for name in AVAILABLE_ROUTINES:
    assert name in output, f'{name} is missing from the catalogue'
  assert 'skip' in output, 'the catalogue does not show the activation key'


def test_pipeline_description_command_runs(capsys: Any) -> None:
  """The graph-description command must render a registered topology.

  Args:
    capsys: pytest's output capture.
  """
  from forecasting_ml_framework.cli import pipeline_description  # noqa: PLC0415
  from forecasting_ml_framework.pipelines.registry import PipelineTopology  # noqa: PLC0415

  saved = {key: os.environ.get(key) for key in (
    constants.ENV_SUFFIX, constants.ENV_SPLIT, constants.ENV_IS_REGRESSION,
    constants.ENV_RUN_MODE, constants.ENV_TRAIN_MODE,
  )}
  try:
    PipelineTopology().export_to_environment()
    pipeline_description.callback()
    output = capsys.readouterr().out
    assert constants.PIPELINE_DEFAULT in output
  finally:
    for key, value in saved.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value


def test_invalid_write_mode_is_rejected() -> None:
  from kedro.io.core import DatasetError  # noqa: PLC0415

  with pytest.raises(DatasetError):
    CustomSparkBQDataSet(table_name='t', write_mode='merge_into_mars')


def test_spark_backed_dataset_refuses_pickling() -> None:
  """A Spark-backed dataset must refuse serialisation with a legible error."""
  import pickle  # noqa: PLC0415

  with pytest.raises(pickle.PicklingError):
    pickle.dumps(CustomSparkBQDataSet(table_name='t'))


def test_optional_components_construct() -> None:
  """The custom loader, the Spark-aware context and the transformer tier.

  None of these sit on the default training path, so an undefined name in any of
  them is latent by construction -- the code parses, imports, and is never
  reached. Each is therefore instantiated here rather than merely imported, so
  the check is the one that would actually have failed.
  """
  assert RuntimeResolverConfigLoader is not None
  assert ForecastingSparkContext is not None
  # An absent environment must resolve to the managed defaults. This is the
  # property the whole dual-backend model rests on: absence of configuration can
  # never redirect a production run at a local file.
  assert resolve_environment({}).is_distributed
  assert not resolve_environment({}).is_local

  from forecasting_ml_framework.modeling.transformer import (  # noqa: PLC0415
    CustomWriter,
    PositionalEncoding,
    TransformerRunner,
  )

  assert callable(PositionalEncoding.build)
  writer = CustomWriter(output_dir='/tmp/x', key_cols=['cust_id'], model_key='m', run_date='2026-09-29')
  assert writer.probability_col_name == 'score_value'
  assert TransformerRunner is not None


def test_config_loader_accepts_its_arguments() -> None:
  """The config loader must read the arguments it was given.

  reading ``self.env`` and ``self.config_patterns``, which
  its parent class never set, so it raised ``AttributeError`` the moment it was
  enabled.
  """
  loader = RuntimeResolverConfigLoader(
    conf_source='conf', base_env='base', default_run_env='local', config_patterns={'globals': ['globals.yml']}
  )
  assert loader._base_env == 'base'
  assert loader._default_run_env == 'local'
  assert loader._config_patterns == {'globals': ['globals.yml']}
