#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Regression tests for the four fixes in run_etl_in_levels.py.

Each test pins one of the bugs that used to be live in the runner:

* dry-run and failure rows never reaching the progress table,
* Ctrl-C cancelling the asyncio tasks but not the BigQuery jobs,
* ``ConfigResolver.resolve`` labelling every error as a run-config error,
* ``Any`` / ``Optional`` / ``Set`` used in annotations without being imported.

The third-party dependencies are stubbed out, so this runs without
google-cloud-bigquery, PyYAML, structlog and rich installed.
'''

from __future__ import annotations

import asyncio
import sys
import time
import types
import typing as t
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ---------- Dependency stubs ----------------------------------------------------


def _make_logger() -> t.Any:
  '''Build a discarding structlog stand-in.'''

  def _noop(*unused_args: t.Any, **unused_kwargs: t.Any) -> None:
    return None

  return types.SimpleNamespace(
    debug=_noop,
    info=_noop,
    warning=_noop,
    warn=_noop,
    error=_noop,
    critical=_noop,
    exception=_noop,
  )


def _install_stubs() -> None:
  '''Register minimal stand-ins for the third-party packages.'''
  if 'structlog' not in sys.modules:
    structlog = types.ModuleType('structlog')
    structlog.configure = lambda *a, **kw: None
    structlog.get_logger = lambda *a, **kw: _make_logger()
    structlog.make_filtering_bound_logger = lambda *a, **kw: None
    stdlib = types.ModuleType('structlog.stdlib')
    stdlib.add_log_level = lambda *a, **kw: None
    dev = types.ModuleType('structlog.dev')
    dev.ConsoleRenderer = lambda *a, **kw: None
    structlog.stdlib = stdlib  # type: ignore[attr-defined]
    structlog.dev = dev  # type: ignore[attr-defined]
    sys.modules['structlog'] = structlog
    sys.modules['structlog.stdlib'] = stdlib
    sys.modules['structlog.dev'] = dev

  if 'yaml' not in sys.modules:
    yaml_mod = types.ModuleType('yaml')
    yaml_mod.safe_load = lambda *a, **kw: {}
    sys.modules['yaml'] = yaml_mod

  if 'google.cloud.bigquery' not in sys.modules:
    google = types.ModuleType('google')
    cloud = types.ModuleType('google.cloud')
    bq = types.ModuleType('google.cloud.bigquery')

    class _Client:  # pylint: disable=too-few-public-methods
      '''Stand-in for bigquery.Client.'''

    bq.Client = _Client  # type: ignore[attr-defined]
    bq.QueryJob = object  # type: ignore[attr-defined]
    cloud.bigquery = bq  # type: ignore[attr-defined]
    google.cloud = cloud  # type: ignore[attr-defined]
    sys.modules.setdefault('google', google)
    sys.modules['google.cloud'] = cloud
    sys.modules['google.cloud.bigquery'] = bq

  if 'rich.table' not in sys.modules:
    rich = types.ModuleType('rich')
    for name in ('console', 'live', 'table'):
      setattr(rich, name, types.ModuleType(f'rich.{name}'))

    class _Table:  # pylint: disable=too-few-public-methods
      '''Stand-in for rich.table.Table.'''

      def __init__(self, title: str = '') -> None:
        self.title = title
        self.columns: list[str] = []
        self.rows: list[tuple[str, ...]] = []

      def add_column(self, name: str) -> None:
        self.columns.append(name)

      def add_row(self, *values: str) -> None:
        self.rows.append(values)

    rich.console.Console = lambda *a, **kw: None  # type: ignore[attr-defined]
    rich.live.Live = lambda *a, **kw: None  # type: ignore[attr-defined]
    rich.table.Table = _Table  # type: ignore[attr-defined]
    sys.modules.setdefault('rich', rich)
    sys.modules['rich.console'] = rich.console
    sys.modules['rich.live'] = rich.live
    sys.modules['rich.table'] = rich.table


_install_stubs()

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Stubs must exist before the module under test is imported.
import run_etl_in_levels as rtl  # noqa: E402  pylint: disable=wrong-import-position


# pylint: disable=protected-access


# ---------- Helpers -------------------------------------------------------------


def _write_sql(directory: Path, name: str = 'query.sql') -> Path:
  '''Write a placeholder-free SQL file and return its path.'''
  sql_path = directory / name
  sql_path.write_text('SELECT 1\n')
  return sql_path


def _make_runner(dry_run: bool, **cfg: t.Any) -> rtl.BQRunner:
  '''Build a runner that never talks to a real BigQuery client.'''
  return rtl.BQRunner(
    client=None,  # never dereferenced on these paths
    cfg=rtl.Config(cfg),
    global_sem=asyncio.Semaphore(10),
    dry_run=dry_run,
    universal_time=None,
  )


class FakeJob:
  '''Records whether it was cancelled, standing in for a real QueryJob.'''

  def __init__(self) -> None:
    self.cancel_count = 0

  def cancel(self) -> bool:
    self.cancel_count += 1
    return True


# ---------- Progress rows -------------------------------------------------------


def test_dry_run_reports_dry_success_row(tmp_path: Path) -> None:
  '''The DRY_SUCCESS transition must be awaited, not silently dropped.'''
  sql_path = _write_sql(tmp_path)
  progress = rtl.ProgressTracker()
  runner = _make_runner(dry_run=True)

  asyncio.run(runner._run_with_retry(sql_path, 'run_1', 0, asyncio.Semaphore(1), progress))

  states = [row['state'] for row in progress._data.values()]
  assert states == ['DRY_SUCCESS']


def test_exhausted_retries_reports_failed_row(tmp_path: Path) -> None:
  '''The FAILED transition must be awaited, not silently dropped.'''
  sql_path = _write_sql(tmp_path)
  progress = rtl.ProgressTracker()
  # max_retries=0 gives exactly one attempt; a None client raises on submit.
  runner = _make_runner(dry_run=False, max_retries=0, max_backoff_seconds=0)

  with pytest.raises(Exception):
    asyncio.run(runner._run_with_retry(sql_path, 'run_1', 0, asyncio.Semaphore(1), progress))

  states = [row['state'] for row in progress._data.values()]
  assert states == ['FAILED']


# ---------- BigQuery job cancellation ------------------------------------------


def test_cancel_all_cancels_submitted_jobs() -> None:
  '''Cancelling the asyncio task alone leaves the BigQuery job billing.

  ``cancel_all`` must reach the jobs the runner submitted and cancel them too.
  '''
  runner = _make_runner(dry_run=False)
  job = FakeJob()

  async def scenario() -> None:
    sleeper = asyncio.create_task(asyncio.sleep(100))
    runner._to_cancel.append(sleeper)
    runner._live_jobs[sleeper] = job  # the shape _run_with_retry leaves behind
    await runner.cancel_all()
    return sleeper

  sleeper = asyncio.run(scenario())

  assert job.cancel_count == 1, 'the BigQuery job must be cancelled on shutdown'
  assert sleeper.cancelled(), 'the asyncio task must be cancelled too'


def test_cancel_all_skips_jobs_that_already_finished(tmp_path: Path) -> None:
  '''A job that finished is not tracked, so shutdown never cancels it.'''
  sql_path = _write_sql(tmp_path)
  progress = rtl.ProgressTracker()
  job = FakeJob()
  runner = _make_runner(dry_run=False)

  async def scenario() -> None:
    task = asyncio.create_task(
      rtl.BQRunner._run_with_retry(runner, sql_path, 'run_1', 0, asyncio.Semaphore(1), progress)
    )
    # Simulate a completed job: registered once, then forgotten on success.
    runner._live_jobs[task] = job
    runner._forget_job(task)
    await runner.cancel_all()

  asyncio.run(scenario())

  assert job.cancel_count == 0, 'a finished job must not be cancelled'
  assert not runner._live_jobs, 'the finished job must have been forgotten'


# ---------- ConfigResolver labels -----------------------------------------------


def test_resolver_labels_errors_as_config_not_run_config() -> None:
  '''Error messages must not assume the caller passed a run config.'''
  resolver = rtl.ConfigResolver()

  with pytest.raises(rtl.UnresolvedPlaceholderError) as exc:
    resolver.resolve({'run_id': '${placeholder_01}'})

  assert 'run_config' not in str(exc.value)


def test_resolve_rejects_non_dict_with_generic_message() -> None:
  '''The type error must describe the argument, not a specific config section.'''
  with pytest.raises(ValueError, match='config must be a dict'):
    rtl.ConfigResolver().resolve(['not', 'a', 'dict'])


def test_resolve_substitutes_from_the_config_itself() -> None:
  '''A placeholder resolves against a sibling key in the same dict.'''
  resolved = rtl.ConfigResolver().resolve({'a': '${b}', 'b': 'x', 'c': '${b}_${b}'})
  assert resolved == {'a': 'x', 'b': 'x', 'c': 'x_x'}


# ---------- Annotation hygiene --------------------------------------------------


def test_wait_until_in_the_past_returns_immediately() -> None:
  '''A scheduled time already elapsed must run now, not wait a full day.'''
  past = datetime.now(timezone.utc) - timedelta(hours=1)
  started = time.monotonic()
  asyncio.run(rtl.wait_until(past))
  assert time.monotonic() - started < 1.0


def test_wait_until_target_delegates_to_the_module_helper() -> None:
  '''The runner method must delegate to the one scheduling implementation.'''
  runner = _make_runner(dry_run=True)
  past = datetime.now(timezone.utc) - timedelta(hours=1)
  started = time.monotonic()
  asyncio.run(runner._wait_until_target(past))
  assert time.monotonic() - started < 1.0


def test_typing_names_used_in_annotations_are_imported() -> None:
  '''Every name an annotation references must actually be in scope.

  With ``from __future__ import annotations`` the hints are strings, so a
  missing import only surfaces when something resolves them -- which is exactly
  what a type checker does.
  '''
  hints = t.get_type_hints(rtl.BQRunner.__init__)
  assert hints['universal_time'] is not None

  hints = t.get_type_hints(rtl.ConfigResolver._resolve_structure)
  assert hints['visited'] is not None

  hints = t.get_type_hints(rtl.ConfigResolver._validate_no_placeholders)
  assert hints['data'] is not None


def test_json_module_is_no_longer_imported() -> None:
  '''The unused ``json`` import was removed, so it must not be bound.'''
  assert not hasattr(rtl, 'json')
