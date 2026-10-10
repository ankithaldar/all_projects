#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''BigQuery parallel runner.

A single-file, fully type-hinted Python 3.10+ utility that orchestrates
multi-level, parallel BigQuery runs with per-run configuration deep-merging,
placeholder validation, retry logic, and a live TUI progress table.

Usage:
  python run_etl_in_levels.py --all_configs config.yml [--dry_run]
      [--max-concurrency N] [--time_set ISO8601]
'''

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import re
import signal
import sys
import time
import typing as t
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Set

import structlog
import yaml
from google.cloud import bigquery
from google.cloud.bigquery import QueryJob
from rich.console import Console
from rich.live import Live
from rich.table import Table

SQL_DEBUG = False


# ---------- Types -----------------------------------------------------------------

JsonType = t.Union[str, int, float, bool, None, t.Dict[str, 'JsonType'], t.List['JsonType']]
ConfigDict = t.Dict[str, JsonType]

# ---------- Logging --------------------------------------------------------------

structlog.configure(
  wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
  processors=[
    structlog.stdlib.add_log_level,
    structlog.dev.ConsoleRenderer(colors=True),
  ],
)
logger = structlog.get_logger()

# ---------- Utilities ---------------------------------------------------------

class ConfigurationError(Exception):
  '''Invalid configuration provided.'''
  pass


def parse_time(time_str: str) -> datetime:
  '''
  Parse ISO 8601 timestamp for universal override.

  Args:
    time_str: ISO 8601 string or None

  Returns:
    Parsed datetime or None

  Raises:
    ConfigurationError: If format is invalid
  '''
  if not time_str:
    return None

  try:
    # Handle Z suffix
    time_str = time_str.replace('Z', '+00:00')
    dt = datetime.fromisoformat(time_str)
    # Ensure timezone aware
    if dt.tzinfo is None:
      dt = dt.replace(tzinfo=timezone.utc)
    return dt
  except ValueError as e:
    raise ConfigurationError(f'Invalid ISO 8601 timestamp: {e}') from e



async def wait_until(target_time: datetime):
  '''Sleep until the wall clock reaches target_time, in interruptible chunks.

  A target time already in the past returns immediately with a warning rather
  than waiting a full day, and the wait is split into short sleeps so a signal
  can interrupt it.

  Args:
    target_time: When to resume. Naive datetimes are treated as UTC.
  '''
  now = datetime.now(timezone.utc)
  if target_time.tzinfo is None:
    target_time = target_time.replace(tzinfo=timezone.utc)

  # If the time has passed today, wait is skipped (or could be interpreted as next day)
  # Logic: strict scheduling for toda. If passed, warn and run immediately
  if target_time < now:
    logger.warn('Scheduled time is in the past. Running immediately.')
    return

  delta = target_time - now
  remaining = str(delta).split('.', maxsplit=1)[0]
  logger.info(f'Waiting for {remaining} until {target_time}... ', trigger_time=str(target_time))
  # Sleep in chunks to allow responsiveness to signals
  try:
    chunk_size = 5.0
    wait_seconds = delta.total_seconds()
    while wait_seconds > 0:
      sleep_time = min(chunk_size, wait_seconds)
      await asyncio.sleep(sleep_time)
      wait_seconds -= sleep_time
  except asyncio.CancelledError:
    logger.info('Scheduled wait cancelled')
    raise


# ---------- Configuration ---------------------------------------------------------


class Config:
  '''Thin dict-like wrapper around a merged configuration.'''

  def __init__(self, cfg: ConfigDict) -> None:
    '''Wrap an already-merged configuration dict.

    Args:
      cfg: The merged base_config + run_config mapping.
    '''
    self._cfg = cfg

  def get(self, key: str, default: t.Any = None) -> t.Any:
    '''Return `key` from the config, or `default` when absent.

    Args:
      key: Config key to look up.
      default: Value returned when `key` is not present.

    Returns:
      The configured value, or `default`.
    '''
    return self._cfg.get(key, default)

  def __getitem__(self, key: str) -> t.Any:
    '''Return `key` from the config, raising KeyError when absent.'''
    return self._cfg[key]

  def __setitem__(self, key: str, value: t.Any) -> None:
    '''Set `key` to `value` in the underlying config dict.'''
    self._cfg[key] = value

  def __contains__(self, key: str) -> bool:
    '''Return True when `key` is present in the config.'''
    return key in self._cfg


def deep_merge(base: t.Dict[str, t.Any], override: JsonType) -> JsonType:
  '''Recursively merge `override` into `base` and return the result.

  Dicts merge key by key, lists are concatenated, and an explicit `None`
  override leaves the base value untouched. Neither input is mutated.

  Args:
    base: The lower-precedence mapping, usually `base_config`.
    override: The higher-precedence values, usually one run config.

  Returns:
    The merged structure.
  '''
  if isinstance(base, dict) and isinstance(override, dict):
    merged: dict[str, JsonType] = base.copy()
    for k, v in override.items():
      merged[k] = deep_merge(base.get(k), v)
    return merged
  if isinstance(base, list) and isinstance(override, list):
    return base + override
  return override if override is not None else base


def load_config(path: Path) -> ConfigDict:
  '''Read the YAML config file at `path`.

  Args:
    path: Path to the YAML file holding base_config, run_configs and
      sql_level_maps.

  Returns:
    The parsed config mapping.

  Exits:
    With status 1 when the file cannot be read or is not valid YAML.
  '''
  try:
    with path.open() as fh:
      return t.cast(ConfigDict, yaml.safe_load(fh))
  except (OSError, yaml.YAMLError) as exc:
    logger.error('Failed to load base config', path=str(path), exc=exc)
    sys.exit(1)


class UnresolvedPlaceholderError(Exception):
  '''Raised when a placeholder cannot be resolved from available context.'''
  pass


class CircularReferenceError(Exception):
  '''Raised when circular dependencies are detected during resolution.'''
  pass


class ConfigResolver:
  '''Resolves ``${var}`` placeholders against the config itself.

  The config dict doubles as its own resolution context: a placeholder is
  looked up as a direct key in the same structure being resolved, so
  ``a: '${b}'`` and ``b: x`` resolves to ``a: x``.
  '''

  def __init__(self, placeholder_pattern: str = r'\$\{([^}]+)\}'):
    '''Build a resolver for one placeholder syntax.

    Args:
      placeholder_pattern: Regex whose group 1 captures the variable name.
        Defaults to the `${var}` syntax used in the config files.
    '''
    self.placeholder_pattern = placeholder_pattern
    self._placeholder_regex = re.compile(placeholder_pattern)

  def resolve(self, config: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    '''Resolve every placeholder in `config` and return the resolved copy.

    The input is never mutated. `config` may be any dict -- a base config, a
    run config, or a merged one -- the labels in the error messages follow
    suit rather than assuming a particular caller.
    '''
    # Validate input structure
    if not isinstance(config, dict):
      raise ValueError('config must be a dict')

    # Create deep copies to avoid mutating original inputs
    resolved = copy.deepcopy(config)

    # PHASE 1: SCAN entire structure for placeholders
    if not self._scan_for_placeholders(resolved):
      return resolved

    # PHASE 2: RESOLVE the config against itself (self-contained resolution)
    resolved = self._resolve_structure(
        resolved,
        context=resolved,
        path='config'
    )

    # PHASE 4: FINAL VALIDATION - ensure no unresolved placeholders remain
    self._validate_no_placeholders(resolved, 'config')
    return resolved


  def _scan_for_placeholders(self, data: t.Any) -> bool:
    '''
    Recursively scan data structure for placeholder patterns.
    Returns:
      True if any placeholder pattern is found, False otherwise
    '''
    if isinstance(data, str):
      return bool(self._placeholder_regex.search(data))
    elif isinstance(data, dict):
      return any(self._scan_for_placeholders(v) for v in data.values())
    elif isinstance(data, list):
      return any(self._scan_for_placeholders(item) for item in data)
    return False

  def _validate_no_placeholders(self, data: Any, path: str) -> None:
    '''
    Verify no unresolved placeholders remain after resolution.

    Raises:
      UnresolvedPlaceholderError: With precise location of failure
    '''
    if isinstance(data, str):
      match = self._placeholder_regex.search(data)
      if match:
        var_name = match.group(1)
        raise UnresolvedPlaceholderError(
          f"Unresolved placeholder '${{{var_name}}}' at {path} "
          f"in value: {data!r}"
        )
    elif isinstance(data, dict):
      for key, value in data.items():
        self._validate_no_placeholders(value, f"{path}.{key}")
    elif isinstance(data, list):
      for idx, item in enumerate(data):
        self._validate_no_placeholders(item, f"{path}[{idx}]")

  def _resolve_structure(
    self,
    data: t.Any,
    context: t.Dict[str, t.Any],
    path: str,
    visited: Optional[Set[int]] = None,
    iteration: int = 0,
    max_iterations: int = 10
    ) -> t.Any:
    '''
    Recursively resolve placeholders in nested structures.

    Returns:
      Resolved data structure with same type as input

    Raises:
      UnresolvedPlaceholderError: For missing variables
      CircularReferenceError: For detected circular dependencies
    '''
    # Initialize visited set for circular structure detection (not placeholder cycles)
    if visited is None:
      visited = set()

    # Handle circular data structures (e.g., dict containing itself)
    if id(data) in visited:
      return data  # Return as-is to avoid infinite recursion

    visited.add(id(data))


    # STRING: Resolve placeholders if present
    if isinstance(data, str):
      current_value = data
      prev_value = None

      while iteration < max_iterations:
        # Extract all unique placeholder variables in current value
        placeholders = set(self._placeholder_regex.findall(current_value))

        # Termination condition: no placeholders remain
        if not placeholders:
          return current_value

        # Termination condition: value stabilized with unresolved placeholders
        if current_value == prev_value:
          unresolved = self._placeholder_regex.findall(current_value)
          raise UnresolvedPlaceholderError(
            f"Unresolved placeholders {unresolved} at {path} "
            f"after {iteration} iterations. Available context keys: "
            f"{sorted(k for k in context if isinstance(k, str))}"
          )

        prev_value = current_value

        # Attempt substitution for each placeholder
        for var_name in placeholders:
          # Context lookup: must be direct key match (flat namespace)
          if var_name not in context:
            raise UnresolvedPlaceholderError(
              f"Placeholder '${{{var_name}}}' at {path} "
              f"has no resolution in context. Available keys: "
              f"{sorted(k for k in context if isinstance(k, str))}"
            )

          replacement = context[var_name]

          # Type safety: only allow string replacements in strings
          if not isinstance(replacement, str):
            raise ValueError(
              f'Cannot substitute non-string value {replacement!r} '
              f'(type: {type(replacement).__name__}) for placeholder '
              f"'${{{var_name}}}' at {path}. Only string values allowed "
              'in placeholder substitution.'
            )

          # Perform ALL occurrences substitution in one pass
          current_value = current_value.replace(f'${{{var_name}}}', replacement)

        iteration += 1

      # Max iterations exceeded - likely circular dependency
      raise CircularReferenceError(
        f"Circular reference detected at {path} during placeholder resolution. "
        f"Value after {max_iterations} iterations: {current_value!r}"
      )

    # DICT: Recurse into values (preserve keys exactly)
    elif isinstance(data, dict):
      resolved_dict = {}
      for key, value in data.items():
        # Keys are NEVER resolved - preserve original exactly
        resolved_value = self._resolve_structure(
          value,
          context,
          f"{path}.{key}",
          visited.copy(),  # Copy to isolate recursion paths
          iteration=0  # Reset iteration counter per value
        )
        resolved_dict[key] = resolved_value
      return resolved_dict

    # LIST: Recurse into elements
    elif isinstance(data, list):
      resolved_list = []
      for idx, item in enumerate(data):
        resolved_item = self._resolve_structure(
          item,
          context,
          f"{path}[{idx}]",
          visited.copy(),
          iteration=0
        )
        resolved_list.append(resolved_item)
      return resolved_list

    # OTHER TYPES: Return unchanged (no placeholders possible)
    else:
      return data



# ---------- Placeholder Validation ------------------------------------------------


def collect_placeholders(sql: str) -> set[str]:
  '''Return the set of {var} placeholder names found in `sql`.

  Args:
    sql: SQL text possibly containing {var} placeholders.

  Returns:
    The distinct placeholder names, without braces.
  '''
  return set(re.findall(r'\{(\w+)\}', sql))


def validate_placeholders(sql_files: list[Path], cfg: Config) -> None:
  '''Check every placeholdered SQL file against the config, up front.

  Reading all files before running anything means a config mistake costs no
  BigQuery slots.

  Args:
    sql_files: The SQL files referenced by the level map.
    cfg: The merged config a run will execute with.

  Exits:
    With status 1 when any SQL file is unreadable or references a placeholder
    the config does not define.
  '''
  missing: dict[str, set[str]] = {}
  for sql_path in sql_files:
    try:
      sql = sql_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
      logger.error('Cannot read SQL file', path=str(sql_path), exc=exc)
      sys.exit(1)
    placeholders = collect_placeholders(sql)
    bad = {p for p in placeholders if p not in cfg}
    if bad:
      missing[str(sql_path)] = bad
  if missing:
    for path, bad in missing.items():
      logger.error('Missing placeholders', sql_path=path, placeholders=sorted(bad))
    sys.exit(1)


# ---------- BigQuery Execution ----------------------------------------------------


class BQRunner:
  '''Submits, tracks and retries BigQuery jobs for a single run config.'''

  def __init__(
    self,
    client: bigquery.Client,
    cfg: Config,
    global_sem: asyncio.Semaphore,
    dry_run: bool,
    universal_time: Optional[datetime]
  ) -> None:
    '''Build a runner for one run config.

    Args:
      client: Shared BigQuery client, used from worker threads.
      cfg: The merged config for this run.
      global_sem: Semaphore shared by every runner, capping total concurrency.
      dry_run: When True, submit nothing and report DRY_SUCCESS.
      universal_time: Overrides every run's scheduled_time when set.
    '''
    self.client = client
    self.cfg = cfg
    self.global_sem = global_sem
    self.dry_run = dry_run
    self._to_cancel: list[asyncio.Task[t.Any]] = []
    self.universal_time = universal_time
    # Jobs submitted but not yet finished, keyed by the task that owns them, so
    # shutdown can cancel the BigQuery side and not just the local asyncio side.
    self._live_jobs: dict[t.Optional[asyncio.Task[t.Any]], QueryJob] = {}

  def _interpolate_sql(self, sql: str) -> str:
    '''Substitute {var} placeholders in `sql` with values from the config.

    Args:
      sql: Raw SQL text, possibly containing {var} placeholders.

    Returns:
      The SQL with every placeholder replaced by its config value.

    Raises:
      KeyError: If a placeholder has no corresponding key in the config.
    '''
    placeholders = collect_placeholders(sql)
    ctx = {k: self.cfg[k] for k in placeholders}
    return sql.format(**ctx)

  async def _run_with_retry(
    self,
    sql_path: Path,
    run_id: str,
    level: int,
    semaphore: asyncio.Semaphore,
    progress: 'ProgressTracker',
  ) -> None:
    '''Run one SQL file to completion, retrying with exponential backoff.

    Waits for the run's scheduled time, submits the job through the blocking
    BigQuery client on a worker thread, and retries up to `max_retries` times
    with `min(2 ** attempt, max_backoff_seconds)` seconds between attempts.

    Args:
      sql_path: Path to the .sql file to execute.
      run_id: Identifier of the run config this job belongs to.
      level: Dependency level this job is executing in.
      semaphore: Per-level concurrency limit.
      progress: Tracker to publish job state to.

    Raises:
      Exception: The last failure, once retries are exhausted.
    '''
    max_retries: int = int(self.cfg.get('max_retries', 20))
    max_backoff: int = int(self.cfg.get('max_backoff_seconds', 600))
    sql = sql_path.read_text()
    sql = self._interpolate_sql(sql)

    target_time = self.universal_time if self.universal_time else self.cfg.get('scheduled_time', None)
    if target_time:
      await self._wait_until_target(target_time)

    if SQL_DEBUG:
      sql_dump_path = sql_path.parent / self.cfg['date_string'] / sql_path.name
      sql_dump_path.parent.mkdir(parents=True, exist_ok=True)
      sql_dump_path.write_text(sql)

    async with semaphore, self.global_sem:
      attempt = 0
      task = asyncio.current_task()
      while True:
        job_id = f'bq_parallel_run__{run_id}__{sql_path.stem}__{level}__{int(time.time()*1000)}'
        try:
          logger.debug('Submitting job', job_id=job_id, sql=str(sql_path))
          await progress.update(run_id, str(sql_path), job_id, 'PENDING')
          if self.dry_run:
            await asyncio.sleep(0.1)
            await progress.update(run_id, str(sql_path), job_id, 'DRY_SUCCESS')
            return
          job: QueryJob = await asyncio.to_thread(
            self.client.query,
            sql,
            job_id=job_id,
            project=self.cfg.get('billing_project'),
            # labels=self.cfg.get("labels", {}),
          )
          # Register before waiting so a cancel arriving mid-wait still finds
          # the job to cancel.
          self._live_jobs[task] = job
          await progress.update(
            run_id,
            str(sql_path),
            job_id,
            'RUNNING',
            job=job,
          )
          await asyncio.to_thread(job.result)
          self._forget_job(task)
          await progress.update(
            run_id,
            str(sql_path),
            job_id,
            'SUCCESS',
            job=job,
          )
          return
        # Retry on any failure: transient BigQuery, network and quota errors all
        # arrive as different exception types and the retry loop is the only
        # place that knows how to back off.
        except Exception as exc:  # pylint: disable=broad-exception-caught
          logger.warning(
            'Job failed',
            job_id=job_id,
            attempt=attempt + 1,
            exc=exc,
          )
          if attempt >= max_retries:
            self._forget_job(task)
            await progress.update(run_id, str(sql_path), job_id, 'FAILED')
            raise
          backoff = min(2**attempt, max_backoff)
          await asyncio.sleep(backoff)
          attempt += 1

  def _forget_job(self, task: t.Optional[asyncio.Task[t.Any]]) -> None:
    '''Stop tracking a job once it has reached a terminal state.'''
    self._live_jobs.pop(task, None)

  def create_task(
    self,
    sql_path: Path,
    run_id: str,
    level: int,
    semaphore: asyncio.Semaphore,
    progress: 'ProgressTracker',
  ) -> asyncio.Task[t.Any]:
    '''Spawn the cancellable asyncio task for one SQL file.

    Args:
      sql_path: Path to the .sql file to execute.
      run_id: Identifier of the run config this job belongs to.
      level: Dependency level this job is executing in.
      semaphore: Per-level concurrency limit.
      progress: Tracker to publish job state to.

    Returns:
      The spawned task.
    '''
    task = asyncio.create_task(
      self._run_with_retry(sql_path, run_id, level, semaphore, progress),
      name=f'{run_id}__{sql_path.stem}',
    )
    self._to_cancel.append(task)
    return task

  async def cancel_all(self) -> None:
    '''Cancel every pending task and the BigQuery jobs they submitted.

    Cancelling an asyncio task only unwinds the local wait; a job already
    submitted to BigQuery keeps running and keeps billing, so each live job is
    cancelled too. Jobs that reached a terminal state are skipped by the
    `Job.cancel` call being harmless on them.
    '''
    live_jobs = list(self._live_jobs.values())
    for task in self._to_cancel:
      if not task.done():
        task.cancel()
    await asyncio.gather(
      *(asyncio.to_thread(job.cancel) for job in live_jobs),
      return_exceptions=True,
    )
    await asyncio.gather(*self._to_cancel, return_exceptions=True)

  async def _wait_until_target(self, target_time: datetime):
    '''Sleep until the wall clock reaches `target_time`, in interruptible chunks.

    Thin wrapper over the module-level ``wait_until`` so the scheduling
    behaviour -- past times run immediately, the wait is chunked so a signal
    can interrupt it -- lives in exactly one place.

    Args:
      target_time: When to resume. A time already in the past returns
        immediately after a warning rather than waiting a full day.
    '''
    await wait_until(target_time)


# ---------- Progress TUI ----------------------------------------------------------


class ProgressTracker:
  '''Coroutine-safe store of job state, and the table that renders it.'''

  def __init__(self) -> None:
    '''Create an empty tracker with its own asyncio lock.'''
    self._lock = asyncio.Lock()
    self._data: dict[
      tuple[str, str],
      dict[str, t.Any],
    ] = {}  # (run_id, sql_path) -> row

  async def update(
    self,
    run_id: str,
    sql_path: str,
    job_id: str,
    state: str,
    job: QueryJob | None = None,
  ) -> None:
    '''Record a state transition for one job, under the lock.

    Rows are created on first sight of a (run_id, sql_path) pair, so every
    later transition updates the same row. Cost fields are only refreshed when
    a finished `job` is supplied.

    Args:
      run_id: Identifier of the run config the job belongs to.
      sql_path: Path to the .sql file being executed.
      job_id: BigQuery job_id currently being reported.
      state: New state, e.g. PENDING, RUNNING, SUCCESS or FAILED.
      job: Optional finished job to read byte and slot figures from.
    '''
    async with self._lock:
      key = (run_id, sql_path)
      row = self._data.setdefault(
        key,
        {
          'run_id': run_id,
          'sql_path': sql_path,
          'job_id': job_id,
          'state': state,
          'bytes': 0,
          'slot_ms': 0,
          'runtime_s': 0,
        },
      )
      row.update(state=state, job_id=job_id)
      if job and job.done():
        row['bytes'] = job.total_bytes_processed or 0
        row['slot_ms'] = job.slot_millis or 0
        row['runtime_s'] = (job.ended - job.started).total_seconds() if job.ended and job.started else 0

  def render(self) -> Table:
    '''Build a rich Table of every tracked job, sorted by file then run id.'''
    table = Table(title='BigQuery Parallel Run')
    for col in ['Level', 'Run-ID', 'SQL File', 'Job ID', 'State', 'Bytes', 'Slot ms', 'Runtime s']:
      table.add_column(col)
    for row in sorted(self._data.values(), key=lambda r: (r['sql_path'], r['run_id'])):
      runtime_s = row['runtime_s']
      table.add_row(
        str(row.get('level', '')),
        row['run_id'],
        Path(row['sql_path']).name,
        row['job_id'],
        row['state'],
        str(row['bytes']),
        str(row['slot_ms']),
        f'{runtime_s:.2f}',
      )
    return table


# ---------- Orchestration ---------------------------------------------------------


class Orchestrator:
  '''Runs jobs level by level, gating each level on the previous one's success.'''

  def __init__(
    self,
    project_id: str,
    run_configs: list[tuple[str, ConfigDict]],
    level_map: dict[int, list[str]],
    dry_run: bool,
    max_concurrency: int,
    universal_time: Optional[datetime]
  ) -> None:
    '''Create the orchestrator for one batch of runs.

    Args:
      project_id: Project the BigQuery client runs as.
      run_configs: One merged config per parallel run.
      level_map: Dependency level -> SQL files for that level.
      dry_run: When True, validate and simulate instead of submitting.
      max_concurrency: Job cap applied per level.
      universal_time: When set, overrides every run's scheduled_time.
    '''
    self.run_configs = run_configs
    self.level_map = level_map
    self.dry_run = dry_run
    self.max_concurrency = max_concurrency
    self.client = bigquery.Client(project=project_id)
    self.global_sem = asyncio.Semaphore(10)  # hard global cap
    self.progress = ProgressTracker()
    self.console = Console()
    self._shutting_down = False
    self.universal_time = universal_time
    self.bq_runners: list[BQRunner] = []

  async def run(self) -> bool:
    '''Run every level in order and report overall success.

    Returns:
      True if every level completed with no failed job, False otherwise.
    '''
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
      loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))

    all_sql_files = {Path(p) for lst in self.level_map.values() for p in lst}
    for run_config in self.run_configs:
      validate_placeholders(all_sql_files, Config(run_config))

    with Live(self.progress.render(), refresh_per_second=1, console=self.console) as live:
      self.live = live
      for level in sorted(self.level_map.keys()):
        logger.info('Starting level %s', level)
        if self._shutting_down:
          break
        ok = await self._run_level(level)
        if not ok:
          return False
      return True

  async def _run_level(self, level: int) -> bool:
    '''Run every job in one level and report success.

    Args:
      level: The dependency level to execute.

    Returns:
      True if no job in the level raised, False if any did.
    '''
    sql_files = [Path(p) for p in self.level_map[level]]
    semaphore = asyncio.Semaphore(self.max_concurrency)
    tasks: list[asyncio.Task[t.Any]] = []

    for run_override in self.run_configs:
      cfg = Config(run_override)

      # Run each of the run configs at the exact time set
      if cfg.get('scheduled_time'):
        cfg['scheduled_time'] = parse_time(cfg['scheduled_time'])

      runner = BQRunner(self.client, cfg, self.global_sem, self.dry_run, self.universal_time)
      # Retained so _shutdown can reach the runner's live BigQuery jobs.
      self.bq_runners.append(runner)

      for sql_path in sql_files:
        task = runner.create_task(
          sql_path,
          run_override.get('run_id'),
          level,
          semaphore,
          self.progress,
        )
        tasks.append(task)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failed = [r for r in results if isinstance(r, Exception)]
    for exc in failed:
      logger.error('Level task failed', level=level, exc=exc)
    return not failed

  async def _shutdown(self) -> None:
    '''Cancel every pending task and every BigQuery job they submitted, then exit.

    Delegates to each runner because only the runner knows which jobs it
    submitted; cancelling the asyncio tasks alone would leave those jobs
    running and billing.
    '''
    if self._shutting_down:
      return
    self._shutting_down = True
    logger.warning('Shutting down gracefully...')
    for runner in self.bq_runners:
      await runner.cancel_all()
    sys.exit(1)


# ---------- CLI -------------------------------------------------------------------


def parse_cli() -> argparse.Namespace:
  '''Parse the command line arguments for the runner.

  Returns:
    The parsed namespace holding all_configs, dry_run, max_concurrency and
    time_set.
  '''
  parser = argparse.ArgumentParser(description='Parallel BigQuery runner with per-run configs')
  parser.add_argument('--all_configs', required=True, type=Path, help='Path to base_config.yaml')
  parser.add_argument('--dry_run', action='store_true', help='Validate only, do not execute')
  parser.add_argument('--max-concurrency', type=int, help='Override max concurrency per level')
  parser.add_argument('--time_set', type=parse_time, default=None, help='Wait until specific time to start execution')
  return parser.parse_args()


async def main() -> None:
  '''Load the config, build the orchestrator and run every level.

  Exits:
    With status 0 when every level succeeded, 1 on validation failure or any
    failed job.
  '''
  args = parse_cli()
  all_configs = load_config(args.all_configs)

  base_cfg = all_configs['base_config']
  run_cfgs = all_configs['run_configs']
  level_map = all_configs['sql_level_maps']
  max_concurrency = len(run_cfgs) or args.max_concurrency or int(base_cfg.get('max_concurrency', 3))

  resolver = ConfigResolver()

  orch = Orchestrator(
    base_cfg['billing_project'],
    [resolver.resolve(deep_merge(base_cfg, cfg)) for cfg in run_cfgs],
    level_map,
    dry_run=args.dry_run,
    max_concurrency=max_concurrency,
    universal_time=args.time_set or None
  )

  if args.time_set:
    await wait_until(args.time_set)

  ok = await orch.run()
  sys.exit(0 if ok else 1)


if __name__ == '__main__':
  asyncio.run(main())
