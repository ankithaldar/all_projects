#!/usr/bin/env python3
'''
bq_parallel_run.py

A single-file, fully type-hinted, richly documented Python 3.10+ utility that
orchestrates multi-level, parallel BigQuery runs with per-run configuration
deep-merging, placeholder validation, retry logic, and a live TUI progress
table.

Author: <you>
'''

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import re
import copy
import time
import typing as t
from datetime import datetime, timezone
from pathlib import Path

import structlog
import yaml
from google.cloud import bigquery
from google.cloud.bigquery import QueryJob
from rich.console import Console
from rich.live import Live
from rich.table import Table

SQL_DEBUG = False


# ---------- Types -----------------------------------------------------------------

JSON = t.Union[str, int, float, bool, None, t.Dict[str, "JSON"], t.List["JSON"]]
ConfigDict = t.Dict[str, JSON]

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
    time_str = time_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(time_str)
    # Ensure timezone aware
    if dt.tzinfo is None:
      dt = dt.replace(tzinfo=timezone.utc)
    return dt
  except ValueError as e:
    raise ConfigurationError(f"Invalid ISO 8601 timestamp: {e}")



async def wait_until(target_time: datetime):
  '''Pauses the execution until the system clock reaches target_time'''
  now = datetime.now(timezone.utc)
  if target_time.tzinfo is None:
    target_time = target_time.replace(tzinfo=timezone.utc)

  # If the time has passed today, wait is skipped (or could be interpreted as next day)
  # Logic: strict scheduling for toda. If passed, warn and run immediately
  if target_time < now:
    logger.warn('Scheduled time is in the past. Running immediately.')
    return

  delta = target_time - now
  logger.info(f"Waiting for {str(delta).split('.')[0]} until {target_time}... ", trigger_time=str(target_time))
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
  '''Thin wrapper around the merged configuration.'''

  def __init__(self, cfg: ConfigDict) -> None:
    self._cfg = cfg

  def get(self, key: str, default: t.Any = None) -> t.Any:
    return self._cfg.get(key, default)

  def __getitem__(self, key: str) -> t.Any:
    return self._cfg[key]

  def __setitem__(self, key: str, value: t.Any) -> None:
    self._cfg[key] = value

  def __contains__(self, key: str) -> bool:
    return key in self._cfg


def deep_merge(base: t.Dict[str, t.Any], override: JSON) -> JSON:
  '''Deep-merge `override` into `base`. Lists are extended.'''
  if isinstance(base, dict) and isinstance(override, dict):
    merged: dict[str, JSON] = base.copy()
    for k, v in override.items():
      merged[k] = deep_merge(base.get(k), v)
    return merged
  if isinstance(base, list) and isinstance(override, list):
    return base + override
  return override if override is not None else base


def load_config(path: Path) -> ConfigDict:
  '''Load and validate base configuration.'''
  try:
    with path.open() as fh:
      return t.cast(ConfigDict, yaml.safe_load(fh))
  except Exception as exc:
    logger.error('Failed to load base config', path=str(path), exc=exc)
    sys.exit(1)


class UnresolvedPlaceholderError(Exception):
  '''Raised when a placeholder cannot be resolved from available context.'''
  pass


class CircularReferenceError(Exception):
  '''Raised when circular dependencies are detected during resolution.'''
  pass


class ConfigResolver:
  '''Initialize resolver with configurable placeholder syntax.'''

  def __init__(self, placeholder_pattern: str = r'\$\{([^}]+)\}'):
    self.placeholder_pattern = placeholder_pattern
    self._placeholder_regex = re.compile(placeholder_pattern)

  def resolve(self, config: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    # Validate input structure
    if not isinstance(config, dict):
      raise ValueError('run_configs must be a dict')

    # Create deep copies to avoid mutating original inputs
    run_copies = copy.deepcopy(config)

    # PHASE 1: SCAN entire structure for placeholders
    if not self._scan_for_placeholders(run_copies):
      return run_copies

    # PHASE 2: RESOLVE base_config (self-contained resolution)
    resolved_base = self._resolve_structure(
        run_copies,
        context=run_copies,
        path='run_config'
    )

    # PHASE 4: FINAL VALIDATION - ensure no unresolved placeholders remain
    self._validate_no_placeholders(resolved_base, 'run_config')
    return resolved_base


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
              f"Cannot substitute non-string value {replacement!r} "
              f"(type: {type(replacement).__name__}) for placeholder "
              f"'${{{var_name}}}' at {path}. Only string values allowed "
              "in placeholder substitution."
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
  '''Return all {var} placeholders found in SQL.'''
  return set(re.findall(r'\{(\w+)\}', sql))


def validate_placeholders(sql_files: list[Path], cfg: Config) -> None:
  '''Fail fast if any placeholder is missing.'''
  missing: dict[str, set[str]] = {}
  for sql_path in sql_files:
    try:
      sql = sql_path.read_text()
    except Exception as exc:
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
  '''Handles BigQuery job submission, retry, and cancellation.'''

  def __init__(
    self,
    client: bigquery.Client,
    cfg: Config,
    global_sem: asyncio.Semaphore,
    dry_run: bool,
    universal_time: Optional[datetime]
  ) -> None:
    self.client = client
    self.cfg = cfg
    self.global_sem = global_sem
    self.dry_run = dry_run
    self._to_cancel: list[asyncio.Task[t.Any]] = []
    self.universal_time = universal_time

  def _interpolate_sql(self, sql: str) -> str:
    '''Replace {var} with values from config.'''
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
    '''Core retry loop for a single SQL file.'''
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
      while True:
        job_id = f'bq_parallel_run__{run_id}__{sql_path.stem}__{level}__{int(time.time()*1000)}'
        try:
          logger.debug('Submitting job', job_id=job_id, sql=str(sql_path))
          await progress.update(run_id, str(sql_path), job_id, 'PENDING')
          if self.dry_run:
            await asyncio.sleep(0.1)
            progress.update(run_id, str(sql_path), job_id, 'DRY_SUCCESS')
            return
          job: QueryJob = await asyncio.to_thread(
            self.client.query,
            sql,
            job_id=job_id,
            project=self.cfg.get('billing_project'),
            # labels=self.cfg.get("labels", {}),
          )
          await progress.update(
            run_id,
            str(sql_path),
            job_id,
            'RUNNING',
            job=job,
          )
          await asyncio.to_thread(job.result)
          await progress.update(
            run_id,
            str(sql_path),
            job_id,
            'SUCCESS',
            job=job,
          )
          return
        except Exception as exc:
          logger.warning(
            'Job failed',
            job_id=job_id,
            attempt=attempt + 1,
            exc=exc,
          )
          if attempt >= max_retries:
            progress.update(run_id, str(sql_path), job_id, 'FAILED')
            raise
          backoff = min(2**attempt, max_backoff)
          await asyncio.sleep(backoff)
          attempt += 1

  def create_task(
    self,
    sql_path: Path,
    run_id: str,
    level: int,
    semaphore: asyncio.Semaphore,
    progress: 'ProgressTracker',
  ) -> asyncio.Task[t.Any]:
    '''Create cancellable task.'''
    task = asyncio.create_task(
      self._run_with_retry(sql_path, run_id, level, semaphore, progress),
      name=f'{run_id}__{sql_path.stem}',
    )
    self._to_cancel.append(task)
    return task

  async def cancel_all(self) -> None:
    '''Cancel all running tasks gracefully.'''
    for t in self._to_cancel:
      if not t.done():
        t.cancel()
    await asyncio.gather(*self._to_cancel, return_exceptions=True)

  async def _wait_until_target(self, target_time: datetime):
    '''Pauses the execution until the system clock reaches target_time'''
    now = datetime.now(timezone.utc)
    if target_time.tzinfo is None:
      target_time = target_time.replace(tzinfo=timezone.utc)

    # If the time has passed today, wait is skipped (or could be interpreted as next day)
    # Logic: strict scheduling for toda. If passed, warn and run immediately
    if target_time < now:
      logger.warn('Scheduled time is in the past. Running immediately.')
      return

    delta = target_time - now
    logger.info(f"Waiting for {str(delta).split('.')[0]} until {target_time}... ", trigger_time=str(target_time))

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


# ---------- Progress TUI ----------------------------------------------------------


class ProgressTracker:
  '''Thread-safe progress store and rich table renderer.'''

  def __init__(self) -> None:
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
    '''Build rich table for live display.'''
    table = Table(title='BigQuery Parallel Run')
    for col in ['Level', 'Run-ID', 'SQL File', 'Job ID', 'State', 'Bytes', 'Slot ms', 'Runtime s']:
      table.add_column(col)
    for row in sorted(self._data.values(), key=lambda r: (r['sql_path'], r['run_id'])):
      table.add_row(
        str(row.get('level', '')),
        row['run_id'],
        Path(row['sql_path']).name,
        row['job_id'],
        row['state'],
        str(row['bytes']),
        str(row['slot_ms']),
        f"{row['runtime_s']:.2f}",
      )
    return table


# ---------- Orchestration ---------------------------------------------------------


class Orchestrator:
  '''Top-level orchestrator for level-by-level execution.'''

  def __init__(
    self,
    project_id: str,
    run_configs: list[tuple[str, ConfigDict]],
    level_map: dict[int, list[str]],
    dry_run: bool,
    max_concurrency: int,
    universal_time: Optional[datetime]
  ) -> None:
    self.run_configs = run_configs
    self.level_map = level_map
    self.dry_run = dry_run
    self.max_concurrency = max_concurrency
    self.client = bigquery.Client(project=project_id)
    self.global_sem = asyncio.Semaphore(10)  # hard global cap
    self.progress = ProgressTracker()
    self.console = Console()
    self._shutting_down = False
    self.all_tasks: list[asyncio.Task[t.Any]] = []
    self.universal_time = universal_time

  async def run(self) -> bool:
    '''Return True if all levels succeeded.'''
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
    '''Run single level; return True if all succeeded.'''
    sql_files = [Path(p) for p in self.level_map[level]]
    semaphore = asyncio.Semaphore(self.max_concurrency)
    tasks: list[asyncio.Task[t.Any]] = []

    for run_override in self.run_configs:
      cfg = Config(run_override)

      # Run each of the run configs at the exact time set
      if cfg.get('scheduled_time'):
        cfg['scheduled_time'] = parse_time(cfg['scheduled_time'])

      runner = BQRunner(self.client, cfg, self.global_sem, self.dry_run, self.universal_time)

      for sql_path in sql_files:
        task = runner.create_task(
          sql_path,
          run_override.get('run_id'),
          level,
          semaphore,
          self.progress,
        )
        tasks.append(task)
        self.all_tasks.append(task)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failed = [r for r in results if isinstance(r, Exception)]
    for exc in failed:
      logger.error('Level task failed', level=level, exc=exc)
    return not failed

  async def _shutdown(self) -> None:
    '''Graceful shutdown.'''
    if self._shutting_down:
      return
    self._shutting_down = True
    logger.warning('Shutting down gracefully...')
    for ts in self.all_tasks:
      if not ts.done():
        ts.cancel()
    await asyncio.gather(*self.all_tasks, return_exceptions=True)
    sys.exit(1)


# ---------- CLI -------------------------------------------------------------------


def parse_cli() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Parallel BigQuery runner with per-run configs')
  parser.add_argument('--all_configs', required=True, type=Path, help='Path to base_config.yaml')
  parser.add_argument('--dry_run', action='store_true', help='Validate only, do not execute')
  parser.add_argument('--max-concurrency', type=int, help='Override max concurrency per level')
  parser.add_argument('--time_set', type=parse_time, default=None, help='Wait until specific time to start execution')
  return parser.parse_args()


async def main() -> None:
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


if __name__ == "__main__":
  asyncio.run(main())
