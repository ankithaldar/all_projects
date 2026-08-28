#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Production logging bootstrap: one flat *.log per stage invocation.

Design goals (user-facing): every failure must be traceable from a
single file, without grepping kernel scrollback. The bootstrap

* attaches a timestamped FILE handler to the ROOT logger, so every
  ``get_logger()`` consumer lands in the file;
* tees ``sys.stdout``/``sys.stderr`` into the same file, which captures
  tqdm progress bars, Lightning prints, child-process output and raw
  tracebacks that never pass through ``logging``;
* writes a structured header (stage, experiment, UTC start, pid, argv)
  so a file identifies its run without opening it elsewhere;
* is idempotent per process (a second call only re-banners), so DDP
  re-execs and repeated stages in one kernel each get their own run
  file while never double-wrapping the streams.

Log location: ``paths.log_dir`` from the composed config (default
``/kaggle/working/logs``), override with ``KNEE_LOG_DIR``.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

_FORMAT = '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
_ACTIVE_STATE: dict = {'log_path': None, 'tee_installed': False}
_ORIGINALS: dict = {}


class Tee:
  """Duplicate a stdout/stderr stream into a log file handle.

  Keeps the original stream usable (notebook scrollback, tqdm) while
  mirroring every byte into the run log. Writes are flushed eagerly:
  crash-time content must survive a hard kernel kill.
  """

  def __init__(self, original, file_handle) -> None:
    """Wrap one stream.

    Args:
        original: The stream being replaced (sys.stdout/stderr).
        file_handle: Open append-mode text file receiving the mirror.
    """
    self._original = original
    self._file = file_handle

  def write(self, text: str) -> int:
    """Mirror one write into the log, then the original stream.

    Args:
        text: Bytes-as-text chunk from the writer.

    Returns:
        Number of characters written to the original stream.
    """
    try:
      self._file.write(text)
      self._file.flush()
    except (OSError, ValueError):
      # A full disk must never take down the run via a progress bar.
      pass
    return self._original.write(text)

  def flush(self) -> None:
    """Flush both the original stream and the log mirror."""
    try:
      self._file.flush()
    except (OSError, ValueError):
      pass
    self._original.flush()

  def set_target(self, file_handle) -> None:
    """Repoint the mirror at a NEW run file without re-wrapping.

    Args:
        file_handle: Open append-mode text file for the next stage.
    """
    self._file = file_handle

  def isatty(self) -> bool:
    """Preserve tty-ness of the original stream.

    Returns:
        The original stream's isatty() result (keeps tqdm colours).
    """
    try:
      return self._original.isatty()
    except (AttributeError, OSError):
      return False

  def __getattr__(self, name: str):
    """Delegate unknown attributes (encoding, fileno, ...) upstream.

    Args:
        name: Attribute requested on the stream.

    Returns:
        The original stream's attribute.
    """
    return getattr(self._original, name)


def current_log_path() -> str | None:
  """Path of the file the current process is mirroring into.

  Returns:
      Absolute log path, or None before setup_logging() ran.
  """
  return _ACTIVE_STATE['log_path']


def setup_logging(
  log_dir: str,
  stage: str,
  experiment_name: str,
  level: str = 'INFO',
  capture_streams: bool = True,
) -> str:
  """Install the file sink + stream tee for this stage invocation.

  Args:
      log_dir: Destination directory, created when absent.
      stage: CLI stage name (train/selftest/cache/...).
      experiment_name: Experiment label embedded in the filename.
      level: Root logger level name.
      capture_streams: When True (production default) also mirror
          stdout/stderr into the file.

  Returns:
      Absolute path of the flat log file.
  """
  os.makedirs(log_dir, exist_ok=True)
  stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  log_path = os.path.join(
    log_dir, f'knee_{stage}_{experiment_name}_{stamp}.log'
  )
  handle = open(log_path, 'a', encoding='utf-8')  # noqa: SIM115 (run-long)

  root = logging.getLogger()
  root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
  file_handler = logging.FileHandler(log_path)
  file_handler.setFormatter(logging.Formatter(_FORMAT))
  root.addHandler(file_handler)
  logging.captureWarnings(True)

  if not _ACTIVE_STATE['tee_installed'] and capture_streams:
    _ORIGINALS['stdout'] = sys.stdout
    _ORIGINALS['stderr'] = sys.stderr
    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)
    _ACTIVE_STATE['tee_installed'] = True
  elif _ACTIVE_STATE['tee_installed'] and capture_streams:
    # Same process, second stage: keep ONE tee wrapping the NEW file.
    sys.stdout.set_target(handle)
    sys.stderr.set_target(handle)

  _ACTIVE_STATE['log_path'] = log_path
  stamp_fmt = 'seconds'
  banner = (
    f'=== knee stage={stage} experiment={experiment_name} '
    f'utc={datetime.now(timezone.utc).isoformat(timespec=stamp_fmt)} '
    f'pid={os.getpid()} log={log_path} ==='
  )
  root.info(banner)
  print(banner, flush=True)
  return log_path


def reset_logging() -> None:
  """Undo bootstrap effects (test isolation helper)."""
  if _ACTIVE_STATE['tee_installed']:
    sys.stdout = _ORIGINALS.get('stdout', sys.stdout)
    sys.stderr = _ORIGINALS.get('stderr', sys.stderr)
    _ACTIVE_STATE['tee_installed'] = False
  root = logging.getLogger()
  for handler in list(root.handlers):
    if isinstance(handler, logging.FileHandler):
      root.removeHandler(handler)
      handler.close()
  _ACTIVE_STATE['log_path'] = None


__all__ = ['Tee', 'current_log_path', 'reset_logging', 'setup_logging']
