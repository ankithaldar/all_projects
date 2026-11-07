#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''MCP server exposing resume parsing tools.

Run: python -m job_hunter.mcp_servers.resume_server
'''


from __future__ import annotations

import threading
from pathlib import Path

try:
  from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # mcp 2.x renamed FastMCP to MCPServer
  from mcp.server.mcpserver import MCPServer as FastMCP
from job_hunter.mcp_servers.common import dumps, ensure_sys_path
from job_hunter.mcp_servers.common import (
  get_settings_obj as _settings_obj,
)

ensure_sys_path()

# bootstrap kept as legacy monkeypatch target (common.get_settings_obj
# honours a patched resume_server.bootstrap).
from job_hunter.core.bootstrap import bootstrap  # noqa: E402,F401
from job_hunter.core.config import AppSettings  # noqa: E402
from job_hunter.services.resume_parser import ResumeParser  # noqa: E402

mcp = FastMCP('job_hunter-resume')
_parser_lock = threading.Lock()


class _LazyResumeParser:
  '''Lazily create the shared ResumeParser on first use.

  Import must not construct the parser (mirrors the lazy ``_http``
  getter in sources_server): the real parser is built on first
  attribute access and then cached. Attribute access delegates to it,
  so existing ``_parser.parse_file`` / ``_parser.clean`` call sites
  work unchanged, and ``_parser()`` returns the real parser.
  '''

  def __init__(self) -> None:
    '''Create the lazy holder; no parser is constructed yet.'''
    self._instance: ResumeParser | None = None

  def _real(self) -> ResumeParser:
    '''Return the cached parser, constructing it on first use.'''
    if self._instance is None:
      with _parser_lock:
        if self._instance is None:
          self._instance = ResumeParser()
    return self._instance

  def __getattr__(self, name: str) -> object:
    '''Delegate attribute access to the real parser.'''
    return getattr(self._real(), name)

  def __call__(self) -> ResumeParser:
    '''Return the real parser (getter-style access, like ``_http``).'''
    return self._real()


_parser = _LazyResumeParser()
# Legacy hook kept for existing tests that monkeypatch
# resume_server._settings directly (common.get_settings_obj honours it).
# The shared cache lives in common.py; this variable is not the cache.
_settings: AppSettings | None = None


def _get_parser() -> ResumeParser:
  '''Return the lazily created shared ResumeParser.'''
  return _parser._real()


@mcp.tool()
def extract_text(path: str) -> str:
  '''Extract cleaned text and metadata from a PDF resume file.

  Args:
    path: Absolute or app-relative file path.

  Returns:
    JSON with text, pages, sha256, ok.
  '''
  try:
    settings = _settings_obj()
    base = (settings.data_dir / 'resumes').resolve()
    p = Path(path).resolve()
    if not p.is_relative_to(base):
      return dumps({'ok': False, 'error': 'invalid path'})
    result = _parser.parse_file(p)
    return dumps(result)
  except Exception:  # pylint: disable=broad-exception-caught
    return dumps({'ok': False, 'error': 'invalid path'})


@mcp.tool()
def clean_text(text: str) -> str:
  '''Normalize raw resume text (ligatures, unicode, whitespace).

  Args:
    text: Raw text blob.

  Returns:
    JSON with cleaned text and a plausible-CV flag.
  '''
  cleaned = _parser.clean(text)
  plausible = _parser.looks_like_cv(cleaned)
  return dumps({'text': cleaned, 'plausible_cv': plausible})


def main() -> None:
  '''Prepare data dirs then serve over stdio.'''
  _settings_obj()
  mcp.run()


if __name__ == '__main__':
  main()
