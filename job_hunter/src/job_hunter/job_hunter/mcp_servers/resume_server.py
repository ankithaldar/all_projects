#!/usr/bin/env python
# -- coding: utf-8 --

'''MCP server exposing resume parsing tools.

Run: python -m job_hunter.mcp_servers.resume_server
'''


from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from job_hunter.mcp_servers.common import config_path, dumps, ensure_sys_path

ensure_sys_path()

from job_hunter.core.bootstrap import bootstrap  # noqa: E402
from job_hunter.services.resume_parser import ResumeParser  # noqa: E402

mcp = FastMCP('job_hunter-resume')
_parser = ResumeParser()


@mcp.tool()
def extract_text(path: str) -> str:
  '''Extract cleaned text and metadata from a PDF resume file.

  Args:
    path: Absolute or app-relative file path.

  Returns:
    JSON with text, pages, sha256, ok.
  '''
  try:
    result = _parser.parse_file(path)
    return dumps(result)
  except Exception as exc:
    return dumps({'ok': False, 'error': str(exc)})


@mcp.tool()
def clean_text(text: str) -> str:
  '''Normalize raw resume text (ligatures, unicode, whitespace).

  Args:
    text: Raw text blob.

  Returns:
    JSON with cleaned text and a plausible-CV flag.
  '''
  cleaned = _parser.clean(text)
  return dumps({'text': cleaned, 'plausible_cv': _parser.looks_like_cv(cleaned)})


def main() -> None:
  '''Prepare data dirs then serve over stdio.'''
  bootstrap(config_path())
  mcp.run()


if __name__ == '__main__':
  main()
