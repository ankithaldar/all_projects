#!/usr/bin/env python
# -- coding: utf-8 --

'''Course runner: Chapter 1 MCP demo launcher.

Examples:
  python main.py                          # restock scenario, mock LLM
  python main.py --scenario telecom       # telecom dispatch
  python main.py --scenario unsafe        # safety policy demo
  python main.py --scenario restock --live  # real LLM through your gateway
'''


from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src', 'learning_agentic_ai_with_ai'))

# Bootstrap import: must run after the sys.path setup above.
import chapter01_mcp.demo  # noqa: E402  # pylint: disable=wrong-import-position


def main() -> None:
  '''Delegate to the Chapter 1 demo.'''
  chapter01_mcp.demo.main()


if __name__ == '__main__':
  main()
