#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Make ``python -m stock_rl`` the entry point to :mod:`stock_rl.cli`.

A package with subpackages has no ``__main__`` module of its own, so
without this file ``python -m stock_rl`` fails with "No module named
stock_rl.__main__" and the only way in is to already know that
``stock_rl.cli`` exists.

The ``SystemExit`` is what makes a verdict legible to a script: without
it the exit status is 0 whatever the harness decided. The
verdict-to-status mapping lives in :func:`stock_rl.cli._verdict_status`,
so there is exactly one place where it is written down.
'''
# pylint: disable=invalid-name
from stock_rl.cli import main

raise SystemExit(main())
