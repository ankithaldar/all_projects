#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Make ``python -m stock_rl.api`` the entry point to the dashboard API.

``stock_rl/api.py`` was runnable as a script, and a package with
subpackages has no ``__main__`` of its own, so without this file
``python -m stock_rl.api`` fails with "No module named
stock_rl.api.__main__" and the argparse block in
:func:`stock_rl.api.server.main` becomes reachable only from a caller
that already knows it exists.

:func:`stock_rl.api.server.main` is imported rather than re-exported
through the package so this file pulls in exactly the serving half.
'''
# pylint: disable=invalid-name
from stock_rl.api.server import main

raise SystemExit(main())
