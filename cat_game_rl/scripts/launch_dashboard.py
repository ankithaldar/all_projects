#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Launch the Streamlit dashboard for the crafting game.

Binds to 127.0.0.1 by default (SEC-02) via the Streamlit argv list.
'''

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
  app_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'src', 'dashboard', 'app.py',
  )
  subprocess.run(
    [
      sys.executable, '-m', 'streamlit', 'run', app_path,
      '--server.address=127.0.0.1',
    ],
    check=True,
  )


if __name__ == '__main__':
  main()
