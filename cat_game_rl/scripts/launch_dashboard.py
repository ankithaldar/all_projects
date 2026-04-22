#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    app_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "dashboard", "app.py",
    )
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path],
        check=True,
    )


if __name__ == "__main__":
    main()
