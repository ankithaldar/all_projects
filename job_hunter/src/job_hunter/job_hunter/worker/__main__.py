#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Worker entrypoint: python -m job_hunter.worker.'''

from pathlib import Path

from job_hunter.workers.scheduler import start_scheduler


def main() -> None:
  '''Start the scheduler with the default config path.'''
  start_scheduler(Path(__file__).resolve().parents[4] / 'config' / 'app.yaml')


if __name__ == '__main__':
  main()
