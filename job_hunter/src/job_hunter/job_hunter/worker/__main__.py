#!/usr/bin/env python
# -- coding: utf-8 --

'''Worker entrypoint: python -m job_hunter.worker.'''

from job_hunter.workers.scheduler import start_scheduler

if __name__ == '__main__':
  start_scheduler()
