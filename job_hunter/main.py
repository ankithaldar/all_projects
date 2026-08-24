#!/usr/bin/env python
# -- coding: utf-8 --

'''Thin launcher for the Job Hunter app.

Commands: seed-db | api | worker | run-discovery | discover-companies |
mcp <sources|resume|store>.
'''


from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
SRC_ROOT = APP_ROOT / 'src' / 'job_hunter'
for path in (str(SRC_ROOT), str(APP_ROOT)):
  if path not in sys.path:
    sys.path.insert(0, path)


def main() -> int:
  '''Dispatch the requested subcommand.

  Returns:
    Process exit code.
  '''
  parser = argparse.ArgumentParser(prog='job_hunter')
  parser.add_argument('command', choices=[
    'seed-db', 'api', 'worker', 'run-discovery',
    'discover-companies', 'mcp',
  ])
  parser.add_argument('server', nargs='?', default='sources')
  parser.add_argument('--seeds', default=str(APP_ROOT / 'seeds'))
  parser.add_argument('--config', default=str(APP_ROOT / 'config' / 'app.yaml'))
  args = parser.parse_args()

  if args.command == 'api':
    import uvicorn
    from job_hunter.api.main import create_app
    uvicorn.run(
      create_app(args.config),
      host='127.0.0.1',
      port=8088,
      log_level='warning',
    )
    return 0

  if args.command == 'worker':
    from job_hunter.workers.scheduler import start_scheduler
    start_scheduler(args.config)
    return 0

  if args.command == 'seed-db':
    from job_hunter.core.bootstrap import bootstrap
    bootstrap(args.config, seeds_dir=args.seeds)
    print('database seeded')
    return 0

  if args.command == 'run-discovery':
    import asyncio
    from job_hunter.workers.jobs import execute_pending_run, enqueue_run
    config_path = args.config
    run_id = asyncio.run(enqueue_run(config_path, kind='discovery'))
    result = asyncio.run(execute_pending_run(config_path, run_id))
    print(f'run {result["run_id"]} finished: {result["status"]}')
    return 0

  if args.command == 'discover-companies':
    import asyncio
    from job_hunter.services.company_discovery import run_seed_ingestion
    count = asyncio.run(run_seed_ingestion(args.config, Path(args.seeds)))
    print(f'companies ingested/updated: {count}')
    return 0

  if args.command == 'mcp':
    servers = {
      'sources': 'job_hunter.mcp_servers.sources_server',
      'resume': 'job_hunter.mcp_servers.resume_server',
      'store': 'job_hunter.mcp_servers.store_server',
    }
    if args.server not in servers:
      print(f'unknown server: {args.server}')
      return 2
    import runpy
    sys.argv = [args.server]
    runpy.run_module(servers[args.server], run_name='__main__')
    return 0

  return 2


if __name__ == '__main__':
  raise SystemExit(main())
