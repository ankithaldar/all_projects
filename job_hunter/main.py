#!/usr/bin/env python
# -- coding: utf-8 --

'''Thin launcher for the Job Hunter app.

Commands: seed-db | api | worker | run-discovery | discover-companies |
verify-ats | mcp <sources|resume|store>.
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
    'discover-companies', 'verify-ats', 'scout-companies', 'fetch-jobs', 'mcp',
  ])
  parser.add_argument('server', nargs='?', default='sources')
  parser.add_argument('--seeds', default=str(APP_ROOT / 'seeds'))
  parser.add_argument('--config', default=str(APP_ROOT / 'config' / 'app.yaml'))
  parser.add_argument(
    '--chunk', type=int, default=30,
    help='Companies to ATS-verify per pass for verify-ats.',
  )
  parser.add_argument(
    '--rounds', type=int, default=4,
    help='Max verify-ats passes before stopping.',
  )
  parser.add_argument(
    '--queries', type=int, default=0,
    help='Search queries for scout-companies (0 = config scout.search_queries).',
  )
  parser.add_argument(
    '--max', type=int, default=0,
    help='Cap on companies written by scout-companies (0 = no extra cap).',
  )
  parser.add_argument(
    '--sources', default='',
    help='Comma-separated source keys for fetch-jobs (default: all aggregators).',
  )
  parser.add_argument(
    '--limit', type=int, default=2000,
    help='Max records to fetch per source for fetch-jobs.',
  )
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
    import signal
    from job_hunter.workers.jobs import (
      enqueue_run, execute_pending_run, recover_orphans,
    )
    config_path = args.config
    orphans = recover_orphans(config_path, ttl_minutes=5)
    if orphans:
      print(f'recovered stale runs: {orphans}')

    def _graceful(signum, frame):
      raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, _graceful)
    run_id = enqueue_run(config_path, kind='discovery')
    try:
      result = execute_pending_run(config_path, run_id)
      print(f'run {result["run_id"]} finished: {result["status"]}')
    except BaseException as exc:  # noqa: B036 - deliberate broad catch
      print(f'run {run_id} aborted: {type(exc).__name__}: {exc}')
    finally:
      from job_hunter.db.repositories.runs import RunsRepository
      from job_hunter.workers.jobs import _settings
      settings = _settings(config_path)
      row = RunsRepository(settings.db_path).get(run_id) or {}
      if row.get('status') == 'running':
        RunsRepository(settings.db_path).finish(
          run_id, 'failed', {}, error_text='runner terminated',
        )
        print(f'run {run_id} finalized as failed')
    return 0

  if args.command == 'discover-companies':
    import asyncio
    from job_hunter.services.company_discovery import run_seed_ingestion
    count = asyncio.run(run_seed_ingestion(args.config, Path(args.seeds)))
    print(f'companies ingested/updated: {count}')
    return 0

  if args.command == 'verify-ats':
    import asyncio
    from job_hunter.core.bootstrap import bootstrap
    from job_hunter.services.company_discovery import verify_pending
    settings = bootstrap(args.config, seeds_dir=args.seeds)
    totals = {'verified': 0, 'failed': 0}
    for index in range(max(1, args.rounds)):
      result = asyncio.run(verify_pending(settings, chunk=max(1, args.chunk)))
      for key, value in result.items():
        totals[key] = totals.get(key, 0) + value
      if not result.get('verified'):
        break
      print(f'pass {index + 1}: {result}')
    print(f'ats verified: {totals["verified"]} | failed: {totals["failed"]}')
    return 0

  if args.command == 'fetch-jobs':
    import asyncio
    from job_hunter.core.bootstrap import bootstrap
    from job_hunter.services.bulk_fetch import bulk_fetch
    settings = bootstrap(args.config, seeds_dir=args.seeds)
    sources = [key for key in (args.sources or '').split(',') if key.strip()]
    report = asyncio.run(bulk_fetch(
      settings,
      source_keys=sources,
      limit_per_source=max(1, args.limit),
    ))
    print(report.summary())
    for source_key, count in sorted(report.per_source.items()):
      print(f'  {source_key:18s} {count}')
    return 0

  if args.command == 'scout-companies':
    import asyncio
    from job_hunter.core.bootstrap import bootstrap
    from job_hunter.services.linkedin_scout import (
      build_queries,
      make_scout,
      write_seed_file,
    )
    settings = bootstrap(args.config, seeds_dir=args.seeds)
    scout_cfg = settings.scout
    scout, search, http = make_scout(settings, Path(args.seeds))

    async def _run():
      '''Execute the scout and optionally write the seed file.

      Returns:
        Exit code for the process.
      '''
      try:
        limit = int(scout_cfg.get('search_queries', 12))
        if args.queries > 0:
          limit = args.queries
        queries = build_queries(limit=limit)
        report = await scout.scout(
          queries,
          result_limit=int(scout_cfg.get('results_per_query', 25)),
          verify_ats=bool(scout_cfg.get('verify_ats', True)),
          ingest=bool(scout_cfg.get('ingest', True)),
        )
      finally:
        await search.close()
        await http.close()
      if args.max > 0:
        report.companies = report.companies[: args.max]
      print(report.summary())
      for company in report.companies:
        ats = f'{company.ats_provider}:{company.board_ref}' if company.ats_provider else '-'
        print(
          f'  {company.name[:38]:40s} {company.domain[:34]:36s} '
          f'{(company.vertical or "?"):20s} {ats[:30]:32s} seen={company.sightings}'
        )
      if report.companies:
        seed_path = settings.seeds_dir / str(scout_cfg.get('seed_file', 'companies_scouted.yaml'))
        added = write_seed_file(
          seed_path,
          report.companies,
          header='Companies discovered from LinkedIn job pages via the search index.',
        )
        print(f'seed file updated: {seed_path} (+{added})')
      return 0

    return asyncio.run(_run())

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
