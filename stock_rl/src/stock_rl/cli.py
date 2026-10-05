#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Command-line entry point: the one process that reaches every module.

Thirty library modules existed with no way to invoke any of them. A
backtest is a function call, so a shell could only reach it by writing
Python. This module is the missing edge.

Five subcommands, and each one is a thin argument parser over work that
already exists:

``serve``
  :func:`stock_rl.api.serve`, loopback-bound, refusing anything else.
``experiment``
  :func:`stock_rl.experiment.run_experiment` -- the three-arm A/B test,
  printed with its verdict and its kill-criteria failures verbatim.
``baselines``
  The five providers in :mod:`stock_rl.baselines` on one backtester,
  as a table. Equal weight is in that table for the same reason it is in
  the API: an untried control arm is indistinguishable from a missing
  row.
``health``
  Version, dependency status and whatever data is actually on disk.
``skills``
  Every shipped ``.md`` with its evidence tier and whether the gate
  would let it run right now.

The exit code is the product
----------------------------
:attr:`stock_rl.experiment.harness.Verdict` becomes a process status:

==========  ======  ==============================================
Verdict     Exit    Meaning
==========  ======  ==============================================
``KEEP``    0       every pre-registered threshold cleared
``KILL``    1       at least one failed
n/a         2       ``INCONCLUSIVE``, or an argparse usage error
n/a         3       nothing could be run: no data, bad arguments
==========  ======  ==============================================

``2`` carries two meanings on purpose. argparse already exits 2 on a
usage error, so mapping ``INCONCLUSIVE`` onto the same code means a
script needs exactly one branch for "you asked wrongly or there is not
enough history", and one for "the measurement says stop". Anything that
cannot be measured is deliberately *not* 0: a CI step that ran the
harness on 200 bars of history must fail, because the harness never
evaluated a criterion.

No fabricated data, ever
------------------------
There is no code path here that invents prices. Panels come from a
directory of vendor CSV exports (:func:`stock_rl.api.load_panels`), and
when there are none the command says so and exits 3. The one demo mode,
``--synthetic``, requires being asked for by name, stamps
``panel_kind=synthetic`` on the report so the verdict cannot be quoted as
a finding, and **exits 3 whatever the verdict** -- a KEEP computed on
generated prices is a test of the harness, not evidence about the market,
and a script must not be able to read one as a pass.

Note what the demo mode can and cannot show. The generated panel is a
geometric random walk, so it has no cross-sectional structure for the
context block to explain, and ``--llm-scalar`` is one constant for every
symbol. All three arms therefore rank identically, the paired difference
is exactly zero, and the expected verdict is KILL. That is the useful
outcome: it shows the harness discriminates and that no pass can be
manufactured out of noise. It says nothing about the context layer.

The same rule covers ``baselines``: a table measured on generated bars is
a demonstration of the backtester, so it prints ``panel_kind=synthetic``
and exits 3. ``health`` and ``skills`` report on things that cannot be
fabricated -- files on disk and files in the package -- so they exit 0.

Verbatim means verbatim
-----------------------
The kill-criteria failures are printed as the harness wrote them, one per
line, with no prefix, no count, no "3 of 4 thresholds failed", and no
rounding of a near-miss. The pre-registered thresholds are the whole
point of the experiment; a CLI that summarised them could make a near
miss readable as a pass, which is the specific failure the harness was
built to prevent.

PONYTAIL: no config file, no plugin registry, no environment variables.
Ceiling: every parameter is a flag, so a twenty-parameter run is a long
command line and there is no named "experiment profile" to share with a
colleague. Upgrade path: a ``--preset NAME`` argument backed by one
module-level table of flag defaults in this file; nothing else has to
change, because every parameter already arrives through ``add_argument``.

PONYTAIL: stdout is the report and stderr is everything else, with no
switch to change it. Ceiling: piping stdout into ``less`` shows only the
report, and there is no way to send the verdict to a file while
diagnostics keep scrolling. Upgrade path: add ``--out PATH``, which
already exists for the experiment report, and extend it to the other
subcommands if that ceiling is ever reached.

PONYTAIL: argparse rather than a hand-rolled parser or a framework.
Ceiling: the help text is argparse's layout, so the epilog is the only
place a table fits and the ``--help`` output wraps the way argparse wraps.
Upgrade path: none needed -- the parser is the standard library's and the
help text is generated from the same ``add_argument`` calls that do the
parsing, so it cannot drift from reality.
'''

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

from stock_rl import __version__, api, baselines
from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY, CostModel
from stock_rl.experiment import ArmConfig, Verdict, run_experiment
from stock_rl.graph import nifty50_seed
from stock_rl.portfolio import run_portfolio
from stock_rl.skills import SkillRegistry
from stock_rl.weights import (
  WeightProvider,
  accepts_weight_cap,
  applied_weight_cap,
  bind_weight_cap,
)

__all__ = [
  'main',
]

#: The Indian Standard Time offset, fixed at +05:30 and never a DST
#: zone. Used for the generated demo panel's timestamps so a synthetic
#: bar is stamped in exchange local time like a real one.
_ist: Final[timedelta] = timedelta(hours=5, minutes=30)

#: Where a generated demo panel starts. Fixed rather than "now" so two
#: runs of the same command are diffable, and so ``--out`` files can be
#: compared across days.
_demo_start: Final[datetime] = datetime(2026, 1, 1, 15, 30,
                                         tzinfo=timezone(_ist))

#: Symbols in a generated demo panel. Named AAA/BBB/CCC on purpose: a
#: demo that printed RELIANCE would invite somebody to believe it.
_demo_symbols: Final[tuple[str, ...]] = ('AAA', 'BBB', 'CCC')

#: Seed for the generated demo panel's price walk. A fixed seed is what
#: makes the demo reproducible; an unseeded one would make a KILL on one
#: run and an INCONCLUSIVE on the next for reasons nobody could find.
_demo_seed: Final[int] = 3

#: Panel kind stamped on generated panels. The one string that stops a
#: demo verdict being quoted as a finding, and it comes from the harness
#: rather than from this file.
_demo_kind: Final[str] = 'synthetic'

#: Exit status for a verdict that cleared every threshold.
_exit_keep: Final[int] = 0

#: Exit status for at least one failed threshold.
_exit_kill: Final[int] = 1

#: Exit status for INCONCLUSIVE and for an argparse usage error. Both,
#: because argparse already exits 2 on a usage error and a caller should
#: not need two branches for "this run cannot tell you anything".
_exit_inconclusive: Final[int] = 2

#: Exit status for "nothing was run": no data, unusable data, unusable
#: arguments, or a generated panel. Never 0, because a command that
#: measured nothing has not succeeded.
_exit_no_data: Final[int] = 3

#: Exit status for an argument the command will not act on: a bind
#: address outside loopback, or a data directory that cannot be read.
#: Same code argparse uses, because from a script's point of view it is
#: the same event -- the invocation was wrong and nothing ran.
_exit_usage: Final[int] = 2

#: Exit status when stdout was closed early by a reader such as ``head``.
#: 128 + SIGPIPE, the same number a shell reports for a process killed by
#: the signal, so a pipeline that truncates a report is not mistaken for a
#: run that produced nothing.
_exit_broken_pipe: Final[int] = 141

#: Exit-code table printed in ``--help``. Kept as one string because the
#: table *is* the interface contract; a reader who cannot see it in
#: ``--help`` has to read the source to learn whether exit 1 means "the
#: context layer failed" or "you typed it wrong".
_verdict_exit_help: Final[str] = (
  'exit codes:\n'
  '  0  KEEP, every pre-registered kill criterion cleared\n'
  '  1  KILL, at least one criterion failed (failures printed verbatim)\n'
  '  2  INCONCLUSIVE, the length gate refused to evaluate anything;\n'
  '     also argparse usage errors, which argparse exits 2 on\n'
  '  3  nothing was run: no data, unusable data or arguments, or a\n'
  '     --synthetic panel, whose verdict is never evidence\n')

#: Exit-code table for the subcommands that report no verdict. Separate
#: from :data:`_verdict_exit_help` rather than a subset of it, because
#: printing the KEEP/KILL rows under ``skills`` would imply a reading of
#: the exit status that the command does not have. The top-level epilog
#: carries the verdict table instead, since it lists ``experiment``
#: alongside the others and a caller reading it wants the whole range.
_report_exit_help: Final[str] = (
  'exit codes:\n'
  '  0  the command ran and reported\n'
  '  2  argparse usage error\n'
  '  3  nothing was run: no data, unusable data or arguments, or a\n'
  '     --synthetic panel, whose numbers are never evidence\n')


def main(argv: list[str] | None = None) -> int:
  '''Parse arguments and dispatch to one subcommand.

  Args:
    argv: Argument list, defaulting to ``sys.argv[1:]``.

  Returns:
    The process exit status. See the module docstring and the epilog of
      ``--help`` for what each value means. Never returns for
      ``--help``, ``--version`` or a usage error: argparse exits those
      itself.
  '''
  arguments = _build_parser().parse_args(argv)
  handlers = {
    'serve': _run_serve,
    'experiment': _run_experiment,
    'baselines': _run_baselines,
    'health': _run_health,
    'skills': _run_skills,
  }
  try:
    return handlers[arguments.command](arguments)
  except BrokenPipeError:
    # `stock_rl experiment ... | head` closes the pipe early, and the
    # interpreter's shutdown flush raises it again after this frame has
    # returned. Redirecting fd 1 to devnull before returning is the
    # documented way to stop that second, noisier failure, and the exit
    # status is 141 (128 + SIGPIPE) because that is what the shell would
    # report for a process killed by SIGPIPE.
    os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return _exit_broken_pipe


def _build_parser() -> argparse.ArgumentParser:
  '''Return the parser for every subcommand.

  Returns:
    The top-level parser. Subparsers are ``required``, so an invocation
      with no subcommand is a usage error rather than a silent exit 0.
  '''
  parser = argparse.ArgumentParser(
    prog='python -m stock_rl',
    description=(
      'Command-line entry point for the stock_rl backtest core. Plain '
      'text on stdout, diagnostics on stderr, no colour and no progress '
      'bars.'),
    epilog=_verdict_exit_help,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    '--version', action='version', version=f'stock_rl {__version__}',
    help='print the package version and exit')
  subparsers = parser.add_subparsers(dest='command', metavar='COMMAND',
                                     required=True)
  _add_serve_parser(subparsers)
  _add_experiment_parser(subparsers)
  _add_baselines_parser(subparsers)
  _add_health_parser(subparsers)
  _add_skills_parser(subparsers)
  return parser


def _add_serve_parser(subparsers: object) -> None:
  '''Register the ``serve`` subcommand.

  Args:
    subparsers: The ``add_subparsers`` result.
  '''
  parser = subparsers.add_parser(
    'serve',
    help='run the read-only HTTP API on loopback',
    description=(
      'Run the existing read-only API. It binds 127.0.0.1 by default and '
      'refuses any other address: it has no authentication and serves '
      'positions, signals and audit state, so exposing it needs a '
      'TLS-terminating reverse proxy with a client certificate, not a '
      'different --host.'),
    epilog=_report_exit_help,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    '--host', default=api.default_host,
    help=('bind address; loopback only, and any other value is refused '
          f'(default: {api.default_host})'))
  parser.add_argument(
    '--port', type=int, default=api.default_port,
    help=f'bind port (default: {api.default_port})')
  parser.add_argument(
    '--data-dir', default=None, metavar='DIR',
    help=('directory of one CSV per symbol to serve; without it the '
          'dashboard starts empty and says so on /api/health'))


def _add_experiment_parser(subparsers: object) -> None:
  '''Register the ``experiment`` subcommand.

  Args:
    subparsers: The ``add_subparsers`` result.
  '''
  parser = subparsers.add_parser(
    'experiment',
    help='run the three-arm A/B experiment and print its verdict',
    description=(
      'Run arms A (price only), B (plus fused context) and C (plus one '
      'injected LLM scalar) over one panel set, then print the '
      'pre-registered verdict and every failed kill criterion verbatim. '
      'Arm C needs one injected scalar per symbol; this command makes no '
      'model call and supplies the constant you ask for, which is why '
      'arm C measures the wiring and not the idea.'),
    epilog=_verdict_exit_help,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  _add_panel_arguments(parser)
  parser.add_argument(
    '--seeds', type=int, nargs='+', default=[1, 2, 3], metavar='N',
    help=('seeds every arm is run under; each threshold is judged on '
          'the least favourable seed (default: 1 2 3)'))
  parser.add_argument(
    '--llm-scalar', type=float, default=0.1, metavar='X',
    help=('the scalar injected into arm C for every symbol; this '
          'command calls no model, so a constant is the only honest '
          'default (default: 0.1)'))
  parser.add_argument(
    '--shock', default=None, metavar='NODE_KEY',
    help=('traverse the Nifty-50 dependency graph from this node, e.g. '
          'commodity:crude, and feed the depth into the context block; '
          'without it the graph half of the context is zeros'))
  parser.add_argument(
    '--capital', type=float, default=10_000_000.0, metavar='RUPEES',
    help='starting cash (default: 10000000.0)')
  parser.add_argument(
    '--rebalance-days', type=int, default=21, metavar='N',
    help='bars between rebalances (default: 21)')
  parser.add_argument(
    '--holdings', type=int, default=5, metavar='N',
    help='names funded per rebalance (default: 5)')
  parser.add_argument(
    '--max-weight', type=float, default=0.10, metavar='F',
    help='cap on any one symbol (default: 0.10)')
  parser.add_argument(
    '--history', type=int, default=60, metavar='N',
    help='warm-up bars before the first decision (default: 60)')
  parser.add_argument(
    '--free-costs', action='store_true',
    help=('zero every charge so a turnover difference cannot be blamed '
          'on the cost model; DELIVERY costs are the default because '
          'that is what a real trade pays'))
  parser.add_argument(
    '--json', action='store_true',
    help='print the report as JSON instead of text; the verdict and the '
         'verbatim failures are carried under "verdict" and "failures"')
  parser.add_argument(
    '--out', default=None, metavar='PATH',
    help='also write the JSON report to this path')


def _add_panel_arguments(parser: argparse.ArgumentParser) -> None:
  '''Register the panel-source arguments shared by two subcommands.

  Args:
    parser: The subcommand parser to add to.
  '''
  source = parser.add_argument_group('panel source')
  source.add_argument(
    '--data-dir', default=None, metavar='DIR',
    help=('directory of one CSV per symbol, named after it, loaded by '
          'stock_rl.bars.load_csv; without it and without --synthetic '
          'the command exits 3 rather than inventing prices'))
  source.add_argument(
    '--synthetic', action='store_true',
    help=('generate a deterministic demo panel instead. The verdict is '
          'stamped panel_kind=synthetic and the command exits 3 whatever '
          'the verdict: a result on generated prices is a test of this '
          'harness, not a finding about the market'))
  source.add_argument(
    '--symbols', nargs='+', default=None, metavar='NAME',
    help='restrict the panel to these symbols (default: every one found)')
  source.add_argument(
    '--bars', type=int, default=700, metavar='N',
    help='--synthetic only: bars per symbol (default: 700)')


def _add_baselines_parser(subparsers: object) -> None:
  '''Register the ``baselines`` subcommand.

  Args:
    subparsers: The ``add_subparsers`` result.
  '''
  parser = subparsers.add_parser(
    'baselines',
    help='run all five baselines over one panel and print a table',
    description=(
      'Run every provider in stock_rl.baselines over one panel on one '
      'backtester with one cost model. Equal weight is in the table '
      'because DeMiguel, Garlappi & Uppal (2009) found 1/N is not '
      'reliably beaten by optimised allocation: a policy that cannot '
      'clear it has not demonstrated anything. A provider that could '
      'not run is printed with its reason rather than dropped.'),
    epilog=_report_exit_help,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  _add_panel_arguments(parser)
  parser.add_argument(
    '--capital', type=float, default=10_000_000.0, metavar='RUPEES',
    help='starting cash (default: 10000000.0)')
  parser.add_argument(
    '--rebalance-days', type=int, default=21, metavar='N',
    help='bars between rebalances (default: 21)')
  parser.add_argument(
    '--max-weight', type=float, default=0.10, metavar='F',
    help='cap on any one symbol (default: 0.10)')
  parser.add_argument(
    '--history', type=int, default=60, metavar='N',
    help='warm-up bars before the first decision (default: 60)')
  parser.add_argument(
    '--free-costs', action='store_true',
    help='zero every charge instead of charging DELIVERY costs')
  parser.add_argument(
    '--json', action='store_true',
    help='print the table as JSON instead of text')


def _add_health_parser(subparsers: object) -> None:
  '''Register the ``health`` subcommand.

  Args:
    subparsers: The ``add_subparsers`` result.
  '''
  parser = subparsers.add_parser(
    'health',
    help='print version, dependency status and available data',
    description=(
      'Report what this installation actually is: its version, whether '
      'it carries runtime dependencies, and which price panels are on '
      'disk. Reports on files, so it cannot fabricate a result and '
      'exits 0.'))
  parser.add_argument(
    '--data-dir', default=None, metavar='DIR',
    help=('directory of one CSV per symbol to report on; without it the '
          'available-data section says so rather than looking for data '
          'somewhere the operator did not name'))
  parser.add_argument(
    '--json', action='store_true',
    help='print the report as JSON instead of text')


def _add_skills_parser(subparsers: object) -> None:
  '''Register the ``skills`` subcommand.

  Args:
    subparsers: The ``add_subparsers`` result.
  '''
  parser = subparsers.add_parser(
    'skills',
    help='list shipped skill files, their tier and their gate status',
    description=(
      'List every shipped skill markdown file with its declared evidence '
      'tier and whether the registry would let it run under the flags '
      'given here. The gate is not a setting: a rejected skill never '
      'runs, an experimental one runs only in experiment mode, and a '
      'supported one needs an explicit enable *and* a passed kill '
      'criterion, which no skill in this repository has.'),
    epilog=_report_exit_help,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    '--enable', action='append', default=[], metavar='NAME',
    help=('operator enable set; repeatable, and independent of each '
          "file's own enabled key because both must agree"))
  parser.add_argument(
    '--experiment-mode', action='store_true',
    help='decide under experiment rules rather than live trading rules')
  parser.add_argument(
    '--json', action='store_true',
    help='print the table as JSON instead of text')


def _run_serve(arguments: argparse.Namespace) -> int:
  '''Run the HTTP API.

  Args:
    arguments: Parsed arguments.

  Returns:
    0 on a clean shutdown, 2 when the requested address is refused.

  Raises:
    OSError: If the port cannot be bound at all. Left to propagate: that
      is the kernel refusing a specific port, and the traceback names it.
  '''
  panels = {}
  if arguments.data_dir:
    try:
      panels = api.load_panels(arguments.data_dir)
    except (OSError, ValueError) as exc:
      _fail(f'--data-dir {arguments.data_dir} could not be loaded '
            f'({type(exc).__name__}: {exc})')
      return _exit_usage
    _warn(f'loaded {len(panels)} symbol(s) from {arguments.data_dir}')
  _warn(f'binding {arguments.host}:{arguments.port} -- loopback only, '
        'no authentication, not financial advice')
  service = api.ApiService(panels=panels)
  # The dependency graph reads its sentiment readings from beside the price
  # CSVs. Recorded on the service, not in module state, so two services in
  # one process cannot share a directory by accident.
  service.directory = arguments.data_dir
  try:
    api.serve(service, arguments.host, arguments.port)
  except ValueError as exc:
    # The refusal is a *usage* error rather than a failure: the operator
    # typed an address this service will not bind, and no socket was ever
    # opened. Reporting it as a traceback would bury a sentence they need
    # to read under a stack they cannot act on, so the sentence is printed
    # and the status is the same 2 argparse uses for a bad argument.
    _fail(str(exc))
    return _exit_usage
  return 0


def _run_experiment(arguments: argparse.Namespace) -> int:
  '''Run the three-arm experiment and print its verdict.

  Args:
    arguments: Parsed arguments.

  Returns:
    The exit status for the verdict: 0 for KEEP, 1 for KILL, 2 for
      INCONCLUSIVE, and 3 when nothing could be run or the panel was
      generated. Never summarises a failure string.
  '''
  try:
    panels, kind = _panels(arguments)
  except ValueError as exc:
    _fail(str(exc))
    return _exit_no_data
  llm_scalars = {symbol: arguments.llm_scalar for symbol in panels}
  # Imported at module scope above rather than here so a missing graph
  # table is an import error the test suite catches, not a NameError
  # behind a --shock nobody typed in CI.
  graph = nifty50_seed() if arguments.shock else None
  config = ArmConfig(
    capital=arguments.capital,
    costs=_cost_model(arguments.free_costs),
    history=arguments.history,
    rebalance_days=arguments.rebalance_days,
    max_weight=arguments.max_weight,
    holdings=arguments.holdings,
    seeds=tuple(dict.fromkeys(arguments.seeds)),
    panel_kind=kind,
  )
  try:
    report = run_experiment(
      panels, graph=graph, shock=arguments.shock,
      llm_scalars=llm_scalars, config=config)
  except ValueError as exc:
    _fail(f'the harness refused these panels: {exc}')
    return _exit_no_data
  if arguments.out:
    _write(arguments.out, report.to_json())
  if arguments.json:
    sys.stdout.write(report.to_json())
  else:
    _print_experiment(report, kind, len(panels))
  status = _verdict_status(report.verdict)
  if kind == _demo_kind:
    # A KILL on generated prices says the harness discriminates. A KEEP
    # says nothing at all, and a script must not be able to read one as
    # evidence, so a generated panel is never a success.
    _warn('panel_kind=synthetic: this verdict is not evidence, so the '
          'exit status is 3 regardless of what it says')
    return _exit_no_data
  return status


def _print_experiment(
  report: object,
  kind: str,
  symbols: int,
) -> None:
  '''Print a human-readable experiment report.

  Args:
    report: The :class:`stock_rl.experiment.ExperimentReport`.
    kind: The panel kind, printed because a verdict without it invites
      a reader to quote a generated panel as a finding.
    symbols: How many symbols were run, which the report itself does
      not carry.
  '''
  verdict = report.verdict
  out = sys.stdout.write
  out('experiment  three-arm A/B, pre-registered kill criteria\n')
  out(f'panel_kind  {kind}\n')
  if kind == _demo_kind:
    out('            GENERATED PRICES. Nothing below is a finding about '
        'the market.\n')
  out(f'symbols     {symbols}\n')
  out(f'run_at      {_utcnow()}\n')
  seeds = ', '.join(str(seed) for seed in report.config.seeds)
  out(f'seeds       {seeds}\n')
  out(f'settings    capital={report.config.capital:.0f} '
      f'rebalance_days={report.config.rebalance_days} '
      f'holdings={report.config.holdings} '
      f'max_weight={report.config.max_weight} '
      f'history={report.config.history}\n')
  out('arms        sharpe mean +/- stdev over seeds, growth multiple\n')
  for summary in report.arms:
    out(f'  {summary.arm.value:<14} sharpe '
        f'{summary.sharpe_mean:+.3f} +/- {summary.sharpe_stdev:.3f}   '
        f'growth {summary.total_return_mean:.3f}\n')
  out(f'benchmark   equal weight 1/N sharpe '
      f'{report.benchmark_sharpe:+.3f}\n')
  out('comparisons treatment minus the price-only control\n')
  for item in report.comparisons:
    turnover = item.monthly_one_sided_turnover
    out(f'  {item.label:<22} dSharpe {item.sharpe_delta_mean:+.3f}  '
        f'alpha {item.alpha_mean:+.5f}  alpha_p {item.alpha_pvalue:.3f}  '
        f'|t| {item.abs_t:.2f}  folds {item.folds_passed}/'
        f'{item.folds_total}  turnover/mo {turnover:.3f}\n')
  out(f'length gate {report.gate_reason}\n')
  out(f'verdict     {verdict.value.upper()}\n')
  out(f'exit        {_verdict_status(verdict)}\n')
  if report.kill is None:
    out('kill        no criterion was evaluated, so no claim may be made '
        'in either direction\n')
    return
  if report.kill.passed:
    out('kill        every pre-registered threshold cleared\n')
    return
  out('kill        FAILED. The strings below are the harness\'s own, '
      'verbatim:\n')
  for failure in report.kill.failures:
    out(f'  {failure}\n')


def _run_baselines(arguments: argparse.Namespace) -> int:
  '''Run every baseline over the panel and print a table.

  Args:
    arguments: Parsed arguments.

  Returns:
    0 when the table came from loaded panels, 3 when no panel was
      available or the panel was generated. Never 0 on generated bars.
  '''
  try:
    panels, kind = _panels(arguments)
  except ValueError as exc:
    _fail(str(exc))
    return _exit_no_data
  costs = _cost_model(arguments.free_costs)
  rows = []
  for name in baselines.__all__:
    rows.append(_baseline_row(
      name, getattr(baselines, name), panels, arguments, costs))
  scored = [row for row in rows if row['status'] == 'ok']
  best = max(scored, key=lambda row: row['sharpe']) if scored else None
  if arguments.json:
    sys.stdout.write(json.dumps({
      'panel_kind': kind,
      'capital': arguments.capital,
      'free_costs': arguments.free_costs,
      # The cap that was asked for. What each book was actually built at
      # is on its own row, under applied_max_weight, so the two can be
      # compared rather than one standing in for the other.
      'max_weight': arguments.max_weight,
      'rebalance_days': arguments.rebalance_days,
      'history': arguments.history,
      'not_advice': api.not_advice,
      'rows': rows,
      'symbols': sorted(panels),
      'best_sharpe': None if best is None else best['name'],
    }, indent=2, sort_keys=True) + '\n')
  else:
    _print_baselines(rows, best, kind, len(panels), arguments)
  if kind == _demo_kind:
    _warn('panel_kind=synthetic: these Sharpes are a demonstration of '
          'the backtester, so the exit status is 3 regardless')
    return _exit_no_data
  return 0 if scored else _exit_no_data


def _baseline_row(
  name: str,
  provider: WeightProvider,
  panels: dict[str, list[Bar]],
  arguments: argparse.Namespace,
  costs: CostModel,
) -> dict[str, object]:
  '''Run one baseline and return its row, or the reason it failed.

  The requested cap is bound to the provider as well as to the engine,
  through :func:`stock_rl.weights.bind_weight_cap`. Every provider in
  :mod:`stock_rl.baselines` carries its own ``max_weight`` default of
  0.10, so passing the provider unbound made that default the binding
  constraint: ``--max-weight 0.1`` and ``--max-weight 0.3`` printed the
  same Sharpe, and the payload echoed a 0.30 the book never used. The
  cap is bound by **keyword** because the five signatures disagree about
  where it sits -- ``trend_filtered_momentum``'s fifth parameter is
  ``ma_window`` -- and a positional call has already fed one to a
  moving-average window.

  Binding alone would still leave the payload asserting a number the
  provider never saw, so the row carries both: ``cap_bound`` says the
  provider was handed the cap at all, and ``applied_max_weight`` is read
  back off the book's own snapshots. Equality between the two is expected
  only while the cap is the binding constraint; see
  :func:`stock_rl.weights.applied_weight_cap`.

  Args:
    name: Provider name, as exported by :mod:`stock_rl.baselines`.
    provider: The provider callable.
    panels: The panel to run over.
    arguments: Parsed arguments, for the backtester settings.
    costs: The cost model every row is measured under.

  Returns:
    A mapping with a ``status`` of ``'ok'``, ``'skipped'`` or
      ``'error'``. The three are different claims and are kept apart:
      ``skipped`` is the backtester refusing the data, ``error`` is a
      provider raising, and an absent row would be indistinguishable
      from an untried control arm.
  '''
  row: dict[str, object] = {
    'name': name,
    # Stamped before anything runs, so a row that never got as far as a
    # book still reports whether its provider could take a cap.
    'cap_bound': accepts_weight_cap(provider),
  }
  try:
    result = run_portfolio(
      panels,
      bind_weight_cap(provider, arguments.max_weight),
      capital=arguments.capital,
      costs=costs,
      rebalance_days=arguments.rebalance_days,
      max_weight=arguments.max_weight,
      history=arguments.history,
    )
  except ValueError as exc:
    row['status'] = 'skipped'
    row['error'] = str(exc)
    return row
  except Exception as exc:  # pylint: disable=broad-exception-caught
    row['status'] = 'error'
    row['error'] = repr(exc)
    return row
  row.update({
    'sharpe': round(result.sharpe, 4),
    'max_drawdown': round(result.max_drawdown, 4),
    'growth': round(result.total_return_multiple, 4),
    'turnover': round(result.turnover, 4),
    'total_cost': round(result.total_cost, 2),
    'rebalances': result.rebalances,
    'points': len(result.equity),
    # Measured, not echoed: the requested cap is the payload's own field
    # and this is what the book was built at.
    'applied_max_weight': applied_weight_cap(result.weights),
    'status': 'ok',
  })
  return row


def _print_baselines(
  rows: list[dict[str, object]],
  best: dict[str, object] | None,
  kind: str,
  symbols: int,
  arguments: argparse.Namespace,
) -> None:
  '''Print the baseline table.

  Args:
    rows: One row per provider, in :mod:`stock_rl.baselines` order.
    best: The highest-Sharpe row, or None when none ran.
    kind: Panel kind, printed on the same lines as the numbers.
    symbols: How many symbols were run.
    arguments: Parsed arguments, for the echoed settings.
  '''
  out = sys.stdout.write
  out('baselines  one backtester, one cost model, one panel\n')
  out(f'panel_kind {kind}\n')
  if kind == _demo_kind:
    out('           GENERATED PRICES. Nothing below is a finding about '
        'the market.\n')
  charged = 'zero' if arguments.free_costs else 'DELIVERY'
  out(f'settings   capital={arguments.capital:.0f} '
      f'rebalance_days={arguments.rebalance_days} '
      f'max_weight={arguments.max_weight} history={arguments.history} '
      f'costs={charged}\n')
  out(f'symbols    {symbols}\n')
  blank = f'{'-':>9}'
  out(f'{'strategy':<26}{'sharpe':>9}{'growth':>9}{'drawdown':>10}'
      f'{'turnover':>10}{'cost Rs':>14}  status\n')
  for row in rows:
    name = str(row['name'])
    if row['status'] != 'ok':
      out(f'{name:<26}{blank}{blank}{'-':>10}{'-':>10}{'-':>14}'
          f'  {row['status']}: {row.get('error', '')}\n')
      continue
    marker = ' *' if best is not None and row is best else '  '
    out(f'{name + marker:<26}{row['sharpe']:>9.3f}'
        f'{row['growth']:>9.3f}{row['max_drawdown']:>10.3f}'
        f'{row['turnover']:>10.3f}{row['total_cost']:>14,.0f}  ok\n')
  if best is not None:
    out(f'\n* highest Sharpe under these settings: {best['name']}. '
        'DeMiguel et al. (2009) is the bar to clear, and equal weight '
        'is in this table.\n')
  out(f'{api.not_advice}\n')


def _run_health(arguments: argparse.Namespace) -> int:
  '''Print version, dependency status and available data.

  Args:
    arguments: Parsed arguments.

  Returns:
    0. Every fact reported here is read from a file or from the
      installed metadata, so a failure to read one is reported rather
      than papered over.
  '''
  payload = _health_payload(arguments)
  if arguments.json:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return 0
  out = sys.stdout.write
  out('health\n')
  out(f'  version            {payload['version']}\n')
  out(f'  python             {payload['python']}\n')
  out(f'  runtime_deps       {payload['runtime_dependencies']}\n')
  out(f'  run_at             {payload['run_at']}\n')
  out('  data\n')
  _print_data(out, payload)
  out(f'  skills             {payload['skills']} shipped markdown files\n')
  return 0


def _print_data(
  out: Callable[[str], object],
  payload: dict[str, object],
) -> None:
  '''Print the available-data section of a health report.

  Args:
    out: The stream writer, which is ``sys.stdout.write`` in production.
    payload: The report from :func:`_health_payload`.

  Raises:
    KeyError: Never; the payload is built by the function above.
  '''
  data = payload['data']
  directory = payload['data_dir']
  if directory is None:
    out('    none named        pass --data-dir to report on a directory '
        'of vendor CSVs\n')
    return
  if data['error'] is not None:
    out(f'    unusable          {data['error']}\n')
    return
  rows = data['symbols']
  if not rows:
    out(f'    empty             {directory} holds no *.csv\n')
    return
  out(f'    directory         {directory}\n')
  out(f'    symbols           {len(rows)} ({data['bars']} bars total)\n')
  for entry in rows:
    out(f'      {entry['symbol']:<12} {entry['bars']:>6} bars  '
        f'{entry['first']} .. {entry['last']}\n')
  out('    calendar          no exchange holiday list is bundled in this '
      'project, so a date cannot be declared a closure here\n')
  if data['misaligned']:
    out(f'    first bars differ across symbols: {data['misaligned']}. '
        'run_portfolio refuses a misaligned panel rather than '
        'forward-filling bars that did not trade.\n')


def _health_payload(arguments: argparse.Namespace) -> dict[str, object]:
  '''Return the health report as JSON-safe primitives.

  Args:
    arguments: Parsed arguments.

  Returns:
    Mapping with the version, the interpreter, the declared runtime
      dependency count and whatever ``--data-dir`` holds. A directory
      that cannot be read is reported as an ``error`` string rather
      than as an empty panel, because "no data" and "the path is wrong"
      are different operator problems.
  '''
  directory = arguments.data_dir
  symbols: list[dict[str, object]] = []
  error: str | None = None
  if directory is not None:
    try:
      panels = api.load_panels(directory)
    except (OSError, ValueError) as exc:
      panels = {}
      error = f'{type(exc).__name__}: {exc}'
    else:
      for symbol in sorted(panels):
        bars = panels[symbol]
        symbols.append({
          'symbol': symbol,
          'bars': len(bars),
          'first': _stamp(bars[0].timestamp),
          'last': _stamp(bars[-1].timestamp),
        })
  return {
    'data_dir': directory,
    'data': {
      'bars': sum(int(entry['bars']) for entry in symbols),
      'error': error,
      'misaligned': _misaligned(symbols),
      'symbols': symbols,
    },
    'python': '.'.join(str(part) for part in sys.version_info[:3]),
    'run_at': _utcnow(),
    'runtime_dependencies': _dependency_status(),
    'skills': len(SkillRegistry.load()),
    'version': __version__,
  }


def _dependency_status() -> str:
  '''Return a sentence about runtime dependencies.

  Returns:
    A short status string. Derived from the *installed metadata* rather
      than from a claim in a README, because a claim is exactly the
      thing that goes stale when someone adds a dependency.
  '''
  try:
    required = importlib.metadata.requires('stock-rl')
  except importlib.metadata.PackageNotFoundError:
    return 'unknown: the distribution is not installed, run make sync'
  if not required:
    return 'none declared; imports are stdlib only'
  return f'{len(required)} declared: {'; '.join(required)}'


def _misaligned(symbols: list[dict[str, object]]) -> list[str]:
  '''Return symbols whose first bar is not the earliest across the panel.

  Args:
    symbols: Per-symbol entries carrying a first-bar stamp.

  Returns:
    Names of symbols whose panel does not start with the earliest bar in
      the set. Reported rather than refused, because
      :func:`stock_rl.portfolio.run_portfolio` is what refuses it and a
      health report that exited non-zero for a fact a caller may already
      know would be a bad tool.
  '''
  if len(symbols) < 2:
    return []
  earliest = min(str(entry['first']) for entry in symbols)
  return [str(entry['symbol']) for entry in symbols
          if entry['first'] != earliest]


def _run_skills(arguments: argparse.Namespace) -> int:
  '''List every shipped skill with its tier and gate status.

  Args:
    arguments: Parsed arguments.

  Returns:
    0 when the registry loaded. A broken skill file raises out of
      :meth:`stock_rl.skills.SkillRegistry.load` and is not swallowed:
      a skill directory that validates three files and silently drops
      the fourth is how a capability disappears without anybody noticing.
  '''
  registry = SkillRegistry.load()
  statuses = registry.statuses(
    enabled=arguments.enable,
    experiment_mode=arguments.experiment_mode)
  if arguments.json:
    sys.stdout.write(json.dumps({
      'enable': list(arguments.enable),
      'experiment_mode': arguments.experiment_mode,
      'fingerprint': registry.registration_fingerprint(),
      'not_advice': api.not_advice,
      'skills': [{
        'active': status.active,
        'allowed_live': status.allowed_live,
        'evidence_tier': status.evidence_tier,
        'experiment_only': status.experiment_only,
        'fingerprint': status.fingerprint,
        'kill_criteria': list(registry.get(status.name).criteria_ids),
        'name': status.name,
        'reason': status.reason,
        'source': registry.get(status.name).source,
        'version': status.version,
      } for status in statuses],
    }, indent=2, sort_keys=True) + '\n')
    return 0
  out = sys.stdout.write
  out('skills  every shipped markdown file, with the gate as it stands\n')
  out(f'flags    enable={list(arguments.enable) or 'none'} '
      f'experiment_mode={arguments.experiment_mode}\n')
  out(f'registry fingerprint {registry.registration_fingerprint()}\n')
  out(f'{'name':<16}{'tier':<14}{'v':>2}  {'activatable':<12}reason\n')
  for status in statuses:
    out(f'{status.name:<16}{status.evidence_tier:<14}{status.version:>2}  '
        f'{_activatable(status):<12}{status.reason}\n')
  out('\nA skill marked not activatable is refused by the gate, not by a '
      'missing file. No kill criterion in this repository has a recorded '
      'pass, so no supported skill runs live.\n')
  return 0


def _activatable(status: object) -> str:
  '''Return how a skill's gate status reads in one word.

  Args:
    status: A :class:`stock_rl.skills.SkillStatus`.

  Returns:
    ``'yes'``, ``'experiment'`` or ``'no'``. The middle value exists
      because "runs in experiment mode" and "may influence a live order"
      are different permissions, and collapsing them into yes/no is the
      mistake :attr:`SkillStatus.allowed_live` was written to prevent.
  '''
  if status.active:
    return 'yes'
  return 'experiment' if status.experiment_only else 'no'


def _panels(arguments: argparse.Namespace) -> tuple[dict[str, list[Bar]],
                                                     str]:
  '''Return the panel to run on, and what it is.

  Args:
    arguments: Parsed arguments, carrying the panel-source flags.

  Returns:
    Tuple of (panels keyed by symbol, panel kind). The kind is
      ``'synthetic'`` for a generated panel and
      ``f'{vendor}/csv'`` for a loaded one, so the string that stops a
      demo being quoted as a finding travels with the data rather than
      being remembered by the reader.

  Raises:
    ValueError: If neither ``--data-dir`` nor ``--synthetic`` was given,
      the directory yielded no CSV, the requested symbols are not
      present, or the panel is too short for the warm-up. Never
      fabricates prices to avoid raising.
  '''
  if arguments.synthetic:
    if arguments.data_dir:
      raise ValueError(
        '--synthetic and --data-dir are mutually exclusive: a generated '
        'panel and a vendor export are different claims about where the '
        'prices came from, and --synthetic wins silently is exactly the '
        'kind of quiet this project refuses')
    panels = _demo_panels(arguments.bars)
  else:
    if not arguments.data_dir:
      raise ValueError(
        'no panel available: pass --data-dir with one CSV per symbol, '
        'or --synthetic to generate a labelled demo panel. This command '
        'will not invent prices, because a backtest on invented prices '
        'is a test of nothing')
    try:
      loaded = api.load_panels(arguments.data_dir)
    except (OSError, ValueError) as exc:
      raise ValueError(
        f'no panel available: {arguments.data_dir} could not be loaded '
        f'({type(exc).__name__}: {exc})') from exc
    panels = {symbol: bars for symbol, bars in loaded.items() if bars}
    if not panels:
      raise ValueError(
        f'no panel available: {arguments.data_dir} holds no readable '
        '*.csv, and this command will not invent prices')
  wanted = arguments.symbols
  if wanted:
    missing = sorted(set(wanted) - set(panels))
    if missing:
      raise ValueError(
        f'no panel available for {missing}; the panel holds '
        f'{sorted(panels)}')
    panels = {symbol: panels[symbol] for symbol in wanted}
  minimum = arguments.history + 1
  short = sorted(symbol for symbol, bars in panels.items()
                 if len(bars) < minimum)
  if short:
    raise ValueError(
      f'panel too short for --history {arguments.history}: {short} '
      f'need more than {minimum} bars each')
  kind = _demo_kind if arguments.synthetic else 'vendor-csv/replay'
  return panels, kind


def _demo_panels(count: int) -> dict[str, list[Bar]]:
  '''Return a deterministic generated panel, for the demo mode only.

  Seeded and reproducible, so two runs of the same command are diffable
  and a KILL does not become an INCONCLUSIVE on the next run for reasons
  nobody can find. Timestamps are timezone-aware IST bar closes, one
  calendar day apart.

  **Arms A, B and C will score identically here, and that is not a bug.**
  A geometric random walk has no cross-sectional structure for the context
  block to explain, and ``--llm-scalar`` is one constant applied to every
  symbol, so all three arms rank the same names and the paired difference
  is exactly zero. A KILL is therefore the expected result of
  ``--synthetic``, and it is the useful one: it shows the harness
  discriminates and that a pass cannot be manufactured out of noise. It
  is not evidence about the context layer, which is why the panel is
  stamped ``synthetic`` and the command exits 3 regardless.

  Args:
    count: Bars per symbol.

  Returns:
    Mapping of symbol to bars on a shared timeline. Every panel is a
      geometric random walk, so there is no real market structure in it
      and no result computed on it is a finding.
  '''
  panels: dict[str, list[Bar]] = {}
  for index, symbol in enumerate(_demo_symbols):
    source = random.Random(_demo_seed + index)
    bars: list[Bar] = []
    price = 100.0 + 10.0 * index
    for step in range(count):
      price = round(price * (1.0 + source.gauss(0.0004, 0.013)), 4)
      opening = bars[-1].close if bars else price
      bars.append(Bar(
        timestamp=_demo_start + timedelta(days=step),
        open=opening,
        high=max(opening, price),
        low=min(opening, price),
        close=price,
        volume=1.0,
      ))
    panels[symbol] = bars
  return panels


def _cost_model(free: bool) -> CostModel:
  '''Return the cost model to measure under.

  Args:
    free: Whether ``--free-costs`` was given.

  Returns:
    An all-zero :class:`stock_rl.costs.CostModel` when ``free``, and
      :data:`stock_rl.costs.DELIVERY` otherwise. DELIVERY is the default
      because it is what a real trade pays; zeroing it is a diagnostic
      for isolating turnover, and it is a named flag rather than a
      default so it cannot happen by accident.
  '''
  if not free:
    return DELIVERY
  # Every field named explicitly, including the two with non-zero
  # defaults. A "free" run that quietly kept its rupee DP charge or its
  # Rs 20 brokerage cap would still be a charged run, and the number it
  # produced would be attributed to the wrong cost model.
  return CostModel(brokerage_pct=0.0, brokerage_cap=0.0, stt_buy=0.0,
                   stt_sell=0.0, exchange_pct=0.0, sebi_pct=0.0,
                   stamp_duty_buy=0.0, gst_pct=0.0, dp_charge=0.0,
                   slippage_bps=0.0)


def _verdict_status(verdict: Verdict) -> int:
  '''Return the exit status a verdict maps to.

  Args:
    verdict: What the harness decided.

  Returns:
    0 for KEEP, 1 for KILL, 2 for INCONCLUSIVE. The mapping is the one
      thing a script depends on, so it lives in one function and both
      the text report and the process status read from it; a verdict
      printed one way and exited another is the failure this prevents.
  '''
  if verdict is Verdict.KEEP:
    return _exit_keep
  if verdict is Verdict.KILL:
    return _exit_kill
  return _exit_inconclusive


def _utcnow() -> str:
  '''Return the current instant as ISO 8601 UTC.

  Returns:
    An ISO 8601 string with an explicit ``Z``. Never a naive stamp and
      never a local-time one: a run record that omits its zone is
      unreadable two months later, and this project's whole
      look-ahead argument is about timestamps that do not say what zone
      they are in.
  '''
  return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _stamp(moment: datetime) -> str:
  '''Return one timestamp as ISO 8601 UTC, converting a naive one.

  A naive vendor stamp is read as IST and the report says so in words,
  because printing a naive value through unchanged would put two
  different clocks in one column of a table a reader is expected to
  compare.

  Args:
    moment: A bar timestamp. Vendor CSV exports are usually naive, and
      this project reads naive as exchange local time.

  Returns:
    ISO 8601 in UTC, with a ``(read as IST)`` suffix when an assumption
      was made.
  '''
  if moment.tzinfo is None:
    assumed = moment.replace(tzinfo=timezone(_ist)).astimezone(timezone.utc)
    return f'{assumed:%Y-%m-%dT%H:%M:%SZ} (read as IST)'
  return f'{moment.astimezone(timezone.utc):%Y-%m-%dT%H:%M:%SZ}'


def _write(path: str, document: str) -> None:
  '''Write a report to a path, or refuse loudly.

  Args:
    path: Destination.
    document: The text to write.

  Raises:
    OSError: Never caught. A report that could not be written is a
      failure the caller must see; silently skipping it would leave a
      CI step believing it archived an experiment it never ran.
  '''
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(document, encoding='utf-8')
  _warn(f'wrote {target}')


def _warn(message: str) -> None:
  '''Write one diagnostic line to stderr.

  Args:
    message: The text, without a trailing newline.
  '''
  sys.stderr.write(f'{message}\n')


def _fail(message: str) -> None:
  '''Write one error line to stderr.

  Args:
    message: The text, without a trailing newline.
  '''
  sys.stderr.write(f'error: {message}\n')


if __name__ == '__main__':
  raise SystemExit(main())
