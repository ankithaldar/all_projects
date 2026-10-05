#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the command-line entry point.

Every test here is written against behaviour a script would depend on,
not against the fact that a function runs. The properties that matter,
each with a test that fails if it is removed:

* **A failing kill criterion produces a non-zero exit.** A CLI that
  always exits 0 is decoration. ``test_experiment_kill_exits_one`` and
  ``test_experiment_inconclusive_exits_two`` pin two different non-zero
  statuses to two different verdicts, and
  ``test_kill_and_inconclusive_exit_codes_differ`` asserts they are not
  the same number -- collapsing them back into one "not a pass" code is
  the regression.
* **The failure strings are verbatim.** The test compares against
  :func:`stock_rl.context.fuse.evaluate_kill_criteria` called
  independently, so a CLI that paraphrased, counted, or rounded a
  near-miss into a pass would fail.
* **Nothing is fabricated.** With no ``--data-dir`` and no
  ``--synthetic``, ``experiment`` must refuse and exit non-zero rather
  than inventing a price, and ``--synthetic`` must be stamped
  ``panel_kind=synthetic`` in both the text and the JSON output and must
  never exit 0.
* **The exit status and the printed verdict cannot disagree.** ``--json``
  is parsed and compared with the text run.
* **Every flag is documented**, and the top-level epilog carries the
  exit-code table, because that table *is* the interface contract.

Panels here are written as real CSV files and loaded through
:func:`stock_rl.api.load_panels`, so the tests exercise the real vendor
ingestion path rather than a stub. The prices are generated, which is why
every panel kind is reported as synthetic and why no verdict here is a
claim about the market: a KILL on a random walk is a statement about the
harness.
'''

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stock_rl import __version__, api, baselines, cli
from stock_rl.context.fuse import KillCriteria, evaluate_kill_criteria
from stock_rl.experiment import Verdict
from stock_rl.graph import nifty50_seed
from stock_rl.skills import discover_skills, load_skills, tiers

#: Exchange local time, so the fixture stamps bars the way an NSE vendor
#: export does rather than in UTC.
_ist = timezone(timedelta(hours=5, minutes=30))

#: Symbols in the generated panel. Deliberately not real NSE names: a test
#: fixture printed as ``RELIANCE`` would invite somebody to believe it.
SYMBOLS = ('AAA', 'BBB', 'CCC')

#: Per-symbol per-bar drift and volatility of the generated walk. The drift
#: is small and positive so the panel has a positive claimed Sharpe, which
#: is what makes the length gate a live question rather than a trivially
#: satisfied one.
DRIFT = 0.0006
VOLATILITY = 0.011

#: Bars in a panel long enough to clear the harness's length gate at its
#: own defaults, so a run on it is judged on the kill criteria.
LONG_BARS = 420

#: Bars in the panel used for the length-gate tests. 130 leaves 70 after
#: the 60-bar warm-up, which is 0.28 years against the several years a
#: claimed Sharpe needs.
GATE_BARS = 130

#: The single step the gated panel takes. Small enough that the claimed
#: Sharpe stays near 0.2 and the gate refuses, large enough that it is
#: positive -- ``minimum_backtest_length`` is undefined at or below zero,
#: and a panel with no return at all would be refused for a different
#: reason than the one being tested.
GATE_BUMP = 0.002


def write_panel(
  directory: Path,
  count: int = LONG_BARS,
  seed: int = 5,
  symbols: tuple[str, ...] = SYMBOLS,
) -> Path:
  '''Write one randomly-walked CSV per symbol, as ``load_csv`` expects.

  Timestamps are ISO 8601 with an explicit ``+05:30`` and advance one
  calendar day per bar. The explicit offset is deliberate: a vendor export
  carrying no zone is a real input shape, and this fixture should not be
  the one thing in the suite that never exercises it.

  Bars are identical across symbols, so the panel is aligned and
  ``run_portfolio`` does not refuse it.

  Args:
    directory: Directory to write into, created if absent.
    count: Bars per symbol. Overrides the walk's own length.
    seed: Seed for the price walk.
    symbols: Symbols to generate.

  Returns:
    The directory, so a test can pass it straight to ``--data-dir``.
  '''
  return _write(directory, count, symbols, _walk(seed, count))


def write_gated_panel(directory: Path) -> Path:
  '''Write a panel whose claimed Sharpe the available history cannot support.

  Flat prices with one small step in the middle, rather than a short
  random walk. Both produce a verdict, but only this one produces it for
  a reason that is *deterministic*: the gate compares available years
  against the years the claimed Sharpe needs, and a walk's claimed Sharpe
  is whatever its seed happened to produce, so a fixture built that way
  silently becomes a KILL the day the drift changes. Here the Sharpe is
  pinned at roughly 0.2 and 70 bars cannot support it, which is the
  question the INCONCLUSIVE verdict answers.

  Args:
    directory: Directory to write into, created if absent.

  Returns:
    The directory.
  '''

  return _write(directory, GATE_BARS, SYMBOLS, _flat(), bump=GATE_BUMP)


def _flat() -> list[list[float]]:
  '''Return per-symbol series that never move.

  The prices are irrelevant: :func:`_write` steps them at the midpoint when
  ``bump`` is given, which is the only return this fixture needs.

  Returns:
    One list per symbol, ``GATE_BARS`` long, of a constant price.
  '''
  return [[100.0 + 10.0 * index] * GATE_BARS
          for index in range(len(SYMBOLS))]


def _walk(seed: int, count: int) -> list[list[float]]:
  '''Return per-symbol closing-price series from a random walk.

  Args:
    seed: Base seed; each symbol takes ``seed + index``.
    count: Bars per symbol.

  Returns:
    One list per symbol, ``count`` long, of closing prices.
  '''
  series: list[list[float]] = []
  for index in range(len(SYMBOLS)):
    source = random.Random(seed + index)
    price = 100.0 + 10.0 * index
    closes = []
    for _ in range(count):
      price = round(price * (1.0 + source.gauss(DRIFT, VOLATILITY)), 4)
      closes.append(price)
    series.append(closes)
  return series


def _write(
  directory: Path,
  count: int,
  symbols: tuple[str, ...],
  series: list[list[float]],
  bump: float | None = None,
) -> Path:
  '''Write one CSV per symbol from closing-price series.

  Args:
    directory: Directory to write into, created if absent.
    count: Bars per symbol.
    symbols: Symbols to generate.
    series: One closing-price list per symbol. Its length sets ``count``,
      so a caller cannot ask for more bars than it supplied.
    bump: When given, the single fractional step taken at the midpoint of
      each series instead of whatever ``series`` holds. Present so
      :func:`write_gated_panel` can pin a Sharpe rather than hope for one.

  Returns:
    The directory.

  Raises:
    ValueError: If the series do not cover ``symbols``.
  '''
  if len(series) != len(symbols):
    raise ValueError('one closing-price series per symbol')
  directory.mkdir(parents=True, exist_ok=True)
  for symbol, closes in zip(symbols, series):
    price = closes[0]
    lines = ['timestamp,open,high,low,close,volume']
    for step in range(count):
      opening = round(price, 4)
      if bump is None:
        price = round(closes[step], 4)
      elif step == count // 2:
        price = round(opening * (1.0 + bump), 4)
      high = round(max(opening, price) * 1.002, 4)
      low = round(min(opening, price) * 0.998, 4)
      stamp = (datetime(2026, 1, 1, 15, 30, tzinfo=_ist)
               + timedelta(days=step))
      lines.append(
        f'{stamp.isoformat()},{opening},{high},{low},{price},1000')
    (directory / f'{symbol}.csv').write_text(
      '\n'.join(lines) + '\n', encoding='utf-8')
  return directory


def captured(
  argv: list[str],
  capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
  '''Run the CLI in-process and return its status with both streams.

  Calling :func:`stock_rl.cli.main` rather than spawning a subprocess is
  what makes a failure legible: the status, stdout and stderr arrive as
  three values instead of as a wall of captured text.

  Args:
    argv: Argument list.
    capsys: pytest capture fixture.

  Returns:
    Tuple of (exit status, stdout, stderr).
  '''
  status = cli.main(argv)
  streams = capsys.readouterr()
  return status, streams.out, streams.err


def run_panel(argv: list[str], directory: Path,
              capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
  '''Run the CLI against a written panel, with ``--data-dir`` prepended.

  Args:
    argv: Argument list, without the panel flags.
    directory: The panel directory.
    capsys: pytest capture fixture.

  Returns:
    Tuple of (exit status, stdout, stderr).
  '''
  return captured([*argv, '--data-dir', str(directory)], capsys)


def verdict_of(payload: dict[str, object]) -> str:
  '''Return the verdict a report carries, uppercased.

  Args:
    payload: A parsed experiment JSON report.

  Returns:
    The verdict spelled the way the text report spells it.
  '''
  assert 'verdict' in payload, payload
  return str(payload['verdict']).upper()


def _failure_block(text: str) -> list[str]:
  '''Return the indented failure lines the report prints verbatim.

  The block is identified by the header the CLI writes immediately above
  it, so a line that merely happens to be indented -- an arm row, a
  comparison row -- cannot be mistaken for a failure. That distinction is
  the test: a CLI that printed the failures anywhere but under this header
  would be reporting them differently from what was asked for.

  Args:
    text: The whole stdout of an experiment run.

  Returns:
    Each failure line, stripped of its two-space indent, in print order.
  '''
  lines = text.splitlines()
  start = next(
    index for index, line in enumerate(lines)
    if line.startswith('kill        FAILED'))
  block = []
  for line in lines[start + 1:]:
    if not line.startswith('  '):
      break
    block.append(line.strip())
  return block


def test_experiment_kill_exits_non_zero(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''A panel that fails a kill criterion must not exit 0.

  This is the load-bearing assertion of the whole command: a CI step that
  runs the harness has to fail when the harness says stop, and the only
  way that is guaranteed is that the status is derived from the verdict.
  '''
  panel = write_panel(tmp_path / 'panel')
  status, out, err = run_panel(['experiment'], panel, capsys)
  assert 'KILL' in out, out
  assert status == 1, (status, out, err)


def test_experiment_inconclusive_exits_two(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Too little history yields INCONCLUSIVE and exit 2, not 0 and not 1.

  The gate compares available years against the years the claimed Sharpe
  needs. 70 bars of returns is 0.28 years, and a Sharpe near 0.2 needs
  several, so the gate refuses: no criterion is evaluated and no claim may
  be made in either direction. A script must still be able to tell that
  apart from a KILL, and it must not be able to read it as a pass.
  '''
  panel = write_gated_panel(tmp_path / 'gated')
  status, out, _ = run_panel(['experiment'], panel, capsys)
  assert 'INCONCLUSIVE' in out, out
  assert 'no criterion was evaluated' in out, out
  assert status == 2, (status, out)


def test_kill_and_inconclusive_exit_codes_differ(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''KILL and INCONCLUSIVE must not collapse onto one status.

  ``KILL`` says the measurement says stop; ``INCONCLUSIVE`` says there
  was not enough history to measure. A single "did not pass" code would
  lose that, and a pipeline branching on the difference -- retry with more
  data versus abandon the layer -- depends on it.
  '''
  long_panel = write_panel(tmp_path / 'long')
  gated_panel = write_gated_panel(tmp_path / 'gated')
  kill = run_panel(['experiment'], long_panel, capsys)[0]
  other = run_panel(['experiment'], gated_panel, capsys)[0]
  assert kill != other, (kill, other)
  assert {kill, other} == {1, 2}, (kill, other)


def test_experiment_prints_failures_verbatim(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Every failure string appears in stdout exactly as the harness wrote it.

  The expected text is produced by calling
  :func:`stock_rl.context.fuse.evaluate_kill_criteria` on the report's own
  ``criteria_inputs``, not by copying a literal. A CLI that summarised
  the failures, counted them, or rounded a near-miss into "4 of 5 passed"
  would fail here, and that is the specific failure mode the harness
  exists to prevent: a threshold that can be read as softer than it is.
  '''
  panel = write_panel(tmp_path / 'panel')
  status, out, _ = run_panel(['experiment', '--json'], panel, capsys)
  payload = json.loads(out)
  assert payload['kill_passed'] is False
  inputs = payload['criteria_inputs']
  # The expected strings come from the criteria function itself, given the
  # report's own inputs, rather than from literals copied out of a run.
  expected = evaluate_kill_criteria(
    alpha_pvalue=float(inputs['alpha_pvalue']),
    sharpe_pvalue=float(inputs['sharpe_pvalue']),
    abs_t=float(inputs['abs_t']),
    folds_passed=int(float(inputs['folds_passed'])),
    monthly_one_sided_turnover=float(inputs['monthly_one_sided_turnover']),
    criteria=KillCriteria(),
  ).failures
  assert expected, 'the fixture must actually fail something'
  assert list(payload['failures']) == list(expected)
  text_status, text_out, _ = run_panel(['experiment'], panel, capsys)
  assert text_status == status
  # Verbatim means verbatim: each string appears in stdout byte for byte,
  # and the CLI prints nothing else in that block. A summary line, a
  # count, or a reordered list would break this.
  printed = _failure_block(text_out)
  assert printed == list(expected), (printed, expected)


def test_experiment_json_agrees_with_text_verdict(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''The JSON verdict is the text verdict, and both are the exit status.

  Two code paths print the verdict and a third returns it. If they can
  disagree, one of them is lying to whoever reads it, and the JSON is what
  a CI job parses.
  '''
  panel = write_panel(tmp_path / 'panel')
  json_status, json_out, _ = run_panel(['experiment', '--json'], panel,
                                       capsys)
  payload = json.loads(json_out)
  text_status, text_out, _ = run_panel(['experiment'], panel, capsys)
  verdict = verdict_of(payload)
  assert Verdict(payload['verdict']) in list(Verdict)
  assert f'verdict     {verdict}' in text_out, text_out
  assert json_status == text_status == verdict_status(verdict)


def verdict_status(verdict: str) -> int:
  '''Return the exit status the documented table maps ``verdict`` to.

  Read from the ``--help`` table rather than written here, so the test
  cannot pass against a status mapping the help text does not describe.

  Args:
    verdict: A harness verdict word, any case.

  Returns:
    0 for keep, 1 for kill, 2 for inconclusive.
  '''
  published = _help_text(['experiment', '--help'])
  assert 'exit codes:' in published
  return {
    'KEEP': 0 if '0  KEEP' in published else -1,
    'KILL': 1 if '1  KILL' in published else -1,
    'INCONCLUSIVE': 2 if '2  INCONCLUSIVE' in published else -1,
  }[verdict.upper()]


def _help_text(argv: list[str]) -> str:
  '''Return what ``--help`` writes to stdout, letting argparse exit.

  Args:
    argv: Argument list ending in ``--help``.

  Returns:
    The captured stdout.
  '''
  buffer = io.StringIO()
  with contextlib.redirect_stdout(buffer):
    with pytest.raises(SystemExit) as caught:
      cli.main(argv)
  assert caught.value.code == 0, argv
  return buffer.getvalue()


def test_unknown_subcommand_is_a_usage_error(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''An unknown command exits non-zero and puts usage on stderr.

  argparse owns this path, and the test exists because "exits 2 with
  usage on stderr" is a contract a script depends on: it must not be a
  traceback on stdout, and it must not be a silent exit 0.
  '''
  with pytest.raises(SystemExit) as caught:
    cli.main(['backtest'])
  streams = capsys.readouterr()
  assert caught.value.code != 0
  assert streams.err, streams.out
  assert 'usage' in streams.err.lower()


def test_missing_subcommand_is_a_usage_error(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''No subcommand at all is an error, not an implicit ``--help``.

  Required subparsers make ``python -m stock_rl`` on its own a usage
  error; otherwise a CI job that dropped its arguments would "pass".
  '''
  with pytest.raises(SystemExit) as caught:
    cli.main([])
  streams = capsys.readouterr()
  assert caught.value.code != 0
  assert 'usage' in streams.err.lower(), streams.err


def test_experiment_without_a_panel_refuses(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''No panel source means no run, a message, and a non-zero status.

  The one rule this command cannot bend: it will not invent prices. A
  fabricated panel that produced a KILL would be indistinguishable from a
  real one in a CI log, and a fabricated panel that produced a KEEP would
  be worse.
  '''
  status, out, err = captured(['experiment'], capsys)
  assert out == '', out
  assert status == 3, (status, err)
  assert 'no panel available' in err, err
  assert 'will not invent prices' in err, err


def test_serve_with_an_unusable_data_dir_is_a_usage_error(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''``serve`` reports a bad ``--data-dir`` instead of starting empty.

  A dashboard that came up with zero symbols because the vendor export
  path was wrong looks exactly like a dashboard with nothing to show, and
  the operator who has to fix it is the one who can least afford that
  ambiguity.
  '''
  status, out, err = captured(
    ['serve', '--data-dir', str(tmp_path / 'nowhere')], capsys)
  assert out == '', out
  assert status == 2, (status, err)
  assert 'could not be loaded' in err, err


def test_empty_data_dir_refuses(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''A directory holding no CSV is "no data", not "an empty panel".

  ``load_panels`` returns an empty mapping rather than raising for an
  empty directory, and an empty panel would backtest to nothing and print
  a table of failures that reads like a finding.
  '''
  empty = tmp_path / 'empty'
  empty.mkdir()
  status, out, err = captured(
    ['experiment', '--data-dir', str(empty)], capsys)
  assert out == ''
  assert status == 3, (status, err)
  assert 'no panel available' in err, err


def test_synthetic_is_labelled_and_never_exits_zero(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''``--synthetic`` is stamped in the output and cannot be read as a pass.

  Three things have to hold at once: the label is in the text output, the
  label is in the JSON output, and the status is non-zero whatever the
  verdict. A KILL on generated prices would otherwise exit 0 if the
  harness ever reported one.
  '''
  status, out, err = captured(
    ['experiment', '--synthetic', '--bars', '400'], capsys)
  assert 'panel_kind  synthetic' in out, out
  assert 'GENERATED PRICES' in out, out
  # The printed `exit` line is the verdict's status, and the status the
  # process returns is 3 regardless: a KEEP on generated prices must not be
  # readable as a pass by anything at all.
  assert 'exit        1' in out, out
  assert status == 3, (status, out, err)
  json_status, json_out, _ = captured(
    ['experiment', '--synthetic', '--bars', '400', '--json'], capsys)
  assert json.loads(json_out)['panel_kind'] == 'synthetic'
  assert json_status == 3


def test_synthetic_and_data_dir_are_mutually_exclusive(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Asking for both is refused rather than one silently winning.

  A generated panel that quietly replaced a vendor export would put
  fabricated prices in a run whose output says ``vendor-csv/replay``.
  '''
  panel = write_panel(tmp_path / 'panel')
  status, out, err = captured(
    ['experiment', '--synthetic', '--data-dir', str(panel)], capsys)
  assert out == ''
  assert status == 3, (status, err)
  assert 'mutually exclusive' in err, err


def test_experiment_writes_out_file(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''``--out`` archives the same report ``--json`` prints.

  A CI step that reads the verdict from a file rather than from a stream
  needs the two to be identical bytes, and it needs the file written even
  though the command's status is non-zero.
  '''
  panel = write_panel(tmp_path / 'panel')
  target = tmp_path / 'nested' / 'report.json'
  status, json_out, _ = run_panel(['experiment', '--json'], panel, capsys)
  status_again, _, _ = run_panel(
    ['experiment', '--out', str(target)], panel, capsys)
  assert status == status_again
  assert target.read_text(encoding='utf-8') == json_out


def test_serve_help_exits_zero_and_names_loopback() -> None:
  '''``serve --help`` exits 0 and says which address it binds.

  The default address is the security-relevant default of the whole
  service, so it belongs in the help text rather than in the source: a
  reader who has to read the source to learn that this server binds
  loopback and has no authentication is a reader who will not learn it.
  '''
  published = _help_text(['serve', '--help'])
  assert api.default_host in published
  assert '127.0.0.1' in published
  assert 'loopback only' in published
  for flag in ('--host', '--port', '--data-dir'):
    assert flag in published, (flag, published)


def test_serve_refuses_a_non_loopback_bind(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''A routable bind is refused with the reason, and no socket is opened.

  The refusal belongs to :func:`stock_rl.api.build_server`, so this test
  asserts the CLI surfaces it as a usage error rather than a traceback:
  the message is what tells the operator to put a proxy in front instead
  of widening ``--host``, and a stack trace buries it. Nothing is mocked,
  so this also proves the real guard still fires.
  '''
  status, out, err = captured(['serve', '--host', '0.0.0.0'], capsys)
  assert out == '', out
  assert status == 2, (status, err)
  assert 'refusing to bind' in err, err
  assert 'reverse proxy' in err, err


def test_top_level_help_documents_the_exit_code_table() -> None:
  '''The exit-code table is in ``--help``, not only in the source.

  The table is the interface contract: without it a script's author has
  to read this module to learn whether 1 means "the context layer failed"
  or "you typed it wrong", and by then they are already depending on it.
  '''
  published = _help_text(['--help'])
  for token in ('exit codes:', 'KEEP', 'KILL', 'INCONCLUSIVE'):
    assert token in published, (token, published)
  for command in ('serve', 'experiment', 'baselines', 'health', 'skills'):
    assert command in published, (command, published)


def test_experiment_help_documents_every_flag() -> None:
  '''Every experiment flag appears in its help.

  An argparse help text is generated from the same ``add_argument`` calls
  that do the parsing, so this test cannot drift from the parser. It is
  here because the failure it catches is silent: a flag nobody can
  discover is a flag nobody sets.
  '''
  published = _help_text(['experiment', '--help'])
  for flag in ('--data-dir', '--synthetic', '--symbols', '--bars',
               '--seeds', '--llm-scalar', '--shock', '--capital',
               '--rebalance-days', '--holdings', '--max-weight',
               '--history', '--free-costs', '--json', '--out'):
    assert flag in published, (flag, published)
  assert 'panel_kind=synthetic' in published


def test_baselines_help_documents_every_flag() -> None:
  '''Every baselines flag is discoverable, including the panel source.

  ``baselines`` shares the panel-source group with ``experiment``, and a
  flag that exists on one subcommand but is undocumented on the other is
  a flag whose absence is a mystery at the moment somebody needs it.
  '''
  published = _help_text(['baselines', '--help'])
  for flag in ('--data-dir', '--synthetic', '--symbols', '--bars',
               '--capital', '--rebalance-days', '--max-weight',
               '--history', '--free-costs', '--json'):
    assert flag in published, (flag, published)


def test_baselines_lists_every_provider(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''The table carries every provider, equal weight included.

  The names are asserted against :mod:`stock_rl.baselines.__all__` rather
  than written out, so adding a sixth baseline fails this test until the
  table shows it. DeMiguel, Garlappi & Uppal (2009) found 1/N is not
  reliably beaten, so a missing control arm is the failure worth
  catching: an untried arm and an absent one look the same.
  '''
  panel = write_panel(tmp_path / 'panel')
  status, out, _ = run_panel(['baselines'], panel, capsys)
  assert status == 0, out
  assert len(baselines.__all__) == 5, baselines.__all__
  for name in baselines.__all__:
    assert name in out, (name, out)
  for column in ('strategy', 'sharpe', 'growth', 'drawdown', 'turnover'):
    assert column in out, (column, out)
  assert out.count('  ok') == 5, out
  assert '* highest Sharpe' in out, out


def test_baselines_json_rows_carry_the_status(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Every row names its status, so an untried arm is distinguishable.

  A row that could be missing makes an untried control arm
  indistinguishable from an absent one, which is the whole failure the
  endpoint and this table exist to prevent.
  '''
  panel = write_panel(tmp_path / 'panel')
  status, out, _ = run_panel(['baselines', '--json'], panel, capsys)
  assert status == 0, out
  payload = json.loads(out)
  assert len(payload['rows']) == len(baselines.__all__)
  assert [row['status'] for row in payload['rows']] == ['ok'] * 5
  assert [row['name'] for row in payload['rows']] == baselines.__all__
  assert payload['panel_kind'] == 'vendor-csv/replay'
  assert payload['not_advice'] == api.not_advice


def test_baselines_without_a_panel_refuses(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''The no-fabrication rule covers ``baselines`` too.

  A table of five Sharpes measured on invented prices looks exactly like a
  result, so the command has to refuse rather than fill the columns.
  '''
  status, out, err = captured(['baselines'], capsys)
  assert out == ''
  assert status == 3, (status, err)
  assert 'no panel available' in err


def test_synthetic_baselines_exit_non_zero(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''A demonstration of the backtester is not a result, so it exits 3.

  The table is printed, because the point of the demo is to show the
  backtester works, and the status is non-zero because the numbers are
  not evidence.
  '''
  status, out, _ = captured(
    ['baselines', '--synthetic', '--bars', '400'], capsys)
  assert 'GENERATED PRICES' in out
  assert status == 3, (status, out)


def test_health_reports_version_and_zero_runtime_deps(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Health names the version and says whether anything is installed on top.

  The dependency line is read from the installed metadata rather than
  from a claim in a README, because a claim is exactly what goes stale
  when somebody adds a dependency.
  '''
  status, out, _ = captured(['health'], capsys)
  assert status == 0
  assert __version__ in out
  assert 'runtime_deps' in out
  assert 'version' in out
  assert 'python' in out


def test_health_json_dependency_line_is_not_a_claim(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''The dependency line is derived, and names its source when unknown.

  ``importlib.metadata`` is the only thing that can be trusted here. A
  hard-coded "0 dependencies" would be true on the day it was written and
  silently false the day somebody added one, which is exactly the failure
  a health check exists to catch.
  '''
  _, out, _ = captured(['health', '--json'], capsys)
  reported = str(json.loads(out)['runtime_dependencies'])
  assert reported, 'the field must never be empty'
  try:
    required = importlib.metadata.requires('stock-rl')
  except importlib.metadata.PackageNotFoundError:
    assert 'not installed' in reported, reported
    return
  if not required:
    assert 'none declared' in reported, reported
  else:
    assert f'{len(required)} declared' in reported, reported


def test_health_reports_data_availability(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Health says what is on disk, and says so when nothing was named.

  Two different facts: a directory that was asked about and held CSVs, and
  an operator who named no directory at all. The second must not be
  answered by searching somewhere they did not point at.
  '''
  panel = write_panel(tmp_path / 'panel', GATE_BARS)
  status, out, _ = captured(['health', '--data-dir', str(panel)], capsys)
  assert status == 0
  for symbol in SYMBOLS:
    assert symbol in out, out
  total = len(SYMBOLS) * GATE_BARS
  assert f'({total} bars total)' in out, out
  for symbol in SYMBOLS:
    assert f'{GATE_BARS} bars' in out, out
  _, plain, _ = captured(['health'], capsys)
  assert 'none named' in plain, plain


def test_health_reports_an_unusable_directory(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''A wrong path is reported as unusable, not as an empty panel.

  "No data" and "the path is wrong" are different operator problems, and
  health reports the difference rather than collapsing both into an empty
  symbol list. It still exits 0: the report succeeded, and the fact it
  reports is a problem with the data, not with the tool.
  '''
  missing = tmp_path / 'nowhere'
  status, out, _ = captured(['health', '--data-dir', str(missing)], capsys)
  assert status == 0
  assert 'unusable' in out, out


def test_health_json_carries_iso_utc_timestamp(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Every timestamp the CLI prints is ISO 8601 in UTC.

  A naive stamp in a run record is unreadable later, and this project's
  whole look-ahead argument is about timestamps that do not say what zone
  they are in.
  '''
  _, out, _ = captured(['health', '--json'], capsys)
  payload = json.loads(out)
  stamp = str(payload['run_at'])
  assert stamp.endswith('Z'), stamp
  assert 'T' in stamp, stamp


def test_skills_lists_every_shipped_file_with_a_tier(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Every shipped ``.md`` is listed with its declared tier.

  Both halves are derived rather than written out. The names come from
  ``discover_skills``, so a file that is shipped but absent from the table
  fails; the tiers come from the schema's own tuple, so a fourth tier --
  which would be a change to the governance model rather than a typo --
  fails too.
  '''
  status, out, _ = captured(['skills'], capsys)
  assert status == 0, out
  shipped = {path.stem: path for path in discover_skills()}
  assert shipped, 'the package must ship at least one skill file'
  for name in shipped:
    assert name in out, (name, out)
  for tier in tiers:
    assert tier in out, (tier, out)
  assert 'activatable' in out, out


def test_skills_json_carries_tier_and_activatability(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''The JSON form carries the tier and the gate's two separate answers.

  ``allowed_live`` is separate from ``active`` on purpose: a skill running
  in experiment mode is genuinely running and must still answer False
  here, which is what stops an experiment result being wired into
  production by a caller that only checked ``active``.
  '''
  _, out, _ = captured(['skills', '--json'], capsys)
  payload = json.loads(out)
  rows = payload['skills']
  assert {row['name'] for row in rows} == {
    skill.name for skill in load_skills()}
  assert {row['name'] for row in rows} == {
    path.stem for path in discover_skills()}
  for row in rows:
    assert row['evidence_tier'] in set(tiers), row
    assert isinstance(row['active'], bool), row
    assert isinstance(row['allowed_live'], bool), row
    assert row['experiment_only'] is False or not row['active'], row
    assert row['kill_criteria'], row
    assert row['reason'], row
    assert row['fingerprint'], row


def test_skills_rejected_never_activate(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''A rejected skill is refused even with every flag in its favour.

  This is the gate's central promise, and it is the one flag combination
  that must not reach it: enable the file, allow experiment mode, and the
  tier still says no. The flags are repeated per name because
  ``--enable`` is ``append``, so one flag takes exactly one value.
  '''
  rejected = [skill.name for skill in load_skills()
              if skill.evidence_tier == 'rejected']
  assert rejected, 'the fixture set must ship a rejected skill'
  flags: list[str] = []
  for name in rejected:
    flags.extend(['--enable', name])
  _, out, _ = captured(
    ['skills', '--json', '--experiment-mode', *flags], capsys)
  for row in json.loads(out)['skills']:
    if row['name'] in rejected:
      assert row['active'] is False, row
      assert row['allowed_live'] is False, row
      assert 'rejected' in row['reason'], row


def test_skills_experimental_runs_only_in_experiment_mode(
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''An experimental skill is active in experiment mode and never live.

  Both halves are asserted, because either alone is satisfiable by a CLI
  that printed a constant. This is the middle tier, and the distinction it
  exists to preserve is between "running" and "may reach an order".
  '''
  experimental = [skill.name for skill in load_skills()
                  if skill.evidence_tier == 'experimental']
  assert experimental, 'the fixture set must ship an experimental skill'
  flags: list[str] = []
  for name in experimental:
    flags.extend(['--enable', name])
  _, live, _ = captured(['skills', '--json', *flags], capsys)
  for row in json.loads(live)['skills']:
    if row['name'] in experimental:
      assert row['active'] is False, row
      assert 'experiment mode' in row['reason'], row
  _, mode, _ = captured(
    ['skills', '--json', '--experiment-mode', *flags], capsys)
  for row in json.loads(mode)['skills']:
    if row['name'] in experimental:
      assert row['active'] is True, row
      assert row['experiment_only'] is True, row
      assert row['allowed_live'] is False, row


def test_unknown_symbol_is_refused(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Naming a symbol the panel does not hold is a refusal, not a subset.

  Silently running the two names that do match would report a comparison
  over a universe the operator did not ask for.
  '''
  panel = write_panel(tmp_path / 'panel')
  status, out, err = captured(
    ['experiment', '--data-dir', str(panel), '--symbols', 'ZZZ'], capsys)
  assert out == ''
  assert status == 3, (status, err)
  assert 'ZZZ' in err, err


def test_symbol_subset_is_run_and_echoed(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''A named subset is the universe that gets run, and it is echoed.

  The header prints the count because a three-symbol run over a
  fifty-symbol panel looks identical to a three-symbol panel until a
  reader is told which one it was.
  '''
  panel = write_panel(tmp_path / 'panel')
  status, out, _ = captured(
    ['experiment', '--data-dir', str(panel),
     '--symbols', 'AAA', 'BBB'], capsys)
  assert status == 1, out
  assert 'symbols     2' in out, out
  assert 'symbols     3' not in out, out


def test_shock_flag_runs_over_the_dependency_graph(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''``--shock`` traverses the Nifty-50 graph instead of being ignored.

  The graph half of the context block is zeros without it, so a run that
  named a shock and silently measured zeros would be a report on
  something other than what was asked for. The node key is checked against
  the graph's own keys rather than assumed.
  '''
  graph = nifty50_seed()
  keys = {str(key) for key in graph.nodes}
  assert 'commodity:crude' in keys, sorted(keys)
  panel = write_panel(tmp_path / 'panel')
  status, out, _ = run_panel(
    ['experiment', '--shock', 'commodity:crude'], panel, capsys)
  assert 'KILL' in out, out
  assert status == 1, (status, out)


def test_no_colour_or_control_characters_in_output(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Output is plain text: no ANSI escapes, no carriage returns.

  A CI log containing escape sequences is unreadable in a file, and a
  spinner that redraws a line turns a captured log into a smear.
  '''
  panel = write_panel(tmp_path / 'panel')
  # --data-dir is only meaningful for the two commands that take one, so
  # it is not appended to health or skills.
  invocations = (
    ['experiment', '--data-dir', str(panel)],
    ['baselines', '--data-dir', str(panel)],
    ['health'],
    ['skills'],
  )
  for argv in invocations:
    _, out, err = captured(argv, capsys)
    for text in (out, err):
      assert '\x1b' not in text, argv
      assert '\r' not in text, argv
      assert '\t' not in text, argv


def test_seeds_flag_reaches_the_report(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''``--seeds`` is passed through, and duplicated seeds are collapsed.

  :class:`~stock_rl.experiment.ArmConfig` refuses duplicate seeds, and a
  user typing ``--seeds 1 1`` has made a typo rather than a request for a
  repeated seed.
  '''
  panel = write_panel(tmp_path / 'panel')
  _, out, _ = run_panel(
    ['experiment', '--json', '--seeds', '1', '1', '2'], panel, capsys)
  assert json.loads(out)['config']['seeds'] == [1, 2]


def test_flags_reach_the_backtester_settings(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''Every backtester flag lands in the report's echoed configuration.

  The report echoes the settings so a reader can attribute a number to the
  parameters that produced it. A flag that parsed but was never passed on
  would leave a Sharpe with no configuration attached, which is the drift
  the API's control arm already had to be fixed for.
  '''
  panel = write_panel(tmp_path / 'panel')
  _, out, _ = run_panel(
    ['experiment', '--json', '--capital', '5000000',
     '--rebalance-days', '5', '--holdings', '2', '--max-weight', '0.25',
     '--free-costs'], panel, capsys)
  config = json.loads(out)['config']
  assert config['capital'] == 5_000_000.0, config
  assert config['rebalance_days'] == 5, config
  assert config['holdings'] == 2, config
  assert config['max_weight'] == 0.25, config
  # --free-costs is asserted against the cost model itself rather than a
  # flag, because a zeroed charge and an absent one differ in the payload.
  assert all(value == 0.0 for value in config['costs'].values()), config


def test_llm_scalar_flag_reaches_arm_c(
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  '''``--llm-scalar`` is the value arm C is injected, not a decoration.

  Arm C is refused outright without a scalar per symbol, so this is the
  flag that makes the third arm runnable at all from a shell. The value is
  read back off the comparison, where it shows up as the arm C treatment
  label.
  '''
  panel = write_panel(tmp_path / 'panel')
  _, out, _ = run_panel(
    ['experiment', '--json', '--llm-scalar', '-0.4'], panel, capsys)
  payload = json.loads(out)
  labels = [item['label'] for item in payload['comparisons']]
  assert 'price_context_minus_a' in labels, labels
  assert len(payload['arms']) == 3, payload['arms']


def test_version_flag_prints_and_exits_zero() -> None:
  '''``--version`` reports the package version and stops.

  A version flag that continued into the dispatcher would run a
  subcommand nobody asked for.
  '''
  buffer = io.StringIO()
  with contextlib.redirect_stdout(buffer):
    with pytest.raises(SystemExit) as caught:
      cli.main(['--version'])
  assert caught.value.code == 0
  assert __version__ in buffer.getvalue()


def test_module_entry_point_exists() -> None:
  '''``python -m stock_rl`` reaches :func:`stock_rl.cli.main`.

  Without ``src/stock_rl/__main__.py`` the module exists but cannot be
  run, so every documented invocation form in the docstring is a lie. The
  file is read rather than imported, because importing it would execute
  ``main`` against pytest's own argv.
  '''
  source = Path(cli.__file__).with_name('__main__.py')
  assert source.is_file(), source
  text = source.read_text(encoding='utf-8')
  assert 'from stock_rl.cli import main' in text, text
  assert 'raise SystemExit(main())' in text, text
