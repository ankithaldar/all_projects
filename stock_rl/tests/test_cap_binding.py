#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''One cap, bound by keyword, reported as the cap that was applied.

A caller asks for ``max_weight=0.30`` and is handed a book built at 0.10
under a payload claiming 0.30. Every provider in
:mod:`stock_rl.baselines` carries its own ``max_weight`` default of 0.10,
and :func:`stock_rl.pipeline.run_baselines` and
:func:`stock_rl.cli._baseline_row` each passed their provider to
``run_portfolio`` unbound, so the provider's own default was the binding
constraint and the requested cap was decoration. The measured symptom was
that two different requested caps returned one byte-identical Sharpe.

:mod:`stock_rl.api` fixed this first. This file pins the same property at
all three seams that report a cap -- the HTTP surface, the composition
layer and the command line -- against one shared implementation,
:mod:`stock_rl.weights`, rather than against three that agree today.

What is asserted, and why each is written to bite:

  * **Reported equals applied, per baseline, per seam, at several caps.**
    Both numbers are read off the result: the reported one is what the
    payload claims was asked for and the applied one is
    :func:`stock_rl.weights.applied_weight_cap` over the engine's own
    weight snapshots. A test comparing two *reported* numbers would pass
    on the broken code, which reported every cap faithfully.
  * **Above ``1/N`` the construction binds, so applied drops BELOW
    reported.** A cap is a ceiling and not a target. Four symbols at 0.30
    cannot each hold 0.30, so 1/N is what the book is built at and the
    applied number sits strictly under the reported one. The
    asymmetry is asserted explicitly, and the applied number is never
    above the reported one at any cap.
  * **The cap is bound by keyword.** A spy splits ``*args`` from
    ``**kwargs``, and a fake provider shaped exactly like
    ``trend_filtered_momentum`` proves ``window``, ``ma_window``, ``skip``
    and ``top`` are all unmoved. Positionally, 0.30 landing in
    ``ma_window`` computes ``sma(closes, 0.3)`` and raises nothing.
  * **A provider that cannot take a cap is marked, not refused.** It is
    measured, ``cap_bound`` is False, and the row reports the cap the
    book was really built at rather than the one nobody applied.

Every negative test has a positive control beside it: the same cap run
twice is one book, the shipped providers bind, a ``**kwargs`` provider
does take the cap, and a binding engine still clamps a provider that
overreaches.
'''

from __future__ import annotations

# ``cli._baseline_row`` is private, and this file calls it directly on
# purpose. It is the only seam at which a provider can be handed to the
# command line as an argument rather than discovered from a registry, so
# it is the only way to drive that seam with a spy, with a provider shaped
# like ``trend_filtered_momentum`` and with a provider that takes no cap.
# The public path -- ``baselines --json`` over a CSV panel directory -- is
# exercised in ``tests/test_cli.py``; what is under test here is the
# binding at this particular call boundary.
# pylint: disable=protected-access

import argparse
import inspect
import json
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from random import Random
from types import SimpleNamespace

import pytest

from stock_rl import api, baselines, cli, pipeline, weights
from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY
from stock_rl.data.vendors import Segment
from stock_rl.weights import (
  CAP_KEYWORD,
  accepts_weight_cap,
  applied_weight_cap,
  bind_weight_cap,
)

#: Exchange local time. Bars are stamped at the 15:30 close, inside the
#: session, so nothing here is dropped by a calendar.
IST = timezone(timedelta(hours=5, minutes=30))

#: First date of the generated history, a Monday.
FIRST = datetime(2024, 1, 1, tzinfo=IST)

#: Four symbols. More than two, so the momentum ranking has something to
#: order, and few enough that ``1/N`` is 0.25 -- a number the caps below
#: straddle, which is what makes the ceiling argument testable.
SYMBOLS = ('AAA', 'BBB', 'CCC', 'DDE')

#: Bars per panel. Long enough that ``momentum_ranked`` is live from the
#: first rebalance at :data:`HISTORY`, rather than silently degenerating
#: to equal weight and making the comparison vacuous.
BARS = 900

#: Warm-up bars. At the shipped default of 60 momentum needs 274 visible
#: bars and would not be scored at all until bar 274.
HISTORY = 300

#: The 1/N share of :data:`SYMBOLS`. The pivot of the whole file: below
#: it the cap is the binding constraint, above it the construction is.
EQUAL_SHARE = 1.0 / len(SYMBOLS)

#: Caps below :data:`EQUAL_SHARE`, where the cap binds and the reported
#: number is expected to equal the applied one. 0.10 is the default every
#: provider silently imposed, so a book that matches it under a request
#: for 0.20 is the defect rather than an accident.
BINDING_CAPS = (0.10, 0.15, 0.20)

#: A cap above :data:`EQUAL_SHARE` and above every provider's own
#: construction, so nothing in this file's universe can be built at it.
#: Four symbols cannot each hold a third of capital, and the heaviest
#: inverse-volatility share is well under half, so at this cap the
#: applied number must come out strictly below the reported one for all
#: five baselines. 0.30 would not do: at 0.30 the inverse-volatility
#: book can still reach the cap, and :data:`MIXED_CAP` exists to pin
#: that difference down.
CEILING_CAP = 0.50

#: A cap between :data:`EQUAL_SHARE` and :data:`CEILING_CAP`. It binds
#: for ``low_volatility``, whose heaviest share exceeds it, and not for
#: the four 1/N baselines, whose share does not. One number, two
#: answers, which is what "a cap is a ceiling" looks like from outside.
MIXED_CAP = 0.30

#: The cap the reported-versus-applied comparison is made at for the fake
#: providers. It is :data:`EQUAL_SHARE`, so a fake that returns an equal
#: book reports exactly what it was asked for and the two numbers are
#: expected to agree rather than to be quietly different.
FAKE_CAP = EQUAL_SHARE

#: The seams that report a cap, by name. Each returns the reported cap
#: and one row per provider, in the same shape, so the assertions below
#: are written once and run at all three.
SEAMS = ('api', 'pipeline', 'cli')

#: A weight provider, as this file names one.
Provider = Callable[[dict[str, list[Bar]]], dict[str, float]]


def trading_dates(count: int) -> list[date]:
  '''Return ``count`` weekday dates from :data:`FIRST`.

  Args:
    count: How many dates are needed.

  Returns:
    Dates in ascending order, every one a weekday.
  '''
  found = []
  day = FIRST
  while len(found) < count:
    if day.weekday() < 5:
      found.append(day.date())
    day += timedelta(days=1)
  return found


@lru_cache(maxsize=1)
def universe() -> dict[str, list[Bar]]:
  '''Return the one panel every test in this file is measured over.

  **SYNTHETIC**: these are generated prices. Every number computed on
  them is a statement about this repository's code and about no market at
  all.

  Each symbol has its own drift and its own volatility, so the momentum
  ranking is strict, ``trend_filtered_momentum``'s regime gate has
  something to let through and something to reject, and
  ``low_volatility`` produces a book that is not equal weight. Four
  symbols that all behaved alike would make every cap assertion below
  true for the wrong reason.

  Returns:
    Mapping of symbol to bars on a shared timeline.
  '''
  days = trading_dates(BARS)
  panels = {}
  for index, symbol in enumerate(SYMBOLS):
    source = Random(11 + index)
    drift = 0.0012 - 0.0003 * index
    swing = 0.010 + 0.004 * index
    price = 100.0 + 40.0 * index
    previous = price
    series = []
    for day in days:
      price = max(1.0, price * (1.0 + drift + source.gauss(0.0, swing)))
      series.append(Bar(
        datetime(day.year, day.month, day.day, 15, 30, tzinfo=IST),
        previous,
        max(previous, price) * 1.003,
        min(previous, price) * 0.997,
        price,
        1e5,
      ))
      previous = price
    panels[symbol] = series
  return panels


def shipped() -> dict[str, Provider]:
  '''Return the five shipped baselines, by name.

  Returns:
    Mapping of provider name to provider callable, discovered from
      :data:`stock_rl.baselines.__all__` so a sixth baseline fails the
      count assertions rather than being quietly skipped.
  '''
  return {name: getattr(baselines, name) for name in baselines.__all__}


def make_panel() -> pipeline.Panel:
  '''Wrap :func:`universe` as the panel the composition layer wants.

  Returns:
    A :class:`stock_rl.pipeline.Panel` over the generated bars.
  '''
  panels = universe()
  first = panels[SYMBOLS[0]]
  return pipeline.Panel(
    panels=panels,
    kind=pipeline.PanelKind.SYNTHETIC,
    vendor='cap-binding-probe',
    segments=frozenset({Segment.CASH}),
    symbols=tuple(sorted(panels)),
    bars=len(first),
    start=first[0].timestamp,
    end=first[-1].timestamp,
    token='cap-binding-probe-token',
  )


def service() -> api.ApiService:
  '''Return an API service over :func:`universe`, with frozen clocks.

  Returns:
    The service. Uptime would otherwise vary between runs.
  '''
  return api.ApiService(
    panels=universe(),
    clock=lambda: 0.0,
    now=lambda: datetime(2026, 3, 4, 9, 30),
  )


def settings(cap: float) -> argparse.Namespace:
  '''Return the CLI argument namespace a caller would produce.

  Args:
    cap: The per-symbol cap requested on the command line.

  Returns:
    The namespace, carrying only the fields a baseline row reads.
  '''
  return argparse.Namespace(
    capital=10_000_000.0,
    rebalance_days=21,
    max_weight=cap,
    history=HISTORY,
  )


def read(cap: float) -> tuple[float, dict[str, dict]]:
  '''Return the HTTP rows measured at ``cap``.

  Args:
    cap: Requested per-symbol cap, sent the way a caller sends it.

  Returns:
    The reported cap and one mapping per provider.
  '''
  response = api.dispatch(
    service(), 'GET',
    f'/api/baselines?max_weight={cap}&history={HISTORY}')
  assert response.status == 200, response.body
  payload = json.loads(response.body)
  return payload['max_weight'], {
    row['name']: row for row in payload['strategies']
  }


def compose(cap: float, names: tuple[str, ...] = ()) -> tuple[float,
                                                               dict]:
  '''Return the composition-layer rows measured at ``cap``.

  Args:
    cap: Requested per-symbol cap.
    names: Baselines to measure, or every shipped one when empty.

  Returns:
    The reported cap and one mapping per provider, from
      :func:`stock_rl.pipeline.run_baselines`.
  '''
  table = pipeline.run_baselines(
    make_panel(), pipeline.RunSpec(max_weight=cap, history=HISTORY),
    names=names or None)
  return table.spec.max_weight, {
    run.name: {
      'status': 'ok',
      'cap_bound': run.cap_bound,
      'applied_max_weight': run.applied_max_weight,
      'sharpe': run.sharpe,
    } for run in table.runs
  }


def command(cap: float) -> tuple[float, dict[str, dict]]:
  '''Return the command-line rows measured at ``cap``.

  Args:
    cap: Requested per-symbol cap.

  Returns:
    The reported cap and one mapping per provider, from
      :func:`stock_rl.cli._baseline_row`.
  '''
  namespace = settings(cap)
  rows = {}
  for name, provider in shipped().items():
    rows[name] = cli._baseline_row(
      name, provider, universe(), namespace, DELIVERY)
  return namespace.max_weight, rows


def _api_seam(registry: Mapping[str, Provider],
              cap: float) -> tuple[float, dict[str, dict]]:
  '''Return the HTTP rows for an arbitrary provider registry.

  Args:
    registry: Providers to expose as the control arm.
    cap: Requested per-symbol cap.

  Returns:
    The reported cap and one mapping per provider.
  '''
  with pytest.MonkeyPatch.context() as patch:
    patch.setattr(api, 'strategies', lambda: dict(registry))
    return read(cap)


def _pipeline_seam(registry: Mapping[str, Provider],
                   cap: float) -> tuple[float, dict[str, dict]]:
  '''Return the composition-layer rows for an arbitrary registry.

  Args:
    registry: Providers to expose as the control arm. One name is padded
      out to two, because :func:`stock_rl.pipeline.run_baselines` refuses
      a single row: the length gate needs two configurations to deflate
      against.
    cap: Requested per-symbol cap.

  Returns:
    The reported cap and one mapping per provider.
  '''
  table_registry = dict(registry)
  if len(table_registry) < 2:
    table_registry['equal_weight'] = baselines.equal_weight
  stub = SimpleNamespace(__all__=tuple(table_registry), **table_registry)
  with pytest.MonkeyPatch.context() as patch:
    patch.setattr(pipeline, 'baselines', stub)
    return compose(cap, tuple(table_registry))


def _cli_seam(registry: Mapping[str, Provider],
              cap: float) -> tuple[float, dict[str, dict]]:
  '''Return the command-line rows for an arbitrary provider registry.

  Args:
    registry: Providers to run. This seam takes the callable as an
      argument rather than discovering it, so nothing needs patching.
    cap: Requested per-symbol cap.

  Returns:
    The reported cap and one mapping per provider.
  '''
  namespace = settings(cap)
  return namespace.max_weight, {
    name: cli._baseline_row(name, provider, universe(), namespace, DELIVERY)
    for name, provider in registry.items()
  }


#: Each seam behind one callable, keyed by the name it is parametrised
#: under. Every real-baseline seam is cached per cap, because a
#: backtest over 900 bars costs real time and the answers do not change
#: within a run; the registry seams take an unhashable argument and are
#: deliberately left uncached.
SEAM_RUNNERS = {
  'api': lru_cache(maxsize=None)(read),
  'pipeline': lru_cache(maxsize=None)(compose),
  'cli': lru_cache(maxsize=None)(command),
}

SEAM_REGISTRY_RUNNERS = {
  'api': _api_seam,
  'pipeline': _pipeline_seam,
  'cli': _cli_seam,
}


def run_seam(seam: str, cap: float) -> tuple[float, dict[str, dict]]:
  '''Return the five shipped baselines' rows at one seam and one cap.

  Args:
    seam: One of :data:`SEAMS`.
    cap: Requested per-symbol cap.

  Returns:
    The reported cap and one mapping per provider.
  '''
  return SEAM_RUNNERS[seam](cap)


def run_registry(seam: str, registry: Mapping[str, Provider],
                 cap: float) -> tuple[float, dict[str, dict]]:
  '''Return an arbitrary provider's row at one seam and one cap.

  Args:
    seam: One of :data:`SEAMS`.
    registry: Providers to expose as the control arm.
    cap: Requested per-symbol cap.

  Returns:
    The reported cap and one mapping per provider.
  '''
  return SEAM_REGISTRY_RUNNERS[seam](registry, cap)


@lru_cache(maxsize=None)
def sharpe_at(seam: str, cap: float, name: str) -> float | None:
  '''Return one baseline's Sharpe, read through no field this file added.

  Deliberately narrow. Every adapter above reports the cap by reading
  ``applied_max_weight`` and ``cap_bound``, which is what makes it
  compare a reported number against a measured one -- and also what
  makes it fail with a missing attribute rather than a wrong value if
  run against the unfixed code. This helper reads ``sharpe`` and nothing
  else, so it can be run against the code as it was and its failure is a
  value mismatch: two different requested caps, one number.

  Args:
    seam: One of :data:`SEAMS`.
    cap: Requested per-symbol cap.
    name: Provider name to read.

  Returns:
    The reported Sharpe, or None when the length gate refused the row.
  '''
  if seam == 'api':
    reported, rows = run_seam(seam, cap)
    assert reported == cap
    return rows[name]['sharpe']
  if seam == 'cli':
    return cli._baseline_row(
      name, shipped()[name], universe(), settings(cap), DELIVERY)['sharpe']
  table = pipeline.run_baselines(
    make_panel(), pipeline.RunSpec(max_weight=cap, history=HISTORY))
  return table.get(name).sharpe


def spy(seen: list[tuple[tuple, dict]]) -> Provider:
  '''Return a provider recording how the cap was passed to it.

  Args:
    seen: Appended to with one ``(args, kwargs)`` pair per call.

  Returns:
    A provider taking ``panels`` plus anything else, which is the shape
      that makes a positional cap visible rather than silent.
  '''
  def recorded(
    panels: dict[str, list[Bar]],
    *args,
    **kwargs,
  ) -> dict[str, float]:
    '''Return an equal-weight book, recording the call shape.

    The book is equal weight on purpose and not the requested cap: the
    point of this provider is the *shape* of the call, and an equal book
    at :data:`FAKE_CAP` makes ``applied == reported`` true here for a
    reason that has nothing to do with which argument position was used.

    Args:
      panels: Visible price panels.
      *args: Recorded positionally.
      **kwargs: Recorded by keyword.

    Returns:
      One weight per symbol.
    '''
    seen.append((args, dict(kwargs)))
    return {symbol: 1.0 / len(panels) for symbol in panels}
  return recorded


def shaped(seen: dict) -> Provider:
  '''Return a provider shaped exactly like ``trend_filtered_momentum``.

  Args:
    seen: Updated in place with every parameter the call bound.

  Returns:
    A provider whose cap is the sixth parameter, so a positional cap
      would land on ``ma_window`` rather than on anything named
      ``max_weight``.
  '''
  def trend(panels: dict[str, list[Bar]], window: int = 252,
            skip: int = 21, top: int = 10, ma_window: int = 40,
            max_weight: float = 0.10) -> dict[str, float]:
    '''Return one weight per symbol, recording what the call bound.

    Args:
      panels: Visible price panels.
      window: Momentum lookback in bars.
      skip: Most recent bars excluded from momentum.
      top: Names held.
      ma_window: Moving-average window for the regime gate.
      max_weight: Cap on any single weight.

    Returns:
      One weight per symbol at whatever cap the caller supplied.
    '''
    seen.update(window=window, skip=skip, top=top, ma_window=ma_window,
                max_weight=max_weight)
    return {symbol: max_weight for symbol in panels}
  return trend


def capless(own: float) -> Provider:
  '''Return a provider that takes no cap and applies one of its own.

  Args:
    own: The cap this provider imposes whatever it is asked for.

  Returns:
    A one-argument provider.
  '''
  def fixed(panels: dict[str, list[Bar]]) -> dict[str, float]:
    '''Return a book at the provider's own cap.

    Args:
      panels: Visible price panels.

    Returns:
      One weight per symbol.
    '''
    return {symbol: own for symbol in panels}
  return fixed


def greedy() -> Provider:
  '''Return a provider that reaches four times the requested cap.

  Returns:
    A one-argument provider, funding a single name well above any cap a
      caller might ask for, so the engine's clamp is what has to hold.
  '''
  def overreaching(panels: dict[str, list[Bar]]) -> dict[str, float]:
    '''Return one overweight name and nothing else.

    Args:
      panels: Visible price panels.

    Returns:
      A single-name book at 0.90, which no cap below 0.90 can allow.
    '''
    return {sorted(panels)[0]: 0.90}
  return overreaching


def loose(seen: list[dict]) -> Provider:
  '''Return a provider accepting any keyword.

  Args:
    seen: Appended to with the keywords of each call.

  Returns:
    A provider that builds its book from whatever cap it is handed, and
      is therefore the positive control for ``**kwargs`` counting as
      accepting one.
  '''
  def anything(panels: dict[str, list[Bar]], **kwargs) -> dict[str, float]:
    '''Return a book at whichever cap was supplied.

    Args:
      panels: Visible price panels.
      **kwargs: Recorded by keyword.

    Returns:
      One weight per symbol at the supplied cap.
    '''
    seen.append(dict(kwargs))
    cap = kwargs.get(CAP_KEYWORD, 0.05)
    return {symbol: cap for symbol in panels}
  return anything


def opaque() -> Provider:
  '''Return a provider that will not describe its own signature.

  Returns:
    A callable object standing in for a C builtin behind a decorator,
      whose ``__signature__`` raises rather than answering the question
      :func:`stock_rl.weights.accepts_weight_cap` asks.
  '''
  class Opaque:
    '''A provider that refuses to be introspected.'''

    @property
    def __signature__(self) -> None:
      '''Raise rather than describe.

      Returns:
        Never returns.

      Raises:
        ValueError: Always, which is the point.
      '''
      raise ValueError('this callable will not describe itself')

    def __call__(self, panels: dict[str, list[Bar]]) -> dict[str, float]:
      '''Return one weight per symbol at the provider's own cap.

      Args:
        panels: Visible price panels.

      Returns:
        One weight per symbol.
      '''
      return {symbol: 0.05 for symbol in panels}

  return Opaque()


class TestReportedEqualsApplied:
  '''The number a payload reports is the number the book was built at.'''

  @pytest.mark.parametrize('name', sorted(baselines.__all__))
  @pytest.mark.parametrize('cap', BINDING_CAPS)
  @pytest.mark.parametrize('seam', SEAMS)
  def test_every_baseline_is_built_at_the_cap_it_reports(self, seam, cap,
                                                        name):
    '''Reported equals applied, per baseline, per cap, per seam.

    Three symbols at 0.20 is 0.60 of capital, so the cap rather than the
    1/N share is what binds and equality is the only correct answer.
    Under the defect the applied number was 0.10 whatever the payload
    said, which is what let two caps return one Sharpe.

    Args:
      seam: Which reporting surface is under test.
      cap: Requested per-symbol cap.
      name: Provider name under test.
    '''
    reported, rows = run_seam(seam, cap)
    assert reported == cap
    row = rows[name]
    assert row['status'] == 'ok', row.get('error')
    applied = row['applied_max_weight']
    assert applied == pytest.approx(cap), (
      f'{seam} told {name} the cap was {cap} and built the book at '
      f'{applied}')
    assert row['cap_bound'] is True

  @pytest.mark.parametrize('name', sorted(baselines.__all__))
  @pytest.mark.parametrize('seam', SEAMS)
  def test_a_cap_above_the_construction_binds_the_construction(
      self, seam, name):
    '''A ceiling is not a target: applied drops BELOW reported.

    No provider in this universe can be built at 0.50 -- four symbols
    cannot each hold a third of capital, and the heaviest
    inverse-volatility share is under half -- so the heaviest weight
    must come out strictly under the number that was requested. The
    asymmetry is the property, so it is asserted as a strict inequality
    rather than excused as rounding.

    Args:
      seam: Which reporting surface is under test.
      name: Provider name under test.
    '''
    reported, rows = run_seam(seam, CEILING_CAP)
    assert reported == CEILING_CAP
    row = rows[name]
    assert row['status'] == 'ok', row.get('error')
    applied = row['applied_max_weight']
    assert applied is not None, row
    assert applied < reported, (
      f'{seam} reported {reported} for {name} and built {applied}; nothing '
      f'in this universe can reach that cap, so equality here would mean '
      f'the two numbers are not about the same book')
    assert row['cap_bound'] is True

  @pytest.mark.parametrize('seam', SEAMS)
  def test_the_one_over_n_book_lands_on_one_over_n(self, seam):
    '''The exact number each construction lands on at the ceiling.

    Stated per baseline rather than as one inequality, because "less
    than the cap" is true of any book and therefore weak: the four 1/N
    providers must sit exactly at :data:`EQUAL_SHARE`, and
    ``low_volatility`` must sit at its own heaviest inverse-volatility
    share, which is neither the cap nor the 1/N share. Both are
    consequences of the same fact -- the cap never binds above this
    point -- and both are falsifiable numbers rather than bounds.

    Args:
      seam: Which reporting surface is under test.
    '''
    reported, rows = run_seam(seam, CEILING_CAP)
    equal_share_names = (
      'buy_and_hold', 'equal_weight', 'momentum_ranked',
      'trend_filtered_momentum',
    )
    for name in equal_share_names:
      assert rows[name]['applied_max_weight'] == pytest.approx(
        EQUAL_SHARE), (seam, name, rows[name])
      assert EQUAL_SHARE < reported, (seam, name)
    inverse = rows['low_volatility']['applied_max_weight']
    assert inverse > EQUAL_SHARE, (seam, inverse)
    assert inverse < reported, (seam, inverse)

  @pytest.mark.parametrize('seam', SEAMS)
  def test_only_the_construction_that_can_reach_the_cap_binds(self, seam):
    '''One number, two answers: 0.30 binds for one provider and not four.

    This is the asymmetry stated where it is sharpest. ``low_volatility``
    reaches 0.30 with its heaviest share, so its applied number equals
    its reported one; the four 1/N providers cannot, so theirs sits at
    :data:`EQUAL_SHARE`. A payload that reported a single cap for all
    five would be asserting that the cap bound for all of them, and it
    did not.

    Args:
      seam: Which reporting surface is under test.
    '''
    reported, rows = run_seam(seam, MIXED_CAP)
    assert reported == MIXED_CAP
    inverse = rows['low_volatility']['applied_max_weight']
    assert inverse == pytest.approx(MIXED_CAP), (seam, inverse)
    for name in ('buy_and_hold', 'equal_weight', 'momentum_ranked',
                 'trend_filtered_momentum'):
      assert rows[name]['applied_max_weight'] == pytest.approx(
        EQUAL_SHARE), (seam, name, rows[name])
      assert rows[name]['applied_max_weight'] < reported, (seam, name)

  @pytest.mark.parametrize('cap', BINDING_CAPS + (CEILING_CAP,))
  @pytest.mark.parametrize('seam', SEAMS)
  def test_the_applied_cap_is_never_above_the_reported_one(self, seam, cap):
    '''The engine is a ceiling no provider can talk its way past.

    Args:
      seam: Which reporting surface is under test.
      cap: Requested per-symbol cap.
    '''
    reported, rows = run_seam(seam, cap)
    assert rows
    for name, row in rows.items():
      if row['status'] != 'ok':
        continue
      assert row['applied_max_weight'] <= reported + 1e-12, (seam, name, row)

  @pytest.mark.parametrize('name', sorted(baselines.__all__))
  def test_two_caps_are_two_sharpes_that_differ(self, name):
    '''The symptom as it was measured: one Sharpe for two requested caps.

    Read through :func:`sharpe_at`, which touches no field this file
    added, so the failure against the unfixed code is a value mismatch
    rather than a missing attribute. Every seam is checked, because every
    seam reported the requested cap faithfully while building the same
    book underneath -- so a payload-only test would have passed on the
    broken code, which reported all three caps faithfully.

    Args:
      name: Provider name under test.
    '''
    for seam in SEAMS:
      narrow = sharpe_at(seam, BINDING_CAPS[0], name)
      wide = sharpe_at(seam, BINDING_CAPS[-1], name)
      assert narrow is not None and wide is not None, (
        f'the length gate refused a {name} row at the {seam} seam, so '
        f'there is no Sharpe to compare')
      assert narrow != wide, (
        f'{seam} returned one Sharpe {narrow} for {name} at both '
        f'{BINDING_CAPS[0]} and {BINDING_CAPS[-1]}, so the requested cap '
        f'never reached the weights')

  @pytest.mark.parametrize('name', sorted(baselines.__all__))
  def test_two_caps_are_two_books(self, name):
    '''The same symptom read off the books rather than the returns.

    A Sharpe can coincide for two books on a short window by chance; two
    weight vectors cannot. The applied caps are therefore compared too,
    and both are required to equal the cap that was requested.

    Args:
      name: Provider name under test.
    '''
    for seam in SEAMS:
      narrow_reported, narrow = run_seam(seam, BINDING_CAPS[0])
      wide_reported, wide = run_seam(seam, BINDING_CAPS[-1])
      assert narrow_reported == BINDING_CAPS[0]
      assert wide_reported == BINDING_CAPS[-1]
      assert narrow[name]['applied_max_weight'] == pytest.approx(
        BINDING_CAPS[0]), (seam, name, narrow[name])
      assert wide[name]['applied_max_weight'] == pytest.approx(
        BINDING_CAPS[-1]), (seam, name, wide[name])
      assert (narrow[name]['applied_max_weight']
              != wide[name]['applied_max_weight']), (seam, name)

  def test_one_cap_twice_is_one_book(self):
    '''The positive control, without which the test above proves nothing.

    The seam runners are cached per cap, so this deliberately asks the
    uncached path for the same cap twice and requires the two answers to
    be identical. A cache, a mutated provider or a drifting clock would
    all show up here and none of them would show up in a comparison
    between two different caps.
    '''
    first = compose(0.20)
    second = compose(0.20)
    assert first == second
    assert all(row['applied_max_weight'] == pytest.approx(0.20)
               for row in first[1].values()), first

  def test_the_cap_is_a_function_of_the_request_not_a_constant(self):
    '''Three requested caps, three applied caps, per seam.

    Under the defect the applied number was 0.10 for every requested
    value, at every seam, which is the whole defect stated as one
    sentence.
    '''
    for seam in SEAMS:
      applied = {}
      for cap in BINDING_CAPS:
        for name, row in run_seam(seam, cap)[1].items():
          applied.setdefault(name, []).append(row['applied_max_weight'])
      assert applied, seam
      for name, values in applied.items():
        assert len(values) == len(BINDING_CAPS), (seam, name)
        assert len(set(values)) == len(BINDING_CAPS), (
          f'{seam} built {name} at one weight for {len(BINDING_CAPS)} '
          f'different requested caps: {values}')
        for value in values:
          assert value == pytest.approx(BINDING_CAPS[
              values.index(value)]), (seam, name, values)

  def test_the_three_seams_agree_on_one_book(self):
    '''One implementation, so the three surfaces cannot drift apart.

    The HTTP row, the composition row and the command-line row each
    report a cap for the same five baselines over the same panel. If
    they ever disagree it is because one of them stopped using the
    shared helper, which is exactly the failure a shared helper exists
    to prevent.
    '''
    for cap in BINDING_CAPS:
      api_reported, api_rows = run_seam('api', cap)
      pipe_reported, pipe_rows = run_seam('pipeline', cap)
      cli_reported, cli_rows = run_seam('cli', cap)
      assert api_reported == pipe_reported == cli_reported == cap
      for name in sorted(baselines.__all__):
        applied = {
          api_rows[name]['applied_max_weight'],
          pipe_rows[name]['applied_max_weight'],
          cli_rows[name]['applied_max_weight'],
        }
        assert len(applied) == 1, (cap, name, applied)
        assert applied.pop() == pytest.approx(cap), (cap, name)


class TestTheCapIsBoundByKeyword:
  '''By keyword, because the five signatures do not agree on position.'''

  @pytest.mark.parametrize('seam', SEAMS)
  def test_the_cap_arrives_as_a_keyword_not_a_positional(self, seam):
    '''A spy splits ``*args`` from ``**kwargs``, so the two cannot blur.

    Positionally, 0.30 in fifth place feeds ``trend_filtered_momentum``'s
    ``ma_window`` and computes ``sma(closes, 0.3)``, which raises
    nothing and prints nothing. Only the recorded keywords show it.

    Args:
      seam: Which reporting surface is under test.
    '''
    seen: list[tuple[tuple, dict]] = []
    reported, rows = run_registry(
      seam, {'spy': spy(seen), 'companion': spy([])}, FAKE_CAP)
    assert reported == FAKE_CAP
    assert rows['spy']['status'] == 'ok', rows['spy'].get('error')
    assert seen, f'the {seam} seam never called the provider'
    for args, kwargs in seen:
      assert args == (), (
        f'{seam} passed the cap positionally as {args!r}; one reordering '
        f'of one baseline signature would feed it to the wrong parameter')
      assert kwargs == {CAP_KEYWORD: FAKE_CAP}, (seam, kwargs)
    assert rows['spy']['applied_max_weight'] == pytest.approx(FAKE_CAP)

  @pytest.mark.parametrize('seam', SEAMS)
  def test_a_shaped_provider_keeps_every_window_it_declares(self, seam):
    '''A fake shaped exactly like ``trend_filtered_momentum``.

    Its fifth parameter is ``ma_window`` and its sixth is the cap. A
    positional call lands on the moving-average window and computes an
    average over a fraction of a bar, so the test asserts that
    ``window``, ``ma_window``, ``skip`` and ``top`` all arrive at the
    defaults this module declares, and that only ``max_weight`` moved.

    Args:
      seam: Which reporting surface is under test.
    '''
    seen: dict = {}
    reported, rows = run_registry(
      seam, {'shaped': shaped(seen), 'companion': shaped({})}, FAKE_CAP)
    assert reported == FAKE_CAP
    assert rows['shaped']['status'] == 'ok', rows['shaped'].get('error')
    assert seen, f'the {seam} seam never called the provider'
    assert seen['window'] == 252, seen
    assert seen['ma_window'] == 40, seen
    assert seen['skip'] == 21, seen
    assert seen['top'] == 10, seen
    assert seen['max_weight'] == pytest.approx(FAKE_CAP), seen
    assert rows['shaped']['applied_max_weight'] == pytest.approx(FAKE_CAP)

  def test_the_real_signatures_put_the_cap_where_a_positional_would_not(
      self):
    '''The five shipped signatures, read rather than assumed.

    The offsets are the reason this file binds by keyword: second for
    ``equal_weight`` and ``buy_and_hold``, third for ``low_volatility``,
    fifth for ``momentum_ranked`` and sixth for
    ``trend_filtered_momentum``. A refactor that moved one of them
    would not raise anything at the call site.
    '''
    offsets = {
      name: list(
        inspect.signature(getattr(baselines, name)).parameters
      ).index(CAP_KEYWORD)
      for name in baselines.__all__
    }
    assert offsets == {
      'buy_and_hold': 1,
      'equal_weight': 1,
      'low_volatility': 2,
      'momentum_ranked': 4,
      'trend_filtered_momentum': 5,
    }, offsets
    assert (
      inspect.signature(
        baselines.trend_filtered_momentum).parameters['ma_window'].default
      == 40), 'the moving-average window moved, so this hazard is stale'

  @pytest.mark.parametrize('name', sorted(baselines.__all__))
  def test_every_shipped_provider_binds_the_cap_by_keyword(self, name):
    '''The positive control: all five admit it, and admit it by name.

    ``signature.bind`` would raise here rather than swallow the cap, so
    this is checked against the real objects rather than against a list
    of parameter names written out by hand.
    '''
    provider = getattr(baselines, name)
    assert accepts_weight_cap(provider) is True, name
    assert bind_weight_cap(provider, FAKE_CAP)(universe()) != {}
    inspect.signature(provider).bind({}, **{CAP_KEYWORD: FAKE_CAP})


class TestAProviderThatCannotTakeACap:
  '''Measured and marked, rather than refused.'''

  @pytest.mark.parametrize('seam', SEAMS)
  def test_it_is_measured_and_marked_unbound(self, seam):
    '''An unbound provider still gets a row, and the row says why.

    The control arm is discovered from ``baselines.__all__``, so a
    provider these seams cannot introspect may simply derive weights
    from the panels and hold no opinion about the cap at all. Refusing
    the row would file that as a defect in the wrong module.

    Args:
      seam: Which reporting surface is under test.
    '''
    provider = capless(0.05)
    reported, rows = run_registry(
      seam, {'plain': provider, 'companion': capless(0.05)}, CEILING_CAP)
    assert reported == CEILING_CAP
    row = rows['plain']
    assert row['status'] == 'ok', row.get('error')
    assert row['cap_bound'] is False
    assert accepts_weight_cap(provider) is False
    assert row['applied_max_weight'] == pytest.approx(0.05)

  @pytest.mark.parametrize('seam', SEAMS)
  def test_its_applied_cap_is_disclosed_rather_than_the_requested(
      self, seam):
    '''The disagreement is reported, not papered over.

    This provider imposes 0.05 and the caller asked for 0.30. Both
    numbers have to survive to the payload, so the mismatch is visible in
    the row that has it rather than only in a module nobody reads.

    Args:
      seam: Which reporting surface is under test.
    '''
    reported, rows = run_registry(
      seam, {'tight': capless(0.05), 'companion': capless(0.05)},
      CEILING_CAP)
    row = rows['tight']
    assert reported == CEILING_CAP
    assert row['cap_bound'] is False
    assert row['applied_max_weight'] == pytest.approx(0.05)
    assert row['applied_max_weight'] < reported, (
      f'{seam} reported {reported} for a provider that never saw it')

  @pytest.mark.parametrize('seam', SEAMS)
  def test_unbound_does_not_mean_unchecked(self, seam):
    '''A provider that cannot be capped is still clamped by the engine.

    Args:
      seam: Which reporting surface is under test.
    '''
    reported, rows = run_registry(
      seam, {'greedy': greedy(), 'companion': capless(0.05)}, CEILING_CAP)
    assert reported == CEILING_CAP
    row = rows['greedy']
    assert row['status'] == 'ok', row.get('error')
    assert row['cap_bound'] is False
    held = row['applied_max_weight']
    assert held == pytest.approx(CEILING_CAP), (
      f'{seam} let an unbound provider hold '
      f'{held} against a {CEILING_CAP} cap')

  @pytest.mark.parametrize('seam', SEAMS)
  def test_a_kwargs_provider_does_take_the_cap(self, seam):
    '''``**kwargs`` counts as accepting it, so the cap is bound.

    This is the positive control beside the three tests above: the same
    harness marks a capless provider unbound and a keyword-flexible one
    bound, so an ``applied_max_weight`` of 0.05 cannot be explained by
    the disclosure being broken rather than by the provider.

    Args:
      seam: Which reporting surface is under test.
    '''
    seen: list[dict] = []
    reported, rows = run_registry(
      seam, {'loose': loose(seen), 'companion': loose([])}, FAKE_CAP)
    assert reported == FAKE_CAP
    row = rows['loose']
    assert row['status'] == 'ok', row.get('error')
    assert row['cap_bound'] is True
    assert accepts_weight_cap(loose([])) is True
    assert row['applied_max_weight'] == pytest.approx(FAKE_CAP)
    assert seen, f'the {seam} seam never called the provider'
    for kwargs in seen:
      assert kwargs == {CAP_KEYWORD: FAKE_CAP}, (seam, kwargs)

  @pytest.mark.parametrize('seam', SEAMS)
  def test_a_provider_that_will_not_describe_itself_is_marked_unbound(
      self, seam):
    '''Refusing to be introspected is not the same as refusing to run.

    :func:`stock_rl.weights.accepts_weight_cap` asks rather than assumes,
    and a callable that will not answer -- a C builtin behind a decorator,
    anything whose ``__signature__`` raises -- is reported as not
    accepting the cap. Guessing "surely it takes one" would be the
    original defect wearing a different hat.

    Args:
      seam: Which reporting surface is under test.
    '''
    provider = opaque()
    reported, rows = run_registry(
      seam, {'opaque': provider, 'companion': capless(0.05)}, CEILING_CAP)
    assert reported == CEILING_CAP
    row = rows['opaque']
    assert row['status'] == 'ok', row.get('error')
    assert row['cap_bound'] is False
    assert row['applied_max_weight'] == pytest.approx(0.05)
    assert accepts_weight_cap(provider) is False


class TestTheMeasurementItself:
  '''The number the disclosure is measured from.'''

  def test_a_book_that_never_rebalanced_applied_no_cap(self):
    '''An absence is None, not zero.

    Reporting 0.0 would be a claim about a portfolio that does not
    exist, and it would read as "this provider could not breach a cap",
    which is a different and stronger claim.
    '''
    assert applied_weight_cap([]) is None
    assert applied_weight_cap(()) is None
    assert applied_weight_cap([{}]) is None
    assert applied_weight_cap([{'AAA': 0.2}, {'AAA': 0.3}]) == 0.3

  def test_a_positive_control_for_the_measurement(self):
    '''The heaviest weight across every snapshot, not the last one.

    A provider's cap can bind on one rebalance and not on the next, and
    the reported number is the worst the book ever held -- the ceiling a
    reader has to believe, not the friendliest snapshot.
    '''
    snapshots = [{'AAA': 0.30, 'BBB': 0.10}, {'AAA': 0.10, 'BBB': 0.10}]
    assert applied_weight_cap(snapshots) == pytest.approx(0.30)
    assert applied_weight_cap(list(reversed(snapshots))) == pytest.approx(
      0.30)

  def test_the_shared_helper_is_the_one_all_three_seams_use(self):
    '''One name in one module, rather than three copies.

    The three seams each report a cap, and the whole point of lifting
    this out of :mod:`stock_rl.api` is that a fourth seam cannot arrive
    with its own opinion about how a provider takes one.
    '''
    assert pipeline.bind_weight_cap is bind_weight_cap
    assert cli.bind_weight_cap is bind_weight_cap
    assert pipeline.applied_weight_cap is applied_weight_cap
    assert cli.applied_weight_cap is applied_weight_cap
    assert pipeline.accepts_weight_cap is accepts_weight_cap
    assert cli.accepts_weight_cap is accepts_weight_cap
    assert CAP_KEYWORD == 'max_weight'
    for name in ('accepts_weight_cap', 'applied_weight_cap',
                 'bind_weight_cap'):
      assert name in weights.__all__, (
        f'{name} is the shared implementation and is not exported, so a '
        f'fourth seam would reach in past the module boundary for it')
