#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The momentum construction, reported honestly.

``GET /api/signals`` reports one thing: a 12-1 cross-sectional momentum
ranking, with the bar it was measured on named in the payload. The other
four families of strategy are measured rather than asserted, by
``GET /api/baselines``, and a signal endpoint that quietly omitted the
control arm would make that comparison impossible to check.

Two things in here are deliberately weaker than they look. ``confidence``
is a midrank percentile of the score, not a probability: nothing has been
fitted to outcome frequency, so a high value means "ranked near the top of
this panel" and nothing more. And a panel whose symbols all score
identically produces a flat ``HOLD`` at confidence 0.0 for every symbol,
because manufacturing a ladder out of ties would invent conviction.

Split out of the former single-module ``stock_rl.api`` without change.
Each reason string cites the ``lookback`` and ``skip`` the score was
actually built from rather than the module defaults, and that is the part
of this file a refactor must not quietly make configurable.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence

from stock_rl.api.constants import (
  signal_lookback,
  signal_max_weight,
  signal_skip,
  signal_top,
  weight_epsilon,
)
from stock_rl.api.models import Signal, _stamp
from stock_rl.bars import Bar
from stock_rl.indicators import momentum


def momentum_signals(
  panels: Mapping[str, Sequence[Bar]],
  held: Mapping[str, float] | None = None,
  top: int = signal_top,
  max_weight: float = signal_max_weight,
  lookback: int = signal_lookback,
  skip: int = signal_skip,
) -> list[Signal]:
  '''Return one signal per symbol from the 12-1 cross-sectional ranking.

  Both the rank and the target weight come from the same score, so they
  cannot disagree. The score is
  :func:`stock_rl.indicators.momentum` over ``lookback`` bars with the
  most recent ``skip`` bars excluded.

  ``confidence`` is the midrank percentile of that score. It is an
  ordering, not a probability: nothing here has been calibrated against
  outcome frequency, so a high value means "near the top of this panel"
  and nothing more. When every score is identical there is no
  dispersion to rank, so every symbol is ``HOLD`` at confidence 0.0 --
  a flat ranking is the honest answer and manufacturing a ladder out of
  ties would invent conviction.

  Args:
    panels: Price panels keyed by symbol.
    held: Currently held weight per symbol.
    top: How many leaders to hold.
    max_weight: Cap on any single weight.
    lookback: Bars in the momentum window.
    skip: Most recent bars excluded from it.

  Returns:
    One :class:`Signal` per symbol, ordered by rank then symbol.
  '''
  held_weights = dict(held or {})
  scores = {symbol: _score(panel, lookback, skip)
            for symbol, panel in panels.items()}
  scored = {symbol: value for symbol, value in scores.items()
            if value is not None}
  ordered = sorted(scored, key=lambda name: (-scored[name], name))
  dispersed = len(set(scored.values())) > 1
  targets = _targets(ordered, panels, top, max_weight)
  rows = []
  for symbol in sorted(panels):
    rows.append(_signal_for(
      symbol, scores[symbol], ordered, scored, targets,
      held_weights.get(symbol, 0.0), dispersed, lookback, skip))
  return rows


def _last_bar(panels: Mapping[str, Sequence[Bar]]) -> str | None:
  '''Return the timestamp of the most recent bar across every panel.

  The bar a signal is measured on, which is a property of the data rather
  than of the clock. Panels are aligned by
  :func:`stock_rl.portfolio.run_portfolio`, so in practice they share a
  final bar and the maximum is that one; taking the maximum rather than
  the first panel's last bar keeps the answer truthful if they ever do
  not.

  Args:
    panels: Loaded price panels.

  Returns:
    ISO 8601 timestamp of the latest bar, or None with no panels.
  '''
  stamps = [panel[-1].timestamp for panel in panels.values() if panel]
  return _stamp(max(stamps)) if stamps else None


def _score(panel: Sequence[Bar], lookback: int, skip: int) -> float | None:
  '''Return the 12-1 trailing return of one panel.

  Args:
    panel: Bars in ascending time order.
    lookback: Bars in the momentum window.
    skip: Most recent bars excluded from it.

  Returns:
    Fractional return, or None when the panel is too short to score.
  '''
  closes = [bar.close for bar in panel]
  window = closes[:len(closes) - skip] if skip else closes
  return momentum(window, lookback)


def _targets(
  ordered: list[str],
  panels: Mapping[str, Sequence[Bar]],
  top: int,
  max_weight: float,
) -> dict[str, float]:
  '''Return target weights: the leaders share the cap, the rest to cash.

  Args:
    ordered: Scored symbols, best momentum first.
    panels: Loaded panels, which name the symbols that cannot be scored.
    top: How many leaders to hold.
    max_weight: Cap on any single weight.

  Returns:
    Weight per symbol across every loaded symbol, including unscorable
    ones at zero.
  '''
  chosen = ordered[:top]
  if not chosen:
    return {symbol: 0.0 for symbol in panels}
  weight = min(1.0 / len(chosen), max_weight)
  return {symbol: weight if symbol in chosen else 0.0
          for symbol in panels}


def _window_clause(lookback: int, skip: int) -> str:
  '''Return the momentum window as one clause, whatever the arguments.

  ``skip`` is stated even at zero. "Skipping the most recent 0 bars" reads
  oddly, but it is the number the caller passed and the number the score
  used, and a reader comparing two payloads needs to see that one excluded
  nothing rather than infer it from a sentence that changed shape.

  Args:
    lookback: Bars in the momentum window.
    skip: Most recent bars excluded from it.

  Returns:
    A clause naming the window, e.g. ``'252 bars, skipping the most
    recent 21'``.
  '''
  return f'{lookback} bars, skipping the most recent {skip}'


def _signal_for(
  symbol: str,
  score: float | None,
  ordered: list[str],
  scored: Mapping[str, float],
  targets: Mapping[str, float],
  held: float,
  dispersed: bool,
  lookback: int,
  skip: int,
) -> Signal:
  '''Build one signal, choosing the action and stating the evidence.

  The window named in each reason is the ``lookback`` and ``skip`` the
  score was actually built from, never the module defaults. A reason is
  the evidence for the action beside it, and one that cites a 252-bar
  window for a score computed over five bars describes a computation
  that did not happen -- which is precisely the sort of drift this
  project's audit research is against.

  Args:
    symbol: Symbol the signal is about.
    score: Its 12-1 momentum, or None when unscorable.
    ordered: Scored symbols, best momentum first.
    scored: Momentum per scored symbol.
    targets: Target weight per symbol.
    held: Currently held weight.
    dispersed: Whether the panel had any dispersion to rank.
    lookback: Bars in the momentum window actually used.
    skip: Most recent bars excluded from it.

  Returns:
    The signal.
  '''
  target = float(targets.get(symbol, 0.0))
  window = _window_clause(lookback, skip)
  if score is None:
    return Signal(
      symbol=symbol,
      action='HOLD',
      confidence=0.0,
      reasons=(f'fewer than {lookback + skip + 1} bars, so the {window} '
               'momentum window is not yet available',),
      target_weight=0.0,
      held_weight=held,
    )
  rank = ordered.index(symbol) + 1
  total = len(scored)
  confidence = _midrank(symbol, scored) if dispersed else 0.0
  reasons = [
    f'momentum {score:+.2%} over {window}',
    f'rank {rank} of {total} by momentum',
  ]
  if not dispersed:
    reasons.append('every symbol scored identically, so the ranking '
                   'carries no information')
  if target - held > weight_epsilon:
    action = 'BUY'
    reasons.append(f'target {target:.2%} against {held:.2%} held, so the '
                   'signal asks to add')
  elif held - target > weight_epsilon:
    action = 'SELL'
    reasons.append(f'target {target:.2%} against {held:.2%} held, so the '
                   'signal asks to reduce')
  else:
    action = 'HOLD'
    reasons.append(f'target {target:.2%} matches the {held:.2%} held, so '
                   'no trade is implied')
  return Signal(
    symbol=symbol,
    action=action,
    confidence=confidence,
    reasons=tuple(reasons),
    target_weight=target,
    held_weight=held,
    score=score,
    rank=rank,
  )


def _midrank(symbol: str, scored: Mapping[str, float]) -> float:
  '''Return the midrank percentile of one score within the panel.

  Ties share the midpoint of the ranks they span, which is why an
  identical panel produces one shared value instead of an arbitrary
  ladder.

  Args:
    symbol: Symbol being ranked.
    scored: Momentum per scored symbol.

  Returns:
    Percentile in ``[0, 1]``, or 0.0 for a panel of one symbol, where
    no comparison exists.
  '''
  total = len(scored)
  score = scored[symbol]
  lower = sum(1 for other in scored.values() if other < score)
  tied = sum(1 for other in scored.values() if other == score)
  # The denominator is floored at one so a single-symbol panel scores
  # 0.0 rather than dividing by zero, which keeps this callable from
  # anywhere without a guard the caller would have to remember.
  return (lower + 0.5 * (tied - 1)) / max(1, total - 1)
