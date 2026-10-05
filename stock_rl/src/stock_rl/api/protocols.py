#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The shapes this API accepts, declared rather than imported.

Two seams live here and they are different kinds of thing.

The ``Protocol`` classes are structural: :class:`TripLike`,
:class:`ThresholdsLike` and :class:`KillSwitchLike` say what the API
needs to *read* out of a kill switch without importing
:mod:`stock_rl.risk.killswitch`, so a dashboard cannot fail to start
because the risk module moved, and so the read-only property is visible
in the type rather than only in a reviewer's memory.

The type aliases are the injected seams: :data:`Clock`,
:data:`WallClock`, :data:`BacktestRunner` and :data:`SignalSource` are how
a test supplies a known clock and a stub backtest instead of a socket and
a price history. ``WeightProvider`` is re-exported rather than declared,
because the API is a caller of
:func:`stock_rl.portfolio.run_portfolio` and not its author.

Split out of the former single-module ``stock_rl.api`` without change;
the forward references inside the two ``Callable`` aliases stayed as they
were, since they are resolved by whoever reads them and not at import
time.
'''

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol

from stock_rl.bars import Bar
from stock_rl.portfolio import WeightProvider

__all__ = [
  'BacktestRunner',
  'Clock',
  'KillSwitchLike',
  'SignalSource',
  'ThresholdsLike',
  'TripLike',
  'WallClock',
  'WeightProvider',
]


#: Injected monotonic-seconds source. Tests supply a stub so uptime is a
#: known number and no test waits for time to pass.
Clock = Callable[[], float]

#: Injected wall-clock source, so a recorded timestamp is deterministic.
WallClock = Callable[[], datetime]


class TripLike(Protocol):
  '''Structural type for one recorded kill-switch trip.

  Declared rather than imported so this module has no import-time
  coupling to :mod:`stock_rl.risk.killswitch`. The two modules are built
  separately, and a dashboard that cannot start because the risk module
  moved is a dashboard nobody uses.
  '''

  code: str
  detail: str
  at: str
  observed: float


class ThresholdsLike(Protocol):
  '''Structural type for the pre-defined trip conditions.'''

  index_fall: float
  max_drawdown: float
  ack_latency: float
  fno_ban: bool


class KillSwitchLike(Protocol):
  '''Structural type for the subset of the kill switch this API reads.

  Read-only by construction: every member here is a getter, so the API
  has no vocabulary for halting or resuming trading even if a future
  change tried to give it one.
  '''

  algo_id: str

  @property
  def tripped(self) -> bool:
    '''Return whether trading is halted.'''

  @property
  def trading_enabled(self) -> bool:
    '''Return whether an order may be released.'''

  @property
  def thresholds(self) -> ThresholdsLike:
    '''Return the pre-defined trip conditions.'''

  def breached(
    self,
    index_change: float,
    drawdown: float,
    fno_banned: bool,
    ack_latency: float | None,
  ) -> tuple[str, ...]:
    '''Return the conditions currently breached, in detail text.'''



#: A backtest trigger. Receives a validated request, returns a
#: JSON-safe mapping. Injected so a test needs neither panels nor a
#: backtest to exercise the HTTP layer.
BacktestRunner = Callable[['BacktestRequest'], Mapping[str, Any]]

#: A signal source. Receives the panels and the held weights, returns one
#: row per symbol. Injected so a test can pin the action vocabulary
#: without constructing price history.
SignalSource = Callable[[Mapping[str, Sequence[Bar]],
                         Mapping[str, float]], list['Signal']]
