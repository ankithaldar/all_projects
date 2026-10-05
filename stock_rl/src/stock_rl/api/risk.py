#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Risk as the API is allowed to see it: read-only, and lossy on purpose.

Two things live here. :func:`historical_var` is the empirical quantile of
realised returns -- the only VaR the standard library can compute without
assuming a distribution, and no normal assumption is made because
assuming one for a book of Indian mid-caps understates the tail those
books produce. The switch views below are how
:class:`stock_rl.risk.killswitch` is read: never written, never triped,
never cleared from here.

The failure mode these views exist to prevent is a safety surface that
raises. A kill switch reads a file, a socket and a lock on every call, and
a state file on a dead mount raises ``OSError``, so every read is
contained and a switch that cannot be read is reported as ``unreadable``
with ``tripped: null`` -- never as untripped. Treating a refusal as
healthy would restore, at the last layer, exactly the defect the switch
class exists to prevent.

Split out of the former single-module ``stock_rl.api`` without change.
The thresholds are read from the switch rather than re-derived here,
because a second copy of the threshold comparison is a second definition
of a safety condition and the two will drift.
'''

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from stock_rl.api.constants import var_confidence
from stock_rl.api.protocols import KillSwitchLike, TripLike


def historical_var(
  returns: Sequence[float],
  confidence: float = var_confidence,
) -> tuple[float | None, float]:
  '''Return the historical one-period VaR and the sample size behind it.

  The empirical quantile of realised returns, which is the only VaR the
  standard library can compute without assuming a distribution. No
  normal assumption is made, because assuming one for a book of Indian
  mid-caps understates the tail those books actually produce.

  Args:
    returns: Per-period returns as fractions.
    confidence: Confidence level, e.g. 0.95.

  Returns:
    Tuple of (loss as a positive fraction or None, observation count).
    With no usable return the fraction is None rather than 0.0: a VaR of
    0.0 means "the worst of the returns we have was a gain", which is a
    measurement, whereas a count of 0 means there is no history at all,
    and that is the absence of a measurement. A caller that cannot tell
    those apart will print "VaR Rs 0" beside a book that has never run.

  Raises:
    ValueError: If ``confidence`` is outside ``(0, 1)``.
  '''
  if not 0.0 < confidence < 1.0:
    raise ValueError(f'confidence must be in (0, 1), got {confidence}')
  usable = sorted(value for value in returns if math.isfinite(value))
  if not usable:
    return None, 0.0
  index = int(math.floor((1.0 - confidence) * len(usable)))
  index = max(0, min(index, len(usable) - 1))
  return max(0.0, -usable[index]), float(len(usable))


def _switch_view(
  switch: KillSwitchLike | None,
  drawdown: float,
) -> dict[str, Any]:
  '''Describe the kill switch, including a refusal to read it.

  A switch whose state cannot be read -- typically a corrupt state file,
  which :class:`stock_rl.risk.killswitch.KillSwitch` raises on purpose --
  is reported as an error with ``tripped: null``. Treating that as
  untripped would be the dangerous reading: the class exists precisely to
  refuse rather than re-arm, and a dashboard that smoothed the refusal
  away would restore the defect at the last layer.

  The catch is broad on purpose and it is the whole point of this
  function: the switch reads a file, a socket or a lock on every call,
  and ``OSError``, ``KeyError`` and ``TypeError`` are all things a real
  deployment raises there. Naming a subset would mean the one that fires
  in production is the one this endpoint does not catch, and a safety
  surface that raises is a safety surface that is simply absent.

  Args:
    switch: The switch, or None when none is wired in.
    drawdown: Current drawdown, fed to the threshold comparison.

  Returns:
    Mapping describing the switch. ``status`` is ``'unreadable'`` when any
    read raised, which is not the same answer as ``'armed'``.
  '''
  if switch is None:
    return {
      'wired': False,
      'tripped': None,
      'trading_enabled': None,
      'algo_id': None,
      'status': 'not_wired',
      'trips': [],
    }
  try:
    tripped = bool(switch.tripped)
    return {
      'wired': True,
      'tripped': tripped,
      'trading_enabled': bool(switch.trading_enabled),
      'algo_id': switch.algo_id,
      'status': 'TRIPPED' if tripped else 'armed',
      'trips': [_trip_view(trip) for trip in switch.history]
      if hasattr(switch, 'history') else [],
    }
  except Exception as exc:  # pylint: disable=broad-exception-caught
    # KillSwitchError derives from RuntimeError, and a refusal here is
    # the state worth surfacing rather than swallowing.
    return {
      'wired': True,
      'tripped': None,
      'trading_enabled': None,
      'algo_id': getattr(switch, 'algo_id', None),
      'status': 'unreadable',
      'error': str(exc),
      'drawdown': drawdown,
      'trips': [],
    }


def _trip_view(trip: TripLike) -> dict[str, Any]:
  '''Return one recorded trip as JSON-safe primitives.

  Args:
    trip: Recorded trip.

  Returns:
    Mapping with the code, the detail, the timestamp and the observation.
  '''
  return {
    'code': getattr(trip, 'code', ''),
    'detail': getattr(trip, 'detail', ''),
    'at': getattr(trip, 'at', ''),
    'observed': getattr(trip, 'observed', 0.0),
  }


def _switch_status(switch: KillSwitchLike | None) -> str:
  '''Return a one-word kill-switch state for the health payload.

  Args:
    switch: The switch, or None.

  Returns:
    ``'armed'``, ``'TRIPPED'``, ``'unreadable'`` or ``'not_wired'``.
  '''
  return str(_switch_view(switch, 0.0)['status'])


def _thresholds(switch: KillSwitchLike | None) -> dict[str, Any]:
  '''Return the pre-defined trip limits, or an empty mapping.

  The limits are supplementary detail: ``breached`` is the answer that
  decides whether trading halts, so a limits read that raises costs the
  caller four numbers and must not cost them the whole payload. The catch
  is broad for the same reason :func:`_switch_view`'s is, and the same
  cost applies in reverse: the switch that cannot be read is reported as
  unreadable by its own view, so an empty mapping here never becomes a
  claim that there are no limits.

  Args:
    switch: The switch, or None.

  Returns:
    Mapping of limit name to value, empty when no switch is wired in or
    the limits could not be read.
  '''
  if switch is None:
    return {}
  try:
    limits = switch.thresholds
    return {
      'index_fall': limits.index_fall,
      'max_drawdown': limits.max_drawdown,
      'ack_latency': limits.ack_latency,
      'fno_ban': limits.fno_ban,
    }
  except Exception:  # pylint: disable=broad-exception-caught
    return {}


def _breached(switch: KillSwitchLike | None,
              drawdown: float) -> list[str]:
  '''Return the conditions the switch considers breached.

  The switch is asked rather than re-derived, because a second copy of
  the threshold comparison is a second definition of a safety condition
  and the two will drift. The live metrics it needs are the ones this
  service holds; the session index change and the acknowledgement
  latency belong to a live feed and are not held here, so they are not
  invented.

  A read that raises yields an empty list, and that is the one place this
  module accepts a silent answer. It does so because the alternative is
  worse than silence: ``breached`` is supplementary detail on a payload
  whose ``kill_switch`` block already refuses loudly, and a partially
  readable switch must still be reportable. The empty list is a list of
  conditions this process was able to read, next to a status that says
  the state could not be.

  Args:
    switch: The switch, or None.
    drawdown: Current drawdown as a positive fraction.

  Returns:
    Detail text per breached condition, empty when clear, unwired or
    unreadable.
  '''
  if switch is None:
    return []
  try:
    return list(switch.breached(0.0, drawdown, False, None))
  except Exception:  # pylint: disable=broad-exception-caught
    return []
