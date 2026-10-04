#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Pre-trade risk, circuit bands, the kill switch and VaR.

Four concerns that share one package because they share one property:
**every one of them is allowed to stop an order**, and a risk control
that does not stop orders is documentation. Kept as four modules because
they rest on four different bases, and conflating them is how a
compliance register ends up citing the wrong instrument for a number:

``checks``
  The NSE operational modalities para 11.1 pre-trade RMS checks. Fourteen
  of the sixteen are implemented, with the two omissions named in the
  module docstring. Every check fails closed: a check that cannot decide
  rejects, because a check that answers "probably fine" is worse than no
  check at all, since the caller cannot tell an absence from a verdict.

``circuit``
  Circuit bands and corporate-action adjustment. Both are the same job --
  reconciling a price series against the rules that move prices -- and
  the corporate-action half is where most backtests die, silently, because
  an unadjusted series reads a 1:1 bonus as a 50 percent crash.

``killswitch``
  A latching, persistent, automatically-tripped halt. SEBI requires the
  automatic trip on pre-defined conditions rather than a button, the
  latch must survive a restart, and the state must refuse to be
  reinterpreted when it is corrupt rather than defaulting to armed.

``var``
  Parametric and historical VaR and CVaR. Explicitly **not** one of the
  para 11.1 pre-trade checks, which the module docstring says out loud.

The package deliberately depends on nothing but the standard library and
:mod:`stock_rl.bars`, and the risk layer is imported by the execution
layer rather than the reverse: risk runs *before* execution, and the
dependency arrow points that way so a caller cannot wire them the other
way round by accident.
'''

from stock_rl.risk.checks import (
  AccountState,
  CheckResult,
  OrderRequest,
  PriceBand,
  RiskRejected,
  RiskReport,
  RmsChecker,
  RmsLimits,
  SecurityLimits,
  check_ids,
  passed_status,
  rejected_status,
  statuses,
)
from stock_rl.risk.circuit import (
  AdjustedSeries,
  CircuitBand,
  CorporateAction,
  band_for,
  circuit_states,
  classify,
  corporate_actions,
  ex_price,
  halted_states,
  round_to_tick,
  split_adjusted_bars,
  states,
)
from stock_rl.risk.killswitch import (
  KillSwitch,
  KillSwitchError,
  KillSwitchLatched,
  KillThresholds,
  Trip,
  default_state_file,
  kill_switch_schema,
  trip_codes,
)
from stock_rl.risk.var import (
  VarEstimate,
  covariance_matrix,
  historical_cvar,
  historical_var,
  parametric_cvar,
  parametric_var,
  portfolio_var,
  var_from_returns,
  z_score,
)

__all__ = [
  'AccountState',
  'AdjustedSeries',
  'CheckResult',
  'CircuitBand',
  'CorporateAction',
  'KillSwitch',
  'KillSwitchError',
  'KillSwitchLatched',
  'KillThresholds',
  'OrderRequest',
  'PriceBand',
  'RiskRejected',
  'RiskReport',
  'RmsChecker',
  'RmsLimits',
  'SecurityLimits',
  'Trip',
  'VarEstimate',
  'band_for',
  'check_ids',
  'circuit_states',
  'classify',
  'corporate_actions',
  'covariance_matrix',
  'default_state_file',
  'ex_price',
  'halted_states',
  'historical_cvar',
  'historical_var',
  'kill_switch_schema',
  'parametric_cvar',
  'parametric_var',
  'passed_status',
  'portfolio_var',
  'rejected_status',
  'round_to_tick',
  'split_adjusted_bars',
  'states',
  'statuses',
  'trip_codes',
  'var_from_returns',
  'z_score',
]
