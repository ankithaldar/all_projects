#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Deep hedging environment over a predefined, continuous instrument set.

Three design decisions here are corrections to the design docs, not
preferences.

**No Discrete action, ever.** SAC's squashed-Gaussian actor is unimodal
and continuous, so the proposed ``{Box(continuous), Discrete(strike),
Discrete(expiry)}`` action space cannot be learned by it. This
environment therefore has exactly three continuous box variables, one
target coverage ratio per member of a **predefined** instrument set. The
alternative designs in the literature are worse: an autoregressive
discrete-then-continuous head is learnable but has no evidence behind it,
and a monthly hierarchical two-stage controller has roughly 120 monthly
decisions in a decade of Nifty data, which is statistically hopeless. In
practice strike and expiry are a contract-design decision made by the desk
and revisited quarterly, so they belong in the instrument list, not in the
action.

**Hard mask, not a penalty.** "Do not buy protection when implied vol is
above 30 percent" is a business rule. A Lagrangian lets the agent trade it
off against reward, which means the rule holds only while the penalty
happens to exceed the edge of breaking it. The bound is applied as a
state-dependent admissible action set, and
:meth:`HedgeEnv.coverage_bounds` is public so a critic can mask its
target computation with exactly the same bounds: a discontinuity in the
actor's policy map makes the critic's values for masked actions wrong.

**Reward is a coherent terminal risk measure minus cost.** Per-step
reward is the cost charged and nothing else; the objective is applied once,
at the end of the episode, to the whole path of the self-financing
hedging error. There is no per-step drawdown penalty, because an
asymmetric penalty that punishes loss without crediting gain makes the
doubling strategy optimal: raise exposure after a loss because the
expected recovery beats the incremental penalty, which is a martingale.
Francois et al. (2025) needed extra anti-speculation machinery for exactly
this. If CVaR is selected, the tail confidence must be at least 0.85:
their table shows the learned difference strategy earning +1.37
unconditionally, roughly 43 percent of the option's initial price on
every path, at CVaR alpha of 20 percent or below. At that setting the agent
has stopped hedging. :func:`cvar` enforces the floor so the mistake cannot
be made by accident.

**No look-ahead.** A decision at the close of bar ``t`` fills at the open
of bar ``t+1``, and the option leg is transacted at its Black-Scholes
value using the implied vol estimated at bar ``t``. Using the fill bar's
vol would price the trade with information the desk did not have.

PONYTAIL: implied vol is trailing realised vol, because this project has
no vendor option chain and offline reproducibility matters more than a
fitted surface. The deep-hedging evidence supports a single reference IV:
Ruf & Wang (2022) show linear regressions on BS Greeks beat BS delta by
15-20 percent of MSHE and neural networks add nothing. Ceiling: realised
vol is not implied vol, so premium levels are approximate and the stated
mask thresholds should be re-derived against real quotes before they are
relied on. Upgrade path: inject an IV series the same way panels are
injected; nothing else in this module changes.
'''

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, stdev

from stock_rl.bars import Bar
from stock_rl.env.gym import Info, Obs
from stock_rl.indicators import realized_volatility
from stock_rl.metrics import TRADING_DAYS_PER_YEAR
from stock_rl.rl.policy import Action

__all__ = [
  'Greeks',
  'HedgeCosts',
  'HedgeEnv',
  'HedgeRisk',
  'Instrument',
  'bs_price',
  'bs_greeks',
  'cvar',
  'default_instruments',
  'min_tail_confidence',
  'norm_cdf',
  'norm_pdf',
  'option_kinds',
  'semi_rmse',
]

#: Lowest tail confidence a CVaR objective may be evaluated at. Francois
#: et al. (2025) report the deep-hedging difference strategy earning
#: +1.37 unconditionally at alpha of 20 percent or below, i.e. no hedge at
#: all, and a genuine delta modification only from alpha 85 percent up.
min_tail_confidence = 0.85

#: Recognised option kinds.
option_kinds = ('call', 'put')

#: Recognised instrument kinds.
instrument_kinds = ('future', 'option')

#: Recognised terminal risk measures.
risk_measures = ('cvar', 'semi_rmse')


def norm_cdf(value: float) -> float:
  '''Return the standard normal cumulative distribution function.

  ``math.erf`` is used rather than a lookup table or an approximation:
  accuracy matters here because delta is a probability, and a Gaussian
  table would degrade exactly the region the hedge is decided in.

  Args:
    value: Standard normal deviate.

  Returns:
    Probability in ``[0, 1]``.
  '''
  return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def norm_pdf(value: float) -> float:
  '''Return the standard normal probability density function.

  Args:
    value: Standard normal deviate.

  Returns:
    Density at ``value``.
  '''
  return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class Greeks:
  '''First and second order spot sensitivities of an option.

  Attributes:
    delta: Spot sensitivity. In ``[0, 1]`` for a call and ``[-1, 0]`` for
      a put while time remains.
    gamma: Second spot sensitivity, always non-negative, which is the
      whole reason a long option is worth holding.
  '''

  delta: float
  gamma: float


def _doubles(spot: float, strike: float, tau: float, rate: float,
             vol: float) -> tuple[float, float]:
  '''Return the Black-Scholes ``d1`` and ``d2`` terms.

  Args:
    spot: Spot price.
    strike: Strike price.
    tau: Time to expiry in years.
    rate: Continuously compounded risk-free rate.
    vol: Annualised volatility.

  Returns:
    Tuple of ``(d1, d2)``.

  Raises:
    ValueError: If any price is non-positive or tau or vol is not.
  '''
  if spot <= 0.0 or strike <= 0.0:
    raise ValueError(f'spot and strike must be positive, got {spot}, {strike}')
  if tau <= 0.0:
    raise ValueError(f'tau must be positive, got {tau}')
  if vol <= 0.0:
    raise ValueError(f'vol must be positive, got {vol}')
  root = math.sqrt(tau)
  d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * tau) / (vol * root)
  return d1, d1 - vol * root


def _expired_delta(spot: float, strike: float, kind: str) -> float:
  '''Return the delta of an option with no time left.

  Args:
    spot: Spot price.
    strike: Strike price.
    kind: ``'call'`` or ``'put'``.

  Returns:
    Intrinsic delta: 0, 1 or -1.
  '''
  if kind == 'call':
    return 1.0 if spot > strike else 0.0
  return -1.0 if spot < strike else 0.0


def bs_greeks(
  spot: float,
  strike: float,
  tau: float,
  rate: float,
  vol: float,
  kind: str = 'call',
) -> Greeks:
  '''Return Black-Scholes delta and gamma.

  Closed form, from d1 = ``(ln(S/K) + (r + sigma^2/2) T) / (sigma sqrt(T))``
  with delta ``N(d1)`` for a call and ``N(d1) - 1`` for a put, and gamma
  ``n(d1) / (S sigma sqrt(T))``. Both are exact for the model and only
  approximate for a real option: they assume log-normal returns and a
  constant volatility, so they carry no smile, no term structure and no
  jump risk. That is acceptable here because the state deliberately uses
  one reference IV per the hedging evidence, and it is why no fitted
  surface is attempted.

  LIMITATIONS, stated because they bound what this environment can be
  used to claim. No dividends, so an index that yields 1.3 percent a year
  is priced as if it yields nothing. Continuous exercise, so American
  early-exercise value is absent; this matters most for short puts on
  futures. No volatility smile, so deep out-of-the-money puts are priced
  as if the distribution had no skew.

  Args:
    spot: Spot price.
    strike: Strike price.
    tau: Time to expiry in years. At or below zero the option has expired
      and gamma is returned as 0.0 rather than divided by zero.
    rate: Continuously compounded risk-free rate.
    vol: Annualised volatility.
    kind: ``'call'`` or ``'put'``.

  Returns:
    ``Greeks`` for the option.

  Raises:
    ValueError: If prices are non-positive, vol is not positive while
      time remains, or ``kind`` is unknown.
  '''
  if kind not in option_kinds:
    raise ValueError(f'kind must be one of {option_kinds}, got {kind!r}')
  if spot <= 0.0 or strike <= 0.0:
    raise ValueError(f'spot and strike must be positive, got {spot}, {strike}')
  if tau <= 0.0:
    return Greeks(_expired_delta(spot, strike, kind), 0.0)
  d1, _ = _doubles(spot, strike, tau, rate, vol)
  gamma = norm_pdf(d1) / (spot * vol * math.sqrt(tau))
  delta = norm_cdf(d1)
  return Greeks(delta if kind == 'call' else delta - 1.0, gamma)


def bs_price(
  spot: float,
  strike: float,
  tau: float,
  rate: float,
  vol: float,
  kind: str = 'call',
) -> float:
  '''Return the Black-Scholes price of a European option.

  Same model and same limitations as :func:`bs_greeks`. This is the whole
  pricing engine for the option leg: there is no option price series in
  this project, so every terminal risk statistic here is a model statistic
  and inherits that model's error.

  Args:
    spot: Spot price.
    strike: Strike price.
    tau: Time to expiry in years.
    rate: Continuously compounded risk-free rate.
    vol: Annualised volatility.
    kind: ``'call'`` or ``'put'``.

  Returns:
    Option premium per unit of underlying.

  Raises:
    ValueError: If prices are non-positive, tau is not positive while
      needed, or ``kind`` is unknown.
  '''
  if kind not in option_kinds:
    raise ValueError(f'kind must be one of {option_kinds}, got {kind!r}')
  if tau <= 0.0:
    if spot <= 0.0 or strike <= 0.0:
      raise ValueError(
        f'spot and strike must be positive, got {spot}, {strike}')
    payoff = spot - strike if kind == 'call' else strike - spot
    return max(payoff, 0.0)
  d1, d2 = _doubles(spot, strike, tau, rate, vol)
  discount = math.exp(-rate * tau)
  if kind == 'call':
    return spot * norm_cdf(d1) - discount * strike * norm_cdf(d2)
  return discount * strike * norm_cdf(-d2) - spot * norm_cdf(-d1)


def cvar(values: list[float], alpha: float = 0.95) -> float:
  '''Return the historical conditional value at risk at high confidence.

  The mean of the worst ``1 - alpha`` fraction of observations. Coherent
  and tail-sensitive, which is what the hedging objective needs, but only
  above ``min_tail_confidence``: at a low alpha the CVaR objective stops
  measuring hedge quality at all. Francois et al. (2025) found the
  deep-hedging difference strategy earning +1.37 unconditionally on every
  path at alpha of 20 percent or below, against a delta hedge that earned
  nothing, which is the signature of an agent that abandoned hedging. The
  floor is enforced here rather than documented and hoped for.

  Args:
    values: Sample of the quantity being risked, one entry per period.
    alpha: Tail confidence. Must be at least ``min_tail_confidence``.

  Returns:
    The mean of the tail, or 0.0 for an empty sample.

  Raises:
    ValueError: If ``alpha`` is outside ``[min_tail_confidence, 1]``.
  '''
  if not min_tail_confidence <= alpha <= 1.0:
    raise ValueError(
      f'alpha must be in [{min_tail_confidence}, 1.0]; a low-tail CVaR does '
      f'not produce a hedge, got {alpha}')
  if not values:
    return 0.0
  count = max(1, math.ceil((1.0 - alpha) * len(values)))
  worst = sorted(values)[:count]
  return fmean(worst)


def semi_rmse(values: list[float]) -> float:
  '''Return the semi-root-mean-square error, downside dispersion only.

  The semideviation ``sqrt(E[min(0, x)^2])``. It gives no credit for
  gains, which is precisely what Carbonneau and Godin (2023) recommend
  when a paper finds a drawdown-only reward has stopped producing a hedge,
  and it is the objective whose reward Zernikov's default reward collapses
  into. It is positively homogeneous, subadditive and monotone, so it is a
  coherent risk measure rather than a proxy.

  Args:
    values: Sample of the quantity being risked.

  Returns:
    Downside dispersion, or 0.0 for an empty sample.
  '''
  if not values:
    return 0.0
  shortfall = [min(0.0, value) ** 2 for value in values]
  return math.sqrt(fmean(shortfall))


@dataclass(frozen=True, slots=True)
class Instrument:
  '''One member of the predefined hedge instrument set.

  Attributes:
    name: Identifier and action key.
    kind: ``'future'`` or ``'option'``. Controls pricing, Greeks and cost.
    option_kind: ``'call'`` or ``'put'``; only read for options.
    strike_ratio: Strike as a fraction of spot at the roll. Only read for
      options.
    tenor_bars: Bars from the roll to expiry. Only read for options.
    coverage_cap: Largest coverage ratio this instrument may carry.
  '''

  name: str
  kind: str
  option_kind: str = 'call'
  strike_ratio: float = 0.0
  tenor_bars: int = 0
  coverage_cap: float = 1.0

  def __post_init__(self) -> None:
    '''Validate the instrument definition at construction.

    Raises:
      ValueError: If the kind, option kind, strike ratio, tenor or cap is
        unusable.
    '''
    if self.kind not in instrument_kinds:
      raise ValueError(
        f'kind must be one of {instrument_kinds}, got {self.kind!r}')
    if self.kind == 'option':
      if self.option_kind not in option_kinds:
        raise ValueError(
          f'option_kind must be one of {option_kinds}, '
          f'got {self.option_kind!r}')
      if self.strike_ratio <= 0.0:
        raise ValueError(
          f'strike_ratio must be positive, got {self.strike_ratio}')
      if self.tenor_bars < 1:
        raise ValueError(f'tenor_bars must be >= 1, got {self.tenor_bars}')
    if not 0.0 < self.coverage_cap <= 1.0:
      raise ValueError(
        f'coverage_cap must be in (0, 1], got {self.coverage_cap}')


def default_instruments() -> tuple[Instrument, ...]:
  '''Return the recommended Nifty hedge instrument set.

  Three continuous coverage ratios, which is the reparameterisation the
  deep-hedging review calls the strongly preferred option: short futures
  as the workhorse, a one-month 95 percent strike put spread as convexity,
  and the collar's upside financing as a separate continuous leg.

  The put spread is represented by its long 95 percent leg. Its net delta
  is dominated by that leg and the short leg only offsets premium, so a
  two-instrument spread would double the action dimension for no evidence
  that it earns its keep. Adding a second ``Instrument`` with
  ``strike_ratio=0.98`` models the short leg exactly, and nothing in this
  environment needs to change.

  Returns:
    Tuple of instruments in fixed action order.
  '''
  return (
    Instrument('nifty_future', 'future', coverage_cap=1.0),
    Instrument('put_spread_95', 'option', option_kind='put',
               strike_ratio=0.95, tenor_bars=21),
    Instrument('collar_upside', 'option', option_kind='call',
               strike_ratio=1.05, tenor_bars=21),
  )


@dataclass(frozen=True, slots=True)
class HedgeCosts:
  '''One-way cost rates for the hedge legs.

  Attributes:
    future_pct: Fraction of futures notional traded. The default is half
      the roughly 22 basis point Indian round trip, one way.
    option_pct: Fraction of option premium traded. The default 1 percent
      sits in the 0.5 to 1.5 percent band Francois et al. (2025) use for
      index options and near Chaudhury (2019)'s 0.95 percent average
      index call cost. Option hedging has a named failure mode when
      friction is ignored -- the agent simply never trades -- so this is
      charged on every rebalance, not just the roll.
  '''

  future_pct: float = 0.0011
  option_pct: float = 0.01


@dataclass(frozen=True, slots=True)
class HedgeRisk:
  '''Objective and reward weights for the hedge environment.

  Attributes:
    measure: ``'cvar'`` or ``'semi_rmse'``.
    alpha: Tail confidence for ``'cvar'``. Enforced at or above
      ``min_tail_confidence``.
    risk: Scale on the terminal risk penalty.
    cost: Scale on the per-step cost charge.
  '''

  measure: str = 'cvar'
  alpha: float = 0.95
  risk: float = 1.0
  cost: float = 1.0


class HedgeEnv:
  '''Deep hedging environment over a predefined instrument set.

  Implements the ``Env`` protocol of :mod:`stock_rl.env.gym` with the
  continuous action signature of
  :class:`stock_rl.rl.policy.ContinuousEnv`.

  Attributes:
    action_symbols: Instrument names, in the order supplied.
    n_steps: Length of one episode in steps.
  '''

  action_symbols: tuple[str, ...]
  n_steps: int

  def __init__(
    self,
    underlying: list[Bar],
    instruments: tuple[Instrument, ...] | None = None,
    capital: float = 10_000_000.0,
    exposure_units: float | None = None,
    rate: float = 0.06,
    base_vol: float = 0.15,
    iv_window: int = 60,
    iv_ceiling: float = 0.30,
    iv_scale: float = 0.30,
    costs: HedgeCosts = HedgeCosts(),
    risk: HedgeRisk = HedgeRisk(),
    history: int = 60,
  ) -> None:
    '''Build a hedge environment over one underlying bar series.

    The liability is a long position in ``exposure_units`` of the
    underlying, defaulting to a fully invested book. Coverage of 1.0 on
    the future leg therefore neutralises the liability exactly, which is
    what makes the coverage ratios interpretable rather than arbitrary.

    Contract sizes are derived once, at the first decision bar, from the
    liability: see :meth:`_initialise_instruments`, which sizes capacity in
    delta units so that 100 percent coverage on any leg offsets the
    liability's spot delta one for one.

    Args:
      underlying: Bars for one underlying in ascending time order.
      instruments: Instrument set. Defaults to
        :func:`default_instruments`.
      capital: Book notional in rupees, used to set the default liability.
      exposure_units: Long exposure in underlying units. Defaults to a
        fully invested book at the first decision bar.
      rate: Continuously compounded risk-free rate.
      base_vol: Floor on the annualised reference volatility, and the
        volatility the contract capacities are sized against.
      iv_window: Bars in the realised-volatility estimate.
      iv_ceiling: Above this vol, no option coverage may be increased.
      iv_scale: Numerator of the smooth cap ``iv_scale / iv``.
      costs: One-way cost rates.
      risk: Objective and reward weights.
      history: Warm-up bars before the first decision.

    Raises:
      ValueError: If the bar series is unusable, an instrument is invalid,
        the risk measure is unknown, or the CVaR alpha is too low.
    '''
    if len(underlying) <= history + 1:
      raise ValueError(
        f'need more than history + 1 bars, got {len(underlying)}')
    if capital <= 0.0:
      raise ValueError(f'capital must be positive, got {capital}')
    if iv_window < 2:
      raise ValueError(f'iv_window must be >= 2, got {iv_window}')
    if base_vol <= 0.0:
      raise ValueError(f'base_vol must be positive, got {base_vol}')
    if risk.measure not in risk_measures:
      raise ValueError(
        f'measure must be one of {risk_measures}, got {risk.measure!r}')
    if risk.measure == 'cvar' and not min_tail_confidence <= risk.alpha <= 1.0:
      raise ValueError(
        f'alpha must be in [{min_tail_confidence}, 1.0]; a low-tail CVaR '
        f'does not produce a hedge, got {risk.alpha}')
    if iv_ceiling <= 0.0 or iv_scale <= 0.0:
      raise ValueError('iv_ceiling and iv_scale must be positive')
    self.underlying = underlying
    self.instruments = tuple(instruments or default_instruments())
    self.action_symbols = tuple(item.name for item in self.instruments)
    self.capital = capital
    self.rate = rate
    self.base_vol = base_vol
    self.iv_window = iv_window
    self.iv_ceiling = iv_ceiling
    self.iv_scale = iv_scale
    self.costs = costs
    self.risk = risk
    self.history = history
    self.n_steps = len(underlying) - history
    self.timestamps = [bar.timestamp for bar in underlying]
    self.closes = [bar.close for bar in underlying]
    if exposure_units is None:
      exposure_units = capital / self.closes[history - 1]
    self.exposure_units = exposure_units
    self._initialise_instruments()
    self._reset_state()

  def _initialise_instruments(self) -> None:
    '''Set strikes, expiries and contract capacities at the first bar.

    Capacity is expressed in **delta units**: 100 percent coverage on any
    leg offsets the liability's spot delta one for one, which is what makes
    a single coverage ratio meaningful across a future and an option. The
    future capacity is therefore the liability's own unit count, and the
    option capacity is that count divided by the leg's delta. Sizing an
    option leg by premium instead would make coverage incomparable across
    the three legs and would buy 50 million contracts of a cheap 95 percent
    strike to spend the whole book.

    Lot sizes are not modelled. A desk should round the capacities to the
    exchange lot; nothing else in this module depends on the lot.
    '''
    spot = self.closes[self.history - 1]
    tenor = 0.0
    self._strike: dict[str, float] = {}
    self._expiry: dict[str, int] = {}
    self._capacity: dict[str, int] = {}
    for item in self.instruments:
      if item.kind == 'future':
        self._strike[item.name] = 0.0
        self._expiry[item.name] = len(self.underlying)
        self._capacity[item.name] = max(
          1, int(round(self.exposure_units)))
        continue
      strike = spot * item.strike_ratio
      tenor = item.tenor_bars / TRADING_DAYS_PER_YEAR
      self._strike[item.name] = strike
      self._expiry[item.name] = self.history - 1 + item.tenor_bars
      delta = bs_greeks(
        spot, strike, tenor, self.rate, self.base_vol, item.option_kind).delta
      self._capacity[item.name] = max(1, int(round(
        self.exposure_units / max(abs(delta), 1e-9))))

  def _reset_state(self) -> None:
    '''Reset positions, cash and episode bookkeeping to the start.'''
    self._step_index = 0
    self._cash = 0.0
    self._positions = dict.fromkeys(self.action_symbols, 0)
    self._coverage = dict.fromkeys(self.action_symbols, 0.0)
    self._hedge_value = 0.0
    self._liability = 0.0
    self._initial_liability = 0.0
    self._initial_hedge = 0.0
    self._started = False
    self._error = 0.0
    self._errors: list[float] = []
    self._delta_log: list[float] = []
    self._equity = [1.0]
    self._total_cost = 0.0
    self._decisions: list[dict[str, object]] = []

  def _bar_index(self) -> int:
    '''Return the absolute index of the current decision bar.'''
    return self.history + self._step_index - 1

  def _tau(self, name: str, index: int) -> float:
    '''Return years to expiry for an instrument at a bar.

    Args:
      name: Instrument name.
      index: Bar index.

    Returns:
      Time to expiry in years, zero once expired.
    '''
    remaining = self._expiry[name] - index
    return max(0.0, remaining) / TRADING_DAYS_PER_YEAR

  def _implied_vol(self, index: int) -> float:
    '''Return the reference volatility known at a bar.

    The trailing realised volatility of the closes up to ``index``, floored
    at ``base_vol``. The floor is not cosmetic: pricing a one-month 95
    percent strike at a 1.3 percent realised vol returns a premium of
    exactly zero in binary floating point, which deletes the option leg
    from the model without any error being raised. ``base_vol`` is also the
    honest long-run level, and a desk that has an implied-vol series should
    inject it here instead rather than raise the floor further.

    Args:
      index: Bar index whose close has just been observed.

    Returns:
      Annualised volatility used for pricing, Greeks and the mask.
    '''
    series = realized_volatility(
      self.closes[:index + 1], self.iv_window, TRADING_DAYS_PER_YEAR)
    latest = series[-1] if series else None
    if latest is None:
      return self.base_vol
    return max(latest, self.base_vol)

  def reset(self, seed: int | None = None) -> Obs:
    '''Restart the episode from the first decision bar.

    Roll state, that is strikes and expiries, is re-derived at the same
    first bar rather than carried over, so an episode is independent of
    how the previous one ended.

    Args:
      seed: Accepted for interface compatibility. This environment is
        deterministic, so the seed is ignored.

    Returns:
      The first observation.
    '''
    del seed
    self._initialise_instruments()
    self._reset_state()
    return self._observation()

  def coverage_bounds(self, observation: Obs | None = None) -> dict[str, float]:
    '''Return the maximum coverage ratio each instrument may carry.

    This is the mask, and it is public for a specific reason: a critic
    must mask its target computation with exactly these bounds, otherwise
    the values it learns for masked actions are the values of an
    infeasible world and the actor is trained against a fiction.

    Three rules stack here. The structural cap is the instrument's own
    ``coverage_cap``. The smooth cap ``iv_scale / iv`` thins protection as
    it gets expensive, which is the state-dependent admissible set from
    the hedging review. The hard rule refuses any increase in option
    coverage once vol exceeds ``iv_ceiling``: a business rule that must
    hold regardless of how attractive the trade looks, so it is expressed
    as a bound rather than a penalty the agent can trade off.

    Args:
      observation: Optional observation whose ``iv`` should be used.
        Defaults to the volatility known at the current decision bar.

    Returns:
      Upper coverage bound per instrument name.
    '''
    index = self._bar_index()
    vol = float(observation['iv']) if observation is not None \
      else self._implied_vol(index)
    smooth = min(1.0, self.iv_scale / vol) if vol > 0.0 else 1.0
    bounds: dict[str, float] = {}
    for item in self.instruments:
      bound = item.coverage_cap
      if item.kind == 'option':
        bound = min(bound, smooth)
        if vol > self.iv_ceiling:
          bound = min(bound, self._coverage[item.name])
      bounds[item.name] = bound
    return bounds

  def _observation(self) -> Obs:
    '''Build the observation visible at the current decision bar.

    Returns:
      Mapping with spot, reference vol, portfolio Greeks, current
      coverage, the mask bounds and the accumulated hedging error.
    '''
    index = self._bar_index()
    spot = self.closes[index]
    vol = self._implied_vol(index)
    hedge_delta, gamma = self._leg_greeks(index, vol)
    option_names = [
      item.name for item in self.instruments if item.kind == 'option'
    ]
    return {
      'step': self._step_index,
      'timestamp': self.timestamps[index],
      'spot': spot,
      'iv': vol,
      'rate': self.rate,
      'tau_years': self._tau(option_names[0], index)
      if option_names else 0.0,
      'net_delta': self.exposure_units + hedge_delta,
      'liability_delta': self.exposure_units,
      'hedge_delta': hedge_delta,
      'gamma': gamma,
      'coverage': dict(self._coverage),
      'contracts': dict(self._positions),
      'bounds': self.coverage_bounds(),
      'error': self._error,
      'equity': self._equity[-1],
      'total_cost': self._total_cost,
    }

  def _leg_greeks(self, index: int, vol: float) -> tuple[float, float]:
    '''Return the aggregate delta and gamma of the held legs.

    Args:
      index: Bar index whose Greeks are wanted.
      vol: Reference volatility to price them at.

    Returns:
      Tuple of (hedge delta, hedge gamma). A future is short by contract,
      so a held future contributes ``-contracts``.
    '''
    spot = self.closes[index]
    delta = 0.0
    gamma = 0.0
    for item in self.instruments:
      held = self._positions[item.name]
      if item.kind == 'future':
        delta -= held
        continue
      greeks = bs_greeks(
        spot, self._strike[item.name], self._tau(item.name, index),
        self.rate, vol, item.option_kind)
      delta += held * greeks.delta
      gamma += held * greeks.gamma
    return delta, gamma

  def step(self, action: Action) -> tuple[Obs, float, bool, Info]:
    '''Advance one bar, executing at the next bar's open.

    Args:
      action: Target coverage ratio per instrument. Values outside
        ``[0, bounds]`` are clamped by the environment, so an actor that
        ignores the mask cannot trade through it.

    Returns:
      Tuple of (observation, reward, terminated, info). The reward is the
      cost charged on this bar, plus the terminal risk penalty on the last
      bar only.
    '''
    if self._step_index >= self.n_steps:
      return self._observation(), 0.0, True, self._info()
    decision = self._bar_index()
    fill = decision + 1
    self._roll(decision)
    vol = self._implied_vol(decision)
    bounds = self.coverage_bounds()
    self._delta_log.append(
      self.exposure_units + self._leg_greeks(decision, vol)[0])
    targets = self._target_contracts(action, bounds)
    cost = self._execute(targets, fill, vol)
    self._mark(fill)
    reward = -self.risk.cost * cost
    self._step_index += 1
    terminated = self._step_index >= self.n_steps
    if terminated:
      reward -= self.risk.risk * self.risk_measure()
    self._decisions.append({
      'step': self._step_index,
      'timestamp': self.timestamps[decision],
      'coverage': dict(self._coverage),
      'contracts': dict(self._positions),
      'cost': cost,
      'error': self._error,
    })
    return self._observation(), reward, terminated, self._info(cost)

  def _roll(self, decision: int) -> None:
    '''Reset strike and tenor for expired option legs at the decision bar.

    A roll is modelled as closing the old leg and opening a new one at
    ``spot * strike_ratio`` with the full tenor. Marking the position
    against the new strike at the next open is exactly that trade, so no
    residual time value has to be written off by hand, and the roll costs
    the same premium it would cost in the market.

    Args:
      decision: Index of the decision bar.
    '''
    spot = self.closes[decision]
    for item in self.instruments:
      if item.kind != 'option':
        continue
      if decision < self._expiry[item.name]:
        continue
      self._strike[item.name] = spot * item.strike_ratio
      self._expiry[item.name] = decision + item.tenor_bars

  def _target_contracts(
    self, action: Action, bounds: dict[str, float]) -> dict[str, int]:
    '''Convert target coverage ratios into whole contracts.

    Args:
      action: Desired coverage per instrument.
      bounds: Feasible upper coverage per instrument, from
        :meth:`coverage_bounds`.

    Returns:
      Target contracts per instrument, never negative.
    '''
    targets: dict[str, int] = {}
    for name in self.action_symbols:
      raw = action.get(name, 0.0)
      numeric = isinstance(raw, (int, float)) and not isinstance(raw, bool)
      wanted = raw if numeric else 0.0
      targets[name] = int(round(max(0.0, min(bounds[name], wanted))
                               * self._capacity[name]))
    return targets

  def _execute(
    self,
    targets: dict[str, int],
    fill: int,
    vol: float,
  ) -> float:
    '''Rebalance every leg at the fill bar's open and return the cost.

    The option leg transacts at its Black-Scholes value using the vol
    known at the decision bar. Using the fill bar's vol would price the
    trade with information the desk did not have when it decided.

    Args:
      targets: Target contracts per instrument.
      fill: Index of the bar whose open the trades occur at.
      vol: Reference volatility known at the decision bar.

    Returns:
      Transaction cost charged, in rupees.
    '''
    spot = self.underlying[fill].open
    cost = 0.0
    for item in self.instruments:
      name = item.name
      delta = targets[name] - self._positions[name]
      if delta == 0:
        continue
      if item.kind == 'future':
        self._cash -= delta * spot
        cost += abs(delta) * spot * self.costs.future_pct
      else:
        premium = bs_price(
          spot, self._strike[name], self._tau(name, fill), self.rate, vol,
          item.option_kind)
        self._cash -= delta * premium
        cost += abs(delta) * premium * self.costs.option_pct
      self._positions[name] = targets[name]
    self._total_cost += cost
    self._recount_coverage()
    return cost

  def _mark(self, fill: int) -> None:
    '''Mark the hedged book at the fill bar's close and book the error.

    The mark uses the volatility implied by closes up to and including
    this bar, which is the information available at its close. A mark that
    reached further would be a look-ahead in the risk statistic itself,
    which is the easiest place in a hedging backtest to hide one.

    Args:
      fill: Index of the bar whose close is being marked.
    '''
    spot = self.closes[fill]
    vol = self._implied_vol(fill)
    value = self._cash
    for item in self.instruments:
      name = item.name
      if item.kind == 'future':
        value += self._positions[name] * spot
        continue
      premium = bs_price(
        spot, self._strike[name], self._tau(name, fill), self.rate, vol,
        item.option_kind)
      value += self._positions[name] * premium
    liability = self.exposure_units * spot
    if not self._started:
      # The hedging error and the equity line must share one inception.
      # Leaving _liability at 0.0 here makes the error telescope to
      # (L_T - V_T) rather than ((L_T - L_0) - (V_T - V_0)), which is
      # larger than the truth by exactly L_0 / L_0 == 1.0 for EVERY
      # policy. That constant offset does not merely shift the scale --
      # it makes the path start at 1.0, which collapses semi_rmse to
      # zero and hands the never-hedged book a better terminal reward
      # than a perfect hedge.
      self._initial_liability = liability
      self._initial_hedge = value
      self._liability = liability
      self._hedge_value = value
      self._started = True
    self._error += (
      (liability - self._liability) - (value - self._hedge_value)
    ) / self._initial_liability
    self._liability = liability
    self._hedge_value = value
    self._errors.append(self._error)
    hedged = self.capital - (liability - self._initial_liability) \
      + (value - self._initial_hedge)
    self._equity.append(hedged / self.capital)

  def _recount_coverage(self) -> None:
    '''Recompute each leg's coverage ratio from its contract count.'''
    self._coverage = {
      name: self._positions[name] / self._capacity[name]
      for name in self.action_symbols
    }

  def risk_measure(self) -> float:
    '''Return the terminal risk measure of the realised error path.

    Returns:
      The configured measure applied to the hedging-error path so far.
    '''
    if self.risk.measure == 'cvar':
      return cvar(self._errors, self.risk.alpha)
    return semi_rmse(self._errors)

  def _info(self, cost: float = 0.0) -> Info:
    '''Build the auxiliary info mapping for the current step.

    Args:
      cost: Transaction cost charged on the step just taken.

    Returns:
      Info mapping including hedging error, net delta, cost and coverage.
    '''
    index = self._bar_index()
    vol = self._implied_vol(index)
    return {
      'cost': cost,
      'total_cost': self._total_cost,
      'error': self._error,
      'net_delta': self.exposure_units + self._leg_greeks(index, vol)[0],
      'gamma': self._leg_greeks(index, vol)[1],
      'coverage': dict(self._coverage),
      'equity': self._equity[-1],
    }

  def render(self) -> str:
    '''Return a one-line summary of the episode so far.

    Returns:
      Human-readable summary.
    '''
    return (
      f'step {self._step_index}/{self.n_steps} '
      f'error {self._error:.2f} cost {self._total_cost:.0f} '
      f'equity {self._equity[-1]:.4f}'
    )

  @property
  def equity_curve(self) -> list[float]:
    '''Return the hedged book's equity curve, starting at 1.0.'''
    return list(self._equity)

  @property
  def hedging_error(self) -> list[float]:
    '''Return the path of the self-financing hedging error.

    Expressed as a fraction of the liability's initial value, so the
    statistic is scale-free. An error in rupees would make the terminal
    penalty scale with the size of the book and would hand a large book a
    large apparent hedging error for the identical relative failure.
    '''
    return list(self._errors)

  @property
  def coverage(self) -> dict[str, float]:
    '''Return the current coverage ratio per instrument.'''
    return dict(self._coverage)

  @property
  def positions(self) -> dict[str, int]:
    '''Return the current contract count per instrument.

    A future position is positive when short, which is the hedge.
    '''
    return dict(self._positions)

  @property
  def strikes(self) -> dict[str, float]:
    '''Return the current strike per option instrument.'''
    return dict(self._strike)

  @property
  def total_cost(self) -> float:
    '''Return cumulative transaction cost charged so far, in rupees.'''
    return self._total_cost

  @property
  def decisions(self) -> list[dict[str, object]]:
    '''Return the per-step decision log, one entry per executed step.'''
    return list(self._decisions)

  def observation(self) -> Obs:
    '''Return the observation visible right now, without advancing.'''
    return self._observation()

  def info(self) -> Info:
    '''Return auxiliary state without advancing the episode.'''
    return self._info()

  def risk_report(self) -> dict[str, float]:
    '''Return the hedging-error statistics, reported separately.

    Zernikov's most useful methodological point is that accumulated
    reward, downside variance, ordinary variance and CVaR rank the same
    hedge differently, so a paper reporting only the trained objective is
    hiding the failure. All of them are reported here.

    Returns:
      Mapping of statistic name to value.
    '''
    errors = self._errors
    return {
      'mean_error': fmean(errors) if errors else 0.0,
      'error_stdev': stdev(errors) if len(errors) > 1 else 0.0,
      'semi_rmse': semi_rmse(errors),
      'cvar': cvar(errors, self.risk.alpha),
      'max_abs_error': max((abs(value) for value in errors), default=0.0),
      'alpha': self.risk.alpha,
    }

  def average_delta(self) -> float:
    '''Return mean net delta over the episode.

    A hedged book should sit near zero. A positive mean means the policy
    underhedged, which is what Francois et al. observed at low CVaR alpha
    and what Zernikov measured as an average delta haircut against BS.

    Returns:
      Mean net delta, or 0.0 for an empty episode.
    '''
    if not self._delta_log:
      return 0.0
    return fmean(self._delta_log)
