#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Indicator-driven strategies, judged against the simple baselines.

Five strategies, each built only on the seven primitives in
:mod:`stock_rl.indicators`, each written as a weight provider so that
:func:`stock_rl.portfolio.run_portfolio` -- the one backtester in this
repository -- is what measures it. The comparison against
:mod:`stock_rl.baselines` is the whole point: a strategy that cannot beat
equal weight on the identical engine has not demonstrated value, and on
the evidence assembled in
``docs/research/methodology/technical-indicators-nse.md`` losing to it is
the expected outcome for four of these five, not a surprise.

**Every threshold here is declared, not fitted.** The RSI entry level, the
200-bar EMA, the volatility band and the ATR multiple are conventional
readings taken from published descriptions of the indicator, not values
chosen because they backtested well. Nothing in this module was tuned
against a result, and the declared values are repeated verbatim in
``docs/INDICATOR-STRATEGIES.md`` so a reader can check that claim.
:func:`require_history` enforces the consequence: a panel too short to
support a claim is refused with a message rather than turned into a number.

**No performance claim is made anywhere in this module.** The measured
numbers live in ``docs/INDICATOR-STRATEGIES.md``, and every one of them is
a property of a SYNTHETIC price generator rather than a fact about Indian
equities. ``docs/research/methodology/technical-indicators-nse.md``
records that the indicator thresholds behind this project's ranking have
no published Indian evidence and that standalone technical signals have
been measured inverted, so a number computed on generated prices says
nothing about whether these rules would survive real costs.

No provider takes a ``max_weight``. :func:`stock_rl.portfolio.run_portfolio`
clamps and rescales every book it is handed, so a cap supplied here could
only disagree with the one actually enforced, and passing it would let the
two engines measure different books.

Providers return the COMPLETE book: every panel symbol is named, and one
that does not qualify gets an explicit ``0.0``. That matches
:data:`stock_rl.portfolio.omitted_weight`, which already reads an omitted
symbol as an exit, and it sidesteps the partial-mapping ambiguity
:mod:`stock_rl.portfolio` documents at length.

# ponytail: every strategy is long-only and holds at most ``top`` names, so
# none of them can express a short leg, a hedge or an intra-index spread.
# Ceiling: a book that wants to be flat in a regime can only say so with
# cash, and Indian momentum being winner-driven makes the missing short leg
# a real rather than a formal limitation. Upgrade path: accept providers
# that emit negative weights and let ``clamp_weight`` refuse them loudly;
# no engine change is needed to observe the refusal.
'''

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY, CostModel
from stock_rl.indicators import (
  atr,
  ema,
  momentum,
  realized_volatility,
  rsi,
  trend_slope,
)
from stock_rl.metrics import TRADING_DAYS_PER_YEAR, minimum_backtest_length
from stock_rl.portfolio import WeightProvider, run_portfolio

__all__ = [
  'Measurement',
  'atr_breakout',
  'ema_trend_following',
  'measure',
  'require_history',
  'rsi_mean_reversion',
  'volatility_filtered_momentum',
  'volatility_scaled_momentum',
]


@dataclass(frozen=True, slots=True)
class Measurement:
  '''One strategy's measured result on one panel through one engine.

  Attributes:
    strategy: Name the caller supplied for the provider.
    sharpe: Annualised Sharpe over the trading window.
    total_return_multiple: Growth factor, so 1.21 is a 21 percent gain.
    max_drawdown: Deepest peak-to-trough decline as a positive fraction.
    turnover: Total absolute weight traded across all rebalances.
    total_cost: Transaction cost charged, in rupees.
    rebalances: Number of rebalances performed.
    bars: Per-bar observations the Sharpe was computed over.
  '''

  strategy: str
  sharpe: float
  total_return_multiple: float
  max_drawdown: float
  turnover: float
  total_cost: float
  rebalances: int
  bars: int


def _closes(panel: list[Bar]) -> list[float]:
  '''Return closing prices from a bar panel.

  Args:
    panel: Bars in ascending time order.

  Returns:
    List of closing prices.
  '''
  return [bar.close for bar in panel]


def _rank(
  scores: list[tuple[str, float]],
  top: int,
) -> list[tuple[str, float]]:
  '''Return the ``top`` best-scoring pairs, highest score first.

  Ties keep their original order, and the caller iterates symbols in
  sorted order, so the ranking is deterministic: two runs over the same
  panel cannot disagree because of a sort that is not stable.

  Args:
    scores: ``(symbol, score)`` pairs.
    top: How many pairs to keep.

  Returns:
    At most ``top`` pairs, ordered by descending score.
  '''
  return sorted(scores, key=lambda item: item[1], reverse=True)[:top]


def _book(
  symbols: list[str],
  scores: list[tuple[str, float]],
) -> dict[str, float]:
  '''Return the complete target-weight mapping for a scored book.

  Qualifying names share the whole book in proportion to their score, so a
  name the strategy likes most carries the most capital and a weak signal
  leaves the remainder in the cash leg. Nothing is silently leveraged:
  :func:`stock_rl.portfolio.run_portfolio` clamps and rescales whatever
  comes back.

  Args:
    symbols: Every symbol in the panel, in sorted order.
    scores: ``(symbol, positive score)`` pairs for the names that qualify.

  Returns:
    One weight per symbol in ``symbols``. Qualifying names sum to 1.0;
    everything else is named explicitly at ``0.0`` rather than omitted.
  '''
  wanted = dict(scores)
  total = sum(wanted.values())
  if total <= 0.0:
    return {symbol: 0.0 for symbol in symbols}
  return {
    symbol: wanted.get(symbol, 0.0) / total for symbol in symbols
  }


def _skip_momentum(
  closes: list[float],
  lookback: int,
  skip: int,
) -> float | None:
  '''Return the 12-1 style trailing return, or ``None``.

  The skip is applied by dropping the most recent ``skip`` closes before
  calling :func:`stock_rl.indicators.momentum`, rather than by indexing
  backwards from the end. That arithmetic is easy to invert and was
  inverted once in this repository: reading the endpoints the wrong way
  round turns momentum into mean reversion and silently buys the worst
  performer. Truncating a list cannot invert.

  Sehgal & Balakrishnan (2002) found Indian short-horizon returns continue
  rather than reverse, so the most recent bars contaminate a long-horizon
  read; the skip is why the 12-1 construction is not a plain 12-month
  lookback.

  Args:
    closes: Closing prices in ascending time order.
    lookback: Total lookback in bars.
    skip: Most recent bars to exclude from the measurement.

  Returns:
    Fractional return over the window, or ``None`` if there is not enough
    history or the base price is not positive.
  '''
  signal = closes[:-skip] if skip > 0 else closes
  return momentum(signal, lookback)


def rsi_mean_reversion(
  panels: dict[str, list[Bar]],
  window: int = 14,
  entry: float = 30.0,
  top: int = 10,
) -> dict[str, float]:
  '''Buy oversold names, sized by how oversold they are.

  Declared, not fitted: Wilder RSI over ``window`` bars with the entry
  level at the conventional ``entry``. RSI(14) is the reading almost every
  description of the indicator uses, and 30 is the oversold line printed
  on it. Neither value was adjusted after seeing a result.

  The depth of the reading, not its direction, is the size. A name at RSI
  12 is a bigger claim than one at 28 because the distance below the line
  is the whole of the evidence the strategy has. A name sitting exactly on
  the line has none, and is sized at zero.

  Held as a mean-reversion hypothesis, and the repository's evidence says
  to expect it to lose: Rink (2023) tests 600 RSI rules and finds the
  family is never the best outside a handful of markets, and
  ``docs/research/methodology/technical-indicators-nse.md`` lists RSI
  among the indicators with no Indian equity evidence. That is the reason
  it is worth measuring rather than a reason to expect a result.

  Args:
    panels: Price panels, all sharing one timeline.
    window: RSI window in bars. Wilder's own default is 14.
    entry: Oversold threshold. A name is bought at or below it.
    top: How many names to hold when more qualify.

  Returns:
    Target weight per symbol over the whole panel. A name without
    ``window + 1`` bars, or whose RSI cannot be read, is named at ``0.0``:
    an unmeasurable signal is refused, not guessed at.
  '''
  if entry <= 0.0:
    raise ValueError(f'entry must be positive, got {entry}')
  symbols = sorted(panels)
  scores: list[tuple[str, float]] = []
  needed = window + 1
  for symbol in symbols:
    closes = _closes(panels[symbol])
    if not closes or len(closes) < needed:
      continue
    latest = rsi(closes, window)[-1]
    if latest is None or latest > entry:
      continue
    scores.append((symbol, max(entry - latest, 0.0)))
  return _book(symbols, _rank(scores, top))


def ema_trend_following(
  panels: dict[str, list[Bar]],
  slow: int = 200,
  trend: int = 20,
  top: int = 10,
) -> dict[str, float]:
  '''Hold names above a long EMA whose recent slope is still rising.

  Declared, not fitted: the 200-bar EMA is the conventional long window,
  and the 20-bar least-squares slope is the conventional one-month
  direction check. Neither was adjusted after seeing a result.

  The 200-bar EMA needs a name to have cleared a full year of prices before
  it means anything, so a panel shorter than that gets a flat book rather
  than a reading off an EMA still converging on its seed value.

  The two conditions are deliberately not one. Price above a long average
  says a name has already risen; a positive recent slope says it is still
  rising. Requiring both is a statement that a trend which has already
  turned is not a trend, and it is the difference between a moving-average
  regime filter and a crossover rule.

  **This pairing has no Indian evidence and the repository says so.**
  ``docs/research/methodology/technical-indicators-nse.md`` records that
  50/200 is folklore in India, that the only cost-aware peer-reviewed
  Indian study of moving-average rules (Mitra 2011) found short crossovers
  losing 7 to 13 percent annualised before costs, and that the
  200-bar EMA was tested paired with a 9- or 12-bar EMA rather than with a
  50-bar one. 200 is used because it is the declared conventional value,
  not because it is the evidenced one.

  Args:
    panels: Price panels, all sharing one timeline.
    slow: Span of the long EMA in bars.
    trend: Bars fitted by the direction slope.
    top: How many names to hold when more qualify.

  Returns:
    Target weight per symbol over the whole panel, each qualifying name
      holding a share of the book proportional to how far its close sits
      above the long EMA.
  '''
  symbols = sorted(panels)
  scores: list[tuple[str, float]] = []
  needed = max(slow, trend)
  for symbol in symbols:
    closes = _closes(panels[symbol])
    if not closes or len(closes) < needed:
      continue
    # `ema` seeds at its first close rather than returning None, so a
    # length-checked panel always has a reading here. The length check is
    # what keeps that seed value out of the book.
    average = ema(closes, slow)[-1]
    direction = trend_slope(closes, trend)
    if average <= 0.0 or closes[-1] <= average:
      continue
    if direction is None or direction <= 0.0:
      continue
    scores.append((symbol, closes[-1] / average - 1.0))
  return _book(symbols, _rank(scores, top))


def volatility_scaled_momentum(
  panels: dict[str, list[Bar]],
  lookback: int = 252,
  skip: int = 21,
  vol_window: int = 60,
  top: int = 10,
) -> dict[str, float]:
  '''Weight the momentum leaders by how little they move.

  The momentum leg is the same 12-1 cross-sectional construction as
  :func:`stock_rl.baselines.momentum_ranked`, so the comparison against it
  isolates one change: every surviving name's share of the book is
  divided by its realised volatility. A name that earned its place with
  the same return but twice the daily range gets half the capital.

  Declared, not fitted: a 252-bar lookback skipped over the most recent
  21 bars, and a 60-bar volatility window, are the conventional daily-bar
  horizons. None of the three was adjusted after seeing a result.

  Only positive momentum is taken, because a long-only delivery book
  cannot harvest the short leg and Maheshwari & Dhankar (2017) find
  Indian momentum is asymmetric and driven by the winners, so the long leg
  is the part that survives.

  This is the most defensible of the five here, for one reason: it is the
  only one whose risk adjustment rests on the volatility effect rather
  than on an untested oscillator. Volume-scaled volatility is the
  strongest second anomaly in Indian data, and dividing a size by a
  volatility estimate is a position-sizing convention that does not need
  a directional claim about the indicator at all. It is offered as a
  candidate to keep, not as a demonstrated improvement: no comparison
  here is a result.

  # ponytail: realised volatility is estimated from closes only, so an
  # overnight gap is priced as if it were a continuous move, and no
  # Parkinson or Garman-Klass range estimator is used even though the bars
  # carry highs and lows. Ceiling: roughly a fifth of the variance in a
  # real daily bar lives in the intrabar range, so the estimate is noisy
  # where it could be precise, and a single limit-down prints as ordinary
  # volatility. Upgrade path: pass a close series built from an
  # intrabar-consistent estimator; nothing else in this function changes.

  Args:
    panels: Price panels, all sharing one timeline.
    lookback: Momentum lookback in bars.
    skip: Most recent bars excluded from the momentum measurement.
    vol_window: Bars in the realised-volatility estimate.
    top: How many names to hold when more qualify.

  Returns:
    Target weight per symbol over the whole panel, each qualifying name
      holding a share of the book proportional to momentum per unit of
      realised volatility.
  '''
  symbols = sorted(panels)
  scores: list[tuple[str, float]] = []
  needed = max(lookback + skip + 1, vol_window + 1)
  for symbol in symbols:
    closes = _closes(panels[symbol])
    if not closes or len(closes) < needed:
      continue
    value = _skip_momentum(closes, lookback, skip)
    if value is None or value <= 0.0:
      continue
    series = realized_volatility(closes, vol_window, TRADING_DAYS_PER_YEAR)
    latest = series[-1] if series else None
    if latest is None or latest <= 0.0:
      continue
    scores.append((symbol, value / latest))
  return _book(symbols, _rank(scores, top))


def atr_breakout(
  panels: dict[str, list[Bar]],
  window: int = 14,
  lookback: int = 60,
  strength: float = 1.0,
  top: int = 10,
) -> dict[str, float]:
  '''Buy names making a new high on an above-average range.

  Declared, not fitted: Wilder ATR over ``window`` bars, a ``lookback``-bar
  closing high, and an advance of at least ``strength`` average true
  ranges. 14 is Wilder's own default. 60 bars is about a quarter of
  trading, which sits inside the 20-to-60-bar horizon
  ``docs/research/methodology/technical-indicators-nse.md`` identifies as
  the only cost-viable moving-average band in Indian data, rather than the
  retail 52-week high. One average true range is the smallest advance that
  is distinguishable from a rounding tick.

  ATR does two jobs, which is why it is here rather than a directional
  oscillator. It is the confirmation bar: a new closing high reached on a
  move no larger than the name's own recent range is not a breakout, it is
  drift. And it is the unit of size: the score is the advance measured in
  average true ranges, so a 3 percent move means something different for
  a quiet name than for a wild one, and the weight says so.

  ``docs/research/methodology/technical-indicators-nse.md`` is explicit
  that ATR has zero evidence of directional predictive power anywhere and
  is a sizing tool. It is used as one here; the direction comes from the
  high, which is a convention this repository has no Indian evidence for.

  # ponytail: an exit rule is missing. A name enters on a breakout and is
  # only reconsidered at the next rebalance, so the holding period is the
  # rebalance interval rather than a stop. Ceiling: a breakout that fails
  # the same week is held until the next monthly rebalance, which is a real
  # loss of capital and the rule Mitra's breakeven-cost table is most
  # sensitive to. Upgrade path: add an explicit ``stop`` in ATR multiples
  # evaluated in the same provider call, which is a subtraction from the
  # visible panel and therefore costs no new data.

  Args:
    panels: Price panels, all sharing one timeline.
    window: ATR window in bars.
    lookback: Bars of closing history that define "a new high".
    strength: Advance required, in average true ranges.
    top: How many names to hold when more qualify.

  Returns:
    Target weight per symbol over the whole panel. A name without enough
      bars for both the ATR and the high is named at ``0.0``.
  '''
  symbols = sorted(panels)
  scores: list[tuple[str, float]] = []
  needed = max(window, lookback) + 1
  for symbol in symbols:
    panel = panels[symbol]
    closes = _closes(panel)
    if not closes or len(closes) < needed:
      continue
    latest = atr(panel, window)[-1]
    if latest is None or latest <= 0.0:
      continue
    advance = closes[-1] - closes[-2]
    if advance < strength * latest:
      continue
    if closes[-1] < max(closes[-lookback:]):
      continue
    scores.append((symbol, advance / latest))
  return _book(symbols, _rank(scores, top))


def volatility_filtered_momentum(
  panels: dict[str, list[Bar]],
  lookback: int = 252,
  skip: int = 21,
  vol_window: int = 60,
  low: float = 0.15,
  high: float = 0.45,
  top: int = 10,
) -> dict[str, float]:
  '''Hold momentum leaders only inside a declared volatility band.

  Declared, not fitted: the same 12-1 momentum leg as
  :func:`volatility_scaled_momentum`, gated on 60-bar annualised realised
  volatility lying between ``low`` and ``high``. The band's endpoints come
  from published descriptions of Indian large-cap volatility, not from a
  search: a Nifty-style index has spent most of the last two decades
  between roughly 12 and 30 percent annualised, so ``low`` excludes names
  calmer than any broad Indian index and ``high`` admits only crisis
  conditions. Neither endpoint was adjusted after seeing a result.

  Both ends are declared for a reason, and the reasons point opposite
  ways. Han, Yang & Zhou (2013) find moving-average timing works on
  high-volatility portfolios and fails on low-volatility ones, so the calm
  end is where a technical rule is measurably weakest. Maheshwari &
  Dhankar (2017) measure Indian momentum turning negative during crises,
  so the panic end is where the signal itself inverts. A band excludes
  both.

  The repository's own methodological note says to condition on volatility
  rather than on a trend-or-range label, which is why this gate is a band
  on volatility rather than an ADX-style regime read.

  # ponytail: the band is cross-sectional and fixed, so it applies the same
  # absolute thresholds on every date and in every regime. Ceiling: a
  # market that simply re-levels its volatility leaves the book entirely in
  # cash for as long as the regime lasts, and no re-entry rule exists.
  # Upgrade path: compare the reading against its own trailing cross-
  # sectional median, which is computable from the same visible panels.

  Args:
    panels: Price panels, all sharing one timeline.
    lookback: Momentum lookback in bars.
    skip: Most recent bars excluded from the momentum measurement.
    vol_window: Bars in the realised-volatility estimate.
    low: Inclusive lower bound on annualised realised volatility.
    high: Inclusive upper bound on annualised realised volatility.
    top: How many names to hold when more qualify.

  Returns:
    Target weight per symbol over the whole panel. A name outside the
      volatility band, or without enough bars to measure it, is named at
      ``0.0``.
  '''
  symbols = sorted(panels)
  scores: list[tuple[str, float]] = []
  needed = max(lookback + skip + 1, vol_window + 1)
  for symbol in symbols:
    closes = _closes(panels[symbol])
    if not closes or len(closes) < needed:
      continue
    value = _skip_momentum(closes, lookback, skip)
    if value is None or value <= 0.0:
      continue
    series = realized_volatility(closes, vol_window, TRADING_DAYS_PER_YEAR)
    latest = series[-1] if series else None
    if latest is None or latest <= 0.0:
      continue
    if not low <= latest <= high:
      continue
    scores.append((symbol, value))
  return _book(symbols, _rank(scores, top))


def require_history(
  panels: dict[str, list[Bar]],
  strategy: str,
  trials: int,
  target_sharpe: float,
  history: int = 60,
) -> float:
  '''Return the years of history available, or refuse to report at all.

  :func:`stock_rl.metrics.minimum_backtest_length` is the gate, and it is
  asked the only question worth asking: the claim a reader could make from
  a table of results is the best row of it, so the gate is asked whether
  *that* is supportable. Gating one row and quoting another would be
  selecting a Sharpe after seeing it.

  ``trials`` is the honest count of configurations compared, and the caller
  is the only party that knows it. Undercounting it deflates nothing and
  weakens the result silently, which is the reason this is an argument
  rather than a constant.

  Args:
    panels: The panel every row was measured over.
    strategy: Name of the best row, quoted in the refusal.
    trials: Number of configurations compared before selecting a row.
    target_sharpe: Annualised Sharpe of that best row.
    history: Warm-up bars the engine held in cash, matching the call to
      :func:`stock_rl.portfolio.run_portfolio`.

  Returns:
    Years of trading-window returns available.

  Raises:
    ValueError: If the panels are empty, if ``target_sharpe`` is not
      positive, or if the panel is too short for ``trials`` at that
      Sharpe. The message names
      :func:`stock_rl.metrics.minimum_backtest_length` in every case,
      because a refusal nobody can trace to a named gate gets ignored.
  '''
  if not panels:
    raise ValueError('panels must not be empty')
  bars = min(len(panel) for panel in panels.values())
  available = max(bars - history, 0) / TRADING_DAYS_PER_YEAR
  if target_sharpe <= 0.0:
    raise ValueError(
      f'{strategy}: best Sharpe {target_sharpe:.3f} is not positive, so '
      f'there is nothing for stock_rl.metrics.minimum_backtest_length to '
      f'defend')
  required = minimum_backtest_length(trials, target_sharpe)
  if required > available:
    raise ValueError(
      f'{strategy}: {trials} configurations at Sharpe {target_sharpe:.3f} '
      f'need {required:.2f} years of returns per '
      f'stock_rl.metrics.minimum_backtest_length, only {available:.2f} '
      f'available in {bars} bars')
  return available


def measure(
  panels: dict[str, list[Bar]],
  runs: Sequence[tuple[str, WeightProvider]],
  capital: float = 10_000_000.0,
  costs: CostModel = DELIVERY,
  rebalance_days: int = 21,
  max_weight: float = 0.10,
  history: int = 60,
) -> tuple[Measurement, ...]:
  '''Run named providers through one engine, then gate the whole report.

  One engine, one cost model, one specification, one panel for every row.
  That is the entire content of this function: rows measured differently
  are unrelated backtests, and the only comparison the project needs --
  these strategies against the five baselines -- exists if and only if the
  instrument is identical.

  Every parameter is passed to :func:`stock_rl.portfolio.run_portfolio` by
  keyword. The engine decides where the fill is, and a positional call
  would let a re-ordered signature silently change what was measured.

  The gate is applied to the best row, not per row, and it raises rather
  than blanking out a Sharpe. A caller that caught the error would still
  hold the equity curves, which is why the refusal is a raise: the point is
  that no number leaves this function for a panel that cannot support it.

  Args:
    panels: Price panels, all sharing one timeline.
    runs: ``(name, provider)`` pairs to measure, at least two. Names are
      the caller's because this module holds no registry of its own.
    capital: Starting cash in rupees.
    costs: Transaction cost model.
    rebalance_days: Bars between rebalances.
    max_weight: Cap on any single symbol's weight.
    history: Bars held in cash before the first rebalance.

  Returns:
    One :class:`Measurement` per provider, in the order supplied.

  Raises:
    ValueError: If fewer than two providers are supplied, or if the engine
      refuses the panel, or if
      :func:`require_history` refuses the report.
  '''
  if len(runs) < 2:
    raise ValueError(
      f'a comparison of one configuration supports no claim to deflate, '
      f'got {[name for name, _ in runs]}')
  measured = [
    (name, run_portfolio(
      panels, provider,
      capital=capital, costs=costs, rebalance_days=rebalance_days,
      max_weight=max_weight, history=history))
    for name, provider in runs
  ]
  best_name, best = max(measured, key=lambda item: item[1].sharpe)
  require_history(panels, best_name, len(measured), best.sharpe, history)
  return tuple(
    Measurement(
      strategy=name,
      sharpe=result.sharpe,
      total_return_multiple=result.total_return_multiple,
      max_drawdown=result.max_drawdown,
      turnover=result.turnover,
      total_cost=result.total_cost,
      rebalances=result.rebalances,
      bars=len(result.returns),
    )
    for name, result in measured
  )
