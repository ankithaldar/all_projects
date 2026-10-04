#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Baseline strategies: the benchmark every learned policy must beat.

These exist first, not second, and that ordering is the whole point. A
learned policy that cannot beat equal weight has not earned its
complexity, and the only way to know is to have equal weight measured on
the identical backtester, costs, splits and fill assumptions.

Evidence behind the selection, from Indian and cross-market studies:

  equal weight
    DeMiguel, Garlappi & Uppal (2009), *Review of Financial Studies*:
    across 14 optimised models on 7 datasets, "none is consistently
    better than the 1/N rule". The estimation window sample-MVO needs to
    beat 1/N on 50 assets is roughly 6,000 months -- 500 years.

  momentum
    The strongest family in Indian data, replicated four times across
    three institutions. Maheshwari & Dhankar (2017) find it survives
    size, value and illiquidity controls and is driven by winners, so the
    long leg alone works without shorting. The 12-1 construction skips
    the most recent month to avoid the short-term reversal that contaminates
    naive 12-month lookback.

  low volatility
    Agarwalla, Jacob & Varma (IIMA) find BAB dominates size, value and
    momentum in India. A 4,400-stock, 19-year study finds the effect does
    not decay over time and survives factor controls. Implementations
    should weight by realised volatility rather than beta, since the
    volatility effect is stronger than the beta effect in Indian data.

  trend filter
    Mitra (2011) is the one peer-reviewed, cost-aware Indian study of
    moving-average rules. SMA(40) and SMA(60) clear realistic delivery
    costs with breakeven one-way costs of 0.91% and 1.24%. The 50/200
    crossover is untested in India and is therefore not offered here.

Deliberately absent: RSI, Bollinger, ADX, MACD and OBV as standalone
signals. None has Indian equity evidence, and ADX has none anywhere for
equities. See docs/research/methodology/technical-indicators-nse.md.
'''

from __future__ import annotations

from stock_rl.bars import Bar
from stock_rl.indicators import realized_volatility, sma

__all__ = [
  'buy_and_hold',
  'equal_weight',
  'low_volatility',
  'momentum_ranked',
  'trend_filtered_momentum',
]

#: Bars in one trading year, for annualising momentum.
_TRADING_DAYS = 252


def _closes(panel: list[Bar]) -> list[float]:
  '''Return closing prices from a bar panel.

  Args:
    panel: Bars in ascending time order.

  Returns:
    List of closing prices.
  '''
  return [bar.close for bar in panel]


def equal_weight(
  panels: dict[str, list[Bar]],
  max_weight: float = 0.10,
) -> dict[str, float]:
  '''Return equal weights across every symbol, the 1/N baseline.

  This is the control arm. DeMiguel (2009) showed 1/N is not reliably
  beaten by optimised allocation, and the Nifty-50 study found equal
  weight outscored both SAC and buy-and-hold. Any strategy that cannot
  clear it has not demonstrated value.

  Args:
    panels: Price panels, used only for their symbol keys.
    max_weight: Cap on any single weight. If 1/N exceeds the cap the
      weights are scaled down and the remainder stays in cash, which is
      the honest treatment rather than silently over-weighting.

  Returns:
    Target weight per symbol.
  '''
  symbols = sorted(panels)
  if not symbols:
    return {}
  weight = min(1.0 / len(symbols), max_weight)
  return {symbol: weight for symbol in symbols}


def buy_and_hold(
  panels: dict[str, list[Bar]],
  max_weight: float = 0.10,
) -> dict[str, float]:
  '''Return equal weights, the buy-and-hold control.

  Identical to ``equal_weight`` on a long-only universe, and kept as a
  separate name because it is a distinct hypothesis: passive exposure to
  the index. A long-only delivery book cannot harvest the short leg that
  drives much of the academic momentum premium, so momentum should be
  judged against this, not against a long-short backtest.

  Args:
    panels: Price panels, used only for their symbol keys.
    max_weight: Cap on any single weight.

  Returns:
    Target weight per symbol.
  '''
  return equal_weight(panels, max_weight)


def momentum_ranked(
  panels: dict[str, list[Bar]],
  lookback: int = 252,
  skip: int = 21,
  top: int = 10,
  max_weight: float = 0.10,
) -> dict[str, float]:
  '''Weight the highest-momentum symbols equally, the rest to cash.

  Implements the cross-sectional 12-1 construction: rank on trailing
  return over ``lookback`` bars, then skip the most recent ``skip``
  bars. The skip matters -- Sehgal & Balakrishnan (2002) found Indian
  short-horizon returns show *continuation*, and a naive 12-month
  lookback mixes that with the long-horizon reversal, degrading the
  signal.

  Args:
    panels: Price panels, all sharing one timeline.
    lookback: Total lookback in bars, typically 252.
    skip: Most recent bars to exclude, typically 21.
    top: How many symbols to hold.
    max_weight: Cap on any single weight.

  Returns:
    Target weight per symbol, with unheld symbols at zero.
  '''
  symbols = sorted(panels)
  scored: list[tuple[str, float]] = []
  needed = lookback + skip + 1
  for symbol in symbols:
    closes = _closes(panels[symbol])
    if len(closes) < needed:
      continue
    # 12-1 construction: return from `lookback + skip` bars ago forward
    # to `skip` bars ago, skipping the most recent month.
    #
    # The index arithmetic here is easy to invert and was inverted once:
    # reading `end` from further back than `start` yields old/recent
    # rather than recent/old, which is mean reversion wearing the label
    # of momentum and silently buys the worst performer.
    start = closes[-(lookback + skip + 1)]
    end = closes[-(skip + 1)]
    if start <= 0.0 or end <= 0.0:
      continue
    scored.append((symbol, end / start - 1.0))
  if not scored:
    return equal_weight(panels, max_weight)
  scored.sort(key=lambda item: item[1], reverse=True)
  chosen = [symbol for symbol, _ in scored[:top]]
  weight = min(1.0 / len(chosen), max_weight)
  return {symbol: weight if symbol in chosen else 0.0 for symbol in symbols}


def low_volatility(
  panels: dict[str, list[Bar]],
  window: int = 60,
  max_weight: float = 0.10,
) -> dict[str, float]:
  '''Weight symbols inversely to their realised volatility.

  The second-best evidenced anomaly in Indian equities after momentum, and
  it needs only price data. Weighting by volatility rather than beta is
  deliberate: the volatility effect is measurably stronger than the beta
  effect in Indian data.

  Args:
    panels: Price panels, all sharing one timeline.
    window: Lookback in bars for the volatility estimate.
    max_weight: Cap on any single weight.

  Returns:
    Target weight per symbol. Symbols without enough history are excluded.
  '''
  symbols = sorted(panels)
  inverse: list[tuple[str, float]] = []
  for symbol in symbols:
    closes = _closes(panels[symbol])
    series = realized_volatility(closes, window, _TRADING_DAYS)
    latest = series[-1] if series else None
    if latest is None or latest <= 0.0:
      continue
    inverse.append((symbol, 1.0 / latest))
  if not inverse:
    return equal_weight(panels, max_weight)
  total = sum(value for _, value in inverse)
  if total <= 0.0:
    return equal_weight(panels, max_weight)
  weights = {
    symbol: min(value / total, max_weight) for symbol, value in inverse
  }
  scale = sum(weights.values())
  if scale > 1.0:
    weights = {symbol: value / scale for symbol, value in weights.items()}
  return weights


def trend_filtered_momentum(
  panels: dict[str, list[Bar]],
  window: int = 252,
  skip: int = 21,
  top: int = 10,
  ma_window: int = 40,
  max_weight: float = 0.10,
) -> dict[str, float]:
  '''Hold momentum leaders only while they sit above a moving average.

  Uses Mitra (2011)'s SMA(40), which clears Indian delivery costs with a
  0.91 percent breakeven one-way cost, as a regime gate. The gate is the
  point: momentum is documented to turn negative in Indian crises
  (Maheshwari & Dhankar, 2017), and a trend filter is the cheapest
  available guard against buying that drawdown.

  Args:
    panels: Price panels, all sharing one timeline.
    window: Momentum lookback in bars.
    skip: Most recent bars to exclude from momentum.
    top: How many symbols to hold when the regime permits.
    ma_window: Moving-average window for the regime gate.
    max_weight: Cap on any single weight.

  Returns:
    Target weight per symbol. A flat book is returned when nothing
    qualifies, which is the correct answer and not a failure.
  '''
  base = momentum_ranked(panels, window, skip, top, max_weight)
  symbols = sorted(panels)
  filtered: dict[str, float] = {}
  for symbol in symbols:
    weight = base.get(symbol, 0.0)
    if weight <= 0.0:
      continue
    closes = _closes(panels[symbol])
    average = sma(closes, ma_window)
    latest = average[-1] if average else None
    if latest is None or closes[-1] <= latest:
      continue
    filtered[symbol] = weight
  return filtered
