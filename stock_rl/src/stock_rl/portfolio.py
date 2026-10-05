#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cross-sectional portfolio backtest over aligned price panels.

This is the measuring instrument every strategy is judged with, including
the RL environment. It exists because a single-symbol backtest cannot
answer the only question that matters here: does this beat equal weight?

Look-ahead is closed the same way ``engine.py`` closes it. A weight
decision taken at the close of bar ``t`` is filled at the **open** of bar
``t+1``, and the provider is only ever shown bars strictly before the
fill bar, so it structurally cannot price its own fill.

PONYTAIL: rebalancing is a fixed calendar interval rather than a
threshold on weight drift. Ceiling: a drift-triggered rebalance trades
less, but needs the provider to expose its target weights, which couples
the backtester to strategy internals. Upgrade path: accept an optional
``should_rebalance`` callable alongside the provider; nothing else in
this module changes.

**The return series covers the trading window only.** ``equity`` carries
one point per bar, because the equity path is a property of the whole
history, but ``returns`` starts at bar ``history`` -- the first bar on
which a decision can be taken -- so it holds exactly ``len - history``
entries and none of them is one of the warm-up bars of flat cash. That
window is the one :class:`stock_rl.rl.portfolio_env.WeightAllocationEnv`
reports, and the two engines are only comparable over a shared window: a
Sharpe computed over a series padded with ``history`` zeros carries a
mean and a variance the other engine's series does not have.

**A weight mapping is a whole book, not a patch.** A symbol the provider
does not name is a target of ``0.0``, i.e. an exit, and an empty mapping
is the flat book. See :func:`_normalise`. This was chosen over "omitted
means hold" because every provider in this repository expresses
exclusion by omission -- the top-N funding rule in
:mod:`stock_rl.experiment.harness`, the regime gate in
:func:`stock_rl.baselines.trend_filtered_momentum`, the missing
estimate in :func:`stock_rl.baselines.low_volatility` -- and all of
them mean "no money in that name", not "leave it alone". The other
engine now reads an omitted symbol the same way; the two defaults live
in two modules that cannot share one constant, so
``tests/test_weights_partial.py`` pins them to each other.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY, CostModel, Side
from stock_rl.weights import affordable_scale, clamp_weight
from stock_rl.metrics import (
  TRADING_DAYS_PER_YEAR,
  max_drawdown,
  sharpe_ratio,
  total_return,
)

__all__ = ['PortfolioResult', 'WeightProvider', 'run_portfolio']

#: Target weight for a symbol the provider did not name. Zero, i.e. an
#: exit, and named here because the reading is a decision rather than a
#: default: :class:`stock_rl.rl.portfolio_env.WeightAllocationEnv` used to
#: read the same omission as "hold current", so one provider book was
#: liquidated by this engine and held by that one, and the gap landed in
#: the cost line instead of in an error.
omitted_weight = 0.0

#: A weight provider maps visible price history to target weights. Only
#: bars strictly before the fill bar are passed in, so a provider cannot
#: see the price it will be filled at.
WeightProvider = Callable[[dict[str, list[Bar]]], dict[str, float]]


@dataclass(frozen=True, slots=True)
class PortfolioResult:
  '''Outcome of a cross-sectional backtest.

  Attributes:
    equity: Equity curve normalised to 1.0, one entry per bar including
      the warm-up.
    returns: Per-bar portfolio returns over the trading window, i.e. one
      entry per bar from ``history`` onward. Shorter than ``equity`` by
      exactly ``history``, and index-aligned with it from that point, so
      ``returns[index - history]`` is the return of ``equity[index]``.
    weights: Target weight snapshot per rebalance, in order.
    sharpe: Annualised Sharpe of the trading-window return series.
    max_drawdown: Deepest peak-to-trough decline as a positive fraction,
      over the whole equity curve.
    total_return_multiple: Growth factor, so 1.21 is a 21 percent gain.
    turnover: Total absolute weight traded across all rebalances.
    total_cost: Transaction cost charged, in rupees.
    rebalances: Number of rebalances performed.
  '''

  equity: list[float] = field(default_factory=list)
  returns: list[float] = field(default_factory=list)
  weights: list[dict[str, float]] = field(default_factory=list)
  sharpe: float = 0.0
  max_drawdown: float = 0.0
  total_return_multiple: float = 1.0
  turnover: float = 0.0
  total_cost: float = 0.0
  rebalances: int = 0


class _Book:
  '''Mutable portfolio state carried across bars.

  Attributes:
    cash: Uninvested cash in rupees.
    quantity: Share holdings per symbol.
    cost: Cumulative transaction cost charged in rupees.
    traded: Cumulative absolute weight traded.
  '''

  __slots__ = ('cash', 'cost', 'quantity', 'traded')

  def __init__(self, capital: float, symbols: tuple[str, ...]) -> None:
    '''Start a book with all cash and no positions.

    Args:
      capital: Starting cash in rupees.
      symbols: Symbols in the portfolio.
    '''
    self.cash = capital
    self.quantity = dict.fromkeys(symbols, 0)
    self.cost = 0.0
    self.traded = 0.0


def _validate(
  panels: dict[str, list[Bar]],
  capital: float,
  rebalance_days: int,
  history: int,
) -> int:
  '''Validate inputs and return the number of bars per panel.

  Args:
    panels: Mapping of symbol to bars.
    capital: Starting cash.
    rebalance_days: Bars between rebalances.
    history: Minimum bars before the first rebalance.

  Returns:
    Number of bars per panel.

  Raises:
    ValueError: If any argument is invalid or leaves no room to trade.
  '''
  if not panels:
    raise ValueError('panels must not be empty')
  lengths = {len(bars) for bars in panels.values()}
  if len(lengths) != 1:
    raise ValueError(f'panels must be aligned, got {sorted(lengths)}')
  length = lengths.pop()
  if length < 2:
    raise ValueError(f'need at least 2 bars per panel, got {length}')
  if capital <= 0.0:
    raise ValueError(f'capital must be positive, got {capital}')
  if rebalance_days < 1:
    raise ValueError(f'rebalance_days must be >= 1, got {rebalance_days}')
  if history < 1:
    raise ValueError(f'history must be >= 1, got {history}')
  if history + rebalance_days >= length:
    raise ValueError(
      f'history={history} plus rebalance_days={rebalance_days} leaves no '
      f'room in a {length} bar series')
  return length


def _normalise(
  raw: dict[str, float],
  symbols: tuple[str, ...],
  max_weight: float,
) -> dict[str, float]:
  '''Clamp and renormalise weights into a feasible long-only portfolio.

  Feasibility is enforced here rather than inside the provider, so a
  strategy author cannot breach a cap by forgetting it, and every
  strategy is compared on identical rules.

  **A symbol absent from ``raw`` is an exit: a target of
  :data:`omitted_weight`, which is zero, not a hold.** The mapping is
  the whole intended book: a name the provider declines to name gets no
  capital, which is what the top-N funding rule, the trend gate and the
  volatility screen all mean when they leave a name out. The
  alternative -- reading omission as "keep whatever is held" -- was
  measurably worse, because a provider that forgets a name would then
  quietly keep a position its strategy had stopped wanting, invisibly
  and without a cost. It was also the reading
  :class:`stock_rl.rl.portfolio_env.WeightAllocationEnv` used, so the two
  engines traded different books for one strategy and reported the
  difference as turnover and rupees. An empty ``raw`` is therefore the
  flat book, every symbol to zero.

  Args:
    raw: Desired weight per symbol. May omit names, which become
      :data:`omitted_weight`.
    symbols: Symbols in the portfolio, in fixed order.
    max_weight: Cap on any single symbol.

  Returns:
    Weights summing to at most 1.0, one entry per symbol in ``symbols``.
    The remainder is cash.
  '''
  wanted = {
    symbol: clamp_weight(float(raw.get(symbol, omitted_weight)), max_weight)
    for symbol in symbols
  }
  total = sum(wanted.values())
  if total > 1.0:
    scale = 1.0 / total
    return {symbol: weight * scale for symbol, weight in wanted.items()}
  return wanted


def _mark(
  book: _Book,
  panels: dict[str, list[Bar]],
  symbols: tuple[str, ...],
  index: int,
) -> float:
  '''Return portfolio value marked at a bar's close.

  Args:
    book: Current book.
    panels: Price panels.
    symbols: Symbols in the portfolio.
    index: Bar index at which to mark holdings.

  Returns:
    Portfolio value in rupees.
  '''
  held = sum(
    book.quantity[symbol] * panels[symbol][index].close
    for symbol in symbols
  )
  return book.cash + held


def _rebalance(
  book: _Book,
  panels: dict[str, list[Bar]],
  symbols: tuple[str, ...],
  index: int,
  mark: float,
  target: dict[str, float],
  costs: CostModel,
) -> None:
  '''Trade toward the target weights at the bar's open.

  Both ``cash`` and ``quantity`` are updated, so the book stays
  self-consistent: shares are never held without the cash having been
  deducted. Getting that wrong silently inflates every equity curve.

  Args:
    book: Book to mutate in place.
    panels: Price panels.
    symbols: Symbols in the portfolio.
    index: Bar index whose open is the fill price.
    mark: Portfolio value used to size the orders.
    target: Target weights after clamping.
    costs: Transaction cost model.
  '''
  if mark <= 0.0:
    return
  prices = {}
  deltas = {}
  for symbol in symbols:
    price = panels[symbol][index].open
    if price <= 0.0:
      continue
    prices[symbol] = price
    deltas[symbol] = int(target[symbol] * mark / price) - book.quantity[symbol]
  # Affordability. Sizing int(target * mark / price) already bounds the
  # notional by the portfolio value, but it ignores the charges, so a
  # fully-invested book necessarily overspends. Left alone that is
  # unpriced borrowing: correct on a flat series, and free leverage on a
  # rising one. Deltas are therefore scaled to fit the cash actually
  # available after costs.
  scale = affordable_scale(book.cash, prices, deltas, costs)
  for symbol, delta in deltas.items():
    if delta == 0:
      continue
    scaled = int(delta * scale)
    if scaled == 0:
      continue
    notional = abs(scaled) * prices[symbol]
    charge = costs.one_way(
      Side.BUY if scaled > 0 else Side.SELL, notional)
    book.cash -= scaled * prices[symbol] + charge
    book.quantity[symbol] += scaled
    book.cost += charge
    book.traded += notional / mark


def run_portfolio(
  panels: dict[str, list[Bar]],
  provider: WeightProvider,
  capital: float = 10_000_000.0,
  costs: CostModel = DELIVERY,
  rebalance_days: int = 21,
  max_weight: float = 0.10,
  history: int = 60,
) -> PortfolioResult:
  '''Backtest a weight provider over aligned daily panels.

  Args:
    panels: Mapping of symbol to bars, all sharing one timeline.
    provider: Maps visible history to the complete target-weight book.
      A symbol it leaves out is an exit, see :func:`_normalise`.
    capital: Starting cash in rupees.
    costs: Transaction cost model.
    rebalance_days: Bars between rebalances. The default is roughly
      monthly, which keeps turnover near the level where Indian delivery
      costs stop dominating the result.
    max_weight: Cap on any single symbol's weight.
    history: Minimum bars required before the first rebalance.

  Returns:
    A ``PortfolioResult``.

  Raises:
    ValueError: If panels are unusable or the arguments leave no room to
      trade.
  '''
  length = _validate(panels, capital, rebalance_days, history)
  symbols = tuple(sorted(panels))
  book = _Book(capital, symbols)
  equity: list[float] = []
  returns: list[float] = []
  snapshots: list[dict[str, float]] = []

  for index in range(length):
    if index > 0 and index % rebalance_days == 0 and index >= history:
      visible = {
        symbol: panel[:index] for symbol, panel in panels.items()
      }
      target = _normalise(provider(visible), symbols, max_weight)
      _rebalance(
        book, panels, symbols, index,
        _mark(book, panels, symbols, index - 1), target, costs)
      snapshots.append(dict(target))
    equity.append(_mark(book, panels, symbols, index) / capital)
    returns.append(
      0.0 if index == 0 else equity[index] / equity[index - 1] - 1.0)

  # Trimmed to the trading window, see the module docstring. The warm-up
  # bars are flat cash, so they contribute exactly 0.0 and only dilute the
  # mean while inflating nothing: they are the reason the Sharpe of this
  # engine disagreed with the environment's for reasons that had nothing
  # to do with the strategy.
  return PortfolioResult(
    equity=equity,
    returns=returns[history:],
    weights=snapshots,
    sharpe=sharpe_ratio(returns[history:], periods=TRADING_DAYS_PER_YEAR),
    max_drawdown=max_drawdown(equity).depth,
    total_return_multiple=total_return(returns[history:]),
    turnover=book.traded,
    total_cost=book.cost,
    rebalances=len(snapshots),
  )
