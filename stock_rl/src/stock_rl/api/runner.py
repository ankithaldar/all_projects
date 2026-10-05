#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The control arm: which providers exist, and how the cap reaches them.

``strategies()`` discovers the table from :mod:`stock_rl.baselines`'
``__all__`` rather than restating it, because a baseline added there and
missing from the control arm is the exact defect the endpoint exists to
prevent. :func:`run_baseline` is the default runner behind
``POST /api/backtest``, and it reuses
:func:`stock_rl.portfolio.run_portfolio` so the API, the RL environment
and the published baselines are all measured by one instrument with one
cost model. Three code paths would give three Sharpes, and three Sharpes
would be unfalsifiable.

The three helpers between them are what stops a reported cap from being a
requested cap. ``_accepts_cap`` asks a provider's signature rather than
assuming an answer, ``_capped`` binds the caller's cap by **keyword** --
the five providers disagree about where it sits -- ``_guarded`` turns a
provider's own exception into :class:`ProviderFailed` so it cannot be
reported to the caller as a bad request, and ``_applied_cap`` reads the
cap back off the book so the payload cannot claim a number no book backs.

Split out of the former single-module ``stock_rl.api`` without change.
'''

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from stock_rl import baselines
from stock_rl.api.constants import cap_keyword, not_advice
from stock_rl.api.errors import ProviderFailed, Refused
from stock_rl.api.models import BacktestRequest
from stock_rl.bars import Bar
from stock_rl.metrics import max_drawdown, sharpe_ratio
from stock_rl.portfolio import PortfolioResult, WeightProvider, run_portfolio

if TYPE_CHECKING:
  # Imported for the annotation only. ``stock_rl.api.service`` imports
  # :func:`run_baseline` as the default runner, so a runtime import here
  # would be a cycle; ``from __future__ import annotations`` keeps the
  # annotation a string either way.
  from stock_rl.api.service import ApiService


def _guarded(name: str, provider: WeightProvider) -> WeightProvider:
  '''Return ``provider`` wrapped so its own failures are distinguishable.

  :func:`stock_rl.portfolio.run_portfolio` raises ``ValueError`` for its
  own refusals -- unaligned panels, a history that leaves no room -- and
  lets whatever a provider raises pass straight through. That makes a
  ``ValueError`` ambiguous at every catch site above it: an off-by-one
  slice inside a provider is not a bad request, and reporting it as a 422
  tells the caller to change a request that was never the problem.

  The distinction cannot be recovered afterwards from the exception type
  alone, so it is made here, at the only point where the two are still
  separable: the wrapper is this module's, the exception it raises is
  :class:`ProviderFailed`, and a refusal from the backtester itself still
  arrives as the ``ValueError`` it always was.

  PONYTAIL: the wrapper converts rather than diagnoses. Ceiling: the 500
  body carries the exception repr and nothing about where inside the
  provider it came from. Upgrade path: let ``baselines.py`` declare its
  own error type and narrow this wrapper to it, so a genuine
  ``ValueError`` from a provider is reported as itself.

  Args:
    name: Provider name, for the message.
    provider: The callable to wrap.

  Returns:
    A callable with the same signature that reports a provider failure as
    :class:`ProviderFailed`.
  '''
  def guarded(visible: Mapping[str, Sequence[Bar]]) -> dict[str, float]:
    '''Return the provider's weights, or raise :class:`ProviderFailed`.

    Args:
      visible: Price panels visible at the decision bar.

    Returns:
      Target weight per symbol.

    Raises:
      ProviderFailed: Whatever the provider raised, renamed.
    '''
    try:
      return provider(visible)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      raise ProviderFailed(
        f'{name!r} raised instead of returning weights: {exc!r}') from exc
  return guarded


def _accepts_cap(provider: WeightProvider) -> bool:
  '''Return whether ``provider`` can be handed the cap by keyword.

  Every provider in :mod:`stock_rl.baselines` takes ``max_weight``, but
  the table is discovered from that module's ``__all__`` and is therefore
  open to anything registered there later, including a callable with no
  introspectable signature at all. The question is asked rather than
  assumed, because assuming it is what produced the original defect: an
  unbound cap is indistinguishable from a bound one until someone compares
  a requested number against the book that came back.

  A ``**kwargs`` parameter counts as accepting the cap, since a keyword
  cannot collide with anything it declares.

  PONYTAIL: introspection rather than a try/except around the call.
  Ceiling: a callable object whose ``__call__`` hides the keyword behind
  a decorator is reported as not accepting it, and its row then says so.
  Upgrade path: let a provider declare ``cap_keyword`` as an attribute.

  Args:
    provider: The weight provider to inspect.

  Returns:
    True when the cap can be passed as a named argument.
  '''
  try:
    parameters = inspect.signature(provider).parameters
  except (TypeError, ValueError):
    # A C builtin or a deliberately opaque callable. Its weight may not
    # be settable, and the row reports that rather than guessing.
    return False
  by_keyword = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
  )
  return any(
    parameter.kind is inspect.Parameter.VAR_KEYWORD
    or (parameter.name == cap_keyword and parameter.kind in by_keyword)
    for parameter in parameters.values()
  )


def _capped(provider: WeightProvider, max_weight: float) -> WeightProvider:
  '''Return ``provider`` with the caller's cap bound into it.

  The engine's cap is a ceiling, not a target.
  :func:`stock_rl.portfolio.run_portfolio` clamps whatever it is handed to
  ``max_weight``, so a provider that applies a *tighter* cap of its own
  produces a book strictly inside the ceiling and the requested number is
  never the binding constraint. That is not a harmless difference: each
  of the five baselines in :mod:`stock_rl.baselines` defaults its own
  ``max_weight`` to 0.10, so a caller asking for 0.30 used to be handed a
  0.10 book and a payload claiming 0.30, and three different requested
  caps returned a byte-identical Sharpe.

  Binding the cap here is what
  :func:`stock_rl.pipeline._measure_engines` already does for its own
  ``momentum_ranked`` leg, and it is done for both sides of the seam:
  the provider stops imposing its own default and the engine keeps
  enforcing the ceiling.

  A provider that cannot take the keyword is called unchanged rather than
  refused. Refusing would file a provider this module cannot introspect
  as a defect when it may simply derive weights from the panels and have
  no opinion about the cap at all. What is not left unspecified is the
  disclosure: the row reports ``cap_bound`` False alongside the cap the
  book was actually built at, so an unbound provider cannot leave a
  number in the payload that no book backs.

  Args:
    provider: The weight provider to bind.
    max_weight: Cap on any single symbol's weight.

  Returns:
    A one-argument callable, the shape ``run_portfolio`` calls.
  '''
  if _accepts_cap(provider):
    def bound(visible: Mapping[str, Sequence[Bar]]) -> dict[str, float]:
      '''Return the provider's weights under the requested cap.

      Args:
        visible: Price panels visible at the decision bar.

      Returns:
        Target weight per symbol.
      '''
      return provider(visible, **{cap_keyword: max_weight})
    return bound

  def unbound(visible: Mapping[str, Sequence[Bar]]) -> dict[str, float]:
    '''Return the provider's weights, which could not be given a cap.

    Args:
      visible: Price panels visible at the decision bar.

    Returns:
      Target weight per symbol.
    '''
    return provider(visible)
  return unbound


def _applied_cap(result: PortfolioResult) -> float | None:
  '''Return the largest weight the book actually held, or None.

  Measured from the snapshots the backtester recorded rather than from
  the request, so it cannot disagree with the book it describes. This is
  the number that makes ``applied == reported`` checkable: the engine
  guarantees ``applied <= requested``, and a provider whose internal cap
  was tighter is exactly the case where ``applied < requested``.

  A cap is a ceiling and not a target, so equality is the expected
  outcome only when the cap is the binding constraint -- 1/N over three
  symbols tops out at 0.333 no matter how large the requested cap is.
  Callers comparing the two need that in mind; see the module tests.

  Args:
    result: The backtester outcome.

  Returns:
    The heaviest single-symbol weight across every rebalance, or None if
    the run never rebalanced.
  '''
  held = [
    weight
    for snapshot in result.weights
    for weight in snapshot.values()
  ]
  return max(held) if held else None


def strategies() -> dict[str, WeightProvider]:
  '''Return every provider in :mod:`stock_rl.baselines`, by name.

  The table is discovered from the module's own ``__all__`` rather than
  written out here, so adding a baseline cannot leave it missing from
  the control arm. The whole point of the endpoint is that the control
  arm is complete.

  Returns:
    Mapping of provider name to the provider callable.
  '''
  found: dict[str, WeightProvider] = {}
  for name in baselines.__all__:
    found[name] = getattr(baselines, name)
  return found


def run_baseline(
  service: ApiService,
  request: BacktestRequest,
) -> dict[str, Any]:
  '''Run the requested baseline over the requested panels.

  The default runner. It reuses
  :func:`stock_rl.portfolio.run_portfolio`, so the API, the RL
  environment and the published baselines are measured by one
  instrument with one cost model -- three different Sharpes from three
  code paths would be unfalsifiable.

  ``request.max_weight`` is bound to the provider as well as to the
  engine, exactly as ``GET /api/baselines`` binds it, because this is
  the runner that endpoint's rows claim to reproduce. A row measured at
  one cap and a POST run at the same cap that disagreed about the cap
  would reintroduce the split this module exists to close. The payload
  reports ``applied_max_weight``, read back off the book, next to the
  requested cap echoed under ``request``.

  Args:
    service: Service supplying the panels.
    request: Validated trigger.

  Returns:
    JSON-safe mapping with the equity curve, returns, final weights and
    headline metrics.

  Raises:
    Refused: If the strategy is unknown, no panels were selected, or the
      backtester refuses the parameter combination.
    ProviderFailed: If the provider itself raises. Distinct from
      :class:`Refused` on purpose: a 422 says "change your request", and
      the request is not what is wrong here.
  '''
  table = strategies()
  if request.strategy not in table:
    raise Refused(
      f'unknown strategy {request.strategy!r}; known: {sorted(table)}')
  panels = service.panels_for(request.symbols)
  if not panels:
    raise Refused('no price panels matched the requested symbols')
  try:
    result = run_portfolio(
      panels,
      # Wrapped so a provider's own ValueError is not mistaken for the
      # backtester refusing the parameter combination below. The cap is
      # bound inside that wrapper, so the provider still sees exactly one
      # argument and a keyword it recognises.
      _guarded(request.strategy,
               _capped(table[request.strategy], request.max_weight)),
      capital=request.capital,
      rebalance_days=request.rebalance_days,
      max_weight=request.max_weight,
      history=request.history,
    )
  except ValueError as exc:
    # The backtester refusing a parameter combination the caller chose
    # is a 422, not a crash. Letting it out would put a traceback in an
    # HTTP response for an input the caller could have been told about.
    raise Refused(str(exc)) from exc
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise ProviderFailed(
      f'{request.strategy!r} raised instead of returning weights: '
      f'{exc!r}') from exc
  return {
    'strategy': request.strategy,
    'symbols': sorted(panels),
    'capital': request.capital,
    'equity': list(result.equity),
    'returns': list(result.returns),
    'weights': dict(result.weights[-1]) if result.weights else {},
    'sharpe': sharpe_ratio(result.returns),
    'max_drawdown': max_drawdown(result.equity).depth,
    'turnover': result.turnover,
    'total_cost': result.total_cost,
    'total_return_multiple': result.total_return_multiple,
    'rebalances': result.rebalances,
    'points': len(result.equity),
    # Measured off the book rather than echoed from the request, so the
    # two numbers on this payload can be compared against each other.
    'applied_max_weight': _applied_cap(result),
    'not_advice': not_advice,
  }
