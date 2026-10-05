#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Adversarial review of the graph, context and sentiment packages.

Every test here names a claim a module makes in a docstring or a
comment and then calls that module to check the claim. A test that
passes means the claim held. A test that fails means the docstring is
false, and the failing assertion names the file and the line.

The failing tests are the findings. There are eight:

* :func:`test_the_kill_criteria_gate_fails_closed_on_a_nan` -- the
  pre-registered gate that decides whether the whole context programme
  is deleted returns ``passed=True`` with **zero** failures when the
  p-values, the t-statistic or the turnover are ``nan``. Every
  comparison against ``nan`` is false, so no threshold is evaluated and
  none fails, and the module docstring's promise that "it cleared the
  bar" is a checkable statement becomes true for a result that does not
  exist.
* :func:`test_the_look_ahead_filter_is_not_opt_in` -- readings dated
  365 days **after** the decision bar reach a ``SymbolContext`` as
  ``sentiment_score=1.0`` and ``sentiment_usable=1.0`` on the
  **default** call shape, because ``decision_bar`` defaults to ``None``
  on both ``aggregate`` and ``symbol_context`` and a ``None`` bar skips
  the visibility filter instead of refusing.
* :func:`test_a_look_ahead_rejection_refuses_the_aggregate` --
  ``Rejection.LOOK_AHEAD`` documents that the aggregate "is refused,
  because one leak poisons the whole average". The code drops the
  leaked reading and returns ``usable=True``, recording the leak in a
  tuple no caller reads.
* :func:`test_a_contaminated_cached_aggregate_cannot_be_injected` --
  ``SentimentAggregate`` records no decision bar, so the
  ``aggregate_result`` injection point cannot be checked against
  anything, which defeats the stated purpose of ``symbol_context``.
* :func:`test_the_cycle_audit_is_bounded_by_the_graph_size` --
  ``cycles()`` is exponential in path length and unbounded in node
  count. A 235-edge **acyclic** graph, under an eighth of the 2000-edge
  budget the module itself documents, spends single-digit seconds on
  it while ``has_cycle()`` answers in a fraction of a millisecond.
* :func:`test_a_near_miss_typo_is_labelled_a_guess_not_an_exact_match`
  -- the prefix rule matches each query token against a name token, so a
  misspelt or truncated company name resolved as if it were exact, and
  every such resolution carried the same ``reason`` string as a correct
  one. Fixed by labelling a fragment match rather than by forbidding it:
  a whole-token rule cannot reject ``'inf'``/``'infy'`` without also
  rejecting ``'beta'``/``'bet'``, which the shipped suite requires, so
  the prefix layer is documented as the deliberate fuzzy match it is
  and made visible through ``Resolution.truncated`` and ``reason``.
* :func:`test_converged_flags_two_identical_extractors_among_three` --
  the anti-convergence guard measures spread about the mean, so one
  dissenter hides two byte-identical extractors, which is the case the
  docstring says the guard exists to catch.
* :func:`test_the_context_gate_is_readable_through_its_documented_path`
  -- the package re-exports a function called ``fuse``, which shadows
  the submodule of the same name, so ``import stock_rl.context.fuse as
  m`` binds the function and the gate that exists to stop the feature
  is unreadable through the path its own docstring prints.

**One test added rather than only repaired.**
:func:`test_a_typo_in_a_hand_maintained_seed_table_fails_loudly` covers
``edges.py``'s ``_input_kind`` ``ValueError``, which was the single
uncovered line in that file. It is the typo guard for a hand-maintained
seed table, which is the one thing such a table most needs tested, and a
coverage number reports it as already fine.

The rest of the file passes today and is pinned on purpose, so a later
fix that breaks the control arm, the seed graph's acyclicity, the depth
ceiling or the equal-weight mean is caught by this file too.

**One clock-dependent assertion.**
:func:`test_the_cycle_audit_is_bounded_by_the_graph_size` is the only
test that reads a clock and it cannot avoid it: the defect is the
absence of a node budget, so elapsed time is the only observable. It
compares ``cycles()`` against ``has_cycle()`` on the same graph rather
than against an absolute deadline, so it stays meaningful on hardware
of any speed. Nothing else here depends on a clock, a seed or an
ordering.
'''

import ast
import importlib
import math
import struct
import time
import types
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

import stock_rl.context
import stock_rl.graph
import stock_rl.sentiment
from stock_rl.context.fuse import (
  CONTEXT_ENABLED,
  CONTEXT_WIDTH,
  ContextArm,
  SymbolContext,
  context_names,
  evaluate_kill_criteria,
  fuse_vector,
  llm_scalar_enabled,
  state_vector,
  symbol_context,
  zero_context,
)
from stock_rl.graph.edges import (
  DependencyGraph,
  Edge,
  EdgeKind,
  _input_kind,
  _sector_inputs,
  _stock_inputs,
  nifty50_seed,
)
from stock_rl.graph.nodes import Node, NodeKind, make_key
from stock_rl.graph.traversal import (
  affected_stocks,
  cycles,
  dependencies,
  has_cycle,
  max_traversal_depth,
)
from stock_rl.sentiment.entity_link import LinkStatus, resolve
from stock_rl.sentiment.score import (
  LookAheadError,
  Rejection,
  SentimentAggregate,
  SentimentReading,
  aggregate,
  converged,
  require_visible,
  visible_readings,
)

ist = timezone(timedelta(hours=5, minutes=30))
bar = datetime(2026, 1, 2, 15, 30, tzinfo=ist)
before = bar - timedelta(hours=6)
after = bar + timedelta(days=1)
far_future = bar + timedelta(days=365)
crude = 'commodity:crude'
liquidity = 'macro:global_liquidity'

#: Roots that would break the standard-library-only, offline, no-model
#: and no-clock promise the three reviewed packages make in prose.
forbidden_imports = frozenset({
  'aiohttp', 'anthropic', 'cohere', 'http', 'httpx', 'langchain',
  'mistralai', 'openai', 'random', 'requests', 'socket', 'time',
  'tomllib', 'torch', 'transformers', 'urllib', 'uuid',
})


def make_reading(
  score: float = 0.5,
  source: str = 'reuters',
  available_from: datetime = before,
  symbol: str = 'RELIANCE',
  intensity: float = 0.5,
) -> SentimentReading:
  '''Return one reading, so each test varies a single field.

  Args:
    score: Directional score in ``[-1, 1]``.
    source: Source identifier, so distinct sources survive aggregation.
    available_from: When the information became publicly available.
    symbol: Canonical NSE symbol.
    intensity: Event strength in ``[0, 1]``.

  Returns:
    A constructed reading.
  '''
  return SentimentReading(
    symbol=symbol,
    score=score,
    event_type='earnings',
    intensity=intensity,
    available_from=available_from,
    source=source,
  )


def public_pair(symbol: str = 'RELIANCE') -> list[SentimentReading]:
  '''Return two public readings from two distinct sources.

  Args:
    symbol: Canonical NSE symbol the readings are about.

  Returns:
    Readings that clear the two-source minimum on their own.
  '''
  return [make_reading(-0.8, 'reuters', symbol=symbol),
          make_reading(-0.6, 'bloomberg', symbol=symbol)]


def acyclic_chain(nodes: int, fanout: int) -> DependencyGraph:
  '''Return an acyclic directed fan-out, the shape a dependency graph has.

  Node ``i`` depends on the next ``fanout`` nodes. The shape matters:
  it is the seed graph's own shape, a directed fan-out that no
  undirected reading mistakes for a loop, scaled past what the
  existing cycle tests build.

  Args:
    nodes: Number of sector nodes.
    fanout: Out-degree of each node.

  Returns:
    A graph with no cycles at all.
  '''
  graph = DependencyGraph()
  for index in range(nodes):
    key = make_key(NodeKind.SECTOR, f'c{index:05d}')
    graph.add_node(Node(key, NodeKind.SECTOR, f'c{index:05d}'))
  for index in range(nodes):
    source = make_key(NodeKind.SECTOR, f'c{index:05d}')
    for step in range(1, fanout + 1):
      target = make_key(NodeKind.SECTOR, f'c{index + step:05d}')
      if target in graph.nodes:
        graph.add_edge(Edge(source, target, EdgeKind.DEPENDS_ON))
  return graph


def small_loop() -> DependencyGraph:
  '''Return a three-node graph carrying one genuine dependency loop.

  Returns:
    ``macro:x -> stock:B -> stock:A -> macro:x``.
  '''
  graph = DependencyGraph()
  for key, kind in (('stock:A', NodeKind.STOCK),
                    ('stock:B', NodeKind.STOCK),
                    ('macro:x', NodeKind.MACRO)):
    graph.add_node(Node(key, kind, key.split(':')[1]))
  for source, target in (('macro:x', 'stock:B'),
                         ('stock:B', 'stock:A'),
                         ('stock:A', 'macro:x')):
    graph.add_edge(Edge(source, target, EdgeKind.DEPENDS_ON))
  return graph


def as_bytes(values: tuple[float, ...]) -> bytes:
  '''Return the IEEE-754 little-endian encoding of a float vector.

  Args:
    values: A state vector.

  Returns:
    The eight-bytes-per-element packing, so "identical" means identical
    bits and not merely close.
  '''
  return struct.pack(f'<{len(values)}d', *values)


def package_files(package: ModuleType) -> list[Path]:
  '''Return every source file of an imported package.

  Args:
    package: An imported module with a ``__file__``.

  Returns:
    Sorted ``.py`` paths inside the package directory.
  '''
  root = Path(str(package.__file__)).parent
  return sorted(root.glob('*.py'))


def imported_roots(path: Path) -> set[str]:
  '''Return the top-level module names a source file imports.

  Args:
    path: A ``.py`` file to parse.

  Returns:
    Root names from every ``import`` and ``from`` statement, including
    those nested inside function bodies.
  '''
  roots: set[str] = set()
  tree = ast.parse(path.read_text(encoding='utf-8'))
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      roots.update(alias.name.split('.')[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
      roots.add(node.module.split('.')[0])
  return roots


# --- the findings ----------------------------------------------------------

def test_the_kill_criteria_gate_fails_closed_on_a_nan() -> None:
  '''A threshold that cannot be evaluated must not report a pass.

  ``fuse.py:30`` says :func:`evaluate_kill_criteria` exists so "it
  cleared the bar" is a checkable statement, and
  :class:`KillCriteriaResult` says ``passed`` is True "only when every
  threshold cleared". The implementation validates with ``if x < 0.0:
  raise`` and then tests with ``if x > threshold: fail``. Both are
  comparisons against a ``nan``, and **every** comparison involving a
  ``nan`` is false, so a ``nan`` p-value skips validation, skips its
  failure, and is reported as a threshold that cleared. The gate that
  decides whether the entire context and sentiment programme is deleted
  or kept answers ``passed=True, failures=()`` for a result that was
  never produced.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  not_a_number = float('nan')
  honest = {'alpha_pvalue': 0.01, 'sharpe_pvalue': 0.02, 'abs_t': 3.4,
            'folds_passed': 5, 'monthly_one_sided_turnover': 0.30}
  assert evaluate_kill_criteria(**honest).passed, (
      'precondition failed: a genuine pass must still pass')
  for field in ('alpha_pvalue', 'sharpe_pvalue', 'abs_t',
                'monthly_one_sided_turnover'):
    poisoned = dict(honest)
    poisoned[field] = not_a_number
    outcome = evaluate_kill_criteria(**poisoned)
    assert not outcome.passed, (
        f'fuse.py:521 evaluates {field} with a comparison against nan, '
        'which is always false, so an unmeasurable result is reported '
        f'as a threshold that cleared: passed={outcome.passed}, '
        f'failures={outcome.failures}')
  everything = dict.fromkeys(honest, not_a_number)
  outcome = evaluate_kill_criteria(**everything)
  assert not outcome.passed, (
      'fuse.py:521 must not report passed=True for an all-nan result')
  assert outcome.failures, (
      'an unevaluable threshold must be named, not silently skipped')
  # The gate must still reject an honestly bad result, so the fix above
  # cannot be "always fail".
  bad = {'alpha_pvalue': 0.9, 'sharpe_pvalue': 0.9, 'abs_t': 0.1,
         'folds_passed': 0, 'monthly_one_sided_turnover': 9.0}
  assert not evaluate_kill_criteria(**bad).passed
  assert len(evaluate_kill_criteria(**bad).failures) == 5
  assert math.isnan(not_a_number), 'precondition failed: nan is not nan'

def test_the_look_ahead_filter_is_not_opt_in() -> None:
  '''A default call shape must not ingest readings from the future.

  ``score.py:352`` guards the visibility filter with ``if decision_bar
  is not None``, and ``decision_bar`` defaults to ``None`` on both
  ``aggregate`` and on ``symbol_context`` at ``fuse.py:275``. The
  ``score`` module docstring calls a fetch-time stamp "the single most
  effective way to manufacture a fake Sharpe" and says the reading must
  be rejected. Here four readings dated a year after the bar are handed
  to ``symbol_context`` with no bar at all, which is the shape a caller
  reaches for first, and they come out the other side as a usable,
  maximally bullish context.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  future = [make_reading(1.0, source, available_from=far_future)
            for source in ('reuters', 'bloomberg', 'cnbc', 'ft')]
  context = symbol_context('RELIANCE', future)
  assert context.sentiment_usable == 0.0, (
      'fuse.py:275 symbol_context skips the visibility filter when '
      f'decision_bar is None, so {len(future)} readings dated '
      f'{far_future.date()} reached a context as usable sentiment '
      f'(score={context.sentiment_score})')
  assert context.sentiment_score == 0.0, (
      'score.py:352 skips the look-ahead filter entirely when '
      'decision_bar is None rather than refusing the aggregate')


def test_a_look_ahead_rejection_refuses_the_aggregate() -> None:
  '''A recorded ``look_ahead`` must mean the number is not published.

  ``score.py:100`` states that the leaked reading "is dropped and the
  aggregate is refused, because one leak poisons the whole average and
  there is no way to subtract it". The code drops the reading and
  publishes the average of the survivors, so the leak is recorded and
  then ignored. ``fuse.py:327`` reads only ``result.usable`` and never
  looks at ``result.rejected``, so no downstream consumer can notice.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  readings = [*public_pair(), make_reading(1.0, 'oracle',
                                           available_from=after)]
  result = aggregate(readings, 'RELIANCE', bar)
  assert Rejection.LOOK_AHEAD.value in result.rejected, (
      'precondition failed: the leak must at least be recorded')
  assert result.usable is False, (
      'score.py:359 records Rejection.LOOK_AHEAD but still returns '
      f'usable=True with score={result.score} computed from the '
      f'{result.source_count} surviving sources')
  assert result.score == 0.0, (
      'score.py:388 zeroes the score only when the source count is '
      f'short; here it published {result.score} next to a look_ahead '
      'rejection')


def test_a_contaminated_cached_aggregate_cannot_be_injected() -> None:
  '''The cache injection point must not be a hole around the guard.

  ``fuse.py:287`` gives ``symbol_context`` an ``aggregate_result``
  parameter so "the sentiment rules cannot be bypassed", then supplies
  the one path that bypasses them: the aggregate is trusted verbatim,
  and :class:`~stock_rl.sentiment.score.SentimentAggregate` carries no
  decision bar, so there is nothing to check it against.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  names = [item.name for item in fields(SentimentAggregate)]
  assert not [name for name in names if 'bar' in name], (
      f'precondition changed: SentimentAggregate now carries {names}, so '
      'this test needs to be re-argued')
  contaminated = aggregate(
    [*public_pair(), make_reading(1.0, 'oracle', available_from=after)],
    'RELIANCE', bar)
  context = symbol_context('RELIANCE', aggregate_result=contaminated)
  assert context.sentiment_usable == 0.0, (
      'fuse.py:321 trusts aggregate_result without inspecting rejected, '
      'so an aggregate that records look_ahead still reaches a context '
      f'as usable with score={context.sentiment_score}')


def test_the_cycle_audit_is_bounded_by_the_graph_size() -> None:
  '''An audit over an acyclic graph must not scale with path count.

  ``traversal.py:220`` justifies ``max_cycles`` as a cap "so a densely
  tangled graph cannot turn an audit into a hang". The blow-up is not
  tangled graphs, it is **deep acyclic** ones: ``_walk`` enumerates
  every simple path up to ``max_length`` from every node, so the cost is
  ``fanout ** max_length`` however few loops exist. This graph is
  acyclic and has a tenth of the documented edge budget.

  The fix has to answer the acyclic case from :func:`has_cycle` and put
  a ceiling on the rest, because "no loops" and "too many paths" are
  different questions and only one of them needs enumeration.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  graph = acyclic_chain(nodes=50, fanout=5)
  assert not has_cycle(graph), 'precondition failed: graph is acyclic'
  start = time.perf_counter()
  has_cycle(graph)
  reference = time.perf_counter() - start
  start = time.perf_counter()
  found = cycles(graph)
  elapsed = time.perf_counter() - start
  assert not found, 'precondition failed: an acyclic graph has no loops'
  budget = max(0.05, 20.0 * reference)
  assert elapsed < budget, (
      f'traversal.py:252 _walk enumerated every simple path of length '
      f'<= max_length across {graph.node_count} nodes and '
      f'{graph.edge_count} acyclic edges in {elapsed:.2f}s against a '
      f'{budget:.2f}s budget, while has_cycle answered the same '
      f'question in {reference * 1000:.2f}ms')

  # A ceiling nothing raises on is not a ceiling, so the budget is
  # checked here too: a small file carrying one real loop must be
  # nameable on the default budget, and must raise rather than grind
  # when the ceiling is deliberately too small to reach it. Deterministic
  # -- the fan-out is fixed, so the extension count is too.
  looped = acyclic_chain(nodes=20, fanout=3)
  first = make_key(NodeKind.SECTOR, 'c00001')
  second = make_key(NodeKind.SECTOR, 'c00002')
  third = make_key(NodeKind.SECTOR, 'c00003')
  looped.add_edge(Edge(third, first, EdgeKind.DEPENDS_ON))
  assert has_cycle(looped), 'precondition failed: this file does loop'
  assert cycles(looped, max_cycles=1) == ((first, second, third, first),), (
      'the gate must not cost a real loop its name')
  with pytest.raises(RuntimeError) as refusal:
    cycles(looped, max_expansions=50)
  message = str(refusal.value)
  assert 'budget of 50 path extensions' in message, (
      f'the refusal must name the ceiling it hit, got {message!r}')
  assert (f'{looped.node_count} nodes and {looped.edge_count} edges'
          in message), (
      f'the refusal must name the graph it gave up on, got {message!r}')
  with pytest.raises(ValueError, match='max_expansions'):
    cycles(looped, max_expansions=0)


def test_a_near_miss_typo_is_labelled_a_guess_not_an_exact_match() -> None:
  '''A misspelt name must never claim to be an exact resolution.

  ``entity_link.py`` used to promise "no fuzzy matching, no edit
  distance, no spelling correction" while matching every query token as
  a **prefix** of a name token. Prefix matching *is* fuzzy matching, so
  the claim was false, and the damage was specific: ``'Infosy'``,
  ``'inf'``, ``'Reliance Industr'``, ``'Axis Ban'``, ``'Tata Motor'``
  and ``'sun pharm'`` all resolved, every one of them carrying the same
  ``reason='unambiguous prefix match'`` as a correct resolution, so an
  audit reading ``reason`` could not tell a typo from a match.

  A strict whole-token rule was tried and is mathematically unavailable.
  ``'inf'`` is 3 of the 4 characters of ``'infy'`` and ``'bet'`` is 3
  of the 4 characters of ``'beta'``: identical ratios, opposite
  verdicts, because the shipped suite requires
  ``SymbolLinker(...).link('alpha bet').symbol == 'AAB'`` while refusing
  ``'inf'`` would reject it. No fraction or leftover threshold separates
  them. The prefix layer therefore stays, is **documented as a
  deliberate fuzzy match** with its residual risk, and is made visible
  instead:

  * every fragment-based match now sets ``Resolution.truncated``;
  * ``Resolution.reason`` distinguishes ``'whole symbol spelled without
    its spaces'`` and ``'whole name tokens matched, no truncation'``
    from ``'truncated symbol match: 3 of 4 characters, ...'``, so the
    field a reviewer reads can no longer confuse the two;
  * ``min_prefix_chars`` is applied to **each query token**, not only to
    the squashed query, which is what finally refuses ``'Dr Red'``
    (``'dr'`` is two characters).

  Returns:
    Nothing. The assertions are the deliverable.
  '''
  fuzzy = ('Infosy', 'inf', 'Reliance Industr', 'Axis Ban', 'Tata Motor',
           'sun pharm')
  exact = ('Infosys', 'Infosys Limited', 'asian paint', 'Infy',
           'Sun Pharmaceutical Ltd', 'the Tata Consultancy Services')
  unlabelled = {
    text: (resolve(text).reason, resolve(text).truncated)
    for text in fuzzy
    if resolve(text).status is LinkStatus.RESOLVED
    and not resolve(text).truncated}
  assert not unlabelled, (
      'a prefix match that rests on a fragment must set '
      f'Resolution.truncated and say so in reason: {unlabelled}')
  # The reason field must discriminate: nothing that reached a symbol by
  # matching every character may share a reason with a truncation. This
  # is the specific defect -- one undifferentiated "unambiguous prefix
  # match" for both a typo and a correct resolution.
  reasons = {text: resolve(text).reason for text in fuzzy + exact}
  fuzzy_reasons = {reasons[text] for text in fuzzy
                   if resolve(text).status is LinkStatus.RESOLVED}
  exact_reasons = {reasons[text] for text in exact
                   if resolve(text).status is LinkStatus.RESOLVED}
  assert not fuzzy_reasons & exact_reasons, (
      'a truncated match and a whole match are reported under the same '
      f'reason string, so an audit reading reason cannot tell them '
      f'apart: shared={sorted(fuzzy_reasons & exact_reasons)}')
  assert all(reason.startswith('truncated') for reason in fuzzy_reasons), (
      'a fragment-based match must name itself a guess in reason: '
      f'{sorted(fuzzy_reasons)}')
  assert all(not resolve(text).truncated for text in exact), (
      'a whole-symbol or whole-name-token match is not a guess')
  # The floor is per token now. 'Dr Red' is two tokens, one of them two
  # characters, so the squashed query 'drred' clearing min_prefix_chars
  # no longer lets the two-letter fragment through.
  assert resolve('Dr Red').status is LinkStatus.UNKNOWN, (
      'min_prefix_chars must be applied to each query token, not only to '
      'the squashed query, so a two-letter token cannot match anything')
  # The refusal cases this module exists for must survive all of that.
  # Refusal cases under a TABLE LOOKUP. Resolution indexes every written
  # form once and resolves with a single dict hit, so a partial symbol is
  # simply not a key. These previously resolved AMBIGUOUS off a prefix
  # match, and that inference is what let 'AXISBANKING' resolve to
  # AXISBANK and 'ITC-INFRA' to ITC - different companies, answered with
  # whichever symbol shared a prefix.
  for text in ('Tata', 'Rel', 'Relian', 'RELI'):
    assert resolve(text).status is LinkStatus.UNKNOWN, text
    assert not resolve(text).candidates, text
  reliance = resolve('RELIANCE')
  assert reliance.symbol == 'RELIANCE'
  assert 'RELIANCEINFRA' not in reliance.candidates
  # The refusal the inverse-prefix bug was really about.
  for text in ('ITC-INFRA', 'AXISBANKING', 'RELIANCEEXTRA'):
    assert resolve(text).status is LinkStatus.UNKNOWN, text


def test_converged_flags_two_identical_extractors_among_three() -> None:
  '''The anti-convergence guard must detect partial convergence.

  ``score.py:426`` is sold as the implementation of the one
  intervention the research review found significant: independent
  extractors which "produce identical output have stopped being
  independent". Its Returns section promises True when "at least two
  scores agree to within tolerance". What it computes is the mean
  absolute deviation about the mean, which one dissenting outlier
  inflates past any tolerance -- so two cloned extractors plus one
  honest one is scored as *not* converged.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  assert converged([1.0, 1.0, 0.0]) is True, (
      'score.py:459 measures spread about the mean, so a single '
      'dissenter hides two byte-identical extractors, which is the '
      'exact case the docstring says the guard exists to catch')
  assert converged([0.4, 0.4, -0.4]) is True, (
      'score.py:459 must not let one dissent veto the guard')
  assert converged([1.0, -1.0]) is False, (
      'sanity: real disagreement is not convergence')
  assert converged([0.5]) is False, (
      'sanity: one source cannot disagree with itself')


def test_the_context_gate_is_readable_through_its_documented_path() -> None:
  '''The safety gate must be reachable by the name the docs print.

  ``context/__init__.py`` re-exported the function ``fuse``, which
  shadowed the submodule of the same name, so ``import stock_rl.context
  .fuse as m`` bound the *function* and the dotted references in the
  package docstrings -- including the reference to
  ``stock_rl.context.fuse.context_enabled`` itself -- raised
  ``AttributeError`` on the gate that exists to stop the feature.

  Fixed by renaming the function to ``fuse_vector`` and keeping the
  module name. The rename alone would have broken every ``fuse(ctx)``
  call site, so the module is also callable and delegates to
  ``fuse_vector``; both facts are asserted, because "callable" is the
  kind of property that works until someone tidies the module up.

  Returns:
    Nothing. The assertions are the deliverable.
  '''
  # The plain form is the thing under test, so it is not importlib.
  # pylint: disable=import-outside-toplevel,reimported
  import stock_rl.context.fuse as fuse_module
  from stock_rl.context import fuse as legacy
  # pylint: enable=import-outside-toplevel,reimported
  assert isinstance(fuse_module, types.ModuleType), (
      'context/__init__.py:45 re-exports the function fuse, so '
      '`import stock_rl.context.fuse as m` binds the function and the '
      'gate is unreadable through its own documented path (got '
      f'{type(fuse_module).__name__})')
  assert fuse_module.context_enabled is False, (
      'the gate must be readable and must be False')
  # The rename kept the old spelling working, so no caller is left with
  # a broken import. This is the compatibility half of the fix and it is
  # the half a later tidy-up would silently drop.
  context = SymbolContext('RELIANCE', sentiment_score=0.4)
  assert legacy(context, True) == fuse_vector(context, True), (
      'fuse.py no longer delegates the module-level call to fuse_vector, '
      'so `fuse(ctx)` either raises or computes something else')
  assert legacy(context) == zero_context(), (
      'the legacy spelling must still honour the gate')


# --- what is already correct, pinned ---------------------------------------

def test_the_control_arm_is_byte_identical_to_the_no_context_path() -> None:
  '''Arm A must equal the no-context path bit for bit.

  A float epsilon here means the control is not a control, so this
  asserts on packed IEEE-754 bytes rather than ``approx``, and builds
  the context from the real seed graph rather than a fixture.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  graph = nifty50_seed()
  reached = affected_stocks(graph, crude, max_depth=3)
  price = [0.1234567890123, -45.6789, 1e-17, 3.0, 0.1, 2.2, 3.3, 4.4,
           5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 0.30000000000000004]
  context = symbol_context(
    'RELIANCE', public_pair(), bar,
    graph_depth=float(reached.get('RELIANCE', 0)),
    commodity_dependencies=float(
      len(dependencies(graph, 'stock:RELIANCE'))))
  control = state_vector(price, context, ContextArm.CONTROL)
  assert control == tuple(float(value) for value in price)
  assert as_bytes(control) == as_bytes(state_vector(price))
  assert as_bytes(control) == as_bytes(state_vector(
    price, context, ContextArm.CONTROL, True))
  assert as_bytes(control) == as_bytes(state_vector(
    price, context, ContextArm.CONTROL, True, True, 0.75))
  assert all(isinstance(value, float) and not isinstance(value, bool)
             for value in control)


def test_the_control_arm_ignores_a_populated_context_and_both_gates() -> None:
  '''Turning a gate on must not change the control arm.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  context = symbol_context('RELIANCE', public_pair(), bar,
                           graph_depth=3.0, commodity_dependencies=2.0,
                           options_pcr=1.4, options_iv_rank=0.9)
  assert fuse_vector(context, True) != zero_context(), (
      'precondition failed: the context must be non-trivial')
  price = [0.01, -0.02, 0.03]
  assert state_vector(price, context) == tuple(price)
  assert state_vector(price, None) == tuple(price)
  assert CONTEXT_ENABLED is False
  assert llm_scalar_enabled is False


def test_aggregate_is_a_plain_equal_weight_mean_over_sources() -> None:
  '''No judge, no priority, no weighting scheme: sources weigh equally.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  loud = [make_reading(-1.0, 'reuters', before - timedelta(days=2)),
          make_reading(-1.0, 'reuters', before - timedelta(days=1)),
          make_reading(-1.0, 'reuters', before),
          make_reading(1.0, 'cnbc', before)]
  result = aggregate(loud, 'RELIANCE', bar)
  assert result.source_scores == (-1.0, 1.0)
  assert result.score == 0.0, 'each source must carry equal weight'
  assert result.reading_count == 4
  assert result.corroboration == 2
  assert aggregate([make_reading(1.0, 'reuters'),
                    make_reading(1.0, 'cnbc')],
                   'RELIANCE', bar).score == 1.0
  assert aggregate([make_reading(0.5, 'reuters'),
                    make_reading(-0.5, 'cnbc')],
                   'RELIANCE', bar).score == 0.0


def test_no_module_under_review_imports_a_network_a_model_or_a_clock() -> None:
  '''The three reviewed packages must be stdlib-only, offline, model-free.

  Parsed from the real source files rather than ``sys.modules``, so an
  import inside a function body is caught too.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  offenders = {}
  for package in (stock_rl.graph, stock_rl.context, stock_rl.sentiment):
    for path in package_files(package):
      clash = imported_roots(path) & forbidden_imports
      if clash:
        offenders[path.name] = sorted(clash)
  assert not offenders, (
      'a module under review imports a network, model or clock library: '
      f'{offenders}')


def test_the_seed_graph_is_honestly_acyclic() -> None:
  '''The seed graph must be acyclic, and say so in both directions.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  graph = nifty50_seed()
  assert not has_cycle(graph)
  assert not cycles(graph)
  assert not cycles(graph, max_cycles=10_000, max_length=12)
  assert not graph.edge_budget_exceeded
  assert set(affected_stocks(graph, crude, max_depth=1)) == {
    'ONGC', 'RELIANCE', 'ASIANPAINT'}
  assert 'HDFCBANK' in affected_stocks(graph, liquidity, max_depth=3)
  assert 'HDFCBANK' not in affected_stocks(graph, liquidity, max_depth=2)


def test_fan_out_is_not_reported_as_a_dependency_loop() -> None:
  '''One cause with many dependents is fan-out, not a loop.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  graph = DependencyGraph()
  graph.add_node(Node('commodity:crude', NodeKind.COMMODITY, 'Crude'))
  for index in range(12):
    name = f'sector:s{index:02d}'
    graph.add_node(Node(name, NodeKind.SECTOR, name))
    graph.add_edge(Edge(name, 'commodity:crude', EdgeKind.DEPENDS_ON))
  assert not has_cycle(graph)
  assert not cycles(graph)
  assert len(affected_stocks(graph, 'commodity:crude', max_depth=1)) == 0


def test_a_real_two_cycle_is_detected_and_reported() -> None:
  '''Contradictory edges must be found and named.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  graph = DependencyGraph()
  graph.add_node(Node('sector:banking', NodeKind.SECTOR, 'Banking'))
  graph.add_node(Node('macro:repo_rate', NodeKind.MACRO, 'Repo'))
  graph.add_edge(Edge('sector:banking', 'macro:repo_rate',
                       EdgeKind.DEPENDS_ON))
  graph.add_edge(Edge('macro:repo_rate', 'sector:banking',
                       EdgeKind.DEPENDS_ON))
  assert has_cycle(graph)
  assert cycles(graph) == (('macro:repo_rate', 'sector:banking',
                            'macro:repo_rate'),)
  looped = small_loop()
  assert has_cycle(looped)
  assert ('macro:x', 'stock:B', 'stock:A', 'macro:x') in cycles(looped)


def test_a_bad_context_is_refused_at_construction_not_at_fusion() -> None:
  '''The documented ranges are enforced, so they are not decoration.

  ``fuse.py`` said "a bad context is the caller's dataclass validation,
  not this function's" while :class:`SymbolContext` validated nothing.
  It therefore constructed cleanly and ``fuse`` published the values for
  fields documented as ``[-1, 1]`` and ``[0, 1]`` straight into a state
  vector::

      SymbolContext(symbol='   ', sentiment_score=99.0,
                    sentiment_disagreement=-5.0, sentiment_intensity=7.0,
                    sentiment_source_ratio=42.0)   # used to be fine

  Every ``_bounds`` entry is checked, one field at a time, so the error
  names the field that is wrong rather than the first one alphabetically.

  Returns:
    Nothing. The assertions are the deliverable.
  '''
  with pytest.raises(ValueError, match='symbol context needs a symbol'):
    SymbolContext('   ')
  for field, value in (('sentiment_score', 99.0),
                       ('sentiment_score', -99.0),
                       ('sentiment_disagreement', -5.0),
                       ('sentiment_disagreement', 1.5),
                       ('sentiment_intensity', 7.0),
                       ('sentiment_usable', 2.0),
                       ('sentiment_source_ratio', 42.0),
                       ('graph_depth', -1.0),
                       ('commodity_dependencies', -1.0),
                       ('macro_dependencies', -1.0),
                       ('sector_dependencies', -1.0),
                       ('options_pcr', -0.5),
                       ('options_iv_rank', 1.5),
                       ('options_net_oi_change', float('inf'))):
    with pytest.raises(ValueError, match=field):
      SymbolContext('RELIANCE', **{field: value})
  # nan must be refused, and the negated comparison is what does it:
  # every comparison involving a nan is false, so an "is it in range"
  # test would let it through. This is the same trap
  # _reject_underspecified exists to close.
  for field in ('sentiment_disagreement', 'sentiment_score', 'graph_depth',
                'options_pcr', 'options_iv_rank'):
    with pytest.raises(ValueError, match=field):
      SymbolContext('RELIANCE', **{field: float('nan')})
  # A number of the right shape is still accepted, including the ints a
  # graph traversal returns for a hop count.
  clean = SymbolContext('RELIANCE', sentiment_score=-1.0,
                        sentiment_disagreement=1.0, graph_depth=3,
                        options_pcr=1.4, options_iv_rank=0.0,
                        options_net_oi_change=-0.5)
  assert fuse_vector(clean, True)[5] == 3.0
  # A validated context still cannot smuggle a non-finite number in
  # through the one field that is legitimately signed.
  assert len(fuse_vector(SymbolContext('RELIANCE'), True)) == CONTEXT_WIDTH


def test_a_typo_in_a_hand_maintained_seed_table_fails_loudly() -> None:
  '''``_input_kind`` is the typo guard, so the typo guard is tested.

  ``edges.py`` builds the seed from hand-maintained tables of commodity
  and macro names, and ``_input_kind`` is the only thing standing
  between a typo in one of those tables and a node key that simply does
  not exist. Its ``ValueError`` was the single uncovered line in
  ``edges.py`` -- the one check in the file that a hand-maintained table
  most needs, and the one a coverage number calls "already fine".

  The seed builds at all, which is the other half: every name in
  ``_sector_inputs`` and ``_stock_inputs`` resolves to a real node kind.
  Without that, the tables could all drift to garbage together and
  nothing here would notice.

  Returns:
    Nothing. The assertions are the deliverable.
  '''
  assert _input_kind('crude') is NodeKind.COMMODITY
  assert _input_kind('natural_gas') is NodeKind.COMMODITY
  assert _input_kind('usdinr') is NodeKind.MACRO
  assert _input_kind('fii_net') is NodeKind.MACRO
  # An EVENT name is not an input name. This is the near-miss a real
  # editor makes, because _stock_events lists 'opec_supply_cut' and
  # 'monsoon' a few lines above where a commodity name would go.
  for text in ('opec_supply_cut', 'monsoon', 'rbi_rate_decision'):
    with pytest.raises(ValueError, match='not a known macro or commodity'):
      _input_kind(text)
  # A transposition and a wrong case, the other two near-misses.
  for text in ('crued', 'Coal', '', 'sant_gas'):
    with pytest.raises(ValueError, match='not a known macro or commodity'):
      _input_kind(text)
  # The guard is load-bearing: every input name in every seed table
  # resolves, so the failure above is unreachable by accident.
  graph = nifty50_seed()
  for name in {name for sector, names in _sector_inputs.items()
               for name in names} | {name for symbol, names
                                     in _stock_inputs.items()
                                     for name in names}:
    assert graph.has(make_key(_input_kind(name), name)), name


def test_a_self_loop_is_flagged_even_though_it_is_not_enumerable() -> None:
  '''A self-edge must at least be visible to the linear check.

  ``cycles()`` cannot enumerate it, which ``traversal.py:225`` does not
  say: it names "a one-key loop needs a self-edge" without saying the
  reporting tool will not report one.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  graph = DependencyGraph()
  graph.add_node(Node('macro:x', NodeKind.MACRO, 'X'))
  graph.add_edge(Edge('macro:x', 'macro:x', EdgeKind.DEPENDS_ON))
  assert has_cycle(graph), 'traversal.py:154 must flag a self-edge'
  assert not cycles(graph)
  assert not affected_stocks(graph, 'macro:x', max_depth=2)


def test_the_depth_ceiling_is_enforced_and_not_clamped() -> None:
  '''A depth beyond the hard cap must raise, not be silently capped.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  graph = nifty50_seed()
  for depth in (0, -1, max_traversal_depth + 1, 10_000):
    with pytest.raises(ValueError, match='max_depth'):
      affected_stocks(graph, crude, max_depth=depth)
  with pytest.raises(ValueError, match='unknown node'):
    affected_stocks(graph, 'macro:ghost', max_depth=1)
  assert affected_stocks(graph, crude,
                         max_depth=max_traversal_depth) == (
                           affected_stocks(graph, crude))


def test_a_leaked_reading_cannot_move_the_score_when_a_bar_is_given() -> None:
  '''Given a bar, the filter must reject, not down-weight, the leak.

  This is the case the module gets right, pinned so that fixing the
  ``decision_bar=None`` default cannot regress it.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  leaked = [*public_pair(),
            make_reading(1.0, 'oracle', available_from=after)]
  result = aggregate(leaked, 'RELIANCE', bar)
  assert result.source_count == 2
  assert result.source_scores == (-0.8, -0.6)
  # The published `score` is zeroed rather than set to the survivors'
  # mean. That was this test's original expectation (-0.7), which
  # conflicts with a deliberate pre-existing decision documented on the
  # field: "a number that must not be used should not be able to reach a
  # caller that forgets to check the flag." A leak must not be usable, and
  # zeroing is what stops a caller that ignores `usable` from acting on a
  # contaminated average. `source_scores` above is the audit trail and is
  # asserted unchanged, so nothing is lost for an auditor.
  assert result.score == 0.0, (
    'a leaked aggregate must not publish a usable-looking score')
  assert result.usable is False, (
    'the refusal documented on Rejection.LOOK_AHEAD must actually happen')
  assert result.reading_count == 2, 'the leaked reading must be dropped'
  assert Rejection.LOOK_AHEAD.value in result.rejected
  assert len(visible_readings(leaked, bar)) == 2
  with pytest.raises(LookAheadError, match='RELIANCE'):
    require_visible(leaked, bar)
  only_leak = [make_reading(1.0, 'oracle', available_from=after)]
  assert aggregate(only_leak, 'RELIANCE', bar).usable is False


def test_the_reviewed_paths_are_deterministic_across_repeated_calls() -> None:
  '''No clock, no seed, no ordering dependence in the reviewed code.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  readings = public_pair()
  assert aggregate(readings, 'RELIANCE', bar) == aggregate(
    readings, 'RELIANCE', bar)
  graph = acyclic_chain(nodes=20, fanout=3)
  origin = make_key(NodeKind.SECTOR, 'c00000')
  assert affected_stocks(graph, origin) == affected_stocks(graph, origin)
  assert cycles(small_loop()) == cycles(small_loop())
  for text in ('RELIANCE', 'Reliance Infra', 'Tata', 'Infosys', '', 'INFY'):
    assert resolve(text) == resolve(text)
  assert symbol_context('RELIANCE', readings, bar) == symbol_context(
    'RELIANCE', readings, bar)


def test_the_context_width_is_fixed_and_matches_the_field_names() -> None:
  '''A vector whose width moves with the data invalidates every policy.

  Returns:
    Nothing. The assertion is the deliverable.
  '''
  assert CONTEXT_WIDTH == len(context_names()) == 12
  assert len(zero_context()) == CONTEXT_WIDTH
  assert 'symbol' not in context_names()
  context = symbol_context('RELIANCE', public_pair(), bar)
  assert len(fuse_vector(context)) == len(
    fuse_vector(context, True)) == CONTEXT_WIDTH
  assert SymbolContext('RELIANCE').symbol == 'RELIANCE'
  module = importlib.import_module('stock_rl.context.fuse')
  assert module.CONTEXT_WIDTH == CONTEXT_WIDTH
  assert module.context_enabled is False
  assert module.llm_scalar_enabled is False


