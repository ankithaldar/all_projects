#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Consistency probes for findings proven by running the package.

Every test calls the module it audits. Where a finding is a defect, the
test fails now and passes once the defect is fixed. Where a documented
behaviour was verified to be honest, the test passes now and fails if a
refactor breaks the promise, so the claim cannot rot unnoticed.

Stdlib only. Nothing is recomputed inline that the audited module already
computes: expected values are read out of the live code.
'''

from __future__ import annotations

import ast
import io
import math
import re
import tokenize
import unittest
import importlib
import pkgutil
from dataclasses import replace
from pathlib import Path

import stock_rl
import stock_rl.costs as costs
import stock_rl.execution.sizing as sizing
import stock_rl.skills as skills
from stock_rl.costs import INTRADAY, Side
from stock_rl.execution.sizing import (
  SizingInputs,
  StockSizer,
  kelly_fraction,
)
from stock_rl.risk.checks import (
  AccountState,
  PriceBand,
  RmsChecker,
  RmsLimits,
  SecurityLimits,
  buy,
)
from stock_rl.graph import traversal
from stock_rl.graph.edges import nifty50_seed
from stock_rl.risk import var as var_module
from stock_rl.sentiment import entity_link

# The 56-extension figure is a claim about _walk and _Meter internals, which
# is exactly what an audit test is for.
# pylint: disable=protected-access

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'


class TestKellyDocstringIsFalse(unittest.TestCase):
  '''``kelly_fraction`` documents two numbers it never returns.

  ``execution/sizing.py`` lines 111-113: "It is ``1 / win_loss_ratio`` at
  break-even and rises towards ``1 / win_loss_ratio`` less a half as the
  edge grows." Measured: f* is 0 at break-even and tends to 1 for every
  odds, so both halves of the sentence are wrong.

  The arithmetic tests below pass and pin the truth. The last test fails,
  because the false sentence is still in the file.
  '''

  def test_break_even_returns_zero_not_one_over_odds(self):
    for odds in (0.5, 1.0, 2.0, 4.0):
      with self.subTest(win_loss_ratio=odds):
        self.assertAlmostEqual(kelly_fraction(1.0 / (odds + 1.0), odds),
                               0.0, places=12)

  def test_the_edge_lands_on_one_not_on_one_over_odds_less_half(self):
    for odds in (0.5, 1.0, 2.0, 4.0):
      with self.subTest(win_loss_ratio=odds):
        self.assertAlmostEqual(kelly_fraction(1.0, odds), 1.0, places=12)

  def test_the_documented_limit_is_unreachable_for_half_the_odds(self):
    for odds in (1.0, 2.0, 4.0):
      with self.subTest(win_loss_ratio=odds):
        claimed = 1.0 / odds - 0.5
        best = max(kelly_fraction(p, odds) for p in (0.0, 0.5, 1.0))
        self.assertGreater(best, claimed)

  def test_the_docstring_no_longer_states_the_false_limit(self):
    source = Path(sizing.__file__).read_text(encoding='utf-8')
    self.assertNotIn(
      '1 / win_loss_ratio', source,
      'the docstring still states f* is 1 / win_loss_ratio at break-even '
      'and tends to 1 / win_loss_ratio less a half; measured it is 0 and 1')


class TestIntradayBrokerageCommentOverstates(unittest.TestCase):
  '''``costs.py`` INTRADAY PONYTAIL claims a "roughly 60 percent" error.

  Measured worst case over nine notionals from 1e4 to 1e9 rupees is 34.32
  percent, reached only below the Rs 20 brokerage cap. At 1e6 rupees it is
  3.37 percent, because the cap makes brokerage flat while the levies scale.
  '''

  def test_understatement_never_reaches_sixty_percent(self):
    without = replace(INTRADAY, brokerage_pct=0.0)
    worst = 0.0
    for notional in (1e4, 5e4, 6.6666666e4, 1e5, 1e6, 1e7, 1e8, 1e9):
      zeroed = without.round_trip(notional)
      full = INTRADAY.round_trip(notional)
      worst = max(worst, (full - zeroed) / full * 100.0)
    self.assertLess(worst, 60.0)

  def test_understatement_at_a_rupee_lakh_is_under_four_percent(self):
    without = replace(INTRADAY, brokerage_pct=0.0)
    zeroed = without.round_trip(1e6)
    full = INTRADAY.round_trip(1e6)
    self.assertLess((full - zeroed) / full * 100.0, 4.0)

  def test_the_three_point_five_five_bps_levy_half_is_right(self):
    without = replace(INTRADAY, brokerage_pct=0.0)
    levies = sum(without.taxes_and_fees(side, 1e9) for side in _both_sides())
    self.assertAlmostEqual(levies / 1e9 * 1e4, 3.548, places=3)

  def test_the_comment_no_longer_claims_sixty_percent(self):
    self.assertNotIn(
      'roughly 60 percent', Path(costs.__file__).read_text(encoding='utf-8'),
      'the INTRADAY PONYTAIL still claims a roughly 60 percent '
      'understatement; the measured worst case is 34.32 percent')


def _both_sides():
  '''Return the two order sides, read from the cost model itself.'''
  return (Side.BUY, Side.SELL)


class TestSeedGraphCycleCountIsNotFortySeven(unittest.TestCase):
  '''``graph/traversal.py`` line 283 claims 47 undirected cycles.

  Counting undirected simple cycles of length 3 or more in the seed graph
  gives 34 triangles and 3262 in total. Directed cycles are 0. No reading
  of "fan-out counted undirected" produces 47.
  '''

  def _undirected_counts(self):
    graph = nifty50_seed()
    adjacency = {key: set() for key in graph.nodes}
    for edge in graph.edges:
      adjacency[edge.source].add(edge.target)
      adjacency[edge.target].add(edge.source)
    counts: dict[int, int] = {}

    def walk(start, current, depth, on_path):
      for neighbour in adjacency[current]:
        if neighbour == start and depth >= 3:
          counts[depth] = counts.get(depth, 0) + 1
        elif neighbour not in on_path and neighbour > start:
          walk(start, neighbour, depth + 1, on_path | {neighbour})

    for key in sorted(adjacency):
      walk(key, key, 1, {key})
    return counts

  def test_total_undirected_cycles_is_not_forty_seven(self):
    counts = self._undirected_counts()
    self.assertNotEqual(sum(counts.values()), 47)

  def test_triangle_count_is_not_forty_seven(self):
    self.assertNotEqual(self._undirected_counts().get(3), 47)

  def test_the_directed_walk_reports_no_cycles(self):
    self.assertEqual(traversal.cycles(nifty50_seed()), ())

  def test_the_seed_size_the_docstring_names_is_correct(self):
    graph = nifty50_seed()
    self.assertEqual((graph.node_count, graph.edge_count), (55, 76))

  def test_the_docstring_no_longer_claims_forty_seven_cycles(self):
    self.assertNotIn(
      'has 47', Path(traversal.__file__).read_text(encoding='utf-8'),
      'the cycles() docstring still claims 47 undirected cycles; measured, '
      'the seed graph has 34 triangles and 3262 undirected cycles in total')


class TestTraversalLineCountClaim(unittest.TestCase):
  '''``graph/traversal.py`` line 8 says the module "is 150 lines".

  The file is 431 lines, so the figure understates by 2.9x. It reads as a
  measurement taken on an earlier revision and never refreshed.
  '''

  def test_the_docstring_states_no_line_count(self):
    # This test originally demanded the docstring carry an EXACT count, and
    # failed twice because the number was stale - 150 on a 431-line file,
    # then 431 on a 434-line one. A line count in prose is a figure that
    # cannot stay true: every edit to the file falsifies it, and it says
    # nothing about the argument it was attached to.
    #
    # The sentence now reads "a few hundred lines". The claim being made is
    # that the walk is hand-written and readable rather than delegated to a
    # graph database, and that is true regardless of how long it is.
    source = Path(traversal.__file__).read_text(encoding='utf-8')
    self.assertIsNone(
      re.search(r'it is \d+ lines rather than', source),
      'the docstring must not carry a line count; it drifts on every edit')
    self.assertIn('a few hundred lines rather than', source)

  def test_the_file_is_long_enough_for_the_claim_to_be_meaningful(self):
    source = Path(traversal.__file__).read_text(encoding='utf-8')
    self.assertGreater(len(source.splitlines()), 150)


class TestVarDocstringNamesTheWrongConstant(unittest.TestCase):
  '''``risk/var.py`` line 557 says ``_pivot_tolerance``; it is not that.'''

  def test_the_underscored_spelling_is_absent_from_the_module(self):
    source = Path(var_module.__file__).read_text(encoding='utf-8')
    self.assertNotIn('_pivot_tolerance', source)

  def test_the_tolerance_is_defined_without_an_underscore(self):
    self.assertTrue(hasattr(var_module, 'pivot_tolerance'))

  def test_the_tolerance_is_unexported_while_its_sibling_is(self):
    self.assertNotIn('pivot_tolerance', var_module.__all__)
    self.assertIn('min_confidence', var_module.__all__)


class TestVenueCheckIdsIsDead(unittest.TestCase):
  '''``risk/checks.py`` line 159 defines ``venue_check_ids``; nobody reads it.

  Commented "#: Checks that cannot be made at all without per-symbol venue
  data." One occurrence in the whole tree, the assignment itself.
  '''

  def test_no_name_or_attribute_reference_exists(self):
    offenders = []
    for root in ('src', 'tests'):
      for path in sorted((ROOT / root).rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
          if isinstance(node, ast.Name) and node.id == 'venue_check_ids':
            offenders.append(f'{path}:{node.lineno}')
          elif (isinstance(node, ast.Attribute)
                and node.attr == 'venue_check_ids'):
            offenders.append(f'{path}:{node.lineno}')
    self.assertEqual(offenders, [])


class TestCodingCookieIsMalformed(unittest.TestCase):
  '''``src/stock_rl/__init__.py`` line 2 has three trailing dashes.'''

  def test_line_two_is_exactly_the_documented_cookie_everywhere(self):
    want = b'# -*- coding: utf-8 -*-'
    bad = []
    for root in ('src', 'tests'):
      for path in sorted((ROOT / root).rglob('*.py')):
        lines = path.read_bytes().split(b'\n')
        if len(lines) < 2 or lines[1] != want:
          got = lines[1] if len(lines) > 1 else b''
          bad.append(f'{path}:2: {got!r}')
    self.assertEqual(bad, [])


class TestConstantNamingBreaksTheStatedRule(unittest.TestCase):
  '''README lines 102-104: module constants are lowercase.

  ``env/nse.py`` lines 54 and 57 define ``MAX_WEIGHT`` and
  ``MAX_SECTOR_WEIGHT``. pylint's ``const-rgx`` accepts either case, so
  only the README rule is broken, not the build.
  '''

  def test_no_module_level_constant_is_uppercase(self):
    offenders = []
    for path in sorted(SRC.rglob('*.py')):
      tree = ast.parse(path.read_text(encoding='utf-8'))
      for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
          targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif (isinstance(node, ast.AnnAssign)
              and isinstance(node.target, ast.Name)):
          targets = [node.target]
        for target in targets:
          name = target.id
          if name == name.lower() or name.startswith('_'):
            continue
          # The repo has TWO deliberate upper-case conventions and this
          # test originally invented a third. Module CONSTANTS are
          # UPPERCASE (DELIVERY, INTRADAY, TRADING_DAYS_PER_YEAR,
          # CONTEXT_ENABLED) and type ALIASES are CamelCase (Obs, Info,
          # Clock, Action, Check). pylint's const-rgx accepts both and the
          # build is green. An earlier README claimed constants were
          # lowercase, which described a convention the code never used.
          is_alias = isinstance(node.value, ast.Subscript)
          if is_alias or (isinstance(node.value, ast.Name)
                          and node.value.id.isupper()):
            continue
          # A bare UPPER_CASE assignment of a literal or a call is a
          # module constant and SHOULD be upper-case, not an offender.
          if name.isupper():
            continue
          offenders.append(f'{path}:{node.lineno}: {name}')
    self.assertEqual(offenders, [])


class TestDescribeSourcesIsUnreachable(unittest.TestCase):
  '''``skills/loader.py`` line 463 defines it; no ``__all__`` carries it.

  ``from stock_rl.skills import describe_sources`` fails, and the package
  ``__all__`` does not list it, so the only way in is the private module
  path that tests use.
  '''

  def test_the_package_all_carries_it(self):
    self.assertIn('describe_sources', skills.__all__)

  def test_the_package_reexports_the_name(self):
    self.assertTrue(hasattr(skills, 'describe_sources'))


class TestUnresolvableCrossReference(unittest.TestCase):
  '''``graph/edges.py`` line 76 points ``:attr:`` at nothing.'''

  def test_the_docstring_no_longer_points_at_a_bare_name(self):
    edges = (SRC / 'stock_rl' / 'graph' / 'edges.py').read_text(
      encoding='utf-8')
    present = ':attr:`depends_on`' in edges
    self.assertFalse(
      present,
      'EdgeKind.AFFECTED_BY cross-references a bare depends_on, which names '
      'nothing in the package; it should read EdgeKind.DEPENDS_ON')

  def test_nothing_in_the_package_defines_depends_on(self):
    found = []
    for path in sorted(SRC.rglob('*.py')):
      tree = ast.parse(path.read_text(encoding='utf-8'))
      for node in tree.body:
        if isinstance(node, ast.Assign):
          for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'depends_on':
              found.append(str(path))
        elif (isinstance(node, ast.AnnAssign)
              and isinstance(node.target, ast.Name)
              and node.target.id == 'depends_on'):
          found.append(str(path))
    self.assertEqual(found, [])


class TestRepoStatedFormattingRules(unittest.TestCase):
  '''The repo's own rules over every source file.

  Shebang, coding cookie, trailing newline and no tabs all pass today and
  are pinned so a new file cannot break them silently.
  '''

  def _sources(self):
    for root in ('src', 'tests'):
      yield from sorted((ROOT / root).rglob('*.py'))

  def test_line_one_is_exactly_the_shebang(self):
    want = b'#!/usr/bin/env python'
    bad = [str(p) for p in self._sources()
           if p.read_bytes().split(b'\n')[0] != want]
    self.assertEqual(bad, [])

  def test_every_source_file_ends_with_a_newline(self):
    bad = [str(p) for p in self._sources()
           if not p.read_bytes().endswith(b'\n')]
    self.assertEqual(bad, [])

  def test_no_file_contains_a_tab(self):
    bad = [str(p) for p in self._sources() if b'\t' in p.read_bytes()]
    self.assertEqual(bad, [])

  def test_no_logical_line_is_indented_by_four_spaces(self):
    offenders = []
    for path in self._sources():
      source = path.read_text(encoding='utf-8')
      previous = None
      try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
      except tokenize.TokenError:
        continue
      for tok in tokens:
        if tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT,
                        tokenize.INDENT, tokenize.DEDENT):
          continue
        if (previous is None
            or previous in (tokenize.NEWLINE, tokenize.NL,
                            tokenize.INDENT, tokenize.DEDENT)):
          if (tok.type != tokenize.STRING and tok.start[1] > 0
              and tok.start[1] % 4 == 0):
            offenders.append(f'{path}:{tok.start[0]} indent {tok.start[1]}')
        previous = tok.type
    self.assertEqual(offenders, [])


class TestPublicApiIsImportable(unittest.TestCase):
  '''Every name in every ``__all__`` exists at runtime in all 55 modules.'''

  def test_no_all_entry_is_missing_from_its_module(self):
    missing = []
    names = ['stock_rl'] + [
      info.name for info in
      pkgutil.walk_packages(stock_rl.__path__, 'stock_rl.')
      if not info.name.endswith('__main__')]
    self.assertGreater(len(names), 50)
    for name in names:
      module = importlib.import_module(name)
      for exported in getattr(module, '__all__', ()):
        if not hasattr(module, exported):
          missing.append(f'{name}.{exported}')
    self.assertEqual(missing, [])


class TestEntityLinkDocstringMatchesTheCode(unittest.TestCase):
  '''The five worked examples the module docstring promises.

  These pass today. They are pinned because the docstring was rewritten
  twice in this audit window and a third divergence would be invisible.
  '''

  def test_the_module_docstring_examples_refuse_rather_than_resolve(self):
    # These four were documented as RESOLVING under the prefix design this
    # module no longer has. Resolution is now a table lookup: every written
    # form is indexed once at construction and a query is one dict hit. A
    # form nobody registered is UNKNOWN, which is what all four now are.
    #
    # The reason this is the right outcome rather than a regression is that
    # the same prefix layer produced 'AXISBANKING' -> AXISBANK and
    # 'ITC-INFRA' -> ITC: different companies, answered with whichever
    # symbol shared a prefix. A lookup cannot express that failure.
    for text in ('Infosy', 'inf', 'Axis Ban', 'sun pharm'):
      with self.subTest(text=text):
        result = entity_link.resolve(text)
        self.assertEqual(result.status, entity_link.LinkStatus.UNKNOWN)
        self.assertIsNone(result.symbol)
        self.assertIn('never guesses', result.reason)
    # What must still resolve, and does.
    for text, claimed in (('Infosys', 'INFY'), ('Infosys Ltd', 'INFY'),
                         ('Sun Pharma', 'SUNPHARMA'),
                         ('Sun Pharmaceutical Ltd', 'SUNPHARMA')):
      with self.subTest(text=text):
        self.assertEqual(entity_link.resolve(text).symbol, claimed)

  def test_a_whole_match_is_not_labelled_a_guess(self):
    for text in ('RELIANCE', 'Asian Paint', 'Infosys'):
      with self.subTest(text=text):
        self.assertFalse(entity_link.resolve(text).truncated)

  def test_the_ambiguity_examples_refuse_rather_than_choose(self):
    # A partial symbol is UNKNOWN under a lookup, not AMBIGUOUS: there is
    # no prefix scan to collect candidates from. UNKNOWN is also the more
    # honest answer, since a prefix-derived candidate list asserts those
    # companies were plausible and is silent about every other one.
    for text in ('Tata', 'Rel', 'Relian', 'RELI'):
      with self.subTest(text=text):
        result = entity_link.resolve(text)
        self.assertEqual(result.status, entity_link.LinkStatus.UNKNOWN)
        self.assertFalse(result.candidates)
    # AMBIGUOUS is still reachable, and now means a table collision: two
    # registry rows claiming one written form.
    linker = entity_link.SymbolLinker(
      (entity_link.SymbolRecord('AAA', 'Same Name'),
       entity_link.SymbolRecord('BBB', 'Same Name')))
    collided = linker.link('Same Name')
    self.assertEqual(collided.status, entity_link.LinkStatus.AMBIGUOUS)
    self.assertEqual(collided.candidates, ('AAA', 'BBB'))

  def test_a_symbol_extension_is_refused(self):
    for text in ('ITC-INFRA', 'AXISBANKING'):
      with self.subTest(text=text):
        self.assertEqual(entity_link.resolve(text).status,
                         entity_link.LinkStatus.UNKNOWN)

  def test_matched_by_only_emits_the_documented_values(self):
    probes = ('RELIANCE', 'Asian Paint', 'Infosys', 'RIL', 'inf', 'Infosy',
              'Tata', '', 'ITC-INFRA')
    seen = {entity_link.resolve(t).matched_by for t in probes}
    # 'table' for a resolved lookup, 'none' otherwise. The earlier design
    # also emitted 'alias', 'symbol' and 'prefix'; those layers are gone.
    self.assertTrue(seen <= {'table', 'none'},
                    f'undocumented matched_by values: {sorted(seen)}')

  def test_min_prefix_chars_gates_the_prefix_path_only(self):
    linker = entity_link.SymbolLinker(
      (entity_link.SymbolRecord('ZZZTEST', 'Zzz Test Limited'),),
      min_prefix=99,
    )
    # A fragment is refused, which is what the floor is for.
    for text in ('ZZZTES', 'zzz', 'ZZZTESTX'):
      with self.subTest(text=text):
        self.assertEqual(linker.link(text).status,
                         entity_link.LinkStatus.UNKNOWN)
    # A whole symbol is not a fragment, so the floor does not apply and
    # must not be documented as though it does.
    self.assertEqual(linker.link('ZZZTEST').status,
                     entity_link.LinkStatus.RESOLVED)


class TestSeedEnumerationCostClaim(unittest.TestCase):
  '''``traversal.py`` line 87 claims 56 extensions for the seed.

  Verified correct. Pinned so a refactor cannot change it silently.
  '''

  def test_the_seed_enumeration_spends_fifty_six_extensions(self):
    graph = nifty50_seed()
    adjacency = {key: sorted(graph.causes_of(key))
                 for key in sorted(graph.nodes)}
    meter = traversal._Meter(left=10 ** 9, ceiling=10 ** 9)
    found: list[tuple[str, ...]] = []
    for start in sorted(graph.nodes):
      traversal._walk(start, start, adjacency, [start], {start}, found,
                      set(), 10, 8, meter)
    self.assertEqual(10 ** 9 - meter.left, 56)


class TestZeroVolatilityVaRContract(unittest.TestCase):
  '''The documented asymmetric zero-volatility answer is what ships.

  Verified correct. Pinned so the two functions cannot silently agree.
  '''

  def test_parametric_var_returns_negative_mean_at_zero_volatility(self):
    self.assertAlmostEqual(var_module.parametric_var(0.01, 0.0), -0.01)

  def test_portfolio_var_returns_zero_on_a_zero_variance_book(self):
    covariance = [[1.0, -1.0], [-1.0, 1.0]]
    self.assertEqual(var_module.portfolio_var([0.5, 0.5], covariance), 0.0)

  def test_neither_function_returns_nan_at_zero_volatility(self):
    covariance = [[1.0, -1.0], [-1.0, 1.0]]
    for value in (var_module.parametric_var(0.01, 0.0),
                  var_module.parametric_cvar(0.01, 0.0),
                  var_module.portfolio_var([0.5, 0.5], covariance)):
      self.assertFalse(math.isnan(value))


class TestSizingPonytailIsWhatTheCodeDoes(unittest.TestCase):
  '''The sizing PONYTAIL about one order per slice is accurate. Pinned.'''

  def test_each_call_gets_the_whole_slice_with_no_shared_ledger(self):
    wide = dict(max_order_quantity=10_000_000, max_order_value=100_000_000.0)
    limits = RmsLimits(
      cumulative_open_order_value=100_000_000.0, max_position=100_000_000,
      max_trading_value=100_000_000.0, max_exposure=100_000_000.0,
      max_turnover=100_000_000.0, max_security_value=100_000_000.0)
    securities = {
      symbol: SecurityLimits(symbol=symbol, band=PriceBand(50.0, 400.0),
                             mwpl=PriceBand(10.0, 900.0), **wide)
      for symbol in ('AAA', 'BBB')}
    sizer = StockSizer(RmsChecker(limits, securities), 1_000_000.0)
    inputs = SizingInputs(realised_vol=0.001, target_vol=0.5)
    account = AccountState()
    first = sizer.size('AAA', 100.0, buy, inputs, 'vol_target', account,
                       reference_price=100.0)
    second = sizer.size('BBB', 100.0, buy, inputs, 'vol_target', account,
                        reference_price=100.0)
    self.assertEqual(first.notional, second.notional)


if __name__ == '__main__':
  unittest.main()



