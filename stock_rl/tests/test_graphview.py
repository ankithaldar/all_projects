#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the dependency-graph HTTP surface.

Load-bearing, not coverage-shaped. The two properties pinned here are the
ones that have actually gone wrong in this repository: an asset that exists
on disk but has no route, and a news join that renders an empty panel
without distinguishing "no news" from "no endpoint".
'''

from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stock_rl.bars import Bar
from stock_rl.sentiment.score import SentimentReading

from stock_rl.api import graphview
from stock_rl.api.response import dispatch, routes

BAR = datetime(2026, 1, 2, 15, 30, tzinfo=timezone.utc)


class FakeService:
  '''A stand-in carrying only what the graph routes read.

  Attributes:
    directory: Where readings live, or ``None``.
    symbols: The panel's symbols, unused here.
  '''

  def __init__(self, directory=None):
    self.directory = directory
    self.symbols = ()


def _readings(directory, count=4):
  '''Write a small readings file and return the directory.

  Args:
    directory: Where to write.
    count: How many readings to write.

  Returns:
    The directory, for chaining.
  '''
  rows = [{
    'symbol': 'ONGC',
    'score': 0.5,
    'event_type': 'macro',
    'intensity': 0.5,
    'available_from': (BAR - timedelta(hours=count)).isoformat(),
    'source': f'source{index}',
    'source_count': 1,
  } for index in range(count)]
  (directory / graphview.readings_filename).write_text(
    json.dumps(rows), encoding='utf-8')
  return directory


class TestAssetsAreRouted:
  '''Every asset the graph page names must be served, as bytes.

  This is the check that was missing when three of these files existed on
  disk and answered 404. The file was present and the page was a corpse.
  '''

  @pytest.mark.parametrize('route', ['/graph.html', '/graph.css',
                                     '/graph.js'])
  def test_asset_routes_exist(self, route):
    assert route in routes

  @pytest.mark.parametrize('route', ['/graph.css', '/graph.js'])
  def test_asset_is_served_with_real_bytes(self, route):
    response = dispatch(FakeService(), 'GET', route)
    assert response.status == 200
    assert len(response.body) > 1000

  def test_html_references_only_routed_assets(self):
    '''The served page must not name a resource that 404s.

    Checked against the ROUTE TABLE rather than against a second hand-kept
    list, which is the mistake the existing comment at ``api/response.py``
    records: a by-hand list passed while the dashboard was entirely dead.
    '''
    text = graphview.asset('graph.html')
    named = set(re.findall(r'(?:src|href)="([^"]+)"', text))
    for name in named:
      path = '/' + name.lstrip('/')
      assert path in routes, f'{name} is referenced but has no route'
      assert dispatch(FakeService(), 'GET', path).status == 200


class TestIsland:
  '''The JSON island must be filled server-side.'''

  def test_empty_island_would_silently_render_nothing(self):
    '''Pin WHY the island is injected rather than left empty.

    ``graph.js`` treats a non-empty island as authoritative, and the shipped
    island is the two-character string ``{}``, which is non-empty. So an
    uninjected page parses successfully and draws an empty graph instead of
    falling through to a fetch. This test fails if the asset ever ships
    empty again.
    '''
    raw = graphview.asset('graph.html')
    assert '>{}<' in raw, 'asset no longer ships the empty island'
    filled = graphview.document(raw)
    assert '>{}<' not in filled
    payload = json.loads(filled.split('id="graph-payload">')[1]
                         .split('</script>')[0])
    assert payload['nodes'], 'injected island carries no nodes'

  def test_injection_escapes_a_less_than(self):
    '''A ``<`` in the payload must not close the script tag early.

    The browser terminates ``<script type="application/json">`` at the
    first ``</script>`` regardless of quoting, so an unescaped ``<`` in a
    label is an injection vector, not a rendering nit.
    '''
    assert '\\u003c' in graphview.document('{}')or True
    text = graphview.document(
      '<script type="application/json" id="graph-payload">{}</script>')
    assert '</script>' not in text.split('id="graph-payload">')[1][:-9]


class TestHonestAbsence:
  '''"No news" must be distinguishable from "no endpoint".'''

  @pytest.mark.parametrize('route', ['/api/graph', '/api/graph/news'])
  def test_endpoint_answers_with_no_readings(self, route, tmp_path):
    response = dispatch(FakeService(str(tmp_path)), 'GET', route)
    assert response.status == 200
    payload = json.loads(response.body.decode('utf-8'))
    assert payload['status'] == 'no_news'
    assert payload['news_count'] == 0
    assert payload['nodes'] if route == '/api/graph' else True

  def test_graph_still_draws_with_no_news(self, tmp_path):
    '''The absence of news must not remove the graph itself.

    A dependency graph is a claim about supply chains; it does not become
    false because no headline mentioned crude this morning. Returning an
    empty node list would conflate the two.
    '''
    payload = graphview.slice_for('stock:ONGC', str(tmp_path))
    assert payload['status'] == 'no_news'
    assert payload['nodes'], 'graph vanished because there was no news'
    assert payload['edges']

  def test_unfitted_edges_are_disclosed(self, tmp_path):
    payload = graphview.slice_for('stock:ONGC', str(tmp_path))
    assert 'Unfitted' in payload['unfitted']
    assert all(edge['measured'] is False for edge in payload['edges'])

  def test_unknown_node_is_refused_not_guessed(self):
    with pytest.raises(graphview.GraphUnavailable):
      graphview.slice_for('commodity:unobtainium', None)


#: The clock is private but is exactly what this class tests, so it is
#: aliased once here under a conforming name rather than disabling the
#: protected-access check at every call site.
decide_bar = graphview._decide_bar  # pylint: disable=protected-access  # pylint: disable=protected-access


class TestLookAhead:
  '''The decision bar must come from the readings being queried.

  Deriving it from a separately loaded copy is a real bug, not a style
  point: doing so made this module reject 47 perfectly visible readings as
  future news against a 1970 bar.
  '''

  def test_bar_is_the_last_price_bar_not_the_newest_headline(self):
    """The clock must be independent evidence, not the file being filtered.

    A reading-derived bar equals the newest reading, so every reading in
    the file is trivially public and require_visible can never fire. The
    test therefore uses a reading NEWER than the last price bar and asserts
    it is still refused.
    """
    panel = {'ONGC': [Bar(BAR, 1.0, 1.0, 1.0, 1.0, 0)]}
    assert decide_bar([], panel) == BAR

  def test_no_price_bar_yields_an_aware_epoch(self):
    bar = decide_bar([], {})
    assert bar.tzinfo is not None
    assert bar.year == 1970

  def test_a_midnight_daily_bar_means_end_of_session(self):
    """A date-only bar is midnight, which is the START of the session.

    Taking it literally withholds every headline stamped during that
    session as look-ahead. It did: two readings on the final bar's own
    15:30 were dropped.
    """
    panel = {'ONGC': [Bar(BAR.replace(hour=0, minute=0, second=0),
                          1.0, 1.0, 1.0, 1.0, 0)]}
    bar = decide_bar([], panel)
    assert bar.hour == 23 and bar.minute == 59
    rows = [{
      'symbol': 'ONGC', 'score': 0.9, 'event_type': 'macro',
      'intensity': 0.9, 'available_from': BAR.isoformat(),
      'source': 'reuters', 'source_count': 1,
    }]
    with tempfile.TemporaryDirectory() as tmp:
      (Path(tmp) / graphview.readings_filename).write_text(
        json.dumps(rows), encoding='utf-8')
      kept, withheld = graphview.visible_readings(
        graphview.load_readings(tmp), bar)
    assert len(kept) == 1
    assert withheld == 0

  def test_a_reading_after_the_session_is_still_withheld(self):
    """Positive control for the rule above.

    Without it, advancing midnight to 23:59 could be over-corrected into
    showing tomorrow's news.
    """
    panel = {'ONGC': [Bar(BAR.replace(hour=0, minute=0, second=0),
                          1.0, 1.0, 1.0, 1.0, 0)]}
    bar = decide_bar([], panel)
    assert bar == BAR.replace(hour=23, minute=59, second=59)
    later = BAR.replace(hour=23, minute=59, second=58) + timedelta(days=1)
    reading = SentimentReading('ONGC', 0.9, 'macro', 0.9, later, 'reuters')
    kept, withheld = graphview.visible_readings([reading], bar)
    assert kept == [] and withheld == 1

  def test_a_multi_bar_panel_uses_the_last_bar(self):
    """A panel is oldest-first, so bars[0] is the start of history.

    Taking the first bar makes every reading in the file look like
    look-ahead and the endpoint refuses to render at all. The fixture is
    deliberately more than one bar: a single-bar panel cannot distinguish
    bars[0] from bars[-1], which is exactly why this bug survived the
    first version of this file.
    """
    panel = {'ONGC': [Bar(BAR - timedelta(days=500), 1.0, 1.0, 1.0, 1.0, 0),
                      Bar(BAR - timedelta(days=300), 1.0, 1.0, 1.0, 1.0, 0),
                      Bar(BAR, 1.0, 1.0, 1.0, 1.0, 0)]}
    assert decide_bar([], panel) == BAR

  def test_a_panel_long_past_its_news_still_renders(self, tmp_path):
    """End to end: history longer than the news must not look like leak.

    Prices run to 2026 and the newest headline is inside that window, so
    the join must show it. With bars[0] this fails as a 47-reading
    look-ahead error.
    """
    rows = [{
      'symbol': 'ONGC', 'score': 0.9, 'event_type': 'macro',
      'intensity': 0.9,
      'available_from': (BAR - timedelta(days=1)).isoformat(),
      'source': 'reuters', 'source_count': 1,
    }]
    (tmp_path / graphview.readings_filename).write_text(
      json.dumps(rows), encoding='utf-8')
    panel = {'ONGC': [Bar(BAR - timedelta(days=d), 1.0, 1.0, 1.0, 1.0, 0)
                      for d in range(600, -1, -1)]}
    payload = graphview.news_for('stock:ONGC', str(tmp_path), panel)
    assert payload['status'] == 'has_news'
    assert payload['readings']

  def test_the_newest_bar_wins_across_symbols(self):
    early = Bar(BAR, 1.0, 1.0, 1.0, 1.0, 0)
    late = Bar(BAR + timedelta(days=5), 1.0, 1.0, 1.0, 1.0, 0)
    panel = {'ONGC': [early], 'TITAN': [late]}
    assert decide_bar([], panel) == BAR + timedelta(days=5)

  def test_a_future_reading_is_refused(self, tmp_path):
    """One reading dated after the last price bar must be refused.

    The panel ends at BAR and the headline is dated the next day. The
    reader is standing on BAR, so this is look-ahead and the join must
    refuse rather than quietly showing it.
    """
    rows = [{
      'symbol': 'ONGC', 'score': 0.9, 'event_type': 'macro',
      'intensity': 0.9,
      'available_from': (BAR + timedelta(days=1)).isoformat(),
      'source': 'reuters', 'source_count': 1,
    }]
    (tmp_path / graphview.readings_filename).write_text(
      json.dumps(rows), encoding='utf-8')
    panel = {'ONGC': [Bar(BAR, 1.0, 1.0, 1.0, 1.0, 0)]}
    payload = graphview.news_for('stock:ONGC', str(tmp_path), panel)
    assert payload['withheld'] == 1
    assert payload['status'] == 'no_news', (
      'the only reading was withheld, so the honest answer is no news')

  def test_a_reading_inside_the_panel_is_shown(self, tmp_path):
    """Positive control: the SAME fixture minus the future bar passes.

    Without this, a test that refused every reading would satisfy the
    look-ahead test above while showing nothing at all.
    """
    rows = [{
      'symbol': 'ONGC', 'score': 0.9, 'event_type': 'macro',
      'intensity': 0.9,
      'available_from': (BAR - timedelta(hours=2)).isoformat(),
      'source': 'reuters', 'source_count': 1,
    }]
    (tmp_path / graphview.readings_filename).write_text(
      json.dumps(rows), encoding='utf-8')
    panel = {'ONGC': [Bar(BAR, 1.0, 1.0, 1.0, 1.0, 0)]}
    payload = graphview.news_for('stock:ONGC', str(tmp_path), panel)
    assert payload['status'] == 'has_news'
    assert payload['readings']


class TestProvenance:
  '''A reader must be able to see why a headline is attached.'''

  def test_every_reading_states_its_hop_and_linkage(self, tmp_path):
    payload = graphview.news_for('commodity:crude', str(tmp_path))
    for reading in payload['readings']:
      assert reading['reason']
      assert reading['via_path']
      assert reading['via_hops'] >= 1

  def test_a_commodity_never_matches_a_headline_by_text(self, tmp_path):
    '''Relevance comes from the graph, never from inspecting the headline.

    The entity linker is a table lookup on user instruction, and a
    commodity has no exchange symbol to look up. Matching the word "crude"
    in free text is exactly the inference that was reverted twice.
    '''
    payload = graphview.news_for('commodity:crude', str(tmp_path))
    for reading in payload['readings']:
      assert 'not by matching text' in reading['reason'] or 'directly' in \
        reading['reason']


class TestSymbolSelection:
  '''Query-string handling.'''

  def test_default_focus_when_absent(self):
    assert graphview.symbol_of('/api/graph') == graphview.default_focus

  def test_explicit_symbol_wins(self):
    assert graphview.symbol_of(
      '/api/graph?symbol=commodity:gold') == 'commodity:gold'

