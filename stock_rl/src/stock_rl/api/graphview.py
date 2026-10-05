#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Serve the dependency graph and its node-to-news join over HTTP.

Kept out of :mod:`stock_rl.api.service` on purpose. That class is the
read-only book the dashboard already agreed on, and the graph is a
different question with a different failure mode: its honest answer is
frequently "no news", which must not be mistakable for "no endpoint".

Two honesty rules are load-bearing here and both have been got wrong in this
repository before:

- **No placeholder graph.** If the graph cannot be built the endpoint says
  so. It never draws a plausible-looking substitute, because a demo graph
  that renders cleanly is indistinguishable from a real one at a glance.
- **The decision bar is required, never defaulted.** A permissive default
  here would be look-ahead: news dated after the bar the reader is standing
  on must not appear. That bug has been reintroduced twice.
'''

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from stock_rl.bars import Bar
from stock_rl.graph.edges import DependencyGraph, nifty50_seed
from stock_rl.graph.news import node_slice, readings_for_node
from stock_rl.sentiment.score import SentimentReading

#: The Nifty-50 seed. Read-only, and the same graph the sentiment linker
#: resolves against, so a symbol the graph names and a symbol the linker
#: knows are never two different things.
default_focus = 'stock:RELIANCE'

#: Where synthetic readings live, relative to the served data directory.
readings_filename = 'readings.json'

#: Cap on readings parsed, so a large file cannot make every request slow.
readings_cap = 20_000


class GraphUnavailable(RuntimeError):
  '''The graph could not be built, and the caller must be told so.'''


@lru_cache(maxsize=1)
def _graph() -> DependencyGraph:
  '''Return the shared seed graph.

  Returns:
    The Nifty-50 dependency graph.
  '''
  return nifty50_seed()


def _decide_bar(
  readings: list[SentimentReading],
  panels: Mapping[str, Sequence[Bar]] | None = None,
) -> datetime:
  '''Return the bar the dashboard stands on.

  Taken from the LAST PRICE BAR, never from the readings themselves.

  Deriving it from the readings looks reasonable and is wrong in a way that
  silently disables the look-ahead guard: the bar becomes the newest
  headline in the file, so every reading in the file is trivially public
  and ``require_visible`` can never fire. The guard would report "clean"
  on a file containing tomorrow's news. A test asserting that a future
  reading is rejected fails against the reading-derived bar and passes
  against the price-derived one, which is the only reason this is spelled
  out rather than left implicit.

  The price panel is the right clock because it is independent evidence of
  what the reader could have known. A reader standing on the last bar of a
  2024 crash must not be shown 2026 news; that has been reintroduced in
  this repository twice.

  Args:
    readings: Unused for the clock, kept for the caller's readability.
    panels: Price panels, used only for their last timestamp. The LAST
      element, not the first: :mod:`stock_rl.bars` yields bars
      oldest-first, so ``bars[0]`` is the start of history and using it
      makes every reading in the file look like look-ahead.

  Returns:
    A timezone-aware datetime, the epoch when no price bar exists.
  '''
  del readings
  stamps = [bars[-1].timestamp for bars in (panels or {}).values() if bars]
  if not stamps:
    return datetime(1970, 1, 1, tzinfo=timezone.utc)
  latest = max(stamps)
  if latest.tzinfo is None:
    # A vendor CSV carrying a bare date parses to a naive datetime, and the
    # join refuses a naive bar rather than assuming a zone. Assuming UTC
    # here rather than raising, because the alternative is that a whole
    # panel of otherwise-fine daily bars cannot be viewed at all - and the
    # assumption is stated here rather than buried. A feed that publishes
    # genuinely local naive timestamps is mislabelled by its own exporter,
    # not by this module.
    latest = latest.replace(tzinfo=timezone.utc)
  # A date-only daily bar parses to midnight, which is the START of the
  # session, not its end. Treating it as the decision bar would make every
  # headline stamped during that same session look like look-ahead - and it
  # did: two readings dated the final bar's own 15:30 were withheld as
  # future. A reader standing on a completed daily bar has seen everything
  # from that day, so a midnight bar is advanced to the end of its day.
  if (latest.hour, latest.minute, latest.second) == (0, 0, 0):
    return latest.replace(hour=23, minute=59, second=59)
  return latest


def load_readings(directory: str | None) -> list[SentimentReading]:
  '''Parse readings from ``readings.json`` beside the served data.

  Args:
    directory: Directory holding the price CSVs, or ``None`` to try the
      environment variable and then give up.

  Returns:
    Readings in the order the file listed them. Empty when there is no
    file, which the caller reports as an absence rather than an error.
  '''
  path = _readings_path(directory)
  if path is None:
    return []
  try:
    raw = json.loads(path.read_text(encoding='utf-8'))
  except (OSError, ValueError):
    return []
  out: list[SentimentReading] = []
  for row in raw[:readings_cap]:
    try:
      out.append(_reading(row))
    except (KeyError, TypeError, ValueError):
      continue
  return out


def _readings_path(directory: str | None) -> Path | None:
  '''Return the readings file path, or None when there is nothing to read.

  Args:
    directory: Candidate directory.

  Returns:
    An existing path, or ``None``.
  '''
  candidates = []
  if directory:
    candidates.append(Path(directory) / readings_filename)
  env = 'STOCK_RL_READINGS'
  if os.environ.get(env):
    candidates.append(Path(os.environ[env]))
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  return None


def _reading(row: dict[str, Any]) -> SentimentReading:
  '''Build one reading, letting the dataclass refuse a naive timestamp.

  Args:
    row: One decoded JSON object.

  Returns:
    A validated reading.
  '''
  return SentimentReading(
    row['symbol'], float(row['score']), row['event_type'],
    float(row['intensity']), datetime.fromisoformat(row['available_from']),
    row['source'], int(row.get('source_count', 1)))


def visible_readings(
  readings: list[SentimentReading],
  bar: datetime,
) -> tuple[list[SentimentReading], int]:
  '''Return only the readings that were public at ``bar``.

  The join raises :class:`~stock_rl.sentiment.score.LookAheadError` on any
  reading dated after the bar rather than quietly showing it. That is the
  right default for a *training* caller, where a leak silently inflates a
  result. It is the wrong behaviour for a *viewer*, where a news file that
  simply runs past the last price bar is a data condition, not a bug, and
  turning it into an HTTP 500 labelled "this is a bug" makes the page
  unusable while telling the operator nothing true.

  So the leak is stopped here, explicitly, and the count of what was
  withheld is returned so the caller can disclose it. Silently dropping
  would be indistinguishable from having no news.

  Args:
    readings: Candidate readings, of any symbol.
    bar: The decision bar.

  Returns:
    The visible subset, and how many were withheld.
  '''
  kept = [r for r in readings if r.available_from <= bar]
  return kept, len(readings) - len(kept)


def slice_for(
  symbol: str,
  directory: str | None = None,
  panels: Mapping[str, Sequence[Bar]] | None = None,
) -> dict[str, Any]:
  '''Return the graph slice for a node, as the front end consumes it.

  Args:
    symbol: Node key such as ``stock:RELIANCE`` or ``commodity:crude``.
    directory: Directory holding readings.
    panels: Price panels, supplying the decision bar. Passed rather than
      loaded, so the bar is the one the service is already serving prices
      for and cannot disagree with them.

  Returns:
    ``GraphSlice.to_dict()`` output.

  Raises:
    GraphUnavailable: The node is not in the graph.
  '''
  raw = load_readings(directory)
  readings, _ = visible_readings(raw, _decide_bar(raw, panels))
  try:
    return node_slice(
      _graph(), symbol, readings, _decide_bar(raw, panels)).to_dict()
  except ValueError as exc:
    raise GraphUnavailable(str(exc)) from exc


def news_for(
  symbol: str,
  directory: str | None,
  panels: Mapping[str, Sequence[Bar]] | None = None,
) -> dict[str, Any]:
  '''Return the node-to-news join for a node.

  Args:
    symbol: Node key.
    directory: Directory holding readings.
    panels: Price panels, supplying the decision bar.

  Returns:
    ``NodeNews.to_dict()`` output.

  Raises:
    GraphUnavailable: The node is not in the graph.
  '''
  raw = load_readings(directory)
  bar = _decide_bar(raw, panels)
  readings, withheld = visible_readings(raw, bar)
  try:
    payload = readings_for_node(
      _graph(), symbol, readings, bar).to_dict()
  except ValueError as exc:
    raise GraphUnavailable(str(exc)) from exc
  payload['withheld'] = withheld
  return payload


def document(text: str, panels: Mapping[str, Sequence[Bar]] | None = None,
             directory: str | None = None) -> str:
  '''Return ``graph.html`` with its JSON island filled in.

  Args:
    text: The raw asset text.
    panels: Price panels, supplying the decision bar.
    directory: Directory holding readings.

  Returns:
    The document with a populated island, or the original when the marker
    is absent, which is reported by the caller rather than guessed at.
  '''
  island = json.dumps(
    slice_for(default_focus, directory, panels)).replace('<', '\\u003c')
  start = text.find('<script type="application/json" id="graph-payload">')
  if start < 0:
    return text
  open_end = text.find('>', start)
  close = text.find('</script>', open_end)
  if close < 0:
    return text
  return text[:open_end + 1] + island + text[close:]


def asset(name: str) -> str:
  '''Return a web asset by filename.

  Args:
    name: Filename inside :mod:`stock_rl.web`.

  Returns:
    The file's text.

  Raises:
    FileNotFoundError: The asset is absent.
  '''
  return resources.files('stock_rl.web').joinpath(name).read_text(
    encoding='utf-8')


def symbol_of(path: str) -> str:
  '''Return the node key a request asks for.

  Args:
    path: Raw path including query string.

  Returns:
    The requested node key, or the default focus.
  '''
  _, _, query = path.partition('?')
  found = parse_qs(query).get('symbol')
  return found[0] if found else default_focus

