#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''OHLCV price bars and a loader for vendor CSV exports.

The standard library reads CSV, so this module uses ``csv`` and adds no
dependency. The only judgement call is the input format: NSE authorised
vendors (TrueData, Global Datafeeds) deliver tick or minute data as
compressed binary, while the daily end-of-day files most research starts
with are CSV or Parquet.

Parquet is deliberately unsupported. Supporting it would mean taking a
runtime dependency on pyarrow purely to read a file, and the format
choice belongs to the ingestion layer, not the backtest core. Convert
once, at the boundary, and the core stays dependency-free.
'''

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

__all__ = ['Bar', 'load_csv', 'parse_timestamp']

#: Column name aliases, lowercased. Vendor exports disagree on spelling
#: and capitalisation, so each logical field accepts several names.
_FIELD_ALIASES = {
  'timestamp': ('timestamp', 'date', 'datetime', 'time', 'trade_date'),
  'open': ('open', 'o', 'openprice', 'open_price'),
  'high': ('high', 'h', 'highprice', 'high_price'),
  'low': ('low', 'l', 'lowprice', 'low_price'),
  'close': ('close', 'c', 'closeprice', 'close_price', 'ltp'),
  'volume': ('volume', 'v', 'vol', 'totaltradedqty', 'total_traded_qty'),
}


def parse_timestamp(text: str) -> datetime:
  '''Parse a vendor timestamp string into a datetime.

  ``datetime.fromisoformat`` is tried first because it is exact, fast and
  handles most modern exports, then the layouts Indian vendor files use
  most often. Trying formats in order rather than maintaining one parser
  per vendor keeps ingestion tolerant of reformatting between providers.

  PONYTAIL: three extra layouts, chosen because they cover the vendor
  files in practice. Ceiling: an exotic layout raises rather than
  guessing, which is deliberate -- a silently misparsed date is far worse
  than a failed load. Upgrade path: append to ``_FORMATS``; there is no
  other seam in this function.

  Args:
    text: Timestamp string from the CSV.

  Returns:
    Parsed datetime.

  Raises:
    ValueError: If the string matches no known layout.
  '''
  candidate = text.strip()
  if not candidate:
    raise ValueError('empty timestamp')
  try:
    return datetime.fromisoformat(candidate)
  except ValueError:
    pass
  for fmt in _FORMATS:
    try:
      return datetime.strptime(candidate, fmt)
    except ValueError:
      continue
  raise ValueError(f'unrecognised timestamp format: {text!r}')


#: Non-ISO timestamp layouts attempted in order, after fromisoformat.
_FORMATS: tuple[str, ...] = (
  '%Y-%m-%d %H:%M:%S',
  '%d-%m-%Y %H:%M:%S',
  '%Y-%m-%d',
)


@dataclass(frozen=True, slots=True)
class Bar:
  '''One price bar.

  Attributes:
    timestamp: Bar close time. Bars are assumed to arrive in ascending
      order; ``load_csv`` validates this rather than sorting, so that a
      scrambled file fails loudly instead of being silently reordered.
    open: Open price.
    high: High price.
    low: Low price.
    close: Close price.
    volume: Traded quantity. Zero or negative is tolerated because
      illiquid bars legitimately print no volume.
  '''

  timestamp: datetime
  open: float
  high: float
  low: float
  close: float
  volume: float = 0.0

  @property
  def typical_price(self) -> float:
    '''Return (high + low + close) / 3, the average traded price.'''
    return (self.high + self.low + self.close) / 3.0

  @property
  def range(self) -> float:
    '''Return high minus low, the intrabar spread.'''
    return self.high - self.low

  @property
  def is_flat(self) -> float:
    '''Return True when the bar traded no intrabar range.'''
    return self.range <= 0.0

  @property
  def change(self) -> float:
    '''Return close minus open for this bar alone.'''
    return self.close - self.open


def load_csv(path: str | Path) -> list[Bar]:
  '''Load bars from a vendor CSV file.

  Column names are matched case-insensitively against a small alias table
  so that TrueData, Global Datafeeds and hand-built exports all load
  without preprocessing.

  Args:
    path: Path to a CSV with a header row and one bar per line.

  Returns:
    Bars in file order, which must already be ascending by timestamp.

  Raises:
    FileNotFoundError: If the path does not exist.
    ValueError: If the header lacks required columns, a value will not
      parse, or the timestamps are not strictly ascending.
  '''
  rows = _read_rows(path)
  if not rows:
    raise ValueError(f'{path} contains no rows')
  if len(rows) < 2:
    # A header with no data rows would otherwise return an empty bar
    # list, which produces a silently empty backtest downstream. Better
    # to fail here, where the cause is still obvious.
    raise ValueError(f'{path} has a header but no data rows')
  header = rows[0]
  index = _column_index(header)
  bars: list[Bar] = []
  previous: datetime | None = None
  for line_number, row in enumerate(rows[1:], start=2):
    bar = _to_bar(row, index, line_number)
    if previous is not None and bar.timestamp <= previous:
      raise ValueError(
        f'{path} line {line_number}: timestamps must be strictly '
        f'ascending, got {bar.timestamp} after {previous}')
    bars.append(bar)
    previous = bar.timestamp
  return bars


def _read_rows(path: str | Path) -> list[list[str]]:
  '''Return all CSV rows for the given path.

  Args:
    path: Path to a CSV file.

  Returns:
    List of rows, each a list of stripped string cells.

  Raises:
    FileNotFoundError: If the path does not exist.
  '''
  with Path(path).open(newline='', encoding='utf-8') as handle:
    return [[cell.strip() for cell in row]
            for row in csv.reader(handle) if any(row)]


def _column_index(header: list[str]) -> dict[str, int]:
  '''Map logical field names to column positions.

  Args:
    header: Stripped header cells.

  Returns:
    Mapping of field name to zero-based column index.

  Raises:
    ValueError: If a required column is absent.
  '''
  lowered = [cell.strip().lower().replace(' ', '_') for cell in header]
  index: dict[str, int] = {}
  for field, aliases in _FIELD_ALIASES.items():
    for alias in aliases:
      if alias in lowered:
        index[field] = lowered.index(alias)
        break
  missing = {'timestamp', 'open', 'high', 'low', 'close'} - set(index)
  if missing:
    raise ValueError(f'missing required columns: {sorted(missing)}')
  return index


def _to_bar(row: list[str], index: dict[str, int],
            line_number: int) -> Bar:
  '''Convert one CSV row into a Bar.

  Args:
    row: Stripped cells from the data row.
    index: Mapping from logical field name to column position.
    line_number: One-based file line, for error messages.

  Returns:
    The parsed bar.

  Raises:
    ValueError: If a price will not parse as a float.
  '''
  def cell(field: str) -> str:
    return row[index[field]]

  def number(field: str) -> float:
    raw = cell(field)
    try:
      return float(raw)
    except ValueError as exc:
      raise ValueError(
        f'line {line_number}: {field} is not a number: {raw!r}') from exc

  volume = 0.0
  if 'volume' in index and cell('volume'):
    volume = number('volume')
  return Bar(
    timestamp=parse_timestamp(cell('timestamp')),
    open=number('open'),
    high=number('high'),
    low=number('low'),
    close=number('close'),
    volume=volume,
  )
