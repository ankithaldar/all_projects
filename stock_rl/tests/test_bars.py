#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for OHLCV bar loading.'''

from datetime import datetime

import pytest

from stock_rl.bars import Bar, load_csv, parse_timestamp

HEADER = 'timestamp,open,high,low,close,volume'
ROW = '2026-01-01T09:15:00,100.0,101.0,99.0,100.5,5000'


def write(tmp_path, text):
  '''Write a CSV fixture and return its path.

  Args:
    tmp_path: pytest temporary directory.
    text: Full file contents including any header row.

  Returns:
    Path to the written file.
  '''
  target = tmp_path / 'bars.csv'
  target.write_text(text, encoding='utf-8')
  return target


class TestParseTimestamp:
  '''Layout tolerance without silently guessing.'''

  @pytest.mark.parametrize('text,expected', [
    ('2026-01-01T09:15:00', datetime(2026, 1, 1, 9, 15)),
    ('2026-01-01 09:15:00', datetime(2026, 1, 1, 9, 15)),
    ('2026-01-01', datetime(2026, 1, 1)),
    ('  2026-01-01  ', datetime(2026, 1, 1)),
  ])
  def test_known_layouts(self, text, expected):
    assert parse_timestamp(text) == expected

  def test_unparseable_raises(self):
    with pytest.raises(ValueError, match='unrecognised'):
      parse_timestamp('not a date')

  def test_empty_raises(self):
    with pytest.raises(ValueError, match='empty'):
      parse_timestamp('   ')


class TestBarProperties:
  '''Derived fields on the record.'''

  def test_typical_price(self):
    bar = Bar(datetime(2026, 1, 1), 10.0, 12.0, 9.0, 11.0)
    assert bar.typical_price == pytest.approx(32.0 / 3.0)

  def test_range(self):
    bar = Bar(datetime(2026, 1, 1), 10.0, 12.0, 9.0, 11.0)
    assert bar.range == pytest.approx(3.0)

  def test_flat_bar_detected(self):
    bar = Bar(datetime(2026, 1, 1), 10.0, 10.0, 10.0, 10.0)
    assert bar.is_flat

  def test_change_is_intrabar_only(self):
    bar = Bar(datetime(2026, 1, 1), 10.0, 12.0, 9.0, 11.0)
    assert bar.change == pytest.approx(1.0)


class TestLoadCsv:
  '''Parsing vendor exports and rejecting bad input.'''

  def test_loads_a_single_row(self, tmp_path):
    bars = load_csv(write(tmp_path, f'{HEADER}\n{ROW}\n'))
    assert len(bars) == 1
    assert bars[0].close == pytest.approx(100.5)
    assert bars[0].volume == pytest.approx(5000.0)

  def test_loads_multiple_rows_in_order(self, tmp_path):
    text = (
      f'{HEADER}\n'
      '2026-01-01T09:15:00,100,101,99,100,10\n'
      '2026-01-01T09:16:00,100,102,99,101,20\n'
    )
    bars = load_csv(write(tmp_path, text))
    assert [bar.close for bar in bars] == [100.0, 101.0]
    assert bars[0].timestamp < bars[1].timestamp

  def test_column_aliases(self, tmp_path):
    # Vendor exports disagree on spelling; LTP must map to close.
    text = 'DATE,Open,High,Low,LTP\n2026-01-01,1,2,0.5,1.5\n'
    bars = load_csv(write(tmp_path, text))
    assert bars[0].close == pytest.approx(1.5)

  def test_missing_volume_defaults_to_zero(self, tmp_path):
    text = 'timestamp,open,high,low,close\n2026-01-01,1,2,0.5,1.5\n'
    assert load_csv(write(tmp_path, text))[0].volume == 0.0

  def test_blank_volume_cell_defaults_to_zero(self, tmp_path):
    text = 'timestamp,open,high,low,close,volume\n2026-01-01,1,2,0.5,1.5,\n'
    assert load_csv(write(tmp_path, text))[0].volume == 0.0

  def test_descending_timestamps_rejected(self, tmp_path):
    # Must fail loudly rather than silently sorting, which would hide a
    # broken export.
    text = (
      f'{HEADER}\n'
      '2026-01-02T09:15:00,100,101,99,100,10\n'
      '2026-01-01T09:15:00,100,101,99,100,10\n'
    )
    with pytest.raises(ValueError, match='ascending'):
      load_csv(write(tmp_path, text))

  def test_duplicate_timestamps_rejected(self, tmp_path):
    text = f'{HEADER}\n{ROW}\n{ROW}\n'
    with pytest.raises(ValueError, match='ascending'):
      load_csv(write(tmp_path, text))

  def test_missing_column_rejected(self, tmp_path):
    text = 'timestamp,open,high,close\n2026-01-01,1,2,1.5\n'
    with pytest.raises(ValueError, match='missing required columns'):
      load_csv(write(tmp_path, text))

  def test_non_numeric_price_rejected(self, tmp_path):
    text = 'timestamp,open,high,low,close\n2026-01-01,1,2,0.5,abc\n'
    with pytest.raises(ValueError, match='not a number'):
      load_csv(write(tmp_path, text))

  def test_bad_timestamp_names_the_line(self, tmp_path):
    text = f'{HEADER}\nnope,1,2,0.5,1.5,1\n'
    with pytest.raises(ValueError, match='unrecognised'):
      load_csv(write(tmp_path, text))

  def test_empty_file_rejected(self, tmp_path):
    with pytest.raises(ValueError, match='no rows'):
      load_csv(write(tmp_path, ''))

  def test_header_only_rejected(self, tmp_path):
    # Must raise rather than return an empty list, which would produce a
    # silently empty backtest.
    with pytest.raises(ValueError, match='no data rows'):
      load_csv(write(tmp_path, f'{HEADER}\n'))

  def test_missing_file_raises(self, tmp_path):
    with pytest.raises(FileNotFoundError):
      load_csv(tmp_path / 'absent.csv')

  def test_blank_lines_ignored(self, tmp_path):
    text = f'{HEADER}\n\n{ROW}\n\n'
    assert len(load_csv(write(tmp_path, text))) == 1
