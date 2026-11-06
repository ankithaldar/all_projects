#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for dashboard schedule utilities.'''

# pylint: disable=use-implicit-booleaness-not-comparison

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.core.items import NUM_ITEMS
from src.dashboard.utils import (
  export_replay_csv,
  parse_batch_schedule,
  parse_ga_log,
  simulate_schedule,
)


def _write_schedule(tmp_path):
  '''Write a minimal batch schedule file and return its path.'''
  path = os.path.join(str(tmp_path), 'schedule.txt')
  with open(path, 'w', encoding='utf-8') as handle:
    handle.write('header line one\nheader line two\n')
    handle.write('0 0 string 5\n')
    handle.write('1 5 string 3\n')
    handle.write('bad line\n')
    handle.write('2 10 unknownitem 2\n')
  return path


def _schedule_frame():
  '''Return a schedule DataFrame with a string action.'''
  return pd.DataFrame(
    [
      {
        'tick_index': 0,
        'elapsed_minutes': 0,
        'item_name': 'string',
        'batch_size_decided': 2,
      }
    ]
  )


class TestParseBatchSchedule:
  '''Tests for parse_batch_schedule.'''

  def test_parses_rows_and_skips_short_lines(self, tmp_path):
    '''Parsed rows contain tick, item and batch size.'''
    path = _write_schedule(tmp_path)
    frame = parse_batch_schedule(path)
    # F5: lines with an unknown item are skipped (with a warning),
    # so only the two valid 'string' rows remain.
    assert len(frame) == 2
    assert frame.iloc[0]['item_name'] == 'string'
    assert int(frame.iloc[0]['batch_size_decided']) == 5
    assert int(frame.iloc[1]['tick_index']) == 1

  def test_empty_schedule_returns_empty_frame(self, tmp_path):
    '''A header-only file yields an empty frame.'''
    path = os.path.join(str(tmp_path), 'empty.txt')
    with open(path, 'w', encoding='utf-8') as handle:
      handle.write('h1\nh2\n')
    frame = parse_batch_schedule(path)
    assert frame.empty


class TestParseGaLog:
  '''Tests for parse_ga_log.'''

  def test_parses_json_lines_and_skips_blank(self, tmp_path):
    '''Each non-blank line becomes one entry.'''
    path = os.path.join(str(tmp_path), 'log.jsonl')
    with open(path, 'w', encoding='utf-8') as handle:
      handle.write(json.dumps({'gen': 0, 'best_cost': 1.5}) + '\n')
      handle.write('\n')
      handle.write(json.dumps({'gen': 1, 'best_cost': 2.5}) + '\n')
    entries = parse_ga_log(path)
    assert len(entries) == 2
    assert entries[1]['gen'] == 1

  def test_empty_file_returns_empty_list(self, tmp_path):
    '''An empty log file yields no entries.'''
    path = os.path.join(str(tmp_path), 'log.jsonl')
    with open(path, 'w', encoding='utf-8') as handle:
      handle.write('')
    assert parse_ga_log(path) == []


class TestSimulateSchedule:
  '''Tests for simulate_schedule.'''

  def test_simulate_produces_histories(self, crafting_tree):
    '''Simulation returns arrays with the requested ticks.'''
    result = simulate_schedule(
      _schedule_frame(), crafting_tree, max_ticks=10,
      initial_coins=100000,
    )
    assert result['stash_history'].shape == (10, NUM_ITEMS)
    assert result['coin_history'].shape == (10,)
    assert result['slot_history'].shape[0] == 10
    assert result['active_slot_history'].shape == (10,)

  def test_simulate_with_initial_stash(self, crafting_tree):
    '''Initial stash entries are honoured at tick zero.'''
    result = simulate_schedule(
      _schedule_frame(),
      crafting_tree,
      max_ticks=3,
      initial_coins=100000,
      initial_stash={'string': 4, 'bogus': 9, 'cotton': 0},
    )
    assert result['stash_history'][0, 4] >= 0
    assert result['stash_history'].shape == (3, NUM_ITEMS)

  def test_simulate_unknown_item_is_skipped(self, crafting_tree):
    '''Unknown items do not crash or consume coins.'''
    frame = pd.DataFrame(
      [
        {
          'tick_index': 0,
          'elapsed_minutes': 0,
          'item_name': 'nosuchitem',
          'batch_size_decided': 5,
        }
      ]
    )
    result = simulate_schedule(
      frame, crafting_tree, max_ticks=2, initial_coins=1000
    )
    # Coins only grow from the generator; no spend happened.
    assert result['coin_history'][-1] >= 1000

  def test_simulate_zero_coins_blocks_crafting(self, crafting_tree):
    '''With no coins no batch is started.'''
    result = simulate_schedule(
      _schedule_frame(), crafting_tree, max_ticks=2, initial_coins=0
    )
    assert result['active_slot_history'][0] == 0


class TestExportReplayCsv:
  '''Tests for export_replay_csv.'''

  def test_export_has_header_and_rows(self, crafting_tree):
    '''CSV output has one row per tick plus a header.'''
    sim = simulate_schedule(
      _schedule_frame(), crafting_tree, max_ticks=4,
      initial_coins=10000,
    )
    text = export_replay_csv(sim, max_ticks=4)
    lines = text.strip().splitlines()
    assert lines[0].startswith('tick,elapsed_minutes')
    assert 'stash_string' in lines[0]
    assert len(lines) == 5
    assert lines[1].split(',')[0] == '0'
    assert lines[1].split(',')[1] == '0'

  def test_export_truncates_to_max_ticks(self, crafting_tree):
    '''Export respects max_ticks below the history length.'''
    sim = simulate_schedule(
      _schedule_frame(), crafting_tree, max_ticks=6,
      initial_coins=10000,
    )
    text = export_replay_csv(sim, max_ticks=2)
    assert len(text.strip().splitlines()) == 3

  def test_export_zero_ticks(self):
    '''Export with no ticks returns only the header.'''
    sim = {
      'stash_history': np.zeros((0, NUM_ITEMS)),
      'coin_history': np.zeros((0,)),
      'active_slot_history': np.zeros((0,)),
    }
    text = export_replay_csv(sim, max_ticks=0)
    assert len(text.strip().splitlines()) == 1


if __name__ == '__main__':
  pytest.main([__file__])
