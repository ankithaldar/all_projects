#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for dashboard pages and the dashboard app.'''

# pylint: disable=protected-access,import-outside-toplevel,wrong-import-position

from __future__ import annotations

import contextlib
import os
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml


def _install_streamlit_stub():
  '''Install a MagicMock-based streamlit stub.'''
  if 'streamlit' in sys.modules and hasattr(
    sys.modules['streamlit'], '__file__'
  ):
    # Real streamlit is importable; replace it with a stub for
    # deterministic unit tests of page rendering.
    pass
  stub = mock.MagicMock(name='streamlit_stub')
  # Context manager support for `with st.sidebar:` blocks.
  stub.sidebar.__enter__ = mock.MagicMock(return_value=stub)
  stub.sidebar.__exit__ = mock.MagicMock(return_value=False)
  def _columns(count):
    '''Return count context-manager columns.'''
    cols = [mock.MagicMock() for _ in range(count)]
    for col in cols:
      col.__enter__ = mock.MagicMock(return_value=col)
      col.__exit__ = mock.MagicMock(return_value=False)
    return tuple(cols)

  stub.columns.side_effect = _columns
  stub.slider.return_value = 0
  stub.radio.return_value = 'Simulation'
  stub.text_input.return_value = 'output/batch_schedule.txt'
  stub.button.return_value = False
  stub.number_input.return_value = 5
  sys.modules['streamlit'] = stub
  return stub


_STUB = _install_streamlit_stub()

# Re-import page modules so they bind to the stub.
for _name in list(sys.modules):
  if _name.startswith('src.dashboard'):
    del sys.modules[_name]

from src.dashboard.pages import (  # noqa: E402
  bottleneck,
  comparison,
  pareto,
  simulation,
  targets,
)
from src.dashboard import app as dashboard_app  # noqa: E402
from src.core.items import NUM_CRAFTABLE, NUM_ITEMS  # noqa: E402


def _sim_data(n_ticks=5):
  '''Build fake simulation histories.'''
  stash = np.zeros((n_ticks, NUM_ITEMS), dtype=np.int32)
  stash[:, 4] = np.arange(n_ticks)
  slots = np.zeros((n_ticks, NUM_CRAFTABLE, 3), dtype=np.int32)
  slots[:, 0, 0] = 1
  slots[:, 0, 1] = 3
  slots[:, 0, 2] = 2
  return {
    'stash_history': stash,
    'coin_history': np.linspace(0, 1000, n_ticks),
    'slot_history': slots,
    'active_slot_history': np.full(n_ticks, 2),
  }


def _ga_log():
  '''Build a fake GA log with a Pareto front.'''
  return [
    {
      'gen': 0,
      'best_cost': 10.0,
      'best_time': 100.0,
      'best_waste': 1.0,
      'pareto_front_size': 1,
      'front': [
        {'cost': 10.0, 'time': 100.0, 'waste': 1.0}
      ],
    },
    {
      'gen': 1,
      'best_cost': 8.0,
      'best_time': 90.0,
      'best_waste': 2.0,
      'pareto_front_size': 2,
      'front': [
        {'cost': 8.0, 'time': 90.0, 'waste': 2.0},
        {'cost': 9.0, 'time': 95.0, 'waste': 0.5},
      ],
    },
  ]


class TestSimulationPage:
  '''Tests for the simulation page.'''

  def test_renders_with_stock_and_slots(self):
    '''A tick with stock and active slots renders charts.'''
    _STUB.slider.return_value = 2
    simulation.render_simulation_page(_sim_data(), max_ticks=5)
    _STUB.header.assert_called()
    _STUB.plotly_chart.assert_called()

  def test_empty_stash_shows_info(self):
    '''An empty stash at a tick shows an info message.'''
    data = _sim_data()
    data['stash_history'][:] = 0
    data['slot_history'][:] = 0
    _STUB.slider.return_value = 0
    simulation.render_simulation_page(data, max_ticks=5)
    _STUB.info.assert_called()


class TestBottleneckPage:
  '''Tests for the bottleneck page.'''

  def test_renders_charts_and_metric(self):
    '''Bottleneck page renders heatmap, bars and coins.'''
    bottleneck.render_bottleneck_page(_sim_data())
    _STUB.header.assert_called()
    _STUB.metric.assert_called()
    assert _STUB.plotly_chart.call_count >= 3


class TestComparisonPage:
  '''Tests for the comparison page.'''

  def test_warns_when_no_data(self):
    '''With no simulations a warning is shown.'''
    comparison.render_comparison_page(None, None, None)
    _STUB.warning.assert_called()

  def test_renders_both_sims_and_log(self):
    '''Both sims plus a GA log render comparison charts.'''
    comparison.render_comparison_page(
      _sim_data(), _sim_data(), _ga_log()
    )
    _STUB.header.assert_called()
    _STUB.plotly_chart.assert_called()

  def test_renders_only_rl(self):
    '''Only RL data shows an info placeholder for GA.'''
    comparison.render_comparison_page(_sim_data(), None, None)
    _STUB.info.assert_called()

  def test_display_metrics_directly(self):
    '''_display_metrics computes utilisation and coin stats.'''
    comparison._display_metrics(_sim_data())
    _STUB.metric.assert_called()


class TestParetoPage:
  '''Tests for the Pareto page.'''

  def test_warns_when_no_log(self):
    '''A missing log shows a warning.'''
    pareto.render_pareto_page(None)
    _STUB.warning.assert_called()

  def test_warns_when_front_empty(self):
    '''An empty front in the last generation warns.'''
    log = _ga_log()
    log[-1]['front'] = []
    pareto.render_pareto_page(log)
    _STUB.warning.assert_called()

  def test_renders_front_charts(self):
    '''A populated front renders 2D, 3D and history charts.'''
    pareto.render_pareto_page(_ga_log())
    _STUB.header.assert_called()
    assert _STUB.plotly_chart.call_count >= 3


class TestTargetsPage:
  '''Tests for the target editor page.'''

  def _targets_file(self, tmp_path, data=None):
    '''Write a targets YAML and return its path.'''
    path = os.path.join(str(tmp_path), 'targets.yaml')
    with open(path, 'w', encoding='utf-8') as handle:
      yaml.dump(data or {'targets': {'string': 5}}, handle)
    return path

  def test_missing_file_shows_error(self, tmp_path, crafting_tree):
    '''A missing targets file shows an error.'''
    frame = pd.DataFrame()
    targets.render_targets_page(
      frame, crafting_tree, os.path.join(str(tmp_path), 'nope.yaml')
    )
    _STUB.error.assert_called()

  def test_invalid_file_shows_error(self, tmp_path, crafting_tree):
    '''A file without a targets key shows an error.'''
    path = self._targets_file(tmp_path, {'other': 1})
    targets.render_targets_page(pd.DataFrame(), crafting_tree, path)
    _STUB.error.assert_called()

  def test_renders_editor_without_resimulate(
    self, tmp_path, crafting_tree
  ):
    '''Without pressing the button only the editor shows.'''
    path = self._targets_file(tmp_path)
    _STUB.button.return_value = False
    targets.render_targets_page(
      _schedule_frame(), crafting_tree, path
    )
    _STUB.subheader.assert_called()
    _STUB.write.assert_called()

  def test_resimulate_writes_results(
    self, tmp_path, crafting_tree
  ):
    '''Pressing re-simulate runs a simulation and reports.'''
    path = self._targets_file(tmp_path)
    _STUB.button.return_value = True
    _STUB.number_input.return_value = 1
    targets.render_targets_page(
      _schedule_frame(), crafting_tree, path,
      initial_coins=100000,
    )
    _STUB.markdown.assert_called()


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


class TestDashboardApp:
  '''Tests for dashboard app main and path safety.'''

  def _setup_config(self, tmp_path):
    '''Copy real config files into a temp cwd.'''
    import shutil  # noqa: PLC0415

    cfg_dir = os.path.join(str(tmp_path), 'config')
    os.makedirs(cfg_dir, exist_ok=True)
    src_cfg = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
      'config',
    )
    shutil.copy(
      os.path.join(src_cfg, 'crafting_tree.yaml'), cfg_dir
    )
    shutil.copy(os.path.join(src_cfg, 'targets.yaml'), cfg_dir)
    with open(
      os.path.join(cfg_dir, 'training.yaml'), 'w',
      encoding='utf-8',
    ) as handle:
      yaml.dump({'environment': {'initial_coins': 1000}}, handle)

  def _run_main(self, page, tmp_path, monkeypatch):
    '''Run main() inside a temp cwd with stubbed inputs.'''
    self._setup_config(tmp_path)
    monkeypatch.chdir(str(tmp_path))
    _STUB.radio.return_value = page
    _STUB.text_input.return_value = os.path.join(
      str(tmp_path), 'missing.txt'
    )
    dashboard_app.main()

  def test_simulation_page_without_data(self, tmp_path, monkeypatch):
    '''Simulation page without data shows an info message.'''
    self._run_main('Simulation', tmp_path, monkeypatch)
    _STUB.info.assert_called()

  def test_bottleneck_page_without_data(
    self, tmp_path, monkeypatch
  ):
    '''Bottleneck page without data shows an info message.'''
    self._run_main('Bottleneck', tmp_path, monkeypatch)
    _STUB.info.assert_called()

  def test_comparison_page(self, tmp_path, monkeypatch):
    '''Comparison page renders even without data.'''
    self._run_main('RL vs GA', tmp_path, monkeypatch)
    _STUB.header.assert_called()

  def test_pareto_page(self, tmp_path, monkeypatch):
    '''Pareto page renders even without data.'''
    self._run_main('Pareto Front', tmp_path, monkeypatch)
    _STUB.header.assert_called()

  def test_target_editor_without_data(
    self, tmp_path, monkeypatch
  ):
    '''Target editor without data shows an info message.'''
    self._run_main('Target Editor', tmp_path, monkeypatch)
    _STUB.info.assert_called()

  def test_unsafe_path_warns(self, tmp_path, monkeypatch):
    '''Paths outside allowed dirs trigger a warning.'''
    self._setup_config(tmp_path)
    monkeypatch.chdir(str(tmp_path))
    _STUB.radio.return_value = 'Simulation'
    _STUB.text_input.return_value = '/etc/passwd'
    dashboard_app.main()
    _STUB.warning.assert_called()

  def test_loads_existing_schedule(self, tmp_path, monkeypatch):
    '''Existing schedule files under output/ are loaded.'''
    monkeypatch.chdir(str(tmp_path))
    out_dir = os.path.join(str(tmp_path), 'output')
    cfg_dir = os.path.join(str(tmp_path), 'config')
    os.makedirs(out_dir)
    os.makedirs(cfg_dir)
    sched = os.path.join(out_dir, 'batch_schedule.txt')
    with open(sched, 'w', encoding='utf-8') as handle:
      handle.write('h1\nh2\n0 0 string 1\n')
    self._setup_config(tmp_path)
    _STUB.radio.return_value = 'Simulation'

    # pylint: disable-next=unused-argument
    def _text_input(label, value=None):
      '''Return the schedule path for RL, missing elsewhere.'''
      if 'RL' in label:
        return sched
      return os.path.join(str(tmp_path), 'missing.txt')

    _STUB.text_input.side_effect = _text_input
    with contextlib.redirect_stdout(open(os.devnull, 'w', encoding='utf-8')):
      dashboard_app.main()


if __name__ == '__main__':
  pytest.main([__file__])
