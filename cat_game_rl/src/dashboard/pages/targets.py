#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Target editor and re-simulation page for the dashboard.'''

from __future__ import annotations

import os
import tempfile
from typing import Any

import streamlit as st
import yaml

from src.core.items import CraftingTree, ITEM_NAME_TO_ID
from src.dashboard.utils import simulate_schedule


def render_targets_page(
  schedule_df: Any,
  crafting_tree: CraftingTree,
  targets_path: str,
  initial_coins: int = 0,
  initial_stash: dict | None = None,
) -> None:
  '''Render the target editor and re-simulation page.

  Args:
    schedule_df: DataFrame of scheduled batch actions.
    crafting_tree: Crafting tree used for recipes and slots.
    targets_path: Path of the targets YAML file.
    initial_coins: Coins available at tick zero.
    initial_stash: Optional mapping of item name to quantity.
  '''
  st.header('Target Editor & Re-Simulation')

  if not os.path.exists(targets_path):
    st.error(f'Targets file not found: {targets_path}')
    return
  with open(targets_path, 'r', encoding='utf-8') as f:
    raw = yaml.safe_load(f)
  if not raw or 'targets' not in raw:
    st.error('Invalid targets file: missing \'targets\' key.')
    return
  targets_data = raw['targets']

  st.subheader('Edit Targets')
  new_targets = {}
  cols = st.columns(3)
  for i, (name, count) in enumerate(targets_data.items()):
    with cols[i % 3]:
      new_val = st.number_input(
        f'{name}',
        min_value=0,
        value=count,
        step=1,
        key=f'target_{name}',
      )
      new_targets[name] = new_val

  if st.button('Re-Simulate with New Targets'):
    tmp_targets = {'targets': new_targets}
    with tempfile.NamedTemporaryFile(
      mode='w', suffix='.yaml', delete=False, encoding='utf-8'
    ) as f:
      yaml.dump(tmp_targets, f)
      tmp_path = f.name

    try:
      sim_result = simulate_schedule(
        schedule_df, crafting_tree, max_ticks=8064,
        initial_coins=initial_coins, initial_stash=initial_stash,
      )

      st.subheader('Re-Simulation Results')
      final_stash = sim_result['stash_history'][-1]

      st.write('**Target vs Achieved:**')
      for name, target in new_targets.items():
        item_id = ITEM_NAME_TO_ID.get(name.lower())
        if item_id is not None:
          achieved = int(final_stash[int(item_id)])
          status = 'met' if achieved >= target else 'NOT MET'
          color = 'green' if achieved >= target else 'red'
          st.markdown(
            f'- **{name}**: {achieved}/{target} '
            f':{color}[{status}]'
          )
    finally:
      os.unlink(tmp_path)

  st.subheader('Current Targets')
  for name, count in targets_data.items():
    st.write(f'- **{name}**: {count}')
