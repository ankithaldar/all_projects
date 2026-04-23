from __future__ import annotations

import os
import sys

_root = os.path.dirname(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _root)

import streamlit as st

from src.core.items import CraftingTree
from src.dashboard.utils import (
  parse_batch_schedule,
  parse_ga_log,
  simulate_schedule,
  export_replay_csv,
)
from src.dashboard.pages.simulation import render_simulation_page
from src.dashboard.pages.bottleneck import render_bottleneck_page
from src.dashboard.pages.comparison import render_comparison_page
from src.dashboard.pages.pareto import render_pareto_page
from src.dashboard.pages.targets import render_targets_page


def main() -> None:
  st.set_page_config(
    page_title="Crafting RL Dashboard",
    page_icon="hammer",
    layout="wide",
  )
  st.title("Crafting Game RL Dashboard")

  tree_path = "config/crafting_tree.yaml"
  targets_path = "config/targets.yaml"
  crafting_tree = CraftingTree.from_yaml(tree_path)

  with st.sidebar:
    st.header("Data Sources")

    rl_schedule_path = st.text_input(
      "RL batch_schedule.txt path",
      value="output/batch_schedule.txt",
    )

    ga_schedule_path = st.text_input(
      "GA batch_schedule.txt path",
      value="output/ga_results/ga_batch_schedule.txt",
    )

    ga_log_path = st.text_input(
      "GA population log path",
      value="output/ga_results/ga_population_log.jsonl",
    )

    page = st.radio(
      "Page",
      ["Simulation", "Bottleneck", "RL vs GA", "Pareto Front", "Target Editor"],
    )

  rl_schedule_df = None
  rl_sim = None
  if os.path.exists(rl_schedule_path):
    rl_schedule_df = parse_batch_schedule(rl_schedule_path)
    rl_sim = simulate_schedule(rl_schedule_df, crafting_tree)
  else:
    if page in ["Simulation", "Bottleneck"]:
      st.warning(f"RL schedule not found at {rl_schedule_path}")

  ga_schedule_df = None
  ga_sim = None
  ga_log = None

  if os.path.exists(ga_schedule_path):
    ga_schedule_df = parse_batch_schedule(ga_schedule_path)
    ga_sim = simulate_schedule(ga_schedule_df, crafting_tree)

  if os.path.exists(ga_log_path):
    ga_log = parse_ga_log(ga_log_path)

  active_sim = rl_sim or ga_sim

  if page == "Simulation":
    if active_sim is not None:
      render_simulation_page(active_sim, max_ticks=2016)
    else:
      st.info("Load a batch schedule to view simulation.")

  elif page == "Bottleneck":
    if active_sim is not None:
      render_bottleneck_page(active_sim)
    else:
      st.info("Load a batch schedule to view bottleneck analysis.")

  elif page == "RL vs GA":
    render_comparison_page(rl_sim, ga_sim, ga_log)

  elif page == "Pareto Front":
    render_pareto_page(ga_log)

  elif page == "Target Editor":
    active_df = rl_schedule_df if rl_schedule_df is not None else ga_schedule_df
    if active_df is not None:
      render_targets_page(active_df, crafting_tree, targets_path)
    else:
      st.info("Load a batch schedule to use target editor.")

  with st.sidebar:
    st.markdown("---")
    if active_sim is not None:
      csv_data = export_replay_csv(active_sim)
      st.download_button(
        label="Download Replay CSV",
        data=csv_data,
        file_name="crafting_replay.csv",
        mime="text/csv",
      )


if __name__ == "__main__":
  main()
