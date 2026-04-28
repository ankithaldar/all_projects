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

  import yaml as _yaml
  with open("config/training.yaml", "r") as _f:
    _train_cfg = _yaml.safe_load(_f)
  _env_cfg = _train_cfg.get("environment", {})
  _initial_coins = _env_cfg.get("initial_coins", 0)
  _initial_stash = _env_cfg.get("initial_stash", {})

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

  allowed_dirs = [
    os.path.abspath("output"),
    os.path.abspath("config"),
  ]

  def _is_safe_path(path: str) -> bool:
    abs_path = os.path.abspath(path)
    return any(
      abs_path.startswith(d + os.sep) or abs_path == d
      for d in allowed_dirs
    )

  rl_schedule_df = None
  rl_sim = None
  if _is_safe_path(rl_schedule_path) and os.path.exists(rl_schedule_path):
    rl_schedule_df = parse_batch_schedule(rl_schedule_path)
    rl_sim = simulate_schedule(
      rl_schedule_df, crafting_tree,
      initial_coins=_initial_coins, initial_stash=_initial_stash,
    )
  else:
    if page in ["Simulation", "Bottleneck"]:
      if not _is_safe_path(rl_schedule_path):
        st.warning("Path must be within output/ or config/.")
      else:
        st.warning(f"RL schedule not found at {rl_schedule_path}")

  ga_schedule_df = None
  ga_sim = None
  ga_log = None

  if _is_safe_path(ga_schedule_path) and os.path.exists(ga_schedule_path):
    ga_schedule_df = parse_batch_schedule(ga_schedule_path)
    ga_sim = simulate_schedule(
      ga_schedule_df, crafting_tree,
      initial_coins=_initial_coins, initial_stash=_initial_stash,
    )

  if _is_safe_path(ga_log_path) and os.path.exists(ga_log_path):
    ga_log = parse_ga_log(ga_log_path)

  active_sim = rl_sim or ga_sim

  if page == "Simulation":
    if active_sim is not None:
      render_simulation_page(active_sim, max_ticks=8064)
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
      render_targets_page(
        active_df, crafting_tree, targets_path,
        initial_coins=_initial_coins, initial_stash=_initial_stash,
      )
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
