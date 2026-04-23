from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.core.items import NUM_CRAFTABLE


def render_comparison_page(
  rl_sim: Optional[Dict[str, Any]],
  ga_sim: Optional[Dict[str, Any]],
  ga_log: Optional[List[Dict[str, Any]]],
) -> None:
  st.header("RL vs GA Comparison")

  if rl_sim is None and ga_sim is None:
    st.warning("Load both RL and GA schedules to compare.")
    return

  st.subheader("Metrics Comparison")
  col1, col2 = st.columns(2)

  with col1:
    st.markdown("### RL Agent")
    if rl_sim is not None:
      _display_metrics(rl_sim)
    else:
      st.info("No RL data loaded.")

  with col2:
    st.markdown("### GA Best")
    if ga_sim is not None:
      _display_metrics(ga_sim)
    else:
      st.info("No GA data loaded.")

  if rl_sim is not None and ga_sim is not None:
    st.subheader("Slot Utilization Over Time")
    fig = go.Figure()

    rl_util = rl_sim["active_slot_history"] / NUM_CRAFTABLE * 100
    ga_util = ga_sim["active_slot_history"] / NUM_CRAFTABLE * 100

    fig.add_trace(go.Scatter(
      y=rl_util.tolist(),
      mode="lines",
      name="RL",
      line=dict(color="blue"),
    ))
    fig.add_trace(go.Scatter(
      y=ga_util.tolist(),
      mode="lines",
      name="GA",
      line=dict(color="red"),
    ))
    fig.update_layout(
      height=350,
      margin=dict(l=0, r=0, t=30, b=0),
      xaxis_title="Tick",
      yaxis_title="Utilization %",
      yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Coin Balance Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
      y=rl_sim["coin_history"].tolist(),
      mode="lines",
      name="RL",
      line=dict(color="blue"),
    ))
    fig.add_trace(go.Scatter(
      y=ga_sim["coin_history"].tolist(),
      mode="lines",
      name="GA",
      line=dict(color="red"),
    ))
    fig.update_layout(
      height=300,
      margin=dict(l=0, r=0, t=30, b=0),
      xaxis_title="Tick",
      yaxis_title="Coins",
    )
    st.plotly_chart(fig, use_container_width=True)

  if ga_log:
    st.subheader("GA Training Progress")
    gens = [e["gen"] for e in ga_log]
    best_costs = [e["best_cost"] for e in ga_log]
    best_times = [e["best_time"] for e in ga_log]
    best_wastes = [e["best_waste"] for e in ga_log]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
      x=gens, y=best_costs, name="Best Cost", mode="lines"
    ))
    fig.add_trace(go.Scatter(
      x=gens, y=best_times, name="Best Time", mode="lines"
    ))
    fig.add_trace(go.Scatter(
      x=gens, y=best_wastes, name="Best Waste", mode="lines"
    ))
    fig.update_layout(
      height=400,
      margin=dict(l=0, r=0, t=30, b=0),
      xaxis_title="Generation",
      yaxis_title="Fitness Value",
    )
    st.plotly_chart(fig, use_container_width=True)


def _display_metrics(sim_data: Dict[str, Any]) -> None:
  coin_hist = sim_data["coin_history"]
  active_hist = sim_data["active_slot_history"]
  n_ticks = len(coin_hist)

  avg_util = np.mean(active_hist) / NUM_CRAFTABLE * 100
  max_coins = np.max(coin_hist)
  min_coins = np.min(coin_hist)

  st.metric("Average Slot Utilization", f"{avg_util:.1f}%")
  st.metric("Peak Coins", f"{max_coins:,}")
  st.metric("Min Coins", f"{min_coins:,}")
  st.metric("Total Ticks Used", f"{n_ticks}")
