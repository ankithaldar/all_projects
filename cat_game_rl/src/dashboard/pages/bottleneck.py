from __future__ import annotations

from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.core.items import ItemId, NUM_ITEMS, NUM_CRAFTABLE, CRAFTABLE_ITEM_IDS


def render_bottleneck_page(sim_data: Dict[str, Any]) -> None:
  st.header("Bottleneck Analysis")

  stash_history = sim_data["stash_history"]
  n_ticks = stash_history.shape[0]

  st.subheader("Material Availability Heatmap")
  sample_rate = max(1, n_ticks // 200)
  sampled_ticks = list(range(0, n_ticks, sample_rate))
  sampled_stash = stash_history[sampled_ticks]

  craftable_names = [ItemId(cid).name for cid in CRAFTABLE_ITEM_IDS]
  craftable_stash = sampled_stash[:, CRAFTABLE_ITEM_IDS]

  log_stash = np.log1p(craftable_stash.T.astype(float))

  fig = go.Figure(go.Heatmap(
    z=log_stash.tolist(),
    x=[str(t) for t in sampled_ticks],
    y=craftable_names,
    colorscale="RdYlGn",
    colorbar_title="log(1+count)",
  ))
  fig.update_layout(
    height=500,
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis_title="Tick",
    yaxis_title="Item",
  )
  st.plotly_chart(fig, use_container_width=True)

  st.subheader("Slot Idle Time")
  slot_history = sim_data["slot_history"]
  idle_ticks = []
  for i in range(NUM_CRAFTABLE):
    idle = np.sum(slot_history[:, i, 0] == 0)
    idle_ticks.append(int(idle))

  fig = go.Figure(go.Bar(
    x=craftable_names,
    y=idle_ticks,
    marker_color="indianred",
  ))
  fig.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis_title="Item",
    yaxis_title="Idle Ticks",
    xaxis_tickangle=-45,
  )
  st.plotly_chart(fig, use_container_width=True)

  st.subheader("Coin Pressure Over Time")
  coin_history = sim_data["coin_history"]
  low_coin_threshold = 500
  low_coin_ticks = np.sum(coin_history < low_coin_threshold)
  st.metric(
    "Ticks with coins < 500",
    f"{low_coin_ticks} / {n_ticks}",
    f"{low_coin_ticks / n_ticks:.1%}",
  )

  sampled_coins = coin_history[sampled_ticks]
  fig = go.Figure(go.Scatter(
    x=[str(t) for t in sampled_ticks],
    y=sampled_coins.tolist(),
    mode="lines",
    fill="tozeroy",
    fillcolor="rgba(255, 0, 0, 0.1)",
    line=dict(color="crimson"),
  ))
  fig.update_layout(
    height=300,
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis_title="Tick",
    yaxis_title="Coins",
  )
  st.plotly_chart(fig, use_container_width=True)
