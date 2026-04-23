from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st


def render_pareto_page(ga_log: Optional[List[Dict[str, Any]]]) -> None:
  st.header("Pareto Front Analysis")

  if not ga_log:
    st.warning("No GA population log loaded.")
    return

  last_gen = ga_log[-1]
  front = last_gen.get("front", [])

  if not front:
    st.warning("No Pareto front data in the last generation.")
    return

  costs = [p["cost"] for p in front]
  times = [p["time"] for p in front]
  wastes = [p["waste"] for p in front]

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Cost vs Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
      x=costs,
      y=times,
      mode="markers",
      marker=dict(size=10, color="red"),
      text=[f"waste={w:.0f}" for w in wastes],
      hovertemplate="Cost: %{x:.0f}<br>Time: %{y:.0f}<br>%{text}",
    ))
    fig.update_layout(
      height=400,
      margin=dict(l=0, r=0, t=30, b=0),
      xaxis_title="Total Cost",
      yaxis_title="Completion Tick",
    )
    st.plotly_chart(fig, use_container_width=True)

  with col2:
    st.subheader("Cost vs Waste")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
      x=costs,
      y=wastes,
      mode="markers",
      marker=dict(size=10, color="purple"),
      text=[f"time={t:.0f}" for t in times],
      hovertemplate="Cost: %{x:.0f}<br>Waste: %{y:.0f}<br>%{text}",
    ))
    fig.update_layout(
      height=400,
      margin=dict(l=0, r=0, t=30, b=0),
      xaxis_title="Total Cost",
      yaxis_title="Waste",
    )
    st.plotly_chart(fig, use_container_width=True)

  st.subheader("3D Pareto Front")
  fig = go.Figure(go.Scatter3d(
    x=costs,
    y=times,
    z=wastes,
    mode="markers",
    marker=dict(
      size=6,
      color=times,
      colorscale="Viridis",
      colorbar_title="Time",
    ),
    hovertemplate="Cost: %{x:.0f}<br>Time: %{y:.0f}<br>Waste: %{z:.0f}",
  ))
  fig.update_layout(
    height=600,
    scene=dict(
      xaxis_title="Total Cost",
      yaxis_title="Completion Tick",
      zaxis_title="Waste",
    ),
  )
  st.plotly_chart(fig, use_container_width=True)

  st.subheader("Pareto Front History")
  gens = [e["gen"] for e in ga_log]
  front_sizes = [e["pareto_front_size"] for e in ga_log]

  fig = go.Figure(go.Scatter(
    x=gens,
    y=front_sizes,
    mode="lines",
    fill="tozeroy",
    fillcolor="rgba(128, 0, 128, 0.2)",
    line=dict(color="purple"),
  ))
  fig.update_layout(
    height=300,
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis_title="Generation",
    yaxis_title="Pareto Front Size",
  )
  st.plotly_chart(fig, use_container_width=True)
