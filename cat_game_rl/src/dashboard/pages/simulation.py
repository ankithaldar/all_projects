from __future__ import annotations

from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.core.items import ItemId, NUM_CRAFTABLE, CRAFTABLE_ITEM_IDS


def render_simulation_page(sim_data: Dict[str, Any], max_ticks: int) -> None:
    st.header("Simulation Playback")

    actual_ticks = len(sim_data["coin_history"])
    tick = st.slider(
        "Time (tick)",
        min_value=0,
        max_value=actual_ticks - 1,
        value=0,
        format="Tick %d",
    )

    day = tick // 288
    hour = (tick % 288) // 12
    minute = (tick % 12) * 5
    st.caption(f"Day {day + 1}, {hour:02d}:{minute:02d} | Elapsed: {tick * 5} min")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stash Inventory")
        stash_at_tick = sim_data["stash_history"][tick]
        items_with_stock = [
            (ItemId(i).name, int(stash_at_tick[i]))
            for i in range(len(stash_at_tick))
            if stash_at_tick[i] > 0
        ]
        if items_with_stock:
            names, counts = zip(*items_with_stock)
            fig = go.Figure(go.Bar(
                x=list(counts),
                y=list(names),
                orientation="h",
                marker_color="steelblue",
            ))
            fig.update_layout(
                height=max(300, len(names) * 25),
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Stash is empty at this tick.")

    with col2:
        st.subheader("Coin Balance")
        coin_hist = sim_data["coin_history"][: tick + 1]
        fig = go.Figure(go.Scatter(
            x=list(range(len(coin_hist))),
            y=coin_hist.tolist(),
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(255, 215, 0, 0.3)",
            line=dict(color="gold"),
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Tick",
            yaxis_title="Coins",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Slot Utilization")
    active_hist = sim_data["active_slot_history"][: tick + 1]
    utilization = (active_hist / NUM_CRAFTABLE * 100).tolist()
    fig = go.Figure(go.Scatter(
        x=list(range(len(utilization))),
        y=utilization,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(46, 139, 87, 0.3)",
        line=dict(color="seagreen"),
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Tick",
        yaxis_title="Utilization %",
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Manufacturing Slots (Current)")
    slot_at_tick = sim_data["slot_history"][tick]
    active_slots = []
    for i in range(NUM_CRAFTABLE):
        if slot_at_tick[i, 0] > 0:
            item_name = ItemId(CRAFTABLE_ITEM_IDS[i]).name
            remaining = int(slot_at_tick[i, 1])
            batch = int(slot_at_tick[i, 2])
            active_slots.append(
                f"**{item_name}**: batch={batch}, remaining={remaining} ticks"
            )
    if active_slots:
        for s in active_slots:
            st.markdown(f"- {s}")
    else:
        st.info("No active manufacturing slots at this tick.")
