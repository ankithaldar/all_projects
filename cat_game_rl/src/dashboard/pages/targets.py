from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import numpy as np
import streamlit as st
import yaml

from src.core.items import CraftingTree, ItemId, ITEM_NAME_TO_ID
from src.dashboard.utils import parse_batch_schedule, simulate_schedule


def render_targets_page(
    schedule_df: Any,
    crafting_tree: CraftingTree,
    targets_path: str,
) -> None:
    st.header("Target Editor & Re-Simulation")

    with open(targets_path, "r") as f:
        targets_data = yaml.safe_load(f)["targets"]

    st.subheader("Edit Targets")
    new_targets = {}
    cols = st.columns(3)
    for i, (name, count) in enumerate(targets_data.items()):
        with cols[i % 3]:
            new_val = st.number_input(
                f"{name}",
                min_value=0,
                value=count,
                step=1,
                key=f"target_{name}",
            )
            new_targets[name] = new_val

    if st.button("Re-Simulate with New Targets"):
        tmp_targets = {"targets": new_targets}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(tmp_targets, f)
            tmp_path = f.name

        try:
            sim_result = simulate_schedule(
                schedule_df, crafting_tree, max_ticks=2016
            )

            st.subheader("Re-Simulation Results")
            final_stash = sim_result["stash_history"][-1]

            st.write("**Target vs Achieved:**")
            for name, target in new_targets.items():
                item_id = ITEM_NAME_TO_ID.get(name.lower())
                if item_id is not None:
                    achieved = int(final_stash[int(item_id)])
                    status = "met" if achieved >= target else "NOT MET"
                    color = "green" if achieved >= target else "red"
                    st.markdown(
                        f"- **{name}**: {achieved}/{target} "
                        f":{color}[{status}]"
                    )
        finally:
            os.unlink(tmp_path)

    st.subheader("Current Targets")
    for name, count in targets_data.items():
        st.write(f"- **{name}**: {count}")
