#!/usr/bin/env python3
"""Live Streamlit dashboard for truck-carton RL.

Displays grid world map with moving trucks, real-time
metrics, reward breakdowns, and episode replay.

Run: streamlit run scripts/dashboard.py
"""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from truck_carton.config import AppConfig
from truck_carton.domain.models import (
  CellType,
  TruckState,
)
from truck_carton.env.packing_env import (
  TruckCartonPackingEnv,
)
from truck_carton.evaluation.metrics import (
  MetricsCollector,
)

# --- Page config ---
st.set_page_config(
  page_title='Truck-Carton RL Dashboard',
  page_icon='🚛',
  layout='wide',
)


def _init_state() -> None:
  """Initialize session state on first load."""
  defaults = {
    'running': False,
    'episode_history': [],
    'current_frames': [],
    'step_rewards': [],
    'step_breakdowns': [],
    'episode_count': 0,
    'auto_speed': 0.1,
  }
  for k, v in defaults.items():
    if k not in st.session_state:
      st.session_state[k] = v


_init_state()

# --- Sidebar controls ---
st.sidebar.title('Controls')

stage = st.sidebar.selectbox(
  'Curriculum Stage',
  options=[0, 1, 2, 3, 4],
  format_func=lambda s: [
    '0: Tiny (5x5)',
    '1: Small (16x16)',
    '2: Medium (40x40)',
    '3: Large (80x80)',
    '4: Full (160x160)',
  ][s],
)
seed = st.sidebar.number_input(
  'Seed', value=42, step=1
)
num_episodes = st.sidebar.number_input(
  'Episodes to run', value=5, min_value=1,
  max_value=100, step=1,
)
use_model = st.sidebar.checkbox(
  'Use trained model', value=False
)
model_path = None
if use_model:
  model_path = st.sidebar.text_input(
    'Model path', value='./output/models/best_model'
  )
speed = st.sidebar.slider(
  'Step delay (sec)', 0.0, 1.0, 0.1, 0.05
)
st.session_state['auto_speed'] = speed

run_btn = st.sidebar.button(
  'Run Simulation', type='primary'
)
stop_btn = st.sidebar.button('Stop')
if stop_btn:
  st.session_state['running'] = False

# --- Grid world rendering ---


def render_grid_plotly(
  snapshot: dict,
  title: str = 'Grid World',
) -> go.Figure:
  """Render grid world as a Plotly heatmap with
  truck markers, facility labels, and roads."""
  gw = snapshot.get('grid_world')
  if gw is None:
    return go.Figure()
  grid = gw.grid.astype(float)
  rows, cols = gw.rows, gw.cols

  color_map = {
    0: 'rgb(38,50,56)',     # TERRAIN
    1: 'rgb(100,140,200)',  # ROAD
    2: 'rgb(120,120,120)',  # DEPOT
    3: 'rgb(30,30,120)',    # WAREHOUSE
    4: 'rgb(30,100,30)',    # STORE
  }

  z = np.zeros((rows, cols, 3), dtype=np.uint8)
  for r in range(rows):
    for c in range(cols):
      cell = int(grid[r, c])
      rgb_str = color_map.get(
        cell, 'rgb(38,50,56)'
      )
      vals = rgb_str.replace(
        'rgb(', ''
      ).replace(')', '').split(',')
      z[r, c] = [int(v) for v in vals]

  fig = go.Figure()

  fig.add_trace(go.Image(z=z))

  # Facility annotations
  dp = gw.depot_position
  fig.add_trace(go.Scatter(
    x=[dp[1]], y=[dp[0]],
    mode='markers+text',
    marker=dict(
      size=14, color='white', symbol='square'
    ),
    text=['D'], textposition='middle center',
    textfont=dict(size=10, color='black'),
    name='Depot',
    showlegend=True,
  ))

  warehouses = snapshot.get('warehouses', [])
  if warehouses:
    fig.add_trace(go.Scatter(
      x=[wh.position[1] for wh in warehouses],
      y=[wh.position[0] for wh in warehouses],
      mode='markers+text',
      marker=dict(
        size=12, color='royalblue',
        symbol='triangle-up',
      ),
      text=[
        f'W{wh.warehouse_id}'
        for wh in warehouses
      ],
      textposition='top center',
      textfont=dict(size=9, color='white'),
      name='Warehouses',
      showlegend=True,
    ))

  stores = snapshot.get('stores', [])
  if stores:
    fig.add_trace(go.Scatter(
      x=[s.position[1] for s in stores],
      y=[s.position[0] for s in stores],
      mode='markers+text',
      marker=dict(
        size=12, color='limegreen',
        symbol='circle',
      ),
      text=[
        f'S{s.store_id}' for s in stores
      ],
      textposition='top center',
      textfont=dict(size=9, color='white'),
      name='Stores',
      showlegend=True,
    ))

  # Trucks with color coding by state
  trucks = snapshot.get('trucks', [])
  state_colors = {
    'ROUTING': 'orange',
    'LOADING': 'cyan',
    'AT_DEPOT': 'gray',
  }
  for i, truck in enumerate(trucks):
    state_name = truck.state.name
    fig.add_trace(go.Scatter(
      x=[truck.position[1]],
      y=[truck.position[0]],
      mode='markers+text',
      marker=dict(
        size=16,
        color=state_colors.get(
          state_name, 'yellow'
        ),
        symbol='diamond',
        line=dict(width=2, color='white'),
      ),
      text=[f'T{truck.truck_id}'],
      textposition='bottom center',
      textfont=dict(size=8, color='white'),
      name=f'Truck {truck.truck_id}'
             f' ({state_name})',
      showlegend=True,
    ))

  fig.update_layout(
    title=title,
    xaxis=dict(
      range=[-0.5, cols - 0.5],
      showgrid=False,
      zeroline=False,
    ),
    yaxis=dict(
      range=[rows - 0.5, -0.5],
      showgrid=False,
      zeroline=False,
      scaleanchor='x',
    ),
    plot_bgcolor='rgb(38,50,56)',
    paper_bgcolor='rgb(30,30,30)',
    font=dict(color='white'),
    height=max(400, rows * 6),
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(
      bgcolor='rgba(0,0,0,0.5)',
      font=dict(size=10),
    ),
  )

  return fig


def render_truck_cargo_bar(
  snapshot: dict,
) -> go.Figure:
  """Bar chart of per-truck cargo load."""
  trucks = snapshot.get('trucks', [])
  cargo = snapshot.get('truck_cargo', [])
  travel = snapshot.get('truck_travel', [])

  names = [f'T{t.truck_id}' for t in trucks]
  cargo_counts = [
    len(cargo[i]) if i < len(cargo) else 0
    for i in range(len(trucks))
  ]
  travel_vals = [
    travel[i] if i < len(travel) else 0
    for i in range(len(trucks))
  ]
  states = [t.state.name for t in trucks]

  fig = go.Figure()
  fig.add_trace(go.Bar(
    x=names, y=cargo_counts,
    name='Cargo (cartons)',
    marker_color='cyan',
    text=states,
    textposition='outside',
  ))
  fig.add_trace(go.Bar(
    x=names, y=travel_vals,
    name='Travel distance',
    marker_color='orange',
    opacity=0.7,
  ))
  fig.update_layout(
    title='Per-Truck Status',
    barmode='group',
    plot_bgcolor='rgb(30,30,30)',
    paper_bgcolor='rgb(30,30,30)',
    font=dict(color='white'),
    height=300,
  )
  return fig


def render_reward_breakdown(
  breakdown: dict[str, float],
) -> go.Figure:
  """Polar chart of reward component values."""
  names = list(breakdown.keys())
  values = [abs(v) for v in breakdown.values()]
  colors = [
    'green' if breakdown[n] >= 0 else 'red'
    for n in names
  ]

  fig = go.Figure(go.Barpolar(
    r=values,
    theta=names,
    marker_color=colors,
    opacity=0.8,
  ))
  fig.update_layout(
    title='Reward Components (absolute)',
    polar=dict(
      bgcolor='rgb(30,30,30)',
      radialaxis=dict(
        visible=True, range=[0, 1.2]
      ),
    ),
    paper_bgcolor='rgb(30,30,30)',
    font=dict(color='white'),
    height=350,
  )
  return fig


def render_reward_timeline(
  rewards: list[float],
) -> go.Figure:
  """Cumulative reward over steps."""
  cumulative = np.cumsum(rewards).tolist()
  fig = go.Figure()
  fig.add_trace(go.Scatter(
    y=rewards,
    mode='lines',
    name='Step reward',
    line=dict(color='cyan', width=1),
    opacity=0.5,
  ))
  fig.add_trace(go.Scatter(
    y=cumulative,
    mode='lines',
    name='Cumulative',
    line=dict(color='orange', width=2),
  ))
  fig.update_layout(
    title='Reward Over Steps',
    xaxis_title='Step',
    yaxis_title='Reward',
    plot_bgcolor='rgb(30,30,30)',
    paper_bgcolor='rgb(30,30,30)',
    font=dict(color='white'),
    height=300,
  )
  return fig


def render_episode_history(
  history: list[dict],
) -> go.Figure:
  """Episode metrics over time."""
  if not history:
    return go.Figure()

  eps = list(range(1, len(history) + 1))
  rewards = [h['total_reward'] for h in history]
  completions = [
    h['completion_rate'] for h in history
  ]

  fig = go.Figure()
  fig.add_trace(go.Scatter(
    x=eps, y=rewards,
    mode='lines+markers',
    name='Total Reward',
    line=dict(color='orange'),
  ))
  fig.add_trace(go.Scatter(
    x=eps, y=completions,
    mode='lines+markers',
    name='Completion Rate',
    line=dict(color='limegreen'),
    yaxis='y2',
  ))
  fig.update_layout(
    title='Episode History',
    xaxis_title='Episode',
    yaxis=dict(title='Reward', color='orange'),
    yaxis2=dict(
      title='Completion',
      overlaying='y',
      side='right',
      color='limegreen',
      range=[0, 1.1],
    ),
    plot_bgcolor='rgb(30,30,30)',
    paper_bgcolor='rgb(30,30,30)',
    font=dict(color='white'),
    height=300,
  )
  return fig


# --- Main layout ---
st.title('Truck-Carton RL Dashboard')

grid_col, metrics_col = st.columns([3, 2])

with grid_col:
  grid_placeholder = st.empty()
  cargo_placeholder = st.empty()

with metrics_col:
  status_placeholder = st.empty()
  reward_chart_placeholder = st.empty()
  breakdown_placeholder = st.empty()

timeline_placeholder = st.empty()
history_placeholder = st.empty()


def run_episode(
  env: TruckCartonPackingEnv,
  model: object | None,
  episode_num: int,
) -> dict:
  """Run one episode with live dashboard updates."""
  obs, info = env.reset(seed=seed + episode_num)
  st.session_state['step_rewards'] = []
  st.session_state['step_breakdowns'] = []

  done = False
  total_reward = 0.0
  step_count = 0
  last_breakdown = {}

  while not done and st.session_state['running']:
    masks = env.action_masks()
    if not masks.any():
      break

    if model is not None:
      action, _ = model.predict(
        obs, action_masks=masks,
        deterministic=True,
      )
    else:
      valid = np.where(masks)[0]
      action = int(np.random.choice(valid))

    obs, reward, terminated, truncated, info = (
      env.step(int(action))
    )
    total_reward += reward
    step_count += 1
    done = terminated or truncated

    st.session_state['step_rewards'].append(reward)
    last_breakdown = info.get(
      'reward_breakdown', {}
    )
    st.session_state['step_breakdowns'].append(
      last_breakdown
    )

    # Live grid update
    snap = env.get_render_snapshot()
    with grid_col:
      grid_placeholder.plotly_chart(
        render_grid_plotly(
          snap,
          title=(
            f'Episode {episode_num + 1}'
            f' | Step {step_count}'
          ),
        ),
        use_container_width=True,
        key=f'grid_{episode_num}_{step_count}',
      )
      cargo_placeholder.plotly_chart(
        render_truck_cargo_bar(snap),
        use_container_width=True,
        key=f'cargo_{episode_num}_{step_count}',
      )

    # Live metrics
    delivered = len(
      snap.get('delivered', set())
    )
    total_c = snap.get('total_cartons', 0)
    with metrics_col:
      status_placeholder.markdown(f"""
**Episode {episode_num + 1}** | Step {step_count}

| Metric | Value |
|--------|-------|
| Placed | {info.get('num_placed', 0)} |
| Delivered | {delivered}/{total_c} |
| Cumulative Reward | {total_reward:.2f} |
| Step Reward | {reward:.4f} |
""")
      if last_breakdown:
        breakdown_placeholder.plotly_chart(
          render_reward_breakdown(last_breakdown),
          use_container_width=True,
          key=(
            f'bdown_{episode_num}_{step_count}'
          ),
        )

    reward_chart_placeholder.plotly_chart(
      render_reward_timeline(
        st.session_state['step_rewards']
      ),
      use_container_width=True,
      key=f'rtl_{episode_num}_{step_count}',
    )

    time.sleep(st.session_state['auto_speed'])

  # Episode summary
  collector = MetricsCollector()
  ep_metrics = collector.compute(
    episode_data=env.episode_data,
    spaces=env.spaces,
    placed_cartons=env.placed_cartons,
    current_weights=env.current_weights,
    total_reward=total_reward,
    curriculum_stage=stage,
  )
  return {
    'total_reward': total_reward,
    'completion_rate': ep_metrics.completion_rate,
    'vol_util': (
      ep_metrics.fleet_volumetric_utilization
    ),
    'num_placed': ep_metrics.num_placed,
    'num_total': ep_metrics.num_total,
    'steps': step_count,
  }


# --- Run simulation ---
if run_btn:
  st.session_state['running'] = True
  config = AppConfig()
  env = TruckCartonPackingEnv(
    config=config, curriculum_stage=stage
  )

  model = None
  if use_model and model_path:
    try:
      from sb3_contrib import MaskablePPO
      model = MaskablePPO.load(model_path)
      st.sidebar.success(
        f'Model loaded: {model_path}'
      )
    except Exception as exc:
      st.sidebar.error(
        f'Failed to load model: {exc}'
      )

  for ep in range(num_episodes):
    if not st.session_state['running']:
      break

    ep_result = run_episode(env, model, ep)
    st.session_state['episode_history'].append(
      ep_result
    )
    st.session_state['episode_count'] += 1

    history_placeholder.plotly_chart(
      render_episode_history(
        st.session_state['episode_history']
      ),
      use_container_width=True,
      key=f'hist_{ep}',
    )

  st.session_state['running'] = False
  st.sidebar.success(
    f'Completed {num_episodes} episodes'
  )

# Show episode history if exists
if st.session_state['episode_history']:
  st.subheader('Episode Summary')
  history = st.session_state['episode_history']
  cols = st.columns(4)
  cols[0].metric(
    'Avg Reward',
    f'{np.mean([h["total_reward"] for h in history]):.2f}',
  )
  cols[1].metric(
    'Avg Completion',
    f'{np.mean([h["completion_rate"] for h in history]):.1%}',
  )
  cols[2].metric(
    'Avg Vol Util',
    f'{np.mean([h["vol_util"] for h in history]):.1%}',
  )
  cols[3].metric(
    'Total Episodes',
    len(history),
  )
