#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Gymnasium environment for truck carton packing.

Trusted caller: this environment is not hardened
against mutation of objects returned by properties
(e.g. spaces, placed_cartons, episode_data). Callers
are trusted not to mutate internal state directly;
all safety checks here assume that contract (E-05).
Invalid actions raise ValueError (E-01). Masked or
out-of-range actions are a defined no-op returning
the current observation with 0.0 reward and no state
change, including no step_count increment and no
truck switch (E-02). Calling step() after the
episode terminated or truncated raises RuntimeError.
'''

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from truck_carton.config import AppConfig
from truck_carton.domain.data_generator import (
  DataGenerator,
)
from truck_carton.domain.models import (
  Carton,
  CellType,
  EpisodeData,
  ObservationContext,
  PlacementInfo,
  TruckState,
)
from truck_carton.env.action import (
  ActionManager,
  RoutingCandidate,
)
from truck_carton.env.masking import (
  ActionMaskProvider,
)
from truck_carton.env.observation import (
  ObservationBuilder,
  build_observation_space,
  validate_environment_config,
)
from truck_carton.packing.placement import (
  PlacementCandidate,
  PlacementValidator,
)
from truck_carton.packing.space3d import Space3D
from truck_carton.reward.calculator import (
  EnvironmentState,
  RewardCalculator,
)

if TYPE_CHECKING:
  from truck_carton.domain.models import Truck


class TruckCartonPackingEnv(gym.Env):
  '''Multi truck routing and 3D packing environment.

  One truck acts per step, either routing to a
  facility or packing the current carton. Action
  masks enforce mutual exclusivity of the two
  action types.
  '''

  metadata = {'render_modes': []}

  def __init__(
    self,
    config: AppConfig | None = None,
    curriculum_stage: int = 0,
    max_steps: int = 500,
  ) -> None:
    '''Initialize the environment.

    Args:
      config: Application configuration.
      curriculum_stage: Initial curriculum stage.
       Clamped to the available stage range.
      max_steps: Maximum steps before truncation.
       Must be >= 1.

    Raises:
      ValueError: If max_steps < 1, curriculum_stage
       is not an integer, or EnvironmentConfig caps
       are not sane.
    '''
    super().__init__()
    self._config = config or AppConfig()
    validate_environment_config(self._config.env)
    self._curriculum_stage = self._clamp_stage(curriculum_stage)
    self._max_steps = self._validate_max_steps(max_steps)
    self._episode_done = False
    self._rng = np.random.default_rng(42)
    self._episode_data: EpisodeData | None = None
    self._spaces: list[Space3D] = []
    self._validators: list[PlacementValidator] = []
    self._placed: dict[int, PlacementInfo] = {}
    self._weights: list[float] = []
    self._delivered: set[int] = set()
    self._truck_cargo: list[set[int]] = []
    self._warehouse_cartons: dict[int, list[int]] = {}
    self._carton_lookup: dict[int, Carton] = {}
    self._truck_travel: list[float] = []
    self._truck_warehouse: dict[int, int] = {}
    self._active_idx = 0
    self._step_count = 0
    self._cumulative = 0.0
    self._current_carton: Carton | None = None
    self._reward = RewardCalculator(self._config.rewards)
    self._action_manager = ActionManager(self._config.env)
    self._mask_provider = ActionMaskProvider(self._action_manager)
    self._obs_builder = ObservationBuilder(self._config.env)
    self.observation_space = build_observation_space(self._config.env)
    total = self._action_manager.total_action_size
    self.action_space = spaces.Discrete(total)
    self._episode_id = 0

  def _clamp_stage(self, stage: int) -> int:
    '''Clamp stage to available range.

    Args:
      stage: Requested curriculum stage.

    Returns:
      Clamped stage index.

    Raises:
      ValueError: If stage is not an integer.
    '''
    if isinstance(stage, bool) or not isinstance(
      stage, (int, np.integer),
    ):
      raise ValueError('curriculum_stage must be an int')
    value = int(stage)
    stages = self._config.curriculum.stages
    if not stages:
      return 0
    return int(min(max(value, 0), len(stages) - 1))

  def _validate_max_steps(self, max_steps: int) -> int:
    '''Validate max_steps.

    Args:
      max_steps: Requested maximum steps.

    Returns:
      Validated max_steps.

    Raises:
      ValueError: If max_steps is not an int >= 1.
    '''
    if isinstance(max_steps, bool):
      raise ValueError('max_steps must be an int >= 1')
    try:
      value = int(max_steps)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
      raise ValueError('max_steps must be an int >= 1') from exc
    if value < 1:
      raise ValueError('max_steps must be >= 1')
    return value

  def _validate_action(self, action: int) -> int:
    '''Validate an action is an integer index.

    Args:
      action: Raw action passed to step().

    Returns:
      Integer action index.

    Raises:
      ValueError: For None, str, NaN, inf, bool, or
       any other non-integer action.
    '''
    if isinstance(action, bool):
      raise ValueError(f'invalid action: {action!r}')
    if isinstance(action, (int, np.integer)):
      return int(action)
    if isinstance(action, float):
      if np.isfinite(action) and float(action).is_integer():
        return int(action)
      raise ValueError(f'invalid action: {action!r}')
    raise ValueError(f'invalid action: {action!r}')

  @property
  def curriculum_stage(self) -> int:
    '''Return the current curriculum stage.

    Returns:
      Current curriculum stage index.
    '''
    return self._curriculum_stage

  @property
  def episode_data(self) -> EpisodeData | None:
    '''Return the current episode data.

    Returns:
      Current episode data, or None before reset.
    '''
    return self._episode_data

  @property
  def spaces(self) -> list[Space3D]:
    '''Return per truck 3D spaces.

    Returns:
      List of truck interior spaces.
    '''
    return self._spaces

  @property
  def placed_cartons(
    self,
  ) -> dict[int, PlacementInfo]:
    '''Return placed carton records.

    Returns:
      Mapping of carton id to placement info.
    '''
    return self._placed

  @property
  def current_weights(self) -> list[float]:
    '''Return current load weights per truck.

    Returns:
      Current weight per truck.
    '''
    return self._weights

  @property
  def num_delivered(self) -> int:
    '''Return the number of delivered cartons.

    Returns:
      Count of delivered cartons.
    '''
    return len(self._delivered)

  @property
  def truck_travel(self) -> list[float]:
    '''Return travel distance per truck.

    Returns:
      Travel distance per truck.
    '''
    return self._truck_travel

  @property
  def step_count(self) -> int:
    '''Return the current step count.

    Returns:
      Number of steps taken in this episode.
    '''
    return self._step_count

  @property
  def action_manager(self) -> ActionManager:
    '''Return the action manager.

    Returns:
      Action manager for candidate actions.
    '''
    return self._action_manager

  @property
  def current_carton(self) -> Carton | None:
    '''Return the carton being packed, if any.

    Returns:
      Current carton, or None outside loading.
    '''
    return self._current_carton

  @property
  def carton_lookup(self) -> dict[int, Carton]:
    '''Return the carton lookup mapping.

    Returns:
      Mapping of carton id to carton.
    '''
    return self._carton_lookup

  @property
  def active_truck_idx(self) -> int:
    '''Return the active truck index.

    Returns:
      Index of the truck acting next.
    '''
    return self._active_idx

  @property
  def truck_cargo(self) -> list[set[int]]:
    '''Return cargo carton ids per truck.

    Returns:
      Cargo carton ids per truck.
    '''
    return self._truck_cargo

  @property
  def warehouse_cartons(
    self,
  ) -> dict[int, list[int]]:
    '''Return remaining cartons per warehouse.

    Returns:
      Mapping of warehouse id to carton ids.
    '''
    return self._warehouse_cartons

  def set_curriculum_stage(self, stage: int) -> None:
    '''Set the curriculum stage.

    Args:
      stage: New curriculum stage index.

    Raises:
      ValueError: If stage is not an int.
    '''
    self._curriculum_stage = self._clamp_stage(stage)

  def reset(
    self,
    *,
    seed: int | None = None,
    options: dict | None = None,
  ) -> tuple[dict, dict]:
    '''Reset the environment with fresh episode data.

    Args:
      seed: Random seed for this episode.
      options: Optional reset options, unused.

    Returns:
      Initial observation and stage info.
    '''
    _ = options
    super().reset(seed=seed)
    self._episode_done = False
    if seed is not None:
      self._rng = np.random.default_rng(seed)
    stage = self._config.curriculum.stages[self._curriculum_stage]
    generator = DataGenerator(self._config, self._rng)
    self._episode_data = generator.generate(
      num_trucks=stage.num_trucks,
      num_stores=stage.num_stores,
      num_cartons=stage.num_cartons,
      num_warehouses=stage.num_warehouses,
      grid_rows=stage.grid_rows,
      grid_cols=stage.grid_cols,
    )
    self._init_episode_state()
    self._compute_candidates()
    self._episode_id += 1
    return self._get_obs(), {'stage': self._curriculum_stage}

  def _init_episode_state(self) -> None:
    '''Initialize per episode mutable state.

    Returns:
      None.
    '''
    data = self._episode_data
    if data is None:
      return
    store_ids = [
      store.store_id
      for store in sorted(
        data.stores,
        key=lambda item: item.route_position,
      )
    ]
    for truck in data.trucks:
      truck.route = list(store_ids)
    self._spaces = [
      Space3D(truck.length, truck.width, truck.height) for truck in data.trucks
    ]
    self._placed = {}
    self._validators = [
      PlacementValidator(space, self._placed) for space in self._spaces
    ]
    self._weights = [0.0] * len(data.trucks)
    self._delivered = set()
    self._truck_cargo = [set() for _ in data.trucks]
    self._truck_travel = [0.0] * len(data.trucks)
    self._truck_warehouse = {}
    self._carton_lookup = {carton.carton_id: carton for carton in data.cartons}
    self._warehouse_cartons = {}
    for warehouse in data.warehouses:
      ids = [
        carton.carton_id
        for carton in data.cartons
        if carton.origin_warehouse_id == warehouse.warehouse_id
      ]
      ids.sort(key=self._queue_key)
      self._warehouse_cartons[warehouse.warehouse_id] = ids
    self._active_idx = 0
    self._step_count = 0
    self._cumulative = 0.0
    self._current_carton = None

  def _queue_key(self, carton_id: int) -> tuple:
    '''Return sorting key for the carton queue.

    Args:
      carton_id: Carton identifier.

    Returns:
      Sort key favoring later stores and priority.
    '''
    carton = self._carton_lookup.get(carton_id)
    if carton is None:
      return (0, 0)
    route_pos = 0
    if self._episode_data is not None:
      for store in self._episode_data.stores:
        if store.store_id == carton.destination_store_id:
          route_pos = store.route_position
    return (-route_pos, carton.priority, carton_id)

  def action_masks(self) -> np.ndarray:
    '''Return the mask over valid actions.

    Returns:
      Boolean mask over the unified action space.
    '''
    return self._mask_provider.get_mask()

  def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
    '''Execute one action and advance the episode.

    Args:
      action: Discrete action index.

    Returns:
      Observation, reward, terminated, truncated
      flag, and info for the step.

    Raises:
      ValueError: If action is not an integer index
       (None, str, NaN, inf, bool, ...).
      RuntimeError: If called after terminated or
       truncated.

    Masked or out-of-range actions are a defined
    no-op: current observation, 0.0 reward, no
    state change, no step_count increment, and
    no truck switch.
    '''
    if self._episode_done:
      raise RuntimeError('step() called after episode done')
    action_int = self._validate_action(action)
    mask = self.action_masks()
    if not (0 <= action_int < len(mask)) or not mask[action_int]:
      info: dict = {
        'reward_breakdown': {},
        'num_placed': len(self._placed),
        'stage': self._curriculum_stage,
        'invalid_action': True,
      }
      return (self._get_obs(), 0.0, False, False, info)
    self._step_count += 1
    decoded = self._action_manager.decode_action(action_int)
    step_carton: Carton | None = None
    step_placement: PlacementInfo | None = None
    if decoded is not None:
      action_type, candidate = decoded
      if action_type == 'packing':
        step_carton = self._current_carton
        step_placement = self._execute_packing(
          candidate  # type: ignore[arg-type]
        )
      else:
        self._execute_routing(
          candidate  # type: ignore[arg-type]
        )
    terminated = self._is_terminal()
    reward, breakdown = self._compute_reward(
      step_carton, step_placement, terminated
    )
    self._cumulative += reward
    truncated = False
    if not terminated:
      if self._step_count >= self._max_steps:
        truncated = True
      else:
        next_valid = self._advance_active_truck()
        if not next_valid:
          terminated = True
    else:
      self._compute_candidates()
    info: dict = {
      'reward_breakdown': breakdown,
      'num_placed': len(self._placed),
      'stage': self._curriculum_stage,
    }
    if terminated or truncated:
      info['episode'] = {
        'r': float(self._cumulative),
        'l': int(self._step_count),
      }
      self._episode_done = True
    return (
      self._get_obs(),
      float(reward),
      bool(terminated),
      bool(truncated),
      info,
    )

  def _is_terminal(self) -> bool:
    '''Check whether the episode has ended.

    Returns:
      True when all cartons are delivered or all
      trucks are at the depot.
    '''
    data = self._episode_data
    if data is None or not data.trucks:
      return True
    if data.cartons and (len(self._delivered) >= len(data.cartons)):
      return True
    return all(truck.state == TruckState.AT_DEPOT for truck in data.trucks)

  def _execute_packing(
    self, candidate: PlacementCandidate
  ) -> PlacementInfo | None:
    '''Place the current carton in a truck.

    Re-checks overlap and weight via the validator
    before placing, and never double-counts a carton
    that is already placed. A place() ValueError is
    treated as an invalid action (None, no change).
    Only overlap/weight are re-checked here; gravity
    and fragility were enforced at candidate time.

    Args:
      candidate: Chosen placement candidate.

    Returns:
      Placement record, or None without a carton,
      when already placed, or when invalid.
    '''
    carton = self._current_carton
    if carton is None:
      return None
    if carton.carton_id in self._placed:
      return None
    idx = candidate.truck_id
    if idx < 0 or idx >= len(self._spaces):
      return None
    dl, dw, dh = candidate.oriented_dims
    max_weight = float(self._config.env.max_weight_capacity)
    data = self._episode_data
    if data is not None and idx < len(data.trucks):
      max_weight = float(data.trucks[idx].max_weight)
    current = self._weights[idx] if idx < len(self._weights) else 0.0
    try:
      if idx < len(self._validators):
        validator = self._validators[idx]
        if not validator.check_no_overlap(
          candidate.x, candidate.y, candidate.z, dl, dw, dh
        ):
          return None
        if not validator.check_weight(
          carton.weight, current, max_weight
        ):
          return None
      elif not self._spaces[idx].can_place(
        candidate.x, candidate.y, candidate.z, dl, dw, dh
      ):
        return None
      self._spaces[idx].place(
        carton.carton_id,
        candidate.x,
        candidate.y,
        candidate.z,
        dl,
        dw,
        dh,
      )
    except ValueError:
      return None
    info = PlacementInfo(
      truck_id=idx,
      position=(candidate.x, candidate.y, candidate.z),
      oriented_dims=candidate.oriented_dims,
      rotation=candidate.rotation,
    )
    self._placed[carton.carton_id] = info
    self._weights[idx] += carton.weight
    if idx < len(self._truck_cargo):
      self._truck_cargo[idx].add(carton.carton_id)
    queue = self._warehouse_cartons.get(carton.origin_warehouse_id, [])
    if carton.carton_id in queue:
      queue.remove(carton.carton_id)
    return info

  def _execute_routing(self, candidate: RoutingCandidate) -> None:
    '''Move the active truck to a facility.

    Args:
      candidate: Chosen routing candidate.

    Returns:
      None.
    '''
    data = self._episode_data
    if data is None:
      return
    idx = self._active_idx
    if idx >= len(data.trucks):
      return
    truck = data.trucks[idx]
    dist = candidate.distance
    if not np.isfinite(dist):
      dist = 0.0
    if idx < len(self._truck_travel):
      self._truck_travel[idx] += float(dist)
    truck.position = candidate.position
    if candidate.location_type == CellType.WAREHOUSE:
      truck.state = TruckState.LOADING
      self._truck_warehouse[idx] = candidate.location_id
    elif candidate.location_type == CellType.STORE:
      self._deliver_cargo(idx, candidate.location_id)
      truck.state = TruckState.ROUTING
      if candidate.location_id not in truck.visited_stores:
        truck.visited_stores.append(candidate.location_id)
    elif candidate.location_type == CellType.DEPOT:
      truck.state = TruckState.AT_DEPOT

  def _deliver_cargo(self, truck_idx: int, store_id: int) -> None:
    '''Deliver cargo destined for a store.

    Args:
      truck_idx: Index of the delivering truck.
      store_id: Store receiving the delivery.

    Returns:
      None.
    '''
    if truck_idx >= len(self._truck_cargo):
      return
    cargo = self._truck_cargo[truck_idx]
    for carton_id in list(cargo):
      carton = self._carton_lookup.get(carton_id)
      if carton is not None and carton.destination_store_id == store_id:
        cargo.discard(carton_id)
        self._delivered.add(carton_id)
        self._weights[truck_idx] = max(
          self._weights[truck_idx] - carton.weight,
          0.0,
        )

  def _advance_active_truck(self) -> bool:
    '''Advance round robin to a truck with actions.

    Returns:
      True when a truck with valid actions exists.
    '''
    data = self._episode_data
    if data is None or not data.trucks:
      return False
    count = len(data.trucks)
    for offset in range(1, count + 1):
      idx = (self._active_idx + offset) % count
      truck = data.trucks[idx]
      if truck.state == TruckState.AT_DEPOT:
        continue
      self._active_idx = idx
      self._compute_candidates()
      if self._mask_provider.has_valid_actions:
        return True
    self._compute_candidates()
    return self._mask_provider.has_valid_actions

  def _compute_candidates(self) -> None:
    '''Compute candidates for the active truck.

    Returns:
      None.
    '''
    data = self._episode_data
    if data is None or not data.trucks:
      return
    self._current_carton = None
    idx = self._active_idx
    if idx >= len(data.trucks):
      idx = 0
      self._active_idx = 0
    truck = data.trucks[idx]
    if truck.state == TruckState.LOADING:
      carton = self._next_carton(idx)
      if carton is not None:
        self._current_carton = carton
        self._action_manager.compute_candidates(
          carton,
          [truck],
          [self._spaces[idx]],
          [self._weights[idx]],
          [self._validators[idx]],
          data.cartons,
        )
        if self._action_manager.candidates:
          return
      truck.state = TruckState.ROUTING
    if truck.state == TruckState.ROUTING:
      self._compute_routing(idx, truck)

  def _next_carton(self, truck_idx: int) -> Carton | None:
    '''Return the next carton for a loading truck.

    Args:
      truck_idx: Index of the loading truck.

    Returns:
      Next carton from its warehouse queue.
    '''
    warehouse_id = self._truck_warehouse.get(truck_idx)
    if warehouse_id is None:
      return None
    queue = self._warehouse_cartons.get(warehouse_id, [])
    if not queue:
      return None
    return self._carton_lookup.get(queue[0])

  def _compute_routing(self, idx: int, truck: Truck) -> None:
    '''Compute routing candidates for an active truck.

    Args:
      idx: Index of the active truck.
      truck: Active truck.

    Returns:
      None.
    '''
    data = self._episode_data
    if data is None:
      return
    warehouses = [
      warehouse
      for warehouse in data.warehouses
      if self._warehouse_cartons.get(warehouse.warehouse_id)
    ]
    cargo_map = {
      truck.truck_id: sorted(self._truck_cargo[idx])
      if idx < len(self._truck_cargo)
      else []
    }
    self._action_manager.compute_routing_candidates(
      truck,
      idx,
      warehouses,
      data.stores,
      data.grid_world,
      cargo_map,
      self._delivered,
      self._carton_lookup,
      self._warehouse_cartons,
    )

  def _compute_reward(
    self,
    step_carton: Carton | None,
    step_placement: PlacementInfo | None,
    is_terminal: bool,
  ) -> tuple[float, dict[str, float]]:
    '''Compute the step reward and its breakdown.

    Args:
      step_carton: Carton placed this step.
      step_placement: Placement made this step.
      is_terminal: Whether the episode ended.

    Returns:
      Total reward and per component breakdown.
    '''
    data = self._episode_data
    if data is None:
      return 0.0, {}
    unplaced = [
      carton for carton in data.cartons if carton.carton_id not in self._placed
    ]
    max_dist = self._max_possible_distance()
    state = EnvironmentState(
      trucks=data.trucks,
      spaces=self._spaces,
      placed_cartons=self._placed,
      unplaced_cartons=unplaced,
      all_cartons=data.cartons,
      stores=data.stores,
      current_weights=self._weights,
      step_carton=step_carton,
      step_placement=step_placement,
      is_terminal=is_terminal,
      total_cartons=len(data.cartons),
      total_travel_distance=sum(self._truck_travel),
      max_possible_distance=max_dist,
      delivered_cartons=set(self._delivered),
    )
    return self._reward.compute(state)

  def _max_possible_distance(self) -> float:
    '''Estimate a travel distance normalization scale.

    Returns:
      Positive normalization scale for travel.
    '''
    data = self._episode_data
    if data is None or data.grid_world is None:
      return 1.0
    matrix = data.grid_world.distance_matrix
    finite = matrix[np.isfinite(matrix)]
    scale = float(finite.max()) if finite.size else 1.0
    return max(scale * max(len(data.cartons), 1), 1.0)

  def _get_obs(self) -> dict:
    '''Encode the current observation.

    Returns:
      Dict observation for the current state.
    '''
    data = self._episode_data
    if data is None:
      ctx = ObservationContext(
        trucks=[],
        spaces=[],
        current_carton=None,
        remaining_cartons=[],
        packing_candidates=[],
        current_weights=[],
        num_placed=0,
        total_cartons=0,
        stage_index=self._curriculum_stage,
        step_count=self._step_count,
        max_steps=self._max_steps,
      )
      return self._obs_builder.build(ctx)
    remaining = [
      carton for carton in data.cartons if carton.carton_id not in self._placed
    ]
    ctx = ObservationContext(
      trucks=data.trucks,
      spaces=self._spaces,
      current_carton=self._current_carton,
      remaining_cartons=remaining,
      packing_candidates=list(self._action_manager.candidates),
      current_weights=list(self._weights),
      num_placed=len(self._placed),
      total_cartons=len(data.cartons),
      stage_index=self._curriculum_stage,
      step_count=self._step_count,
      max_steps=self._max_steps,
      grid_world=data.grid_world,
      warehouses=data.warehouses,
      warehouse_cartons=self._warehouse_cartons,
      routing_candidates=list(self._action_manager.routing_candidates),
      active_truck_idx=self._active_idx,
      total_travel=sum(self._truck_travel),
      num_delivered=len(self._delivered),
    )
    return self._obs_builder.build(ctx)

  def get_render_snapshot(self) -> dict:
    '''Return a snapshot for grid rendering.

    Returns:
      Dict snapshot decoupled from the environment.
    '''
    data = self._episode_data
    if data is None:
      return {
        'grid_world': None,
        'trucks': [],
        'warehouses': [],
        'stores': [],
        'warehouse_cartons': {},
        'truck_cargo': [],
        'delivered': set(),
        'step': self._step_count,
        'total_cartons': 0,
        'cumulative_reward': self._cumulative,
        'truck_travel': [],
      }
    return {
      'grid_world': data.grid_world,
      'trucks': list(data.trucks),
      'warehouses': list(data.warehouses),
      'stores': list(data.stores),
      'warehouse_cartons': {
        key: list(value) for key, value in self._warehouse_cartons.items()
      },
      'truck_cargo': [set(cargo) for cargo in self._truck_cargo],
      'delivered': set(self._delivered),
      'step': self._step_count,
      'total_cartons': len(data.cartons),
      'cumulative_reward': self._cumulative,
      'truck_travel': list(self._truck_travel),
    }
