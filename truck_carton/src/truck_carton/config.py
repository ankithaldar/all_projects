from dataclasses import dataclass, field


@dataclass(frozen=True)
class TruckConfig:
  length_range: tuple[int, int] = (8, 16)
  width_range: tuple[int, int] = (4, 8)
  height_range: tuple[int, int] = (4, 8)
  weight_capacity_range: tuple[float, float] = (
    500.0, 2000.0
  )


@dataclass(frozen=True)
class CartonConfig:
  length_range: tuple[int, int] = (1, 4)
  width_range: tuple[int, int] = (1, 4)
  height_range: tuple[int, int] = (1, 4)
  weight_range: tuple[float, float] = (1.0, 50.0)
  fragile_probability: float = 0.2


@dataclass(frozen=True)
class GridConfig:
  """Procedural grid generation parameters."""

  min_facility_spacing: int = 2
  road_extra_edges: int = 2


@dataclass(frozen=True)
class RewardWeights:
  alpha_utilization: float = 1.0
  beta_displacement: float = -2.0
  gamma_grouping: float = 1.5
  delta_fragility: float = -5.0
  epsilon_support: float = -5.0
  zeta_weight: float = -3.0
  eta_completion: float = 10.0
  theta_priority: float = 1.0
  iota_travel_distance: float = -1.0


@dataclass(frozen=True)
class CurriculumStage:
  name: str
  num_trucks: int
  num_stores: int
  num_cartons: int
  num_warehouses: int
  grid_rows: int
  grid_cols: int
  promotion_threshold: float
  promotion_window: int


@dataclass(frozen=True)
class CurriculumConfig:
  stages: tuple[CurriculumStage, ...] = (
    CurriculumStage(
      'toy', 2, 2, 10, 1, 5, 5, 0.7, 100
    ),
    CurriculumStage(
      'small', 3, 3, 20, 2, 7, 7, 0.7, 100
    ),
    CurriculumStage(
      'medium', 5, 4, 40, 3, 10, 10, 0.7, 100
    ),
  )


@dataclass(frozen=True)
class EnvironmentConfig:
  """Max bounds for observation space dims."""

  max_trucks: int = 5
  max_stores: int = 4
  max_cartons: int = 40
  max_truck_length: int = 16
  max_truck_width: int = 8
  max_truck_height: int = 8
  max_weight_capacity: float = 2000.0
  max_carton_weight: float = 50.0
  max_candidates: int = 500
  candidate_feature_dim: int = 18
  max_grid_rows: int = 10
  max_grid_cols: int = 10
  max_warehouses: int = 4
  max_routing_actions: int = 100
  routing_feature_dim: int = 8


@dataclass(frozen=True)
class TrainingConfig:
  """MaskablePPO hyperparameters."""

  total_timesteps: int = 2_000_000
  learning_rate: float = 3e-4
  n_steps: int = 2048
  batch_size: int = 64
  n_epochs: int = 10
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip_range: float = 0.2
  ent_coef: float = 0.01
  vf_coef: float = 0.5
  net_arch: tuple[int, ...] = (256, 256)
  eval_freq: int = 10_000
  eval_episodes: int = 20
  seed: int = 42


@dataclass(frozen=True)
class AppConfig:
  """Top-level configuration container."""

  truck: TruckConfig = field(
    default_factory=TruckConfig
  )
  carton: CartonConfig = field(
    default_factory=CartonConfig
  )
  grid: GridConfig = field(
    default_factory=GridConfig
  )
  rewards: RewardWeights = field(
    default_factory=RewardWeights
  )
  curriculum: CurriculumConfig = field(
    default_factory=CurriculumConfig
  )
  env: EnvironmentConfig = field(
    default_factory=EnvironmentConfig
  )
  training: TrainingConfig = field(
    default_factory=TrainingConfig
  )
