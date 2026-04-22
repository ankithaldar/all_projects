import numpy as np

from truck_carton.config import EnvironmentConfig
from truck_carton.domain.models import Carton, Truck
from truck_carton.env.action import ActionManager
from truck_carton.env.masking import (
  ActionMaskProvider,
)
from truck_carton.packing.placement import (
  PlacementValidator,
)
from truck_carton.packing.space3d import Space3D


def test_mask_shape():
  cfg = EnvironmentConfig()
  am = ActionManager(cfg)
  mask_provider = ActionMaskProvider(am)
  mask = mask_provider.get_mask()
  assert mask.shape == (am.total_action_size,)
  assert not mask.any()


def test_mask_after_candidates():
  cfg = EnvironmentConfig()
  am = ActionManager(cfg)
  mask_provider = ActionMaskProvider(am)

  space = Space3D(8, 4, 4)
  truck = Truck(0, 8, 4, 4, 500.0, [0])
  carton = Carton(
    1, 1, 1, 1, 5.0, False, 2, 0
  )
  validator = PlacementValidator(space, {})

  am.compute_candidates(
    carton, [truck], [space],
    [0.0], [validator],
  )

  mask = mask_provider.get_mask()
  num_valid = mask.sum()
  assert num_valid > 0
  assert num_valid == len(am.candidates)
  assert not mask[len(am.candidates):].any()


def test_has_valid_actions():
  cfg = EnvironmentConfig()
  am = ActionManager(cfg)
  mask_provider = ActionMaskProvider(am)

  assert mask_provider.has_valid_actions is False

  space = Space3D(4, 4, 4)
  truck = Truck(0, 4, 4, 4, 500.0, [0])
  carton = Carton(
    1, 1, 1, 1, 5.0, False, 2, 0
  )
  validator = PlacementValidator(space, {})
  am.compute_candidates(
    carton, [truck], [space],
    [0.0], [validator],
  )

  assert mask_provider.has_valid_actions is True


def test_decode_valid_action():
  cfg = EnvironmentConfig()
  am = ActionManager(cfg)

  space = Space3D(4, 4, 4)
  truck = Truck(0, 4, 4, 4, 500.0, [0])
  carton = Carton(
    1, 1, 1, 1, 5.0, False, 2, 0
  )
  validator = PlacementValidator(space, {})
  am.compute_candidates(
    carton, [truck], [space],
    [0.0], [validator],
  )

  decoded = am.decode_action(0)
  assert decoded is not None
  action_type, candidate = decoded
  assert action_type == 'packing'
  assert candidate.truck_id == 0


def test_decode_invalid_action():
  cfg = EnvironmentConfig()
  am = ActionManager(cfg)
  assert am.decode_action(0) is None
  assert am.decode_action(999) is None


def test_routing_mask():
  """Routing candidates should appear after
  the packing offset in the action mask."""
  from truck_carton.domain.models import (
    GridWorld,
    Store,
    Truck,
    TruckState,
    Warehouse,
  )

  cfg = EnvironmentConfig()
  am = ActionManager(cfg)

  gw = GridWorld(
    rows=5, cols=5,
    grid=np.zeros((5, 5), dtype=np.int8),
    depot_position=(2, 2),
    distance_matrix=np.array([
      [0.0, 3.0], [3.0, 0.0]
    ]),
    facility_positions=[(2, 2), (0, 0)],
  )
  truck = Truck(
    0, 8, 4, 4, 500.0, [],
    position=(2, 2),
    state=TruckState.ROUTING,
  )
  wh = Warehouse(0, (0, 0))
  am.compute_routing_candidates(
    truck, 0, [wh], [],
    gw, {0: [1, 2]}, set(), {},
  )

  mask = am.get_action_mask()
  assert not mask[:cfg.max_candidates].any()
  routing_mask = mask[cfg.max_candidates:]
  assert routing_mask.any()
