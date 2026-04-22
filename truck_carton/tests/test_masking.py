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
    assert mask.shape == (cfg.max_candidates,)
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

    candidate = am.decode_action(0)
    assert candidate is not None
    assert candidate.truck_id == 0


def test_decode_invalid_action():
    cfg = EnvironmentConfig()
    am = ActionManager(cfg)
    assert am.decode_action(0) is None
    assert am.decode_action(999) is None
