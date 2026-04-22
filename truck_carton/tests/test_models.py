from truck_carton.domain.models import (
    Carton,
    EpisodeData,
    PlacementInfo,
    Store,
    Truck,
)
from truck_carton.packing.rotation import Rotation


def test_store_creation():
    s = Store(store_id=0, route_position=0)
    assert s.store_id == 0
    assert s.route_position == 0


def test_truck_creation():
    t = Truck(
        truck_id=0, length=10, width=5,
        height=5, max_weight=1000.0, route=[0, 1],
    )
    assert t.truck_id == 0
    assert t.route == [0, 1]
    assert t.max_weight == 1000.0


def test_carton_creation():
    c = Carton(
        carton_id=1, length=2, width=3,
        height=4, weight=15.0, is_fragile=True,
        priority=3, destination_store_id=0,
    )
    assert c.is_fragile is True
    assert c.priority == 3
    assert c.destination_store_id == 0


def test_placement_info():
    info = PlacementInfo(
        truck_id=0, position=(1, 2, 0),
        oriented_dims=(2, 3, 4),
        rotation=Rotation.LWH,
    )
    assert info.position == (1, 2, 0)
    assert info.oriented_dims == (2, 3, 4)


def test_episode_data(
    sample_episode: EpisodeData,
):
    assert len(sample_episode.trucks) == 2
    assert len(sample_episode.stores) == 2
    assert len(sample_episode.cartons) == 4
