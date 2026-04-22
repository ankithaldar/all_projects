import logging

from truck_carton.logging_config import (
  EpisodeLogger,
  get_logger,
  setup_logging,
)


def test_get_logger():
  log = get_logger('env')
  assert log.name == 'truck_carton.env'
  assert isinstance(log, logging.Logger)


def test_get_logger_subsystems():
  names = [
    'env', 'reward', 'training',
    'curriculum', 'packing',
  ]
  for name in names:
    log = get_logger(name)
    assert log.name == f'truck_carton.{name}'


def test_setup_logging_idempotent():
  setup_logging(level=logging.DEBUG)
  setup_logging(level=logging.DEBUG)
  root = logging.getLogger('truck_carton')
  assert root.level == logging.DEBUG


def test_episode_logger_lifecycle():
  el = EpisodeLogger()

  el.begin_episode(
    episode_id=1,
    stage=0,
    num_trucks=2,
    num_cartons=10,
    grid_size=(5, 5),
  )

  el.log_step(
    step=1,
    action=42,
    action_type='packing',
    reward=0.5,
    breakdown={'utilization': 0.3},
    num_placed=1,
    num_delivered=0,
    active_truck=0,
    truck_states=['LOADING', 'ROUTING'],
    truck_positions=[(2, 2), (2, 2)],
  )

  el.log_routing(
    truck_id=0,
    src=(2, 2),
    dst=(1, 3),
    distance=3.0,
    destination_type='WAREHOUSE',
  )

  el.log_packing(
    carton_id=0,
    truck_id=0,
    position=(0, 0, 0),
    dims=(2, 2, 2),
    weight=10.0,
  )

  el.log_delivery(
    truck_id=0, store_id=1, num_unloaded=3
  )

  summary = el.end_episode(
    total_reward=5.0,
    num_placed=8,
    num_delivered=6,
    total_cartons=10,
    terminated=True,
  )

  assert summary['episode_id'] == 1
  assert summary['total_reward'] == 5.0
  assert summary['num_placed'] == 8
  assert summary['num_delivered'] == 6
  assert summary['terminated'] is True
  assert len(summary['steps']) == 1
  assert summary['steps'][0]['action'] == 42


def test_episode_logger_completion_rate():
  el = EpisodeLogger()
  el.begin_episode(1, 0, 2, 10, (5, 5))
  summary = el.end_episode(
    total_reward=0.0,
    num_placed=5,
    num_delivered=3,
    total_cartons=10,
    terminated=False,
  )
  assert summary['completion_rate'] == 0.5
  assert summary['delivery_rate'] == 0.3
