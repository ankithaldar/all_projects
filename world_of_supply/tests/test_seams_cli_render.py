#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Seam tests for CLI, renderer, status, hardware, and tracker.'''

import numpy as np
import pytest

from pathlib import Path

pytest.importorskip('PIL')

from world_of_supply.analytics.hardware import _init_ray
from world_of_supply.analytics.tracker import SimulationTracker
from world_of_supply.analytics.tracker import _cumulative_rewards
from world_of_supply.analytics.tracker import _first_episode_cumulative
from world_of_supply.analytics.tracker import _plot_tracker, _tick_axis
from world_of_supply.cli import _frame_filename, _run_ticks
from world_of_supply.cli import _should_render_frame, _tick_line
from world_of_supply.cli import _training_curriculum
from world_of_supply.economy import BalanceSheet
from world_of_supply.rendering.renderer import AsciiWorldRenderer
from world_of_supply.rendering.renderer import NotebookAnimator
from world_of_supply.rendering.renderer import _animation_dimensions
from world_of_supply.rendering.renderer import _animation_figure_size
from world_of_supply.rendering.renderer import _blank_layer, _blank_layers
from world_of_supply.rendering.renderer import _canvas_size
from world_of_supply.rendering.renderer import _draw_layers, _draw_status_column
from world_of_supply.rendering.renderer import _draw_status_panels
from world_of_supply.rendering.renderer import _facility_glyph, _layer_text
from world_of_supply.rendering.renderer import _make_animation
from world_of_supply.rendering.renderer import _status_columns_chunk
from world_of_supply.rendering.status import WorldStatusFormatter
from world_of_supply.rendering.status import ascii_progress_bar
from world_of_supply.rendering.status import counter_to_dict
from world_of_supply.scenario import ScenarioConfig, WorldBuilder
from world_of_supply.world import World


def test_cli_should_render_frame_and_filename():
  '''Verify frame cadence and zero-padded frame filenames.

  Returns:
    None: Test completes without assertion failure.
  '''
  assert _should_render_frame(0, 10) is True
  assert _should_render_frame(0, 60) is False
  assert _should_render_frame(5, 60) is True
  assert _should_render_frame(59, 60) is True
  assert _frame_filename(3) == 'frame_00003.png'
  assert _frame_filename(42) == 'frame_00042.png'


def test_cli_training_curriculum():
  '''Verify training curriculum offers warehouse promotion entries.

  Returns:
    None: Test completes without assertion failure.
  '''
  expected = ((0.25, 'WarehouseCell'), (0.35, 'WarehouseCell'))
  assert _training_curriculum(True) == expected
  assert _training_curriculum(False) == ()


def test_cli_run_ticks_advances_world_and_renders(capsys, tmp_path):
  '''Verify run ticks advances world and saves rendered frames.

  Args:
    capsys: Pytest output capture fixture.
    tmp_path: Pytest temporary directory fixture.

  Returns:
    None: Test completes without assertion failure.
  '''

  class FakeWorld:
    '''Minimal world stand-in advancing a balance each tick.'''

    def __init__(self):
      '''Initialize tick counter for the fake world.

      Returns:
        None: Stand-in is mutated in place.
      '''
      self.ticks = 0

    def act(self, control):
      '''Advance the fake world by one tick.

      Args:
        control: Control ignored by the fake world.

      Returns:
        BalanceSheet: Empty sheet recording no monetary effect.
      '''
      self.ticks += 1
      return BalanceSheet()

    @property
    def economy(self):
      '''Expose self as the economy balance provider.

      Returns:
        FakeWorld: Self, which provides global_balance.
      '''
      return self

    def global_balance(self):
      '''Report a balance equal to the elapsed tick count.

      Returns:
        BalanceSheet: Sheet whose profit equals the tick count.
      '''
      return BalanceSheet(self.ticks, 0)

  class FakePolicy:
    '''Minimal policy stand-in returning empty controls.'''

    def compute_control(self, world):
      '''Return an empty control mapping for the fake world.

      Args:
        world: Fake world asking for controls.

      Returns:
        dict: Empty facility controls mapping.
      '''
      return {}

  class FakeRenderer:
    '''Minimal renderer stand-in saving a tiny PIL image.'''

    def render(self, world):
      '''Produce a tiny RGB image for frame saving.

      Args:
        world: Fake world to render.

      Returns:
        PIL.Image.Image: Tiny blank RGB image.
      '''
      from PIL import Image

      return Image.new('RGB', (2, 2))

  world = FakeWorld()
  _run_ticks(world, FakePolicy(), 3, FakeRenderer(), tmp_path)
  output = capsys.readouterr().out
  assert world.ticks == 3
  assert 'tick 1/3' in output and 'tick 3/3' in output
  assert 'saved' in output
  assert _tick_line(0, 3, world).startswith('--- tick 1/3')


def test_renderer_blank_layers_and_helpers():
  '''Verify renderer blanks, glyphs, text, and canvas sizing.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = World(4, 3)
  layer = _blank_layer(world)
  assert len(layer) == 3 and len(layer[0]) == 4
  assert all(cell == ' ' for row in layer for cell in row)
  layers = _blank_layers(world)
  assert len(layers) == 3
  assert _facility_glyph(object()) is None
  assert _layer_text([['a', 'b'], ['c', 'd']]) == 'ab\ncd'
  assert _canvas_size(100, 10, 150) == (400, 40)
  assert _status_columns_chunk([1, 2, 3, 4], 1, 2) == [3, 4]


def test_renderer_draw_helpers_paint_canvas():
  '''Verify draw helpers paint layers and status panels on canvas.

  Returns:
    None: Test completes without assertion failure.
  '''
  from PIL import Image, ImageDraw, ImageFont

  font_path = Path(__file__).resolve().parents[1]
  font_path = font_path / 'src/world_of_supply/assets/FiraMono-Bold.ttf'
  font_path = str(font_path)
  font = ImageFont.truetype(font_path, 12)
  world = WorldBuilder.build(ScenarioConfig(), seed=0)
  image = Image.new('RGB', (400, 300))
  canvas = ImageDraw.Draw(image)
  _draw_layers(canvas, _blank_layers(world), 10, 10, font)
  _draw_status_column(canvas, [['status']], 5, 5, font)
  _draw_status_panels(canvas, world, 400, 50, font, 2)
  assert image.size == (400, 300)
  renderer = AsciiWorldRenderer()
  assert renderer.background == '#263238'


def test_renderer_make_animation_seam():
  '''Verify make animation wires frames through the animator.

  Returns:
    None: Test completes without assertion failure.
  '''
  image_array = np.zeros((3, 72, 144, 3))
  assert _animation_dimensions(image_array) == (72, 144)
  width, height = _animation_figure_size(image_array)
  assert (width, height) == (2.0, 1.0)

  class FakeImage:
    '''Minimal image stand-in recording the latest frame array.'''

    def __init__(self):
      '''Initialize the recorded frame holder.

      Returns:
        None: Stand-in is mutated in place.
      '''
      self.frame = None

    def set_array(self, frame):
      '''Record the frame array handed to the artist.

      Args:
        frame: Frame pixel data assigned by the animation.

      Returns:
        None: Stand-in is mutated in place.
      '''
      self.frame = frame

  class FakeAnimation:
    '''Minimal animation module stand-in recording build arguments.'''

    recorded = None

    @classmethod
    def FuncAnimation(cls, figure, func, frames, interval, repeat_delay,
                      repeat):
      '''Record animation build arguments and return them.

      Args:
        figure: Matplotlib figure displaying the frames.
        func: Animation callback updating one frame.
        frames: Number of frames to animate.
        interval: Delay between frames in milliseconds.
        repeat_delay: Delay before repeating the animation.
        repeat: Whether the animation repeats forever.

      Returns:
        tuple: Recorded animation build arguments.
      '''
      cls.recorded = (func, frames, interval, repeat_delay, repeat)
      return cls.recorded

  image = FakeImage()
  built = _make_animation('figure', image, image_array, FakeAnimation)
  assert built[1] == 3 and built[2] == 200 and built[4] is True
  built[0](1)
  assert image.frame.shape == image_array[1].shape
  assert NotebookAnimator.plot_sequence_images is not None


def test_status_per_unit_formatters():
  '''Verify status formatter builds each per-unit section.

  Returns:
    None: Test completes without assertion failure.
  '''
  formatter = WorldStatusFormatter()
  assert ascii_progress_bar(1, 2, 4) == '==-- 1/2'
  assert counter_to_dict({'a': 0, 'b': 2}) == {'b': 2}
  world = WorldBuilder.build(ScenarioConfig(), seed=0)
  facility = list(world.facilities.values())[0]
  assert formatter._distribution_status(facility)[0][0] == 'Fleet:'
  assert formatter._consumer_status(facility) == []
  assert formatter._seller_status(facility) == []
  storage_lines = formatter._storage_status(facility)
  assert storage_lines[0].startswith('Usage:')
  truck = facility.distribution.fleet[0]
  assert formatter._idle_line(truck) == 'IDLE'
  assert formatter._loading_line(truck).startswith('LOAD ')
  assert formatter._enroute_line(truck).startswith('MOVE ')
  assert formatter._unloading_line(truck).startswith('UNLD ')
  assert formatter._returning_line(truck) == 'BACK home -> home'
  assert formatter._transport_destination(truck) == 'home'
  assert formatter._transport_cargo(truck) == ':0'
  formatters = formatter._transport_formatters()
  state_class = truck.state.__class__
  expected_states = {state_class(1), state_class(2)}
  expected_states |= {state_class(3), state_class(4)}
  assert set(formatters) == expected_states


def test_hardware_init_ray_tolerates_reinit():
  '''Verify init ray forwards the ignore-reinit flag verbatim.

  Returns:
    None: Test completes without assertion failure.
  '''

  class FakeRay:
    '''Minimal ray stand-in recording initialization arguments.'''

    def __init__(self):
      '''Initialize the recorded initialization arguments.

      Returns:
        None: Stand-in is mutated in place.
      '''
      self.kwargs = None

    def init(self, **kwargs):
      '''Record initialization keyword arguments.

      Args:
        **kwargs: Arguments forwarded from the hardware helper.

      Returns:
        None: Stand-in is mutated in place.
      '''
      self.kwargs = kwargs

  fake_ray = FakeRay()
  _init_ray(fake_ray)
  assert fake_ray.kwargs == {'ignore_reinit_error': True}


def test_tracker_plot_and_axis_helpers(monkeypatch):
  '''Verify tracker helpers compute axes and plot without display.

  Args:
    monkeypatch: Pytest monkeypatch fixture.

  Returns:
    None: Test completes without assertion failure.
  '''
  import matplotlib

  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  tracker = SimulationTracker(3, 1, ['facility_a', 'facility_b'])
  samples = {'facility_a': 1.0, 'facility_b': 2.0}
  tracker.add_sample(0, 0, 1.0, samples)
  samples = {'facility_a': 3.0, 'facility_b': 4.0}
  tracker.add_sample(0, 1, 2.0, samples)
  samples = {'facility_a': 5.0, 'facility_b': 6.0}
  tracker.add_sample(0, 2, 3.0, samples)
  assert _tick_axis(3).tolist() == [0.0, 1.5, 3.0]
  balances = tracker.step_balances
  assert _cumulative_rewards(balances).tolist() == [[3.0, 10.0, 21.0]]
  first = _first_episode_cumulative(balances).tolist()
  assert first == [[1.0, 2.0], [4.0, 6.0], [9.0, 12.0]]
  monkeypatch.setattr(plt, 'show', lambda: None)
  _plot_tracker(tracker)
  assert len(plt.get_fignums()) >= 1
  plt.close('all')
