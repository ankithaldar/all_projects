#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the cat_game_bot.adb ADB client and validation helpers."""

import os
import subprocess

import cv2
import numpy as np
import pytest

import cat_game_bot.adb as adb
from cat_game_bot.errors import AdbError, ConfigError, VisionError
from cat_game_bot.models import Point

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


class FakeProc:
  """Minimal Popen stand-in backed by a real OS pipe."""

  def __init__(self, data, *, hold=False):
    """Create a fake process that serves ``data`` from a pipe.

    Args:
      data: Bytes the fake command writes to stdout.
      hold: When true, keep the write end open so readers see no EOF.
    """
    self.killed = False
    self.waited = False
    self.wait_timeout = None
    read_fd, write_fd = os.pipe()
    self._write_fd = None
    if hold:
      self._write_fd = write_fd
    else:
      os.write(write_fd, data)
      os.close(write_fd)
    self.stdout = os.fdopen(read_fd, 'rb', buffering=0)

  def kill(self):
    """Record that the process was killed."""
    self.killed = True

  def wait(self, timeout=None):
    """Record that the process was waited on."""
    self.waited = True
    self.wait_timeout = timeout
    return 0

  def close_write(self):
    """Close the retained write end so readers observe EOF."""
    if self._write_fd is not None:
      os.close(self._write_fd)
      self._write_fd = None


class FakeRunner:
  """Callable Popen replacement that records invocations."""

  def __init__(self, data=b'', *, hold=False):
    """Configure the runner's canned response.

    Args:
      data: Bytes returned by every fake process.
      hold: Forwarded to :class:`FakeProc`.
    """
    self.calls = []
    self.data = data
    self.hold = hold
    self.procs = []

  def __call__(self, cmd, **kwargs):
    """Return a fake process and remember the command and kwargs."""
    self.calls.append((list(cmd), kwargs))
    proc = FakeProc(self.data, hold=self.hold)
    self.procs.append(proc)
    return proc


def make_png(width, height):
  """Encode a solid black PNG of the requested size.

  Args:
    width: Image width in pixels.
    height: Image height in pixels.

  Returns:
    The encoded PNG bytes.
  """
  ok, encoded = cv2.imencode(
    '.png', np.zeros((height, width, 3), dtype=np.uint8))
  assert ok
  return encoded.tobytes()


def make_jpeg(width, height):
  """Encode a solid black JPEG of the requested size.

  Args:
    width: Image width in pixels.
    height: Image height in pixels.

  Returns:
    The encoded JPEG bytes.
  """
  ok, encoded = cv2.imencode(
    '.jpg', np.zeros((height, width, 3), dtype=np.uint8))
  assert ok
  return encoded.tobytes()


def fake_png_header(width, height):
  """Build a minimal byte string with just a PNG IHDR header.

  Args:
    width: Declared width.
    height: Declared height.

  Returns:
    Header bytes sufficient for pre-decode dimension parsing.
  """
  return (PNG_SIGNATURE + (13).to_bytes(4, 'big') + b'IHDR'
          + width.to_bytes(4, 'big') + height.to_bytes(4, 'big')
          + b'\x08\x02\x00\x00\x00' + b'\x00' * 4)


# --- validation helpers ---------------------------------------------------


def test_validate_serial_accepts_typical_serials():
  """Typical emulator, USB and network serials are accepted unchanged."""
  for serial in ('emulator-5554', '0123456789ABCDEF', '192.168.1.1:5555'):
    assert adb.validate_serial(serial) == serial


def test_validate_serial_rejects_empty_and_hostile_input():
  """Empty and shell-hostile serial strings raise ConfigError."""
  for serial in ('', '-abc', 'a b', 'a;b', '$(id)', '../dev', 'a/b'):
    with pytest.raises(ConfigError):
      adb.validate_serial(serial)


def test_validate_package_accepts_valid_package():
  """A normal Android package name is accepted unchanged."""
  package = 'com.minogames.cats.client'
  assert adb.validate_package(package) == package


def test_validate_package_rejects_invalid_input():
  """Malformed or shell-hostile package names raise ConfigError."""
  for package in ('', '-com.x', 'com', 'com..x', 'com x', 'com;x', '1com.x'):
    with pytest.raises(ConfigError):
      adb.validate_package(package)


def test_build_adb_args_returns_discrete_tokens():
  """build_adb_args prefixes -s and the serial without shell quoting."""
  args = adb.build_adb_args('dev-1', ['shell', 'input', 'tap', '1', '2'])
  assert args == ['-s', 'dev-1', 'shell', 'input', 'tap', '1', '2']
  assert all(isinstance(token, str) for token in args)


def test_client_rejects_bad_serial_at_construction():
  """AdbClient validates its serial when constructed."""
  with pytest.raises(ConfigError):
    adb.AdbClient('-bad')


# --- run() ----------------------------------------------------------------


def test_run_returns_decoded_text_and_uses_devnull_stderr():
  """run decodes stdout as text and discards stderr via DEVNULL."""
  runner = FakeRunner(b'device\n')
  client = adb.AdbClient('dev-1', runner=runner)
  assert client.run(['get-state']) == 'device\n'
  cmd, kwargs = runner.calls[0]
  assert cmd == ['adb', '-s', 'dev-1', 'get-state']
  assert kwargs['stderr'] is subprocess.DEVNULL
  assert kwargs['stdout'] is subprocess.PIPE


def test_run_returns_raw_bytes_in_binary_mode():
  """run with binary=True returns the undecoded byte string."""
  payload = b'\x89PNG\r\n\x1a\n'
  client = adb.AdbClient('dev-1', runner=FakeRunner(payload))
  assert client.run(['exec-out'], binary=True) == payload


def test_run_raises_adb_error_and_kills_on_timeout():
  """A silent child is killed and raises AdbError at the deadline."""
  runner = FakeRunner(b'', hold=True)
  client = adb.AdbClient('dev-1', timeout=0.05, runner=runner)
  try:
    with pytest.raises(AdbError):
      client.run(['shell', 'sleep'])
    assert runner.procs[0].killed is True
    assert runner.procs[0].waited is True
  finally:
    runner.procs[0].close_write()


def test_run_checks_deadline_before_select(monkeypatch):
  """A deadline already passed before select kills without selecting."""
  runner = FakeRunner(b'', hold=True)
  client = adb.AdbClient('dev-1', timeout=1.0, runner=runner)
  ticks = iter((0.0, 100.0, 100.0, 100.0))
  monkeypatch.setattr(adb.time, 'monotonic', lambda: next(ticks, 100.0))
  try:
    with pytest.raises(AdbError):
      client.run(['shell', 'sleep'])
    assert runner.procs[0].killed is True
  finally:
    runner.procs[0].close_write()


def test_run_raises_adb_error_on_output_cap():
  """Exceeding max_stdout_bytes kills the child and raises AdbError."""
  runner = FakeRunner(b'x' * 100)
  client = adb.AdbClient(
    'dev-1', max_stdout_bytes=16, runner=runner)
  with pytest.raises(AdbError):
    client.run(['shell', 'yes'])
  assert runner.procs[0].killed is True


def test_run_passes_shell_false_explicitly():
  """The runner is invoked with an explicit shell=False keyword."""
  runner = FakeRunner(b'ok\n')
  adb.AdbClient('dev-1', runner=runner).run(['get-state'])
  assert runner.calls[0][1]['shell'] is False


def test_run_kills_child_when_read_raises(monkeypatch):
  """An OSError during stdout reading still kills and waits the child."""
  runner = FakeRunner(b'')
  client = adb.AdbClient('dev-1', runner=runner)

  def boom(unused_fd, unused_size):
    raise OSError('read failed')

  monkeypatch.setattr(adb.os, 'read', boom)
  try:
    with pytest.raises(OSError):
      client.run(['shell', 'sleep'])
    assert runner.procs[0].killed is True
    assert runner.procs[0].waited is True
  finally:
    runner.procs[0].stdout.close()


# --- screenshot -----------------------------------------------------------


def test_screenshot_decodes_png_and_enforces_dimensions():
  """screenshot decodes a valid PNG into a Screen of the right size."""
  runner = FakeRunner(make_png(1080, 2400))
  client = adb.AdbClient('dev-1', runner=runner)
  screen = client.screenshot()
  assert (screen.width, screen.height) == (1080, 2400)
  assert runner.calls[0][0] == [
    'adb', '-s', 'dev-1', 'exec-out', 'screencap', '-p']


def test_screenshot_decodes_jpeg_using_sof_scan():
  """screenshot reads JPEG dimensions from the first SOF marker."""
  client = adb.AdbClient('dev-1', runner=FakeRunner(make_jpeg(200, 100)))
  screen = client.screenshot()
  assert (screen.width, screen.height) == (200, 100)


def test_screenshot_rejects_empty_data():
  """An empty capture raises VisionError."""
  client = adb.AdbClient('dev-1', runner=FakeRunner(b''))
  with pytest.raises(VisionError):
    client.screenshot()


def test_screenshot_rejects_bad_signature():
  """Data that is neither PNG nor JPEG raises VisionError."""
  client = adb.AdbClient('dev-1', runner=FakeRunner(b'not an image'))
  with pytest.raises(VisionError):
    client.screenshot()


def test_screenshot_rejects_oversized_png_before_decode():
  """Oversized IHDR dimensions are rejected before image decoding."""
  client = adb.AdbClient(
    'dev-1', runner=FakeRunner(fake_png_header(2000, 100)))
  with pytest.raises(VisionError):
    client.screenshot()


def test_screenshot_rejects_oversized_pixel_count():
  """A PNG whose pixel count exceeds the cap is rejected."""
  client = adb.AdbClient(
    'dev-1', runner=FakeRunner(fake_png_header(1080, 2401)))
  with pytest.raises(VisionError):
    client.screenshot()


def test_screenshot_decodes_shell_translated_png():
  """A CRLF-translated PNG is de-CRed back into a decodable image."""
  translated = make_png(100, 50).replace(b'\n', b'\r\n')
  client = adb.AdbClient('dev-1', runner=FakeRunner(translated))
  screen = client.screenshot()
  assert (screen.width, screen.height) == (100, 50)


def test_decr_only_removes_crlf_pairs():
  """The de-CR helper converts CRLF but preserves lone CR bytes."""
  assert adb._decr(b'a\r\nb\r\n') == b'a\nb\n'
  assert adb._decr(b'a\rb') == b'a\rb'


def test_normalise_crlf_preserves_canonical_headers():
  """Canonical PNG/JPEG captures are left untouched by normalisation."""
  png = make_png(4, 4)
  assert adb._normalise_crlf(png) == png
  jpeg = make_jpeg(4, 4)
  assert adb._normalise_crlf(jpeg) == jpeg


# --- device actions -------------------------------------------------------


def test_launch_uses_validated_package():
  """launch issues a monkey command with the validated package."""
  runner = FakeRunner(b'')
  adb.AdbClient('dev-1', runner=runner).launch('com.minogames.cats.client')
  assert runner.calls[0][0] == [
    'adb', '-s', 'dev-1', 'shell', 'monkey', '-p',
    'com.minogames.cats.client', '1']


def test_launch_rejects_hostile_package():
  """launch rejects a hostile package before invoking the runner."""
  runner = FakeRunner(b'')
  with pytest.raises(ConfigError):
    adb.AdbClient('dev-1', runner=runner).launch('com;rm -rf')
  assert runner.calls == []


def test_tap_swipe_back_emit_input_commands():
  """tap, swipe and back emit discrete input shell commands."""
  runner = FakeRunner(b'')
  client = adb.AdbClient('dev-1', runner=runner)
  client.tap(Point(10, 20))
  client.swipe(Point(1, 2), Point(3, 4), 500)
  client.back()
  assert runner.calls[0][0][-4:] == ['input', 'tap', '10', '20']
  assert runner.calls[1][0][-6:] == ['swipe', '1', '2', '3', '4', '500']
  assert runner.calls[2][0][-3:] == ['input', 'keyevent', '4']


def test_is_connected_interprets_get_state():
  """is_connected returns True only when get-state reports a device."""
  assert adb.AdbClient('dev-1', runner=FakeRunner(b'device\n')).is_connected()
  assert not adb.AdbClient(
    'dev-1', runner=FakeRunner(b'unknown\n')).is_connected()


@pytest.mark.parametrize('state', [
    b'no device\n', b'device offline\n', b'device not found\n',
    b'unauthorized\n', b'device product:foo\n',
])
def test_is_connected_rejects_substring_device_matches(state):
  """Only an exact 'device' stdout counts as connected."""
  client = adb.AdbClient('dev-1', runner=FakeRunner(state))
  assert client.is_connected() is False


def test_is_connected_swallows_adb_errors():
  """is_connected returns False when the underlying run raises AdbError."""
  runner = FakeRunner(b'', hold=True)
  client = adb.AdbClient('dev-1', timeout=0.01, runner=runner)
  try:
    assert client.is_connected() is False
  finally:
    runner.procs[0].close_write()


def test_discover_parses_device_lines():
  """discover returns serials listed as devices by adb devices."""
  text = (b'List of devices attached\n'
          b'emulator-5554\tdevice\n'
          b'0123ABC\tunauthorized\n')
  found = adb.AdbClient.discover(runner=FakeRunner(text))
  assert found == ['emulator-5554']
