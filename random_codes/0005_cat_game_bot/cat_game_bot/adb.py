#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Safe ADB subprocess client with deadlines and pre-decode image checks."""

import os
import re
import selectors
import subprocess
import time
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from cat_game_bot.errors import AdbError, ConfigError, VisionError
from cat_game_bot.models import Point, Screen

__all__ = (
  'validate_serial', 'validate_package', 'build_adb_args', 'AdbClient',
)

ADB_BIN = 'adb'
MAX_WIDTH = 1080
MAX_HEIGHT = 2400
MAX_PIXELS = 2592000
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
SOF_MARKERS = frozenset((0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                         0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))

_SERIAL_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]*')
_PACKAGE_RE = re.compile(
  r'[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+')


def validate_serial(serial: str) -> str:
  """Validate an ADB serial against a conservative allow-list.

  Args:
    serial: The candidate serial string.

  Returns:
    The serial unchanged when valid.

  Raises:
    ConfigError: If the serial is empty or contains unsafe characters.
  """
  if not _SERIAL_RE.fullmatch(serial):
    raise ConfigError(f'invalid adb serial: {serial!r}')
  return serial


def validate_package(package: str) -> str:
  """Validate an Android package name against a conservative allow-list.

  Args:
    package: The candidate package name.

  Returns:
    The package unchanged when valid.

  Raises:
    ConfigError: If the package is malformed or contains unsafe characters.
  """
  if not _PACKAGE_RE.fullmatch(package):
    raise ConfigError(f'invalid package name: {package!r}')
  return package


def build_adb_args(serial: str, args: Sequence[str]) -> list[str]:
  """Build discrete ADB argument tokens targeting a validated serial.

  Args:
    serial: The target device serial.
    args: The ADB subcommand and its arguments.

  Returns:
    A list of tokens beginning with ``-s`` and the validated serial.
  """
  return ['-s', validate_serial(serial), *args]


def _kill(proc: subprocess.Popen) -> None:
  """Kill a child process and reap it, tolerating a slow exit.

  Args:
    proc: The process handle to terminate.
  """
  proc.kill()
  try:
    proc.wait(timeout=1.0)
  except subprocess.TimeoutExpired:
    pass


def _drain(proc: subprocess.Popen, data: bytes, deadline: float) -> bytes:
  """Wait for a process that reached EOF and return its buffered output.

  Args:
    proc: The process handle to reap.
    data: Output bytes collected so far.
    deadline: Absolute monotonic deadline for the wait.

  Returns:
    The collected output bytes.

  Raises:
    AdbError: If the process does not exit by the deadline.
  """
  try:
    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
  except subprocess.TimeoutExpired:
    _kill(proc)
    raise AdbError('adb command timed out') from None
  return data


def _run_process(runner: Callable, cmd: list[str], timeout: float,
                 max_stdout_bytes: int) -> bytes:
  """Run a command with a deadline and an output cap, collecting stdout.

  Args:
    runner: A Popen-compatible callable.
    cmd: The full command token list.
    timeout: Maximum runtime in seconds.
    max_stdout_bytes: Cap on collected stdout bytes.

  Returns:
    The bytes read from stdout before EOF.

  Raises:
    AdbError: On timeout or when the output cap is exceeded.
  """
  deadline = time.monotonic() + timeout
  proc = runner(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                shell=False)
  buffer = bytearray()
  selector = selectors.DefaultSelector()
  try:
    selector.register(proc.stdout.fileno(), selectors.EVENT_READ)
    while True:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise AdbError('adb command timed out')
      events = selector.select(max(0.0, remaining))
      if not events:
        raise AdbError('adb command timed out')
      for key, _ in events:
        chunk = os.read(key.fd, 4096)
        if not chunk:
          return _drain(proc, bytes(buffer), deadline)
        buffer.extend(chunk)
        if len(buffer) >= max_stdout_bytes:
          raise AdbError('adb output exceeded cap')
  except BaseException:
    # Any failure while reading must not leave a child process alive.
    _kill(proc)
    raise
  finally:
    selector.close()


def _decr(data: bytes) -> bytes:
  """Replace CRLF pairs with LF, leaving lone carriage returns intact.

  Args:
    data: Raw byte string from the device.

  Returns:
    The byte string with CRLF sequences normalised to LF.
  """
  return data.replace(b'\r\n', b'\n')


def _normalise_crlf(data: bytes) -> bytes:
  """Undo shell-introduced CR translation without corrupting headers.

  ``adb exec-out`` already returns canonical image bytes whose PNG
  signature legitimately contains a CRLF pair.  Only streams that lack
  a canonical PNG or JPEG header are shell-translated and need de-CRing.

  Args:
    data: Raw capture bytes from the device.

  Returns:
    Canonical image bytes ready for validation and decoding.
  """
  if data.startswith(PNG_SIGNATURE) or data.startswith(b'\xff\xd8'):
    return data
  return _decr(data)


def _png_dimensions(data: bytes) -> tuple[int, int]:
  """Read width and height from a PNG IHDR chunk.

  Args:
    data: PNG byte string.

  Returns:
    The declared width and height.

  Raises:
    VisionError: If the header is truncated or malformed.
  """
  if len(data) < 24:
    raise VisionError('truncated PNG header')
  if data[12:16] != b'IHDR':
    raise VisionError('PNG IHDR chunk missing')
  if int.from_bytes(data[8:12], 'big') != 13:
    raise VisionError('PNG IHDR length invalid')
  width = int.from_bytes(data[16:20], 'big')
  height = int.from_bytes(data[20:24], 'big')
  return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
  """Scan JPEG markers to the first SOF and read its dimensions.

  Args:
    data: JPEG byte string.

  Returns:
    The width and height declared by the first SOF marker.

  Raises:
    VisionError: If no SOF marker is found.
  """
  offset = 2
  size = len(data)
  while offset + 4 <= size:
    if data[offset] != 0xFF:
      offset += 1
      continue
    marker = data[offset + 1]
    offset += 2
    if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
      continue
    if marker == 0xDA or offset + 2 > size:
      break
    length = int.from_bytes(data[offset:offset + 2], 'big')
    if length < 2 or offset + length > size:
      break
    if marker in SOF_MARKERS:
      if offset + 7 > size:
        break
      height = int.from_bytes(data[offset + 3:offset + 5], 'big')
      width = int.from_bytes(data[offset + 5:offset + 7], 'big')
      return width, height
    offset += length
  raise VisionError('JPEG SOF marker not found')


def _image_dimensions(data: bytes) -> tuple[int, int]:
  """Read image dimensions from a PNG or JPEG byte string.

  Args:
    data: Encoded image bytes.

  Returns:
    The declared width and height.

  Raises:
    VisionError: If the image format is unsupported.
  """
  if data.startswith(PNG_SIGNATURE):
    return _png_dimensions(data)
  if data.startswith(b'\xff\xd8'):
    return _jpeg_dimensions(data)
  raise VisionError('screenshot is not PNG or JPEG')


class AdbClient:
  """Client for ADB commands with injected process runner and deadlines."""

  def __init__(self, serial: str, timeout: float = 10.0,
               max_stdout_bytes: int = 33554432,
               runner: Callable | None = None) -> None:
    """Initialise the client for one validated device serial.

    Args:
      serial: The target device serial.
      timeout: Per-command deadline in seconds.
      max_stdout_bytes: Cap on collected stdout bytes.
      runner: Optional Popen-compatible callable for tests.
    """
    self._serial = validate_serial(serial)
    self._timeout = timeout
    self._max_stdout_bytes = max_stdout_bytes
    self._runner: Callable = (
      runner if runner is not None else subprocess.Popen)

  @classmethod
  def discover(cls, runner: Callable | None = None) -> list[str]:
    """List serials of connected devices in the ready ``device`` state.

    Args:
      runner: Optional Popen-compatible callable for tests.

    Returns:
      The list of ready device serials, empty when ADB is unavailable.
    """
    actual: Callable = runner if runner is not None else subprocess.Popen
    try:
      text = _run_process(actual, [ADB_BIN, 'devices'], 10.0,
                          1024 * 1024).decode('utf-8', errors='replace')
    except AdbError:
      return []
    serials = []
    for line in text.splitlines()[1:]:
      fields = line.split()
      if len(fields) >= 2 and fields[1] == 'device':
        serials.append(fields[0])
    return serials

  def run(self, args: Sequence[str], *, binary: bool = False) -> str | bytes:
    """Run an ADB subcommand against the configured device.

    Args:
      args: The ADB subcommand and its arguments.
      binary: Return raw bytes when true, otherwise decoded text.

    Returns:
      The command stdout as text or bytes.

    Raises:
      AdbError: On timeout or when the output cap is exceeded.
    """
    cmd = [ADB_BIN, *build_adb_args(self._serial, args)]
    data = _run_process(self._runner, cmd, self._timeout,
                        self._max_stdout_bytes)
    if binary:
      return data
    return data.decode('utf-8', errors='replace')

  def launch(self, package: str) -> None:
    """Launch the given validated package via the monkey tool.

    Args:
      package: The Android package name to launch.
    """
    self.run(['shell', 'monkey', '-p', validate_package(package), '1'])

  def tap(self, point: Point) -> None:
    """Tap the given screen point.

    Args:
      point: The point to tap.
    """
    self.run(['shell', 'input', 'tap', str(point.x), str(point.y)])

  def swipe(self, start: Point, end: Point, duration_ms: int = 300) -> None:
    """Swipe from one point to another over the given duration.

    Args:
      start: The swipe origin.
      end: The swipe destination.
      duration_ms: Swipe duration in milliseconds.
    """
    self.run(['shell', 'input', 'swipe', str(start.x), str(start.y),
              str(end.x), str(end.y), str(duration_ms)])

  def back(self) -> None:
    """Press the hardware back key."""
    self.run(['shell', 'input', 'keyevent', '4'])

  def screenshot(self) -> Screen:
    """Capture and decode the device screen into a bounded Screen.

    Returns:
      The decoded screen frame.

    Raises:
      VisionError: On empty, malformed or oversized captures.
    """
    raw = self.run(['exec-out', 'screencap', '-p'], binary=True)
    if not raw:
      raise VisionError('screenshot returned no data')
    raw = _normalise_crlf(raw)
    width, height = _image_dimensions(raw)
    if (width > MAX_WIDTH or height > MAX_HEIGHT
        or width * height > MAX_PIXELS):
      raise VisionError('screenshot dimensions exceed cap')
    try:
      pixels = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8).copy(), cv2.IMREAD_COLOR)
    except cv2.error as exc:
      raise VisionError('screenshot decode failed') from exc
    if pixels is None:
      raise VisionError('screenshot could not be decoded')
    return Screen(pixels)

  def is_connected(self) -> bool:
    """Report whether ADB sees the configured device in the ready state.

    Returns:
      True when ``get-state`` reports a device.
    """
    try:
      state = self.run(['get-state'])
    except AdbError:
      return False
    return state.strip() == 'device'
