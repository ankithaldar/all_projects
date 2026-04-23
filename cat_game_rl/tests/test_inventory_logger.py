from __future__ import annotations

import glob
import os
import tempfile

import pytest

from src.logging_util.inventory_logger import InventoryLogger


class TestInventoryLogger:
  @pytest.fixture
  def log_dir(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      yield tmpdir

  def test_creates_log_file(self, log_dir: str):
    logger = InventoryLogger(log_dir=log_dir, name="test_log1")
    logger.log_decision(0, {"string": 5})
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    assert len(log_files) == 1

  def test_log_decision(self, log_dir: str):
    logger = InventoryLogger(log_dir=log_dir, name="test_log2")
    logger.log_decision(42, {"string": 5}, {"masked": True})
    log_path = os.path.join(log_dir, "test_log2.log")
    with open(log_path, "r") as f:
      content = f.read()
    assert "decision" in content
    assert "42" in content

  def test_log_transition(self, log_dir: str):
    logger = InventoryLogger(log_dir=log_dir, name="test_log3")
    logger.log_transition(1, {"stash": [0]}, {"stash": [5]})
    log_path = os.path.join(log_dir, "test_log3.log")
    with open(log_path, "r") as f:
      content = f.read()
    assert "transition" in content

  def test_log_reward(self, log_dir: str):
    logger = InventoryLogger(log_dir=log_dir, name="test_log4")
    logger.log_reward(10, 0.5, {"slot_util": 0.3})
    log_path = os.path.join(log_dir, "test_log4.log")
    with open(log_path, "r") as f:
      content = f.read()
    assert "reward" in content
    assert "0.5" in content

  def test_log_warning(self, log_dir: str):
    logger = InventoryLogger(log_dir=log_dir, name="test_log5")
    logger.log_warning(5, "Invalid action attempted")
    log_path = os.path.join(log_dir, "test_log5.log")
    with open(log_path, "r") as f:
      content = f.read()
    assert "warning" in content.lower() or "WARNING" in content

  def test_log_slot_event(self, log_dir: str):
    logger = InventoryLogger(log_dir=log_dir, name="test_log6")
    logger.log_slot_event(7, "start", "STRING", {"batch": 5})
    log_path = os.path.join(log_dir, "test_log6.log")
    with open(log_path, "r") as f:
      content = f.read()
    assert "slot" in content

  def test_log_ga_event(self, log_dir: str):
    logger = InventoryLogger(log_dir=log_dir, name="test_log7")
    logger.log_ga_event(0, "init", {"pop_size": 200})
    log_path = os.path.join(log_dir, "test_log7.log")
    with open(log_path, "r") as f:
      content = f.read()
    assert "ga" in content

  def test_log_frame_skip(self, log_dir: str):
    logger = InventoryLogger(log_dir=log_dir, name="test_log8")
    logger.log_frame_skip(10, 3)
    log_path = os.path.join(log_dir, "test_log8.log")
    with open(log_path, "r") as f:
      content = f.read()
    assert "frame_skip" in content

  def test_timestamp_format(self, log_dir: str):
    logger = InventoryLogger(log_dir=log_dir, name="test_log9")
    logger.log_decision(0, {})
    log_path = os.path.join(log_dir, "test_log9.log")
    with open(log_path, "r") as f:
      content = f.read()
    assert "T" in content  # ISO format separator

  def test_rotation(self, log_dir: str):
    logger = InventoryLogger(
      log_dir=log_dir, max_bytes=500, backup_count=2, name="test_rotation"
    )
    for i in range(100):
      logger.log_decision(i, {"item": "x" * 50})
    files = glob.glob(os.path.join(log_dir, "test_rotation*"))
    assert len(files) >= 2  # at least main + 1 rotated
