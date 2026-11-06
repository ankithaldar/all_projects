#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Security tests for S-01 pickle load and S-02 path traversal.'''

import os
import pickle
import sys
import types
from pathlib import Path

import pytest

from truck_carton.evaluation.visualizer import GridRenderer
from truck_carton.logging_config import (
  is_trusted_load_allowed,
  load_model_safely,
  resolve_contained_path,
  setup_logging,
  validate_model_path,
)


def _evil_pickle(marker: Path) -> bytes:
  '''Craft a pickle that touches marker when unpickled.'''

  class Evil:
    '''Unpickling runs a shell command.'''

    def __reduce__(self):
      '''Reduce.'''
      return (os.system, (f'touch {marker}',))

  return pickle.dumps(Evil())


def _fake_loader_factory(calls: list):
  '''Return a loader that records calls, never unpickles.'''

  def loader(path):
    '''Record and return a sentinel model.'''
    calls.append(Path(path))
    return 'model'

  return loader


# --- resolve_contained_path (S-02 core) ---


def test_resolve_normal_relative(tmp_path, monkeypatch):
  '''Normal relative path resolves under base.'''
  monkeypatch.chdir(tmp_path)
  out = resolve_contained_path('./my_run')
  assert out == (tmp_path / 'my_run').resolve()


def test_resolve_rejects_dotdot_escape(tmp_path, monkeypatch):
  '''Traversal with .. outside base is rejected.'''
  monkeypatch.chdir(tmp_path)
  with pytest.raises(ValueError, match='outside|traversal|escapes'):
    resolve_contained_path('../evil')


def test_resolve_rejects_nested_dotdot_escape(tmp_path, monkeypatch):
  '''Nested a/../../escape is rejected.'''
  monkeypatch.chdir(tmp_path)
  (tmp_path / 'a').mkdir()
  with pytest.raises(ValueError):
    resolve_contained_path('a/../../evil')


def test_resolve_allows_dotdot_staying_inside(tmp_path, monkeypatch):
  '''a/../b stays inside base and is allowed.'''
  monkeypatch.chdir(tmp_path)
  out = resolve_contained_path('a/../b')
  assert out == (tmp_path / 'b').resolve()


def test_resolve_rejects_absolute_outside(tmp_path, monkeypatch):
  '''Absolute path outside base is rejected.'''
  monkeypatch.chdir(tmp_path)
  with pytest.raises(ValueError):
    resolve_contained_path('/etc/passwd')


def test_resolve_allows_absolute_inside(tmp_path, monkeypatch):
  '''Absolute path inside base is allowed.'''
  monkeypatch.chdir(tmp_path)
  out = resolve_contained_path(str(tmp_path / 'run'))
  assert out == (tmp_path / 'run').resolve()


def test_resolve_rejects_symlink_escape(tmp_path, monkeypatch):
  '''Symlink inside base pointing outside is rejected.'''
  monkeypatch.chdir(tmp_path)
  outside = tmp_path.parent / 'outside-secret'
  outside.mkdir(exist_ok=True)
  link = tmp_path / 'link'
  try:
    link.symlink_to(outside, target_is_directory=True)
  except OSError:
    pytest.skip('symlinks not supported')
  with pytest.raises(ValueError):
    resolve_contained_path('link/evil')


def test_resolve_explicit_base_dir(tmp_path):
  '''Explicit base_dir overrides cwd.'''
  base = tmp_path / 'base'
  base.mkdir()
  out = resolve_contained_path('./run', base_dir=base)
  assert out == (base / 'run').resolve()
  with pytest.raises(ValueError):
    resolve_contained_path('../evil', base_dir=base)


def test_resolve_rejects_empty(tmp_path, monkeypatch):
  '''Empty path is rejected.'''
  monkeypatch.chdir(tmp_path)
  with pytest.raises(ValueError):
    resolve_contained_path('')


# --- safe model loading (S-01) ---


def test_load_rejects_traversal_without_executing(tmp_path, monkeypatch):
  '''Traversal model path rejected before any load/exec.'''
  monkeypatch.chdir(tmp_path)
  marker = tmp_path / 'pwned'
  evil = tmp_path / 'evil.zip'
  evil.write_bytes(_evil_pickle(marker))
  calls: list = []
  with pytest.raises(ValueError):
    load_model_safely(
      '../evil.zip',
      allow_trusted=True,
      loader=_fake_loader_factory(calls),
    )
  assert not calls
  assert not marker.exists()


def test_load_rejects_outside_allowlist_even_trusted(tmp_path, monkeypatch):
  '''Path outside allowlisted dirs rejected even if trusted.'''
  monkeypatch.chdir(tmp_path)
  (tmp_path / 'output').mkdir()
  outside = tmp_path.parent / 'elsewhere.zip'
  outside.write_bytes(b'not-a-model')
  calls: list = []
  with pytest.raises(ValueError):
    load_model_safely(
      str(outside),
      allow_trusted=True,
      loader=_fake_loader_factory(calls),
    )
  assert not calls


def test_load_rejects_untrusted_pickle_not_executed(tmp_path, monkeypatch):
  '''Contained malicious pickle rejected without opt-in, not run.'''
  monkeypatch.chdir(tmp_path)
  model_dir = tmp_path / 'output' / 'models'
  model_dir.mkdir(parents=True)
  marker = tmp_path / 'pwned'
  (model_dir / 'best_model.zip').write_bytes(_evil_pickle(marker))
  calls: list = []
  with pytest.raises(ValueError, match='trusted|TRUSTED|opt-in'):
    load_model_safely(
      './output/models/best_model.zip',
      loader=_fake_loader_factory(calls),
    )
  assert not calls
  assert not marker.exists()


def test_load_allows_trusted_contained(tmp_path, monkeypatch):
  '''Contained path + explicit opt-in reaches the loader.'''
  monkeypatch.chdir(tmp_path)
  model_dir = tmp_path / 'output' / 'models'
  model_dir.mkdir(parents=True)
  (model_dir / 'best_model.zip').write_bytes(b'fake')
  calls: list = []
  result = load_model_safely(
    './output/models/best_model.zip',
    allow_trusted=True,
    loader=_fake_loader_factory(calls),
  )
  assert result == 'model'
  assert calls == [(model_dir / 'best_model.zip').resolve()]


def test_load_env_opt_in(tmp_path, monkeypatch):
  '''Env var TRUCK_CARTON_ALLOW_TRUSTED_LOAD is an opt-in.'''
  monkeypatch.chdir(tmp_path)
  monkeypatch.setenv('TRUCK_CARTON_ALLOW_TRUSTED_LOAD', '1')
  assert is_trusted_load_allowed() is True
  model_dir = tmp_path / 'output' / 'models'
  model_dir.mkdir(parents=True)
  (model_dir / 'best_model.zip').write_bytes(b'fake')
  calls: list = []
  result = load_model_safely(
    './output/models/best_model.zip',
    loader=_fake_loader_factory(calls),
  )
  assert result == 'model'


def test_load_rejects_non_zip_extension(tmp_path, monkeypatch):
  '''Only SB3 .zip models load; raw .pkl is rejected.'''
  monkeypatch.chdir(tmp_path)
  model_dir = tmp_path / 'output' / 'models'
  model_dir.mkdir(parents=True)
  (model_dir / 'evil.pkl').write_bytes(b'fake')
  calls: list = []
  with pytest.raises(ValueError):
    load_model_safely(
      './output/models/evil.pkl',
      allow_trusted=True,
      loader=_fake_loader_factory(calls),
    )
  assert not calls


def test_load_rejects_symlink_escape(tmp_path, monkeypatch):
  '''Symlinked model inside allowlist pointing out is rejected.'''
  monkeypatch.chdir(tmp_path)
  model_dir = tmp_path / 'output' / 'models'
  model_dir.mkdir(parents=True)
  outside = tmp_path.parent / 'outside.zip'
  outside.write_bytes(b'fake')
  link = model_dir / 'best_model.zip'
  try:
    link.symlink_to(outside)
  except OSError:
    pytest.skip('symlinks not supported')
  calls: list = []
  with pytest.raises(ValueError):
    load_model_safely(
      './output/models/best_model.zip',
      allow_trusted=True,
      loader=_fake_loader_factory(calls),
    )
  assert not calls


def test_load_custom_allow_dir(tmp_path):
  '''Configurable allowlist dir is honoured.'''
  allow = tmp_path / 'custom'
  allow.mkdir()
  (allow / 'm.zip').write_bytes(b'fake')
  calls: list = []
  result = load_model_safely(
    str(allow / 'm.zip'),
    allow_dirs=[allow],
    allow_trusted=True,
    loader=_fake_loader_factory(calls),
  )
  assert result == 'model'
  other = tmp_path / 'other'
  other.mkdir()
  with pytest.raises(ValueError):
    validate_model_path(str(allow / 'm.zip'), allow_dirs=[other])


def test_validate_model_path_normal(tmp_path, monkeypatch):
  '''validate_model_path returns resolved path for normal input.'''
  monkeypatch.chdir(tmp_path)
  p = validate_model_path('./output/models/best_model.zip')
  assert p == (tmp_path / 'output' / 'models' / 'best_model.zip')


# --- setup_logging S-02 ---


def test_setup_logging_rejects_traversal(tmp_path, monkeypatch):
  '''log_file with .. escape raises ValueError.'''
  monkeypatch.chdir(tmp_path)
  with pytest.raises(ValueError):
    setup_logging(log_file='../evil.log')
  with pytest.raises(ValueError):
    setup_logging(json_log_file='/etc/passwd')


def test_setup_logging_allows_normal(tmp_path, monkeypatch):
  '''Normal log file under cwd still works.'''
  import truck_carton.logging_config as lc

  monkeypatch.chdir(tmp_path)
  lc._LOGGERS_CONFIGURED = False
  try:
    setup_logging(log_file='./logs/app.log')
    assert (tmp_path / 'logs' / 'app.log').exists()
  finally:
    lc._LOGGERS_CONFIGURED = False


# --- GridRenderer.save_gif S-02 ---


def test_save_gif_rejects_traversal(tmp_path, monkeypatch):
  '''save_gif rejects .. and absolute escapes.'''
  monkeypatch.chdir(tmp_path)
  renderer = GridRenderer()
  with pytest.raises(ValueError):
    renderer.save_gif('../evil.gif')
  with pytest.raises(ValueError):
    renderer.save_gif('/tmp/evil.gif')


def test_save_gif_allows_normal(tmp_path, monkeypatch):
  '''Normal relative gif path is accepted (no frames = no-op).'''
  monkeypatch.chdir(tmp_path)
  renderer = GridRenderer()
  renderer.save_gif('./out/anim.gif')
  assert not (tmp_path / 'out' / 'anim.gif').exists()  # no frames


# --- Trainer S-02 (SB3 stubbed; env not installed) ---


def _stub_sb3(monkeypatch):
  '''Install minimal SB3 stubs so trainer imports.'''

  class Dummy:
    '''Accepts any args.'''

    def __init__(self, *args, **kwargs):
      '''Init.'''
      self.args = args
      self.kwargs = kwargs

  sb3 = types.ModuleType('sb3_contrib')
  sb3.MaskablePPO = Dummy
  sbc = types.ModuleType('stable_baselines3.common.callbacks')
  sbc.BaseCallback = Dummy
  sbc.EvalCallback = Dummy
  sbm = types.ModuleType('stable_baselines3.common.monitor')
  sbm.Monitor = Dummy
  monkeypatch.setitem(sys.modules, 'sb3_contrib', sb3)
  monkeypatch.setitem(
    sys.modules, 'stable_baselines3', types.ModuleType('sb3')
  )
  monkeypatch.setitem(
    sys.modules, 'stable_baselines3.common', types.ModuleType('c')
  )
  monkeypatch.setitem(sys.modules, 'stable_baselines3.common.callbacks', sbc)
  monkeypatch.setitem(sys.modules, 'stable_baselines3.common.monitor', sbm)


def test_trainer_rejects_traversal_output(tmp_path, monkeypatch):
  '''Trainer rejects output_dir escaping its base dir.'''
  _stub_sb3(monkeypatch)
  from truck_carton.config import AppConfig
  from truck_carton.training.trainer import Trainer

  with pytest.raises(ValueError):
    Trainer(AppConfig(), output_dir='../evil', base_dir=tmp_path)
  with pytest.raises(ValueError):
    Trainer(AppConfig(), output_dir='/etc/evil', base_dir=tmp_path)
  assert not (tmp_path.parent / 'evil').exists()


def test_trainer_allows_normal_output(tmp_path, monkeypatch):
  '''Trainer accepts ./my_run under its base dir (CLI compat).'''
  _stub_sb3(monkeypatch)
  import truck_carton.logging_config as lc

  lc._LOGGERS_CONFIGURED = False
  try:
    from truck_carton.config import AppConfig
    from truck_carton.training.trainer import Trainer

    trainer = Trainer(
      AppConfig(), output_dir='./my_run', base_dir=tmp_path
    )
    assert (tmp_path / 'my_run').is_dir()
    assert trainer is not None
  finally:
    lc._LOGGERS_CONFIGURED = False
