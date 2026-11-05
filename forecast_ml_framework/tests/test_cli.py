#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for CLI.'''


def test_cli_help_exits_zero(capsys):
  '''CLI --help exits zero.'''
  import pytest
  from forecast_ml_framework.cli import main
  with pytest.raises(SystemExit) as exc:
    main(['--help'])
  assert exc.value.code == 0
  assert 'train' in capsys.readouterr().out


def test_cli_subcommand_help_exits_zero():
  '''CLI subcommand --help exits zero.'''
  import pytest
  from forecast_ml_framework.cli import main
  with pytest.raises(SystemExit) as exc:
    main(['train', '--help'])
  assert exc.value.code == 0


def test_cli_command_mapping():
  '''train->modeling, predict->scoring, run->default (no pipeline).'''
  from forecast_ml_framework.cli import _planned_command
  assert _planned_command('train', 'local') == [
    'kedro', 'run', '--pipeline', 'modeling', '--env', 'local']
  assert _planned_command('predict', 'local') == [
    'kedro', 'run', '--pipeline', 'scoring', '--env', 'local']
  assert _planned_command('run', 'local') == [
    'kedro', 'run', '--env', 'local']


def test_cli_serve_plans_uvicorn_command():
  '''serve maps to a uvicorn serving command, not a kedro pipeline.'''
  from forecast_ml_framework.cli import _planned_command
  command = _planned_command('serve', 'local')
  assert command[0] == 'uvicorn'
  assert 'forecast_ml_framework.serving.app:app' in command
  assert '--host' in command and '127.0.0.1' in command
  assert '--port' in command and '8000' in command


def test_cli_serve_prints_uvicorn_command(capsys, monkeypatch):
  '''CLI serve prints the uvicorn command without kedro.'''
  from forecast_ml_framework import cli
  monkeypatch.setattr(cli.shutil, 'which', lambda name: None)
  assert cli.main(['serve', '--env', 'local']) == 0
  out = capsys.readouterr().out
  assert 'uvicorn' in out
  assert 'forecast_ml_framework.serving.app:app' in out


def test_cli_train_prints_planned_command(capsys, monkeypatch):
  '''CLI train does not crash without kedro.'''
  from forecast_ml_framework import cli
  monkeypatch.setattr(cli.shutil, 'which', lambda name: None)
  code = cli.main(['train', '--env', 'local'])
  assert code == 0
  out = capsys.readouterr().out
  assert 'kedro' in out or 'train' in out


# --- Coverage: __main__ entry point + settings smoke ---


def test_main_entrypoint_parser_and_version():
  import pytest
  from forecast_ml_framework.__main__ import _build_parser, main
  parser = _build_parser()
  assert parser.prog == 'forecast-ml'
  with pytest.raises(SystemExit) as exc:
    parser.parse_args(['--version'])
  assert exc.value.code == 0
  with pytest.raises(SystemExit) as exc:
    main(['--help'])
  assert exc.value.code == 0


def test_main_entrypoint_fallback_help_when_cli_missing(
    capsys, monkeypatch):
  import sys
  monkeypatch.setitem(
      sys.modules, 'forecast_ml_framework.cli', None)
  monkeypatch.setitem(sys.modules, 'kedro.framework.cli', None)
  from forecast_ml_framework.__main__ import main
  assert main([]) == 0
  assert 'forecast' in capsys.readouterr().out.lower()


def test_main_entrypoint_fallback_to_kedro(monkeypatch):
  import sys
  import types
  monkeypatch.setitem(
      sys.modules, 'forecast_ml_framework.cli', None)
  kedro_mod = types.ModuleType('kedro')
  framework_mod = types.ModuleType('kedro.framework')
  cli_mod = types.ModuleType('kedro.framework.cli')
  cli_mod.main = lambda: 42
  kedro_mod.framework = framework_mod
  framework_mod.cli = cli_mod
  monkeypatch.setitem(sys.modules, 'kedro', kedro_mod)
  monkeypatch.setitem(sys.modules, 'kedro.framework', framework_mod)
  monkeypatch.setitem(sys.modules, 'kedro.framework.cli', cli_mod)
  from forecast_ml_framework.__main__ import main
  assert main([]) == 42


def test_settings_import_smoke():
  from forecast_ml_framework import settings
  assert settings.CONF_SOURCE == 'conf'
  assert settings.HOOKS == ()
  assert settings.__all__ == ['CONFIG_LOADER_CLASS', 'CONF_SOURCE', 'HOOKS']
  # Kedro is a declared dependency, so the loader class resolves.
  assert settings.CONFIG_LOADER_CLASS is not None
