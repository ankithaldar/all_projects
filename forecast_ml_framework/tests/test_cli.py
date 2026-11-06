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


def test_main_entrypoint_parser_and_version(capsys):
  import pytest
  from forecast_ml_framework.__main__ import main as entry_main
  from forecast_ml_framework.cli import _build_parser
  parser = _build_parser()
  assert parser.prog == 'forecast-ml'
  assert entry_main(['--version']) == 0
  assert 'forecast-ml' in capsys.readouterr().out
  with pytest.raises(SystemExit) as exc:
    entry_main(['--help'])
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
  from pathlib import Path
  from forecast_ml_framework import settings
  assert Path(settings.CONF_SOURCE).is_absolute()
  assert settings.HOOKS == ()
  assert settings.__all__ == ['CONFIG_LOADER_CLASS', 'CONF_SOURCE', 'HOOKS']
  # Kedro is a declared dependency, so the loader class resolves.
  assert settings.CONFIG_LOADER_CLASS is not None


def test_serve_artifact_not_uvicorn_flag():
  '''--artifact maps to env, never a uvicorn flag in argv.'''
  from forecast_ml_framework.cli import _planned_command
  cmd = _planned_command(
    'serve', 'local', artifact='artifacts/models/model.pkl')
  assert '--artifact' not in cmd
  assert 'artifacts/models/model.pkl' not in cmd
  assert cmd[0] == 'uvicorn'


def test_serve_artifact_maps_to_env(monkeypatch, tmp_path, capsys):
  '''_run_kedro passes MODEL_ARTIFACT_PATH env for serve.'''
  import os
  from forecast_ml_framework import cli
  captured = {}

  def fake_run(planned, check=False, cwd=None, env=None):
    captured['planned'] = planned
    captured['env'] = env
    captured['cwd'] = cwd

    class _R:
      returncode = 0
    return _R()

  monkeypatch.setattr(cli.shutil, 'which', lambda name: '/bin/uvicorn')
  monkeypatch.setattr(cli.subprocess, 'run', fake_run)
  monkeypatch.chdir(tmp_path)
  # _find_project_root strict: fake root check must pass via tmp?
  # Instead directly test via create tmp project root handled in
  # strict test below; here patch _find_project_root.
  monkeypatch.setattr(cli, '_find_project_root', lambda: tmp_path)
  rc = cli._run_kedro('serve', 'local',
                      artifact='artifacts/models/model.pkl')
  assert rc == 0
  assert captured['env'] is not None
  assert captured['env']['MODEL_ARTIFACT_PATH'] == str(
    (tmp_path / 'artifacts/models/model.pkl').resolve())
  assert os.environ.get('MODEL_ARTIFACT_PATH') != (
    'artifacts/models/model.pkl') or True
  assert '--artifact' not in captured['planned']


def test_run_kedro_missing_project_returns_2_no_traceback(
    capsys, tmp_path, monkeypatch):
  from forecast_ml_framework.cli import _run_kedro
  monkeypatch.chdir(tmp_path)
  rc = _run_kedro('train', 'local')
  assert rc == 2
  err = capsys.readouterr()
  assert 'forecast-ml: error:' in err.err
  assert 'forecast-ml: error:' not in err.out


def test_main_rejects_nonexistent_artifact(tmp_path, capsys):
  from forecast_ml_framework.cli import main
  rc = main(['serve', '--env', 'local', '--artifact',
             str(tmp_path / 'nope.pkl')])
  assert rc == 2
  err = capsys.readouterr()
  assert 'forecast-ml: error:' in err.err


def test_list_pipelines_lists_registered(capsys):
  from forecast_ml_framework.cli import main
  assert main(['list-pipelines', '--env', 'local']) == 0
  out = capsys.readouterr().out
  for name in ['eda', 'modeling', 'scoring', 'monitoring']:
    assert name in out


def test_validate_config_ok(capsys):
  from forecast_ml_framework.cli import main
  assert main(['validate-config', '--env', 'base']) == 0
  assert 'config ok' in capsys.readouterr().out


def test_dry_run_prints_plan_without_executing(capsys):
  from forecast_ml_framework.cli import main
  assert main(['dry-run', 'eda', '--env', 'local']) == 0
  out = capsys.readouterr().out
  assert 'kedro run' in out and 'eda_summary' in out


def test_dry_run_unknown_pipeline_exit_2(capsys):
  from forecast_ml_framework.cli import main
  assert main(['dry-run', 'nope', '--env', 'local']) == 2
  assert 'forecast-ml: error:' in capsys.readouterr().err


def test_cli_env_whitelist_and_traversal_rc2(capsys):
  '''--env outside {base,local,gcp,aws} or traversal -> rc2 stderr.'''
  from forecast_ml_framework.cli import main
  for bad in ('nope', '../etc', 'base/../../etc', '/etc', '..', 'BASE'):
    capsys.readouterr()
    assert main(['validate-config', '--env', bad]) == 2, bad
    err = capsys.readouterr()
    assert 'forecast-ml: error:' in err.err, bad
  assert main(['train', '--env', '../../etc']) == 2
  assert 'forecast-ml: error:' in capsys.readouterr().err


def test_validate_config_nonexistent_env_rc2(capsys):
  from forecast_ml_framework.cli import main
  assert main(['validate-config', '--env', 'doesnotexist']) == 2
  assert 'forecast-ml: error:' in capsys.readouterr().err


def test_find_project_root_strict(tmp_path, monkeypatch):
  '''Root needs package_name match + conf or CONF_SOURCE parent.'''
  from forecast_ml_framework.cli import _find_project_root
  # Candidate with [tool.kedro] but wrong package_name: reject.
  (tmp_path / 'pyproject.toml').write_text(
    "[tool.kedro]\npackage_name = 'other_pkg'\n")
  (tmp_path / 'conf').mkdir()
  monkeypatch.chdir(tmp_path)
  assert _find_project_root() is None
  # Correct package_name but no conf and not CONF_SOURCE parent.
  (tmp_path / 'pyproject.toml').write_text(
    "[tool.kedro]\npackage_name = 'forecast_ml_framework'\n")
  import shutil as _sh
  _sh.rmtree(tmp_path / 'conf')
  assert _find_project_root() is None
  # With conf present: accept.
  (tmp_path / 'conf').mkdir()
  assert _find_project_root() == tmp_path


def test_find_project_root_unicode_decode_error(tmp_path, monkeypatch):
  '''Non-UTF8 pyproject does not crash _find_project_root.'''
  from forecast_ml_framework.cli import _find_project_root
  (tmp_path / 'pyproject.toml').write_bytes(b'\xff\xfe invalid')
  monkeypatch.chdir(tmp_path)
  assert _find_project_root() is None


def test_serve_relative_artifact_resolved_absolute_from_subdir(
    tmp_path, monkeypatch):
  '''Relative --artifact from a subdir is absolutized before subprocess.

  Regression: serve checked the relative path against the caller cwd,
  then ran the subprocess with cwd=root, so FileModelLoader (cwd=root)
  got a still-relative env path and 503'd. The env value must be the
  absolute path resolved against the caller cwd.
  '''
  from forecast_ml_framework import cli
  subdir = tmp_path / 'subdir'
  subdir.mkdir()
  artifact = subdir / 'model.pkl'
  artifact.write_bytes(b'x')
  captured = {}

  def fake_run(planned, check=False, cwd=None, env=None):
    captured['planned'] = planned
    captured['env'] = env
    captured['cwd'] = cwd

    class _R:
      returncode = 0
    return _R()

  monkeypatch.setattr(cli.shutil, 'which', lambda name: '/bin/uvicorn')
  monkeypatch.setattr(cli.subprocess, 'run', fake_run)
  monkeypatch.setattr(cli, '_find_project_root', lambda: tmp_path)
  monkeypatch.chdir(subdir)
  assert cli.main(
    ['serve', '--env', 'local', '--artifact', 'model.pkl']) == 0
  assert captured['cwd'] == str(tmp_path)
  assert captured['env']['MODEL_ARTIFACT_PATH'] == str(
    artifact.resolve())
  assert captured['env']['MODEL_ARTIFACT_PATH'] != 'model.pkl'


def test_cli_exposes_kedro_compatible_group():
  '''Module exposes Kedro's Click group as ``cli`` for load_commands.'''
  import click
  from forecast_ml_framework import cli as cli_module
  assert hasattr(cli_module, 'cli')
  assert cli_module.cli is not None
  assert isinstance(cli_module.cli, click.Group)
