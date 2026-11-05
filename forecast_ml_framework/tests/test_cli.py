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


def test_cli_train_prints_planned_command(capsys, monkeypatch):
  '''CLI train does not crash without kedro.'''
  from forecast_ml_framework import cli
  monkeypatch.setattr(cli.shutil, 'which', lambda name: None)
  code = cli.main(['train', '--env', 'local'])
  assert code == 0
  out = capsys.readouterr().out
  assert 'kedro' in out or 'train' in out
