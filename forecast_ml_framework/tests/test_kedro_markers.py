#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for Kedro project markers in pyproject.toml.'''

import tomllib
from pathlib import Path


def _load_pyproject():
  path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
  with path.open('rb') as fh:
    return tomllib.load(fh)


def test_kedro_tool_section():
  '''[tool.kedro] carries only Kedro 1.7 ProjectMetadata keys.

  NOTE: project_version was removed — kedro 1.7 ProjectMetadata
  rejects unexpected keys and `kedro run` exits 1 with
  "unexpected keys" if it is present. Version lives in
  [project] version, not [tool.kedro].
  '''
  data = _load_pyproject()
  kedro = data['tool']['kedro']
  assert kedro['package_name'] == 'forecast_ml_framework'
  assert kedro['project_name'] == 'forecast-ml-framework'
  assert kedro['kedro_init_version'] == '1.7.0'
  assert 'project_version' not in kedro
  allowed = {'package_name', 'project_name', 'kedro_init_version'}
  assert set(kedro) <= allowed, f'unexpected keys: {set(kedro) - allowed}'


def test_console_scripts_alias():
  '''forecast-ml-framework alias exists alongside forecast-ml.'''
  data = _load_pyproject()
  scripts = data['project']['scripts']
  assert scripts['forecast-ml'] == (
      'forecast_ml_framework.__main__:main')
  assert scripts['forecast-ml-framework'] == (
      'forecast_ml_framework.__main__:main')
  assert data['project']['name'] == 'forecast-ml-framework'
  assert data['project']['version'] == '0.1.0'
