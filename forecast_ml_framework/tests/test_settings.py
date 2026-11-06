#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for settings.'''

from pathlib import Path


def test_conf_source_is_absolute_and_exists():
  from forecast_ml_framework.settings import CONF_SOURCE
  assert Path(CONF_SOURCE).is_absolute()
  assert (Path(CONF_SOURCE) / 'base' / 'catalog.yml').exists()


def test_conf_source_independent_of_cwd(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  import importlib
  import forecast_ml_framework.settings as settings
  importlib.reload(settings)
  assert Path(settings.CONF_SOURCE).is_absolute()
  assert (Path(settings.CONF_SOURCE) / 'base' / 'catalog.yml').exists()


def test_settings_oc_error_without_node_content():
  '''ValueError from _reject_oc must not echo node content.'''
  import pytest
  from forecast_ml_framework import settings
  cls = settings.CONFIG_LOADER_CLASS
  assert cls is not None
  secret = 'leaked-secret-node-content-xyz'
  with pytest.raises(ValueError) as excinfo:
    cls._reject_oc({'x': '${oc.env:SECRET} ' + secret})
  assert secret not in str(excinfo.value)
  assert '${oc.env' not in str(excinfo.value)
  with pytest.raises(ValueError) as excinfo2:
    cls._reject_oc(['${oc.create:dict} ' + secret])
  assert secret not in str(excinfo2.value)
