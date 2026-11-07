#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for settings.'''

from pathlib import Path


def test_conf_source_is_absolute_and_exists():
  '''Conf source is absolute and exists.'''
  from forecast_ml_framework.settings import CONF_SOURCE
  assert Path(CONF_SOURCE).is_absolute()
  assert (Path(CONF_SOURCE) / 'base' / 'catalog.yml').exists()


def test_conf_source_independent_of_cwd(tmp_path, monkeypatch):
  '''Conf source independent of cwd.'''
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


def _restore_oc_env_resolver():
  '''Restore the stock oc.env resolver after loader init.'''
  from omegaconf import OmegaConf as _oc
  from omegaconf.resolvers import oc as _oc_resolvers
  if not _oc.has_resolver('oc.env'):
    _oc.register_new_resolver('oc.env', _oc_resolvers.env)


def test_merge_and_resolve_clean_merge(tmp_path):
  '''Clean multi-file config merges and resolves.

  Fails if _merge_and_resolve rejects clean configs or
  drops a file's keys during the merge.
  '''
  import pytest
  pytest.importorskip('kedro')
  from forecast_ml_framework import settings
  loader_cls = settings.CONFIG_LOADER_CLASS
  assert loader_cls is not None
  conf = tmp_path / 'conf'
  base = conf / 'base' / 'parameters'
  base.mkdir(parents=True)
  (base / 'a.yml').write_text('left: 1\n')
  (base / 'b.yml').write_text('right: 2\n')
  (conf / 'base' / 'catalog.yml').write_text('{}\n')
  try:
    loader = loader_cls(str(conf), env='base', base_env='base')
    merged = dict(loader['parameters'])
    assert merged['left'] == 1
    assert merged['right'] == 2
  finally:
    _restore_oc_env_resolver()


def test_merge_and_resolve_cross_file_oc_rejected(
    tmp_path, monkeypatch):
  '''Cross-file recombined oc is rejected during resolve.

  Fails if fragments split across files resolve to oc.env
  instead of raising ValueError.
  '''
  import pytest
  pytest.importorskip('kedro')
  monkeypatch.setenv('SECRET', 'leaked-secret-value')
  from forecast_ml_framework import settings
  loader_cls = settings.CONFIG_LOADER_CLASS
  assert loader_cls is not None
  conf = tmp_path / 'conf'
  base = conf / 'base' / 'parameters'
  base.mkdir(parents=True)
  (base / 'a.yml').write_text('frag_a: \'$\'\n')
  (base / 'b.yml').write_text(
    'frag_b: \'{oc.env:SECRET}\'\njoined: \'${frag_a}${frag_b}\'\n')
  (conf / 'base' / 'catalog.yml').write_text('{}\n')
  try:
    loader = loader_cls(str(conf), env='base', base_env='base')
    with pytest.raises(ValueError, match='oc\\.'):
      loader['parameters']
  finally:
    _restore_oc_env_resolver()


def test_credentials_path_oc_rejected():
  '''Credentials resolver refuses oc interpolations.

  Fails if _resolve_environment_variables lets oc.env
  through on the credentials path.
  '''
  import pytest
  from omegaconf import OmegaConf
  from forecast_ml_framework import settings
  cls = settings.CONFIG_LOADER_CLASS
  assert cls is not None
  assert cls._resolve_environment_variables(
    OmegaConf.create({'a': 'clean'})) is None
  assert cls._resolve_environment_variables(
    OmegaConf.create({'a': '${params:x}'})) is None
  with pytest.raises(ValueError, match='oc\\.'):
    cls._resolve_environment_variables(
      OmegaConf.create({'a': '${oc.env:SECRET}'}))
