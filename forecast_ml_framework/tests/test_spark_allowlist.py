#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the Spark conf allowlist in ``spark_builder_config``.

Cycle-1 F-11 / cycle-2 antisec R-10: arbitrary Spark conf merged
from config (``spark.jars``, ``executorEnv``, ``extraClassPath``)
allowed remote code execution. ``spark_builder_config`` must now
apply an allowlist -- exactly the keys used in shipped conf
(``conf/base/parameters/platform.yml`` ``spark_params``) plus
documented resource/query tuning keys -- and raise ``ValueError``
naming any other key.
'''

import sys
import types

import pytest

from forecast_ml_framework.core.spark_factory import (
  SparkSessionFactory,
  spark_builder_config,
)

# Keys actually used in shipped conf (platform.yml spark_params.spark).
_SHIPPED_KEYS = {
  'spark.sql.shuffle.partitions': 1000,
  'spark.driver.memory': '4g',
  'spark.dynamicAllocation.enabled': 'true',
  'spark.dynamicAllocation.minExecutors': 2,
  'spark.dynamicAllocation.maxExecutors': 20,
  'spark.sql.adaptive.enabled': 'true',
  'spark.sql.adaptive.coalescePartitions.enabled': 'true',
}

# Documented tuning keys: resource sizing + query tuning only.
# ``spark.executor.memory`` is used by test_core.py; the rest are
# the standard executor/driver/query knobs for the 100M-row
# Dataproc/EMR sizing documented in README/infra READMEs.
_ALLOWED_TUNING_KEYS = {
  'spark.executor.memory': '2g',
  'spark.executor.cores': 4,
  'spark.executor.instances': 10,
  'spark.driver.cores': 4,
  'spark.sql.adaptive.skewJoin.enabled': 'true',
  'spark.sql.adaptive.localShuffleReader.enabled': 'true',
  'spark.dynamicAllocation.initialExecutors': 2,
  'spark.dynamicAllocation.executorIdleTimeout': '60s',
  'spark.sql.autoBroadcastJoinThreshold': 10485760,
  'spark.sql.broadcastTimeout': '300',
  'spark.sql.codegen.wholeStage': 'true',
  'spark.sql.files.maxPartitionBytes': 536870912,
}

# Keys that can execute / load arbitrary code if passed to Spark.
_DANGEROUS_KEYS = (
  'spark.jars',
  'spark.jars.packages',
  'foo.jars.bar',
  'spark.executorEnv.PYTHONPATH',
  'spark.executorEnv.MY_VAR',
  'spark.executor.extraClassPath',
  'spark.driver.extraClassPath',
  'spark.executor.extraJavaOptions',
  'spark.driver.extraJavaOptions',
  'spark.sql.classloader',
  'spark.executor.userClassPath',
  'spark.kubernetes.instance',
  'spark.kubernetes.authenticate.driver.serviceAccountName',
  'spark.kubernetes.driverEnv.X',
)

# Unknown keys: not shipped, not documented tuning, not dangerous
# by name -- still rejected (allowlist, not denylist).
_UNKNOWN_KEYS = (
  'a',
  'evil_key',
  'spark.sql.fooBar.baz',
  'spark.ui.port',
  'spark.yarn.am.memory',
  'spark.eventLog.dir',
)


def _via_params(key, value):
  '''Build config carrying one conf key via ``spark_params``.'''
  return {'spark_params': {key: value}}


def _via_config(key, value):
  '''Build config carrying one conf key via spark ``config``.'''
  return {'spark': {'config': {key: value}}}


@pytest.mark.parametrize('key', sorted(_DANGEROUS_KEYS))
def test_dangerous_key_rejected_via_params(key):
  '''Dangerous keys raise ValueError naming the key (params).'''
  with pytest.raises(ValueError) as excinfo:
    spark_builder_config(_via_params(key, 'evil'))
  assert key in str(excinfo.value)


@pytest.mark.parametrize('key', sorted(_DANGEROUS_KEYS))
def test_dangerous_key_rejected_via_config(key):
  '''Dangerous keys raise ValueError naming the key (config).'''
  with pytest.raises(ValueError) as excinfo:
    spark_builder_config(_via_config(key, 'evil'))
  assert key in str(excinfo.value)


@pytest.mark.parametrize('key', sorted(_DANGEROUS_KEYS))
def test_dangerous_key_rejected_direct_in_section(key):
  '''Dangerous keys raise even when placed directly in section.'''
  with pytest.raises(ValueError) as excinfo:
    spark_builder_config({'spark': {key: 'evil'}})
  assert key in str(excinfo.value)


@pytest.mark.parametrize('key', sorted(_UNKNOWN_KEYS))
def test_unknown_key_rejected(key):
  '''Unknown keys raise ValueError naming the key (allowlist).'''
  with pytest.raises(ValueError) as excinfo:
    spark_builder_config(_via_params(key, 'evil'))
  assert key in str(excinfo.value)


@pytest.mark.parametrize('key', sorted(_SHIPPED_KEYS))
def test_shipped_conf_key_allowed(key):
  '''Every shipped-conf key passes and lands in the merged conf.'''
  cfg = spark_builder_config(_via_params(key, _SHIPPED_KEYS[key]))
  assert cfg[key] == _SHIPPED_KEYS[key]


@pytest.mark.parametrize('key', sorted(_ALLOWED_TUNING_KEYS))
def test_documented_tuning_key_allowed(key):
  '''Documented resource/query tuning keys pass.'''
  cfg = spark_builder_config(_via_params(key, _ALLOWED_TUNING_KEYS[key]))
  assert cfg[key] == _ALLOWED_TUNING_KEYS[key]


def test_defaults_preserved():
  '''Autoscaling defaults + shuffle default are unchanged.'''
  cfg = spark_builder_config(None)
  assert cfg['spark.dynamicAllocation.enabled'] == 'true'
  assert cfg['spark.sql.adaptive.enabled'] == 'true'
  assert cfg['spark.sql.adaptive.coalescePartitions.enabled'] == 'true'
  assert cfg['spark.sql.shuffle.partitions'] == 200


def test_shuffle_partitions_alias_still_works():
  '''Non-conf alias ``shuffle_partitions`` still sizes shuffles.'''
  cfg = spark_builder_config({'spark_params': {'shuffle_partitions': 400}})
  assert cfg['spark.sql.shuffle.partitions'] == 400


def test_nested_spark_dict_unwrapped_and_validated():
  '''Shipped platform shape ``spark_params.spark`` is unwrapped.'''
  shipped = {
    'master': 'local[*]',
    'app_name': 'forecast-ml-framework',
    'spark': dict(_SHIPPED_KEYS),
    'jar_packages': [],
  }
  cfg = spark_builder_config({'spark_params': shipped})
  for key, value in _SHIPPED_KEYS.items():
    assert cfg[key] == value
  assert 'spark' not in cfg
  assert 'jar_packages' not in cfg


def test_nested_spark_dict_dangerous_key_rejected():
  '''Dangerous key inside nested ``spark`` dict is rejected.'''
  with pytest.raises(ValueError) as excinfo:
    spark_builder_config(
      {'spark_params': {'spark': {'spark.jars': '/tmp/evil.jar'}}})
  assert 'spark.jars' in str(excinfo.value)


def test_nonempty_jar_packages_rejected():
  '''Non-empty ``jar_packages`` (arbitrary packages) is rejected.'''
  with pytest.raises(ValueError) as excinfo:
    spark_builder_config({'spark_params': {'jar_packages': ['evil:pkg']}})
  assert 'jar_packages' in str(excinfo.value)


def test_get_spark_fail_closed_on_dangerous_key(monkeypatch):
  '''get_spark returns None, never calls builder.config, on RCE key.'''
  configs = {}

  class _Builder:
    '''Fake Spark builder recording config calls.'''

    def master(self, value):
      '''Record master, return self.'''
      configs['master'] = value
      return self

    def appName(self, value):
      '''Record app, return self.'''
      configs['app_name'] = value
      return self

    def config(self, key, value):
      '''Record config entry, return self.'''
      configs[key] = value
      return self

    def getOrCreate(self):
      '''Return sentinel session.'''
      return 'fake-session'

  fake_sql = types.ModuleType('pyspark.sql')
  fake_sql.SparkSession = types.SimpleNamespace(builder=_Builder())
  fake_pyspark = types.ModuleType('pyspark')
  fake_pyspark.sql = fake_sql
  monkeypatch.setitem(sys.modules, 'pyspark', fake_pyspark)
  monkeypatch.setitem(sys.modules, 'pyspark.sql', fake_sql)

  session = SparkSessionFactory.get_spark(
    {'spark': {'enabled': True, 'spark_params': {'spark.jars': 'evil'}}})
  assert session is None
  assert 'spark.jars' not in configs
