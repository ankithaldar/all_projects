#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Direct tests for reject_oc_in_resolved.

Covers clean containers passing through and oc strings at
every depth raising, so a dropped walk step turns red.
'''

import pytest

from forecast_ml_framework.core.config_schema import (
  reject_oc_in_resolved,
)


def test_reject_oc_clean_nested_passes():
  '''Clean nested containers pass through.

  Fails if the walk rejects clean strings.
  '''
  plain = {'a': ['x', (1, 2)], 'b': 3, '${params:x}': 'y'}
  assert reject_oc_in_resolved(plain) is None


def test_reject_oc_clean_dollar_ref_passes():
  '''Non-oc interpolation passes.

  Fails if every interpolation is treated as oc.
  '''
  assert reject_oc_in_resolved({'a': '${params:x}'}) is None


def test_reject_oc_scalars_pass():
  '''Non-dict scalars pass.

  Fails if bare numbers or booleans are treated as oc.
  '''
  assert reject_oc_in_resolved('plain') is None
  assert reject_oc_in_resolved(42) is None
  assert reject_oc_in_resolved(None) is None
  assert reject_oc_in_resolved(3.5) is None


def test_reject_oc_empty_containers_pass():
  '''Empty dict and list pass.

  Fails if empty containers raise instead of passing.
  '''
  assert reject_oc_in_resolved({}) is None
  assert reject_oc_in_resolved([]) is None


def test_reject_oc_direct_string_raises():
  '''Direct oc string raises.

  Fails if walk misses top-level strings.
  '''
  with pytest.raises(ValueError, match='oc\\.'):
    reject_oc_in_resolved('${oc.env:SECRET}')


def test_reject_oc_list_element_raises():
  '''Oc string inside a list raises.

  Fails if the walk skips list elements.
  '''
  with pytest.raises(ValueError, match='oc\\.'):
    reject_oc_in_resolved(['ok', '${oc.create:dict}'])


def test_reject_oc_dict_value_raises():
  '''Oc string as a dict value raises.

  Fails if the walk skips dict values.
  '''
  with pytest.raises(ValueError, match='oc\\.'):
    reject_oc_in_resolved({'k': '${oc.env:X}'})


def test_reject_oc_tuple_element_raises():
  '''Oc string inside a tuple raises.

  Fails if the walk skips tuple elements.
  '''
  with pytest.raises(ValueError, match='oc\\.'):
    reject_oc_in_resolved(('ok', '${ OC.env:X}'))


def test_reject_oc_dict_key_raises():
  '''Oc string as a dict key raises.

  Fails if the walk skips dict keys.
  '''
  with pytest.raises(ValueError, match='oc\\.'):
    reject_oc_in_resolved({'${oc.env:X}': 'v'})


def test_reject_oc_case_variant_raises():
  '''Case-variant oc string raises.

  Fails if the pattern is case-sensitive.
  '''
  with pytest.raises(ValueError, match='oc\\.'):
    reject_oc_in_resolved('${OC.ENV:X}')


def test_reject_oc_without_dot_passes():
  '''Bare ${oc} without a dot passes.

  Fails if the pattern matches the bare word oc.
  '''
  assert reject_oc_in_resolved('${oc}') is None
