#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the logger parameters.

The six keys are the original's ``logger_params``.  They map onto wandb
as follows: ``project_name`` is the project, ``name`` is the run name,
``prefix_name`` is the **group**, ``tags`` are the tags, ``params`` is
the run config, and ``log_every`` is the logging cadence.
'''

from typing import Any

import pytest
from pydantic import ValidationError

from birdcall.config_params.logger_params import LoggerParams


def test_the_defaults_are_the_originals():
  '''The default project is the project of the competition.'''
  params = LoggerParams()
  assert params.project_name == 'birdcall'
  assert params.log_every == 10
  assert params.name == 'run'
  assert params.prefix_name == 'birdcall'


def test_an_omitted_tag_list_becomes_an_empty_one():
  '''The tags default to an empty list, not to None.'''
  params = LoggerParams()
  assert params.tags == []


def test_the_tags_are_kept():
  '''The tags are stored as they are given.'''
  params = LoggerParams(tags=[0, 'example_config', 'sed_dense121att'])
  assert params.tags == [0, 'example_config', 'sed_dense121att']


def test_the_hyper_parameter_payload_is_optional():
  '''The payload is derived from the configuration when it is omitted.'''
  assert LoggerParams().params is None
  payload: dict[str, Any] = {'lr': 0.001}
  assert LoggerParams(params=payload).params == payload


def test_an_unknown_key_is_refused():
  '''The configuration surface is closed.'''
  with pytest.raises(ValidationError):
    LoggerParams(project='birdcall')
