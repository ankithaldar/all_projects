#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the backend selection.

``backend`` says which of the two interchangeable backends a run
exercises, and ``backends`` holds the per-backend settings.
'''

import pytest

from birdcall.config_params.backend import BACKEND_SELECTION_PATH
from birdcall.config_params.backend import BackendSelection
from birdcall.config_params.backend import BackendSettings


def test_the_selection_defaults_to_both_backends():
  '''The default is to run the two backends in turn.'''
  selection = BackendSelection()
  assert selection.backend == 'both'
  assert selection.selected() == ['pl', 'tf']


def test_the_per_backend_defaults_are_filled_in():
  '''The per-backend settings default to the two training entry points.'''
  selection = BackendSelection()
  assert selection.backends['pl'].enabled is True
  assert selection.backends['pl'].entry_point == 'birdcall.train_pl'
  assert selection.backends['tf'].enabled is True
  assert selection.backends['tf'].entry_point == 'birdcall.train_tf'


def test_a_single_backend_selects_only_itself():
  '''``pl`` and ``tf`` select one backend each.'''
  assert BackendSelection(backend='pl').selected() == ['pl']
  assert BackendSelection(backend='tf').selected() == ['tf']


def test_an_unknown_backend_is_refused():
  '''The selection is a closed set of three names.'''
  with pytest.raises(ValueError):
    BackendSelection(backend='jax')


def test_a_disabled_backend_does_not_run():
  '''A backend that is selected but disabled does not run.'''
  selection = BackendSelection(backend='pl',
                               backends={'pl': BackendSettings(enabled=False)})
  assert selection.is_enabled('pl') is False
  assert selection.is_enabled('tf') is False


def test_an_enabled_backend_runs():
  '''A backend that is selected and enabled runs.'''
  selection = BackendSelection(backend='pl')
  assert selection.is_enabled('pl') is True
  assert selection.is_enabled('tf') is False


def test_the_backend_settings_default_to_enabled():
  '''A backend with no settings of its own is enabled.'''
  settings = BackendSettings()
  assert settings.enabled is True
  assert settings.entry_point == ''


def test_the_selection_file_is_the_packaged_one():
  '''The packaged selection lives next to the run directory.'''
  assert BACKEND_SELECTION_PATH.name == 'backend.yaml'
  assert BACKEND_SELECTION_PATH.parent.name == 'run'
