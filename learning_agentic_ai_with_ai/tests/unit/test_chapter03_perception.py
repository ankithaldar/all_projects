#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: the Perception layer.'''


from __future__ import annotations

from chapter03_cognition.perception import PerceptionLayer

RETAIL_TASK = (
  'Check which items are below their reorder point and inspect the '
  'sales trend of the worst one.'
)
TELECOM_TASK = (
  'Check which cell sites are degraded and dispatch technician T-06.'
)
RESTOCK_TASK = (
  'Restock store S01 with 9000 units of R-101 immediately.'
)


def perceive(task: str):
  '''Build a percept for a task.

  Args:
    task: Task text.

  Returns:
    Percept.
  '''
  return PerceptionLayer().perceive(task, 'it-perception')


class TestDomainAndEntities:
  '''Domain classification and identifier extraction.'''

  def test_retail_domain(self) -> None:
    assert perceive(RETAIL_TASK).signals.domain == 'retail'

  def test_telecom_domain(self) -> None:
    assert perceive(TELECOM_TASK).signals.domain == 'telecom'

  def test_unknown_domain(self) -> None:
    assert perceive('Say hello.').signals.domain == 'unknown'

  def test_entities(self) -> None:
    percept = perceive(RESTOCK_TASK)
    assert percept.entities['store_ids'] == ['S01']
    assert percept.entities['skus'] == ['R-101']
    assert '9000' in percept.entities['quantities']

  def test_sku_fragments_not_quantities(self) -> None:
    quantities = perceive(RESTOCK_TASK).entities['quantities']
    assert '101' not in quantities


class TestSignals:
  '''Signal extraction drives strategy selection.'''

  def test_write_intent_and_risk(self) -> None:
    signals = perceive(RESTOCK_TASK).signals
    assert signals.write_intent is True
    assert signals.risk > 0

  def test_read_task_has_no_write_intent(self) -> None:
    signals = perceive(RETAIL_TASK).signals
    assert signals.write_intent is False

  def test_data_need_high_for_live_lookups(self) -> None:
    assert perceive(RETAIL_TASK).signals.data_need > 0

  def test_ambiguity_detected(self) -> None:
    percept = perceive('Investigate why the site is degraded and find the '
                       'root cause.')
    assert percept.signals.ambiguity > 0

  def test_complexity_is_bounded(self) -> None:
    signals = perceive(RETAIL_TASK).signals
    assert 0.0 <= signals.complexity <= 1.0


class TestCapabilities:
  '''Capability inference maps tasks to required tools.'''

  def test_retail_capabilities(self) -> None:
    capabilities = perceive(RETAIL_TASK).required_capabilities
    assert 'sales_read' in capabilities
    assert 'inventory_read' in capabilities

  def test_write_capability(self) -> None:
    capabilities = perceive(RESTOCK_TASK).required_capabilities
    assert 'restock_write' in capabilities

  def test_telecom_capabilities(self) -> None:
    capabilities = perceive(TELECOM_TASK).required_capabilities
    assert 'site_read' in capabilities
    assert 'dispatch_write' in capabilities


class TestNormalization:
  '''Task text is sanitized before it reaches prompts.'''

  def test_control_characters_removed(self) -> None:
    percept = perceive('Check stock\x00\x07 now')
    assert '\x00' not in percept.task
    assert '\x07' not in percept.task

  def test_whitespace_collapsed(self) -> None:
    percept = perceive('Check   stock\n\nnow')
    assert percept.task == 'Check stock now'
