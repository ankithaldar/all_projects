#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Agent-id conventions linking RL agents to facilities.

Every facility is controlled by two agents: a producer (suffix ``p``) that
sets price and production rate, and a consumer (suffix ``c``) that decides
purchases.
'''

from __future__ import annotations

PRODUCER_SUFFIX = 'p'
CONSUMER_SUFFIX = 'c'

RETAILER_FACILITY_TYPES = frozenset({'RetailerCell'})
FACILITY_PRODUCER_PRICE_INDEX = {
  'SteelFactoryCell': 0,
  'LumberFactoryCell': 0,
  'ToyFactoryCell': 2,
  'WarehouseCell': 4,
  'RetailerCell': 6,
}


def producer_agent_id(facility_id: str) -> str:
  '''Build the producer agent id of a facility.

  Args:
    facility_id: Facility string id.

  Returns:
    str: Facility id with the producer suffix appended.
  '''
  return facility_id + PRODUCER_SUFFIX


def consumer_agent_id(facility_id: str) -> str:
  '''Build the consumer agent id of a facility.

  Args:
    facility_id: Facility string id.

  Returns:
    str: Facility id with the consumer suffix appended.
  '''
  return facility_id + CONSUMER_SUFFIX


def facility_id_of(agent_id: str) -> str:
  '''Strip the role suffix from an agent id.

  Args:
    agent_id: Agent id ending in the producer or consumer suffix.

  Returns:
    str: Underlying facility id.
  '''
  return agent_id[:-1]


def facility_type_name_of_id(facility_id: str) -> str:
  '''Extract the facility class name from a facility id.

  Facility ids have the shape ``ClassName_idNum``, so the type is the
  portion before the last underscore.

  Args:
    facility_id: Facility string id.

  Returns:
    str: Facility class name.
  '''
  if '_' in facility_id:
    return facility_id.rsplit('_', 1)[0]
  return facility_id


def facility_type_name(facility) -> str:
  '''Return the facility class name of a facility object.

  Args:
    facility: Facility object carrying a class.

  Returns:
    str: Facility class name.
  '''
  return type(facility).__name__


def is_retailer(facility_id: str) -> bool:
  '''Check whether a facility id denotes a retailer.

  Args:
    facility_id: Facility string id.

  Returns:
    bool: True when the facility type is a registered retailer.
  '''
  return facility_type_name_of_id(facility_id) in RETAILER_FACILITY_TYPES


def producer_price_index(facility_type: str) -> int:
  '''Look up the scripted producer price index of a facility type.

  Args:
    facility_type: Facility class name.

  Returns:
    int: Price index, zero for unregistered types.
  '''
  return FACILITY_PRODUCER_PRICE_INDEX.get(facility_type, 0)


def is_producer(agent_id: str) -> bool:
  '''Check whether an agent id denotes a producer role.

  Args:
    agent_id: Agent string id.

  Returns:
    bool: True when the id ends in the producer suffix.
  '''
  return agent_id.endswith(PRODUCER_SUFFIX)


def is_consumer(agent_id: str) -> bool:
  '''Check whether an agent id denotes a consumer role.

  Args:
    agent_id: Agent string id.

  Returns:
    bool: True when the id ends in the consumer suffix.
  '''
  return agent_id.endswith(CONSUMER_SUFFIX)
