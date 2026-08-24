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


def is_producer(agent_id: str) -> bool:
  '''Check whether an agent id denotes a producer role.'''
  return agent_id.endswith(PRODUCER_SUFFIX)


def is_consumer(agent_id: str) -> bool:
  '''Check whether an agent id denotes a consumer role.'''
  return agent_id.endswith(CONSUMER_SUFFIX)
