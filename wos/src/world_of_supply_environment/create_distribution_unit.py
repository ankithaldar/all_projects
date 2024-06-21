#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Create distribution unit with given configuration.'''

# imports
#   script imports
from .distribution_unit import DistributionUnit
from .transport import Transport
# imports


def create_distribution_unit(facility, config):
  return DistributionUnit(
    facility=facility,
    fleet_size=config.fleet_size,
    distribution_economy=DistributionUnit.Economy(
      wrong_order_penatly = config.wrong_order_penatly,
      pending_order_penalty = config.pending_order_penalty
    ),
    transport_economy=Transport.Economy(config.unit_transport_cost)
  )
