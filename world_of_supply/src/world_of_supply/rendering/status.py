#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Human-readable world status formatting for logs and status panels.'''

from __future__ import annotations

from functools import singledispatchmethod

from world_of_supply.distribution import DistributionUnit, Order, Transport
from world_of_supply.economy import BalanceSheet
from world_of_supply.facility import FacilityCell
from world_of_supply.storage import StorageUnit
from world_of_supply.transport import TransportState
from world_of_supply.world import World


def ascii_progress_bar(done: float, limit: float, bar_length: int = 15) -> str:
  '''Render a textual progress bar.

  Args:
    done: Current value.
    limit: Maximum value.
    bar_length: Width of the bar in characters.

  Returns:
    str: Bar like ``======------- 6/15``.
  '''
  done_chars = round(min(done, limit) / limit * bar_length) if limit else 0
  return '=' * done_chars + '-' * (bar_length - done_chars) + f' {done}/{limit}'


def counter_to_dict(counter: dict) -> dict:
  '''Drop zero entries from a stock/open-order mapping.

  Args:
    counter: Mapping of product id to count.

  Returns:
    dict: Only strictly positive entries.
  '''
  return {key: value for key, value in counter.items() if value}


class WorldStatusFormatter:
  '''Builds nested plain-data structures describing the world state.

  The output contains only strings, numbers, lists and dicts, so it can be
  dumped as YAML/JSON or printed directly.
  '''

  @singledispatchmethod
  def status(self, obj):
    '''Dispatch to the type-specific formatter.

    Args:
      obj: A world, facility, transport, or storage object.

    Returns:
      list: Nested human-readable status lines.
    '''
    raise NotImplementedError(f'No formatter registered for {type(obj)}')

  @status.register
  def _(self, world: World) -> list:
    '''Format the whole world.

    Args:
      world: World snapshot.

    Returns:
      list: Header entry plus one entry per facility.
    '''
    status = [
        [
            'World:',
            [f'Time step: {world.time_step}', f'Global balance: {world.economy.global_balance()}'],
        ]
    ]
    status.extend(self.status(facility) for facility in world.facilities.values())
    return status

  @status.register
  def _(self, facility: FacilityCell) -> list:
    '''Format one facility including its installed units.

    Args:
      facility: Facility snapshot.

    Returns:
      list: Identity line followed by unit sub-statuses.
    '''
    substatus: list = [f'Balance: {facility.economy.total_balance}']
    if isinstance(facility.distribution, DistributionUnit):
      substatus.append(
          ['Fleet:', [self._fleet_line(truck) for truck in facility.distribution.fleet]]
      )
      substatus.append(
          ['Inbound orders:', [self._order_line(order) for order in facility.distribution.order_queue]]
      )
      substatus.append([f'Current unit price: ${facility.distribution.economy.unit_price}'])
      substatus.append([
          'Penalties:',
          (
              f'wrong orders {facility.distribution.economy.total_wrong_order_penalties}, '
              f'pending orders {facility.distribution.economy.total_pending_order_penalties}'
          ),
      ])
    if facility.consumer is not None:
      in_transit = sum(sum(counter.values()) for counter in facility.consumer.open_orders.values())
      substatus.append(['Outbound orders:', [f'{src}: {counter_to_dict(dict(book))}' for src, book in facility.consumer.open_orders.items()]])
      substatus.append([f'Total units purchased: {facility.consumer.economy.total_units_purchased}'])
      substatus.append([f'Total units received: {facility.consumer.economy.total_units_received}'])
    if facility.seller is not None:
      economy = facility.seller.economy
      substatus.append([f'Current unit price: ${economy.unit_price}'])
      substatus.append([f'Current demand: {economy.market_demand(economy.unit_price)}'])
      substatus.append([f'Total units sold: {economy.total_units_sold}'])
    storage_status = self.status(facility.storage) if facility.storage is not None else ['No storage']
    substatus.append(['Storage:', storage_status])
    return [f'{facility.id} ({facility.x}, {facility.y})', substatus]

  @staticmethod
  def _order_line(order: Order) -> str:
    '''Describe a queued distribution order.

    Args:
      order: Queued order.

    Returns:
      str: Compact single-line description.
    '''
    return f'{order.product_id}:{order.quantity} at ${order.unit_price} -> {order.destination.id}'

  def _fleet_line(self, truck: Transport) -> str:
    '''Describe one truck with its route progress bar (legacy format).

    Args:
      truck: Transport snapshot.

    Returns:
      str: Status line followed by a route progress bar.
    '''
    progress = ascii_progress_bar(truck.location_pointer, max(truck.path_len() - 1, 0), 5)
    return f'{self.status(truck)} {progress}'

  @status.register
  def _(self, truck: Transport) -> str:
    '''Describe the current activity of one truck.

    Args:
      truck: Transport snapshot.

    Returns:
      str: One of LOAD/MOVE/UNLD/BACK/IDLE lines.
    '''
    state = truck.state
    destination = truck.destination.id if truck.destination is not None else 'home'
    cargo = f'{truck.product_id}:{truck.payload}'
    if state == TransportState.IDLE:
      return 'IDLE'
    if state == TransportState.LOADING:
      return f'LOAD {truck.product_id}:{truck.requested_quantity} -> {destination}'
    if state == TransportState.EN_ROUTE:
      return f'MOVE {cargo} -> {destination}'
    if state == TransportState.UNLOADING:
      return f'UNLD {cargo} -> {destination}'
    return f'BACK {destination} -> home'

  @status.register
  def _(self, storage: StorageUnit) -> list:
    '''Describe storage utilization and inventory.

    Args:
      storage: Storage snapshot.

    Returns:
      list: Usage bar, cost line, and inventory mapping.
    '''
    return [
        f'Usage: {ascii_progress_bar(storage.used_capacity(), storage.max_capacity)}',
        f'Storage cost/unit: {storage.economy.unit_storage_cost}',
        f'Inventory: {counter_to_dict(storage.stock_levels)}',
    ]
