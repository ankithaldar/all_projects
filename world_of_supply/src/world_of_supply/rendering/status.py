#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Human-readable world status formatting for logs and status panels.'''

from __future__ import annotations

from collections.abc import Callable
from functools import singledispatchmethod

from world_of_supply.distribution import DistributionUnit, Order, Transport
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
  bar = '=' * done_chars + '-' * (bar_length - done_chars)
  return f'{bar} {done}/{limit}'


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

  The output contains only strings, numbers, lists and dicts, so
  it can be dumped as YAML/JSON or printed directly.
  '''

  @singledispatchmethod
  def status(self, obj):
    '''Dispatch to the type-specific formatter.

    Args:
      obj: A world, facility, transport, or storage object.

    Returns:
      list: Nested human-readable status lines.

    Raises:
      NotImplementedError: If no formatter is registered for the
        object's type.
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
    time_line = f'Time step: {world.time_step}'
    balance_line = f'Global balance: {world.economy.global_balance()}'
    status = [['World:', [time_line, balance_line]]]
    facilities = world.facilities.values()
    status.extend(self.status(facility) for facility in facilities)
    return status

  def _distribution_status(self, facility: FacilityCell) -> list:
    '''Build the distribution section for one facility.

    Args:
      facility: Facility snapshot.

    Returns:
      list: Fleet, inbound-order, price, and penalty lines; empty
        when the facility has no distribution unit.
    '''
    if not isinstance(facility.distribution, DistributionUnit):
      return []
    fleet = [self._fleet_line(truck) for truck in facility.distribution.fleet]
    orders = [
        self._order_line(order)
        for order in facility.distribution.order_queue
    ]
    economy = facility.distribution.economy
    price_line = f'Current unit price: ${economy.unit_price}'
    penalties = (
        f'wrong orders {economy.total_wrong_order_penalties}, '
        f'pending orders {economy.total_pending_order_penalties}'
    )
    return [
        ['Fleet:', fleet],
        ['Inbound orders:', orders],
        [price_line],
        ['Penalties:', penalties],
    ]

  def _consumer_status(self, facility: FacilityCell) -> list:
    '''Build the consumer section for one facility.

    Args:
      facility: Facility snapshot.

    Returns:
      list: Outbound-order and purchase/receipt totals; empty when
        the facility has no consumer unit.
    '''
    if facility.consumer is None:
      return []
    outbound = [
        f'{src}: {counter_to_dict(dict(book))}'
        for src, book in facility.consumer.open_orders.items()
    ]
    economy = facility.consumer.economy
    return [
        ['Outbound orders:', outbound],
        [f'Total units purchased: {economy.total_units_purchased}'],
        [f'Total units received: {economy.total_units_received}'],
    ]

  def _seller_status(self, facility: FacilityCell) -> list:
    '''Build the seller section for one facility.

    Args:
      facility: Facility snapshot.

    Returns:
      list: Price, demand, and units-sold lines; empty when the
        facility has no seller unit.
    '''
    if facility.seller is None:
      return []
    economy = facility.seller.economy
    return [
        [f'Current unit price: ${economy.unit_price}'],
        [f'Current demand: {economy.market_demand(economy.unit_price)}'],
        [f'Total units sold: {economy.total_units_sold}'],
    ]

  def _storage_status(self, facility: FacilityCell) -> list:
    '''Build the storage section for one facility.

    Args:
      facility: Facility snapshot.

    Returns:
      list: Storage status lines, or a ``No storage`` placeholder
        when the facility has no storage unit.
    '''
    if facility.storage is None:
      return ['No storage']
    return self.status(facility.storage)

  @status.register
  def _(self, facility: FacilityCell) -> list:
    '''Format one facility including its installed units.

    Args:
      facility: Facility snapshot.

    Returns:
      list: Identity line followed by unit sub-statuses.
    '''
    substatus: list = [f'Balance: {facility.economy.total_balance}']
    substatus.extend(self._distribution_status(facility))
    substatus.extend(self._consumer_status(facility))
    substatus.extend(self._seller_status(facility))
    substatus.append(['Storage:', self._storage_status(facility)])
    return [f'{facility.id} ({facility.x}, {facility.y})', substatus]

  @staticmethod
  def _order_line(order: Order) -> str:
    '''Describe a queued distribution order.

    Args:
      order: Queued order.

    Returns:
      str: Compact single-line description.
    '''
    product = order.product_id
    quantity = order.quantity
    price = order.unit_price
    destination = order.destination.id
    return f'{product}:{quantity} at ${price} -> {destination}'

  def _fleet_line(self, truck: Transport) -> str:
    '''Describe one truck with its route progress bar (legacy format).

    Args:
      truck: Transport snapshot.

    Returns:
      str: Status line followed by a route progress bar.
    '''
    limit = max(truck.path_len() - 1, 0)
    progress = ascii_progress_bar(truck.location_pointer, limit, 5)
    return f'{self.status(truck)} {progress}'

  @staticmethod
  def _transport_destination(truck: Transport) -> str:
    '''Resolve the display destination for one truck.

    Args:
      truck: Transport snapshot.

    Returns:
      str: Destination facility id, or ``home`` when idle at base.
    '''
    if truck.destination is not None:
      return truck.destination.id
    return 'home'

  @staticmethod
  def _transport_cargo(truck: Transport) -> str:
    '''Describe the cargo carried by one truck.

    Args:
      truck: Transport snapshot.

    Returns:
      str: ``product:payload`` cargo description.
    '''
    return f'{truck.product_id}:{truck.payload}'

  def _idle_line(self, unused_truck: Transport) -> str:
    '''Describe an idle truck.

    Args:
      unused_truck: Transport snapshot; unused because idle trucks
        have no destination or cargo to report.

    Returns:
      str: ``IDLE`` line.
    '''
    return 'IDLE'

  def _loading_line(self, truck: Transport) -> str:
    '''Describe a loading truck.

    Args:
      truck: Transport snapshot.

    Returns:
      str: ``LOAD`` line with requested quantity and destination.
    '''
    destination = self._transport_destination(truck)
    product = truck.product_id
    quantity = truck.requested_quantity
    return f'LOAD {product}:{quantity} -> {destination}'

  def _enroute_line(self, truck: Transport) -> str:
    '''Describe an en-route truck.

    Args:
      truck: Transport snapshot.

    Returns:
      str: ``MOVE`` line with cargo and destination.
    '''
    destination = self._transport_destination(truck)
    cargo = self._transport_cargo(truck)
    return f'MOVE {cargo} -> {destination}'

  def _unloading_line(self, truck: Transport) -> str:
    '''Describe an unloading truck.

    Args:
      truck: Transport snapshot.

    Returns:
      str: ``UNLD`` line with cargo and destination.
    '''
    destination = self._transport_destination(truck)
    cargo = self._transport_cargo(truck)
    return f'UNLD {cargo} -> {destination}'

  def _returning_line(self, truck: Transport) -> str:
    '''Describe a truck returning home.

    Args:
      truck: Transport snapshot.

    Returns:
      str: ``BACK`` line with destination and home.
    '''
    destination = self._transport_destination(truck)
    return f'BACK {destination} -> home'

  def _transport_formatters(self) -> dict[TransportState, Callable]:
    '''Map each transport state to its line formatter.

    Returns:
      dict[TransportState, Callable]: State to bound formatter
        method mapping; the returning formatter is the fallback
        for unlisted states.
    '''
    return {
        TransportState.IDLE: self._idle_line,
        TransportState.LOADING: self._loading_line,
        TransportState.EN_ROUTE: self._enroute_line,
        TransportState.UNLOADING: self._unloading_line,
    }

  @status.register
  def _(self, truck: Transport) -> str:
    '''Describe the current activity of one truck.

    Args:
      truck: Transport snapshot.

    Returns:
      str: One of LOAD/MOVE/UNLD/BACK/IDLE lines.
    '''
    formatters = self._transport_formatters()
    formatter = formatters.get(truck.state, self._returning_line)
    return formatter(truck)

  @status.register
  def _(self, storage: StorageUnit) -> list:
    '''Describe storage utilization and inventory.

    Args:
      storage: Storage snapshot.

    Returns:
      list: Usage bar, cost line, and inventory mapping.
    '''
    usage = ascii_progress_bar(storage.used_capacity(), storage.max_capacity)
    cost = storage.economy.unit_storage_cost
    return [
        f'Usage: {usage}',
        f'Storage cost/unit: {cost}',
        f'Inventory: {counter_to_dict(storage.stock_levels)}',
    ]
