#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Core abstractions shared by every acting entity.

Defines the :class:`Agent` protocol and the narrow facility-unit
protocols so units depend on minimal interfaces (Dependency Inversion)
instead of the concrete :class:`FacilityCell` god class.
'''

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
  from world_of_supply.economy import BalanceSheet
  from world_of_supply.facility import FacilityCell, FacilityControl
  from world_of_supply.manufacturing import BillOfMaterials


class Agent(Protocol):
  '''Protocol for anything stepped once per simulation tick.

  Implementations include transport vehicles and all facility units
  (storage, manufacturing, distribution, consumer, seller).

  Attributes:
    None: Protocol; implementations hold their own state.
  '''

  def act(
      self,
      control: 'FacilityControl | None',
  ) -> 'BalanceSheet':
    '''Advance one time step under the given control.

    Args:
      control: Facility-level control carrying this unit's parameters,
        or ``None`` to run with defaults.

    Returns:
      BalanceSheet: Monetary effect of this step.

    Raises:
      NotImplementedError: Implementations must implement this step.
    '''
    raise NotImplementedError


class StorageUnitLike(Protocol):
  '''Narrow interface for storage units.

  Attributes:
    max_capacity: Maximum units the storage can hold.
    stock_levels: Current stock per product id.
  '''

  max_capacity: int
  stock_levels: dict[str, int]

  def used_capacity(self) -> int:
    '''Report currently stored units.

    Returns:
      int: Sum over all product stock levels.

    Raises:
      NotImplementedError: Implementations must report capacity use.
    '''
    raise NotImplementedError

  def available_capacity(self) -> int:
    '''Report free capacity.

    Returns:
      int: ``max_capacity - used_capacity``.

    Raises:
      NotImplementedError: Implementations must report free space.
    '''
    raise NotImplementedError

  def try_add_units(
      self,
      product_quantities: dict[str, int],
      all_or_nothing: bool = True,
  ) -> dict[str, int]:
    '''Deposit units respecting the capacity limit.

    Args:
      product_quantities: Product id to requested quantity.
      all_or_nothing: Deposit only if everything fits when True.

    Returns:
      dict[str, int]: Quantities actually deposited per product.

    Raises:
      NotImplementedError: Implementations must handle deposits.
    '''
    raise NotImplementedError

  def try_take_units(
      self,
      product_quantities: dict[str, int],
  ) -> bool:
    '''Withdraw units atomically.

    Args:
      product_quantities: Product id to requested quantity.

    Returns:
      bool: True when the full withdrawal succeeded.

    Raises:
      NotImplementedError: Implementations must handle removals.
    '''
    raise NotImplementedError

  def take_available(self, product_id: str, quantity: int) -> int:
    '''Withdraw up to ``quantity`` units of one product.

    Args:
      product_id: Product to withdraw.
      quantity: Desired quantity.

    Returns:
      int: Quantity actually withdrawn.

    Raises:
      NotImplementedError: Implementations must handle withdrawals.
    '''
    raise NotImplementedError


class ConsumerUnitLike(Protocol):
  '''Narrow interface for consumer units.

  Attributes:
    sources: Upstream facilities this consumer may order from.
    open_orders: Mapping of source id to outstanding quantities.
  '''

  sources: list['FacilityCell']
  open_orders: dict[str, Counter]

  def on_order_reception(
      self,
      source_id: str,
      product_id: str,
      quantity: int,
  ) -> None:
    '''Record a delivery and clear the open order.

    Args:
      source_id: Id of the shipping facility.
      product_id: Delivered product.
      quantity: Number of units received.

    Returns:
      None: This method mutates the open-order book in place.

    Raises:
      NotImplementedError: Implementations must record delivery.
    '''
    raise NotImplementedError


class DistributionUnitLike(Protocol):
  '''Narrow interface for distribution units.

  Attributes:
    None: Distribution state is private to implementations.
  '''

  def place_order(self, order: 'object') -> int:
    '''Register an incoming order and return the payment.

    Args:
      order: The incoming order.

    Returns:
      int: Positive amount the buyer has to pay.

    Raises:
      NotImplementedError: Implementations must handle orders.
    '''
    raise NotImplementedError


class FacilityLike(Protocol):
  '''Narrow view of a facility used by its units.

  Units depend on this interface rather than the full
  :class:`FacilityCell` so they stay testable in isolation
  (Dependency Inversion Principle).

  Attributes:
    x: Horizontal grid position.
    y: Vertical grid position.
    id: String id of the form ``ClassName_idNum``.
    bom: Production recipe.
    storage: Installed storage unit, or ``None``.
    consumer: Installed consumer unit, or ``None``.
    manufacturing: Installed manufacturing unit, or ``None``.
    distribution: Installed distribution unit, or ``None``.
    seller: Installed seller unit, or ``None``.
  '''

  x: int
  y: int
  id: str
  bom: 'BillOfMaterials'
  storage: 'StorageUnitLike | None'
  consumer: 'ConsumerUnitLike | None'
  manufacturing: 'object | None'
  distribution: 'DistributionUnitLike | None'
  seller: 'object | None'
