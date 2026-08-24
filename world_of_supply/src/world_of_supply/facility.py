#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Facilities: economic agents composed of storage/manufacturing/logistics units.

A :class:`FacilityCell` is a Template Method: subclasses decide which units
to install, while the base class owns identity, bookkeeping, and the fan-out
of a single :class:`FacilityControl` to every installed unit.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from world_of_supply.economy import BalanceSheet
from world_of_supply.geography import Cell
from world_of_supply.manufacturing import BillOfMaterials, ManufacturingEconomy, ManufacturingUnit
from world_of_supply.storage import StorageEconomy, StorageUnit
from world_of_supply.transport import TransportEconomy
from world_of_supply.distribution import DistributionEconomy, DistributionUnit
from world_of_supply.consumer import ConsumerUnit
from world_of_supply.seller import SellerEconomy, SellerUnit

if TYPE_CHECKING:
  from world_of_supply.world import World


@dataclass
class FacilityControl:
  '''Facility-level control issued by policies or RL actions each tick.

  All fields are optional so a single object can drive heterogeneous units;
  absent fields mean keep current behavior / do nothing.

  Attributes:
    unit_price: Offered/selling price for seller and distribution units.
    production_rate: Lots to attempt manufacturing this tick.
    consumer_product_id: Product to order upstream.
    consumer_source_id: Index into the consumer sources list.
    consumer_quantity: Units to order.
  '''

  unit_price: int | None = None
  production_rate: int | None = None
  consumer_product_id: str | None = None
  consumer_source_id: int | None = None
  consumer_quantity: int | None = None


@dataclass
class FacilityConfig:
  '''Complete configuration used to assemble a facility.

  Attributes:
    bill_of_materials: Production recipe.
    max_storage_capacity: Storage capacity in units.
    unit_storage_cost: Holding cost per unit per tick.
    fleet_size: Number of trucks of the distribution unit.
    unit_transport_cost: Freight cost per unit per cell.
    sources: Upstream facilities available to the consumer unit.
    wrong_order_penalty: Penalty per unit ordered with a foreign product.
    pending_order_penalty: Fee per queued order per tick.
    unit_manufacturing_cost: Production cost per unit.
    price_demand_intercept: Seller demand-curve intercept.
    price_demand_slope: Seller demand-curve slope.
    initial_balance: Starting cash position.
  '''

  bill_of_materials: BillOfMaterials
  max_storage_capacity: int = 20
  unit_storage_cost: int = 1
  fleet_size: int = 1
  unit_transport_cost: int = 1
  sources: list['FacilityCell'] | None = None
  wrong_order_penalty: int = 500
  pending_order_penalty: int = 4
  unit_manufacturing_cost: int = 100
  price_demand_intercept: float = 50.0
  price_demand_slope: float = 0.005
  initial_balance: int = 1000


class FacilityEconomy:
  '''Bookkeeper holding the running balance of a facility.

  Attributes:
    total_balance: Cumulative balance since creation.
  '''

  def __init__(self, initial_balance: BalanceSheet) -> None:
    '''Seed the ledger.

    Args:
      initial_balance: BalanceSheet representing the starting capital.
    '''
    self.total_balance = initial_balance

  def deposit(self, balance_sheets: list[BalanceSheet]) -> BalanceSheet:
    '''Book a batch of step sheets into the running balance.

    Args:
      balance_sheets: Sheets returned by all units of one tick.

    Returns:
      BalanceSheet: Aggregated sheet of this deposit.
    '''
    aggregated = sum(balance_sheets, BalanceSheet())
    self.total_balance += aggregated
    return aggregated


class FacilityCell(Cell):
  '''Base class of every economic agent on the grid.

  Attributes:
    id_num: Numeric id assigned by the world.
    id: String id of the form ``ClassName_idNum``.
    world: Owning world.
    economy: Bookkeeper.
    bom: Production recipe.
    storage / consumer / manufacturing / distribution / seller:
      Installed optional units (None when not applicable).
  '''

  def __init__(self, x: int, y: int, world: 'World', config: FacilityConfig) -> None:
    '''Create the facility and install its units.

    Args:
      x: Grid x position.
      y: Grid y position.
      world: Owning world (used for id generation and routing).
      config: Assembly configuration.
    '''
    super().__init__(x, y)
    self.id_num = world.generate_id()
    self.id = f'{self.__class__.__name__}_{self.id_num}'
    self.world = world
    self.economy = FacilityEconomy(BalanceSheet(config.initial_balance, 0))
    self.bom = config.bill_of_materials
    self.storage: StorageUnit | None = None
    self.consumer: ConsumerUnit | None = None
    self.manufacturing: ManufacturingUnit | None = None
    self.distribution: DistributionUnit | None = None
    self.seller: SellerUnit | None = None
    self._install_units(config)

  def _install_units(self, config: FacilityConfig) -> None:
    '''Install the units composing this facility (Template Method hook).

    Args:
      config: Assembly configuration.
    '''
    raise NotImplementedError

  @property
  def _units(self) -> list:
    '''Collect all installed non-None units.

    Returns:
      list: Active agent units in execution order.
    '''
    return [unit for unit in (
        self.storage,
        self.consumer,
        self.manufacturing,
        self.distribution,
        self.seller,
    ) if unit is not None]

  def act(self, control: FacilityControl | None) -> BalanceSheet:
    '''Step every installed unit and deposit their sheets.

    Args:
      control: Facility-level control or ``None``.

    Returns:
      BalanceSheet: Aggregated monetary effect of this tick.
    '''
    sheets = [unit.act(control) for unit in self._units]
    return self.economy.deposit(sheets)


def _distribution_unit(facility: FacilityCell, config: FacilityConfig) -> DistributionUnit:
  '''Assemble a distribution unit from a facility config.

  Args:
    facility: Owning facility.
    config: Source of fleet and cost parameters.

  Returns:
    DistributionUnit: Ready-to-use outbound-logistics unit.
  '''
  return DistributionUnit(
      facility,
      config.fleet_size,
      DistributionEconomy(
          wrong_order_penalty=config.wrong_order_penalty,
          pending_order_penalty=config.pending_order_penalty,
      ),
      TransportEconomy(unit_transport_cost=config.unit_transport_cost),
  )


class RawMaterialsFactoryCell(FacilityCell):
  '''Extracts/produces raw materials without any upstream suppliers.'''

  def _install_units(self, config: FacilityConfig) -> None:
    '''Install storage, manufacturing, and distribution.

    Args:
      config: Assembly configuration.
    '''
    self.storage = StorageUnit(config.max_storage_capacity, StorageEconomy(config.unit_storage_cost))
    self.manufacturing = ManufacturingUnit(self, ManufacturingEconomy(config.unit_manufacturing_cost))
    self.distribution = _distribution_unit(self, config)


class SteelFactoryCell(RawMaterialsFactoryCell):
  '''Produces ``steel`` from nothing (extraction semantics).'''


class LumberFactoryCell(RawMaterialsFactoryCell):
  '''Produces ``lumber`` from nothing (extraction semantics).'''


class ValueAddFactoryCell(FacilityCell):
  '''Transforms purchased inputs into higher-value outputs.'''

  def _install_units(self, config: FacilityConfig) -> None:
    '''Install storage, consumer, manufacturing, and distribution.

    Args:
      config: Assembly configuration.
    '''
    self.storage = StorageUnit(config.max_storage_capacity, StorageEconomy(config.unit_storage_cost))
    self.consumer = ConsumerUnit(self, list(config.sources or []))
    self.manufacturing = ManufacturingUnit(self, ManufacturingEconomy(config.unit_manufacturing_cost))
    self.distribution = _distribution_unit(self, config)


class ToyFactoryCell(ValueAddFactoryCell):
  '''Builds ``toy_car`` out of lumber and steel.'''


class WarehouseCell(FacilityCell):
  '''Pure logistics node: buys, stores, and reships products.'''

  def _install_units(self, config: FacilityConfig) -> None:
    '''Install storage, consumer, and distribution.

    Args:
      config: Assembly configuration.
    '''
    self.storage = StorageUnit(config.max_storage_capacity, StorageEconomy(config.unit_storage_cost))
    self.consumer = ConsumerUnit(self, list(config.sources or []))
    self.distribution = _distribution_unit(self, config)


class RetailerCell(FacilityCell):
  '''Final node selling inventory directly to market demand.'''

  def _install_units(self, config: FacilityConfig) -> None:
    '''Install storage, consumer, and seller (no trucks).

    Args:
      config: Assembly configuration.
    '''
    self.storage = StorageUnit(config.max_storage_capacity, StorageEconomy(config.unit_storage_cost))
    self.consumer = ConsumerUnit(self, list(config.sources or []))
    self.seller = SellerUnit(
        self,
        SellerEconomy(
            price_demand_intercept=config.price_demand_intercept,
            price_demand_slope=config.price_demand_slope,
        ),
    )
