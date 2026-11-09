#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Facilities: agents composed of storage/logistics units.

A :class:`FacilityCell` is a Template Method: subclasses decide
which units to install, while the base class owns identity,
bookkeeping, and fan-out of a single control to every unit.
Unit assembly uses narrow configs so each factory depends only
on the parameters it actually needs.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from world_of_supply.economy import BalanceSheet
from world_of_supply.geography import Cell
from world_of_supply.manufacturing import (
    BillOfMaterials,
    ManufacturingEconomy,
    ManufacturingUnit,
)
from world_of_supply.storage import StorageEconomy, StorageUnit
from world_of_supply.transport import TransportEconomy
from world_of_supply.distribution import (
    DistributionEconomy,
    DistributionUnit,
)
from world_of_supply.consumer import ConsumerUnit
from world_of_supply.seller import SellerEconomy, SellerUnit

if TYPE_CHECKING:
  from world_of_supply.world import World


@dataclass
class FacilityControl:
  '''Facility-level control issued by policies or RL actions.

  All fields are optional so a single object can drive
  heterogeneous units; absent fields mean keep current / no-op.

  Attributes:
    unit_price: Offered/selling price for seller/distribution.
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

  Kept as the public assembly contract; subclasses narrow it
  via :class:`StorageConfig`, :class:`DistributionConfig`, etc.

  Attributes:
    bill_of_materials: Production recipe.
    max_storage_capacity: Storage capacity in units.
    unit_storage_cost: Holding cost per unit per tick.
    fleet_size: Number of trucks of the distribution unit.
    unit_transport_cost: Freight cost per unit per cell.
    sources: Upstream facilities for the consumer unit.
    wrong_order_penalty: Penalty per foreign-product unit.
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


@dataclass
class StorageConfig:
  '''Narrow storage fragment of a :class:`FacilityConfig`.

  Attributes:
    max_storage_capacity: Storage capacity in units.
    unit_storage_cost: Holding cost per unit per tick.
  '''

  max_storage_capacity: int = 20
  unit_storage_cost: int = 1

  @classmethod
  def from_facility_config(
      cls, config: FacilityConfig,
  ) -> 'StorageConfig':
    '''Project storage parameters out of a facility config.

    Args:
      config: Full facility assembly configuration.

    Returns:
      StorageConfig: Storage-only parameters.
    '''
    return cls(config.max_storage_capacity, config.unit_storage_cost)


@dataclass
class DistributionConfig:
  '''Narrow distribution fragment of a :class:`FacilityConfig`.

  Attributes:
    fleet_size: Number of trucks.
    unit_transport_cost: Freight cost per unit per cell.
    wrong_order_penalty: Penalty per foreign-product unit.
    pending_order_penalty: Fee per queued order per tick.
  '''

  fleet_size: int = 1
  unit_transport_cost: int = 1
  wrong_order_penalty: int = 500
  pending_order_penalty: int = 4

  @classmethod
  def from_facility_config(
      cls, config: FacilityConfig,
  ) -> 'DistributionConfig':
    '''Project distribution parameters out of a facility config.

    Args:
      config: Full facility assembly configuration.

    Returns:
      DistributionConfig: Distribution-only parameters.
    '''
    return cls(
        config.fleet_size,
        config.unit_transport_cost,
        config.wrong_order_penalty,
        config.pending_order_penalty,
    )


@dataclass
class ManufacturingConfig:
  '''Narrow manufacturing fragment of a :class:`FacilityConfig`.

  Attributes:
    unit_manufacturing_cost: Production cost per unit.
  '''

  unit_manufacturing_cost: int = 100

  @classmethod
  def from_facility_config(
      cls, config: FacilityConfig,
  ) -> 'ManufacturingConfig':
    '''Project manufacturing parameters out of a facility config.

    Args:
      config: Full facility assembly configuration.

    Returns:
      ManufacturingConfig: Manufacturing-only parameters.
    '''
    return cls(config.unit_manufacturing_cost)


@dataclass
class SellerConfig:
  '''Narrow seller fragment of a :class:`FacilityConfig`.

  Attributes:
    price_demand_intercept: Seller demand-curve intercept.
    price_demand_slope: Seller demand-curve slope.
  '''

  price_demand_intercept: float = 50.0
  price_demand_slope: float = 0.005

  @classmethod
  def from_facility_config(cls, config: FacilityConfig) -> 'SellerConfig':
    '''Project seller parameters out of a facility config.

    Args:
      config: Full facility assembly configuration.

    Returns:
      SellerConfig: Seller-only parameters.
    '''
    return cls(config.price_demand_intercept, config.price_demand_slope)


class FacilityEconomy:
  '''Bookkeeper holding the running balance of a facility.

  Attributes:
    total_balance: Cumulative balance since creation.
  '''

  def __init__(self, initial_balance: BalanceSheet) -> None:
    '''Seed the ledger.

    Args:
      initial_balance: BalanceSheet representing starting capital.

    Returns:
      None: The economy is mutated in place.
    '''
    self.total_balance = initial_balance

  def deposit(
      self, balance_sheets: list[BalanceSheet],
  ) -> BalanceSheet:
    '''Book a batch of step sheets into the running balance.

    Args:
      balance_sheets: Sheets returned by all units of one tick.

    Returns:
      BalanceSheet: Aggregated sheet of this deposit.
    '''
    aggregated = sum(balance_sheets, BalanceSheet())
    self.total_balance += aggregated
    return aggregated


def _storage_unit(config: StorageConfig) -> StorageUnit:
  '''Assemble a storage unit from a narrow storage config.

  Args:
    config: Source of capacity and cost parameters.

  Returns:
    StorageUnit: Ready-to-use storage unit.
  '''
  economy = StorageEconomy(config.unit_storage_cost)
  return StorageUnit(config.max_storage_capacity, economy)


def _manufacturing_unit(
    facility: 'FacilityCell',
    config: ManufacturingConfig,
) -> ManufacturingUnit:
  '''Assemble a manufacturing unit from a narrow config.

  Args:
    facility: Owning facility.
    config: Source of production-cost parameters.

  Returns:
    ManufacturingUnit: Ready-to-use manufacturing unit.
  '''
  economy = ManufacturingEconomy(config.unit_manufacturing_cost)
  return ManufacturingUnit(facility, economy)


def _distribution_unit(
    facility: 'FacilityCell',
    config: DistributionConfig,
) -> DistributionUnit:
  '''Assemble a distribution unit from a narrow config.

  Args:
    facility: Owning facility.
    config: Source of fleet and cost parameters.

  Returns:
    DistributionUnit: Ready-to-use outbound-logistics unit.
  '''
  economy = DistributionEconomy(
      wrong_order_penalty=config.wrong_order_penalty,
      pending_order_penalty=config.pending_order_penalty,
  )
  transport = TransportEconomy(config.unit_transport_cost)
  return DistributionUnit(facility, config.fleet_size, economy, transport)


class FacilityCell(Cell):
  '''Base class of every economic agent on the grid.

  Attributes:
    id_num: Numeric id assigned by the world.
    id: String id of the form ``ClassName_idNum``.
    world: Owning world.
    economy: Bookkeeper.
    bom: Production recipe.
    storage: Installed storage unit, or ``None``.
    consumer: Installed consumer unit, or ``None``.
    manufacturing: Installed manufacturing unit, or ``None``.
    distribution: Installed distribution unit, or ``None``.
    seller: Installed seller unit, or ``None``.
  '''

  def __init__(
      self,
      x: int,
      y: int,
      world: 'World',
      config: FacilityConfig,
  ) -> None:
    '''Create the facility and install its units.

    Args:
      x: Grid x position.
      y: Grid y position.
      world: Owning world (id generation and routing).
      config: Assembly configuration.

    Returns:
      None: The facility is mutated in place.
    '''
    super().__init__(x, y)
    self.id_num = world.generate_id()
    self.id = f'{self.__class__.__name__}_{self.id_num}'
    self.world = world
    balance = BalanceSheet(config.initial_balance, 0)
    self.economy = FacilityEconomy(balance)
    self.bom = config.bill_of_materials
    self.storage: StorageUnit | None = None
    self.consumer: ConsumerUnit | None = None
    self.manufacturing: ManufacturingUnit | None = None
    self.distribution: DistributionUnit | None = None
    self.seller: SellerUnit | None = None
    self._install_units(config)

  def _install_units(self, config: FacilityConfig) -> None:
    '''Install the units composing this facility hook.

    Args:
      config: Assembly configuration.

    Returns:
      None: The facility units are mutated in place.

    Raises:
      NotImplementedError: Subclasses must install their units.
    '''
    raise NotImplementedError

  @property
  def _units(self) -> list:
    '''Collect all installed non-None units in execution order.

    Returns:
      list: Active units (storage, consumer, manufacturing,
        distribution, seller order is preserved).
    '''
    return [
        unit
        for unit in (
            self.storage,
            self.consumer,
            self.manufacturing,
            self.distribution,
            self.seller,
        )
        if unit is not None
    ]

  def require_storage(self) -> StorageUnit:
    '''Return the installed storage unit.

    Returns:
      StorageUnit: The storage unit.

    Raises:
      AssertionError: If no storage unit is installed.
    '''
    assert self.storage is not None
    return self.storage

  def require_consumer(self) -> ConsumerUnit:
    '''Return the installed consumer unit.

    Returns:
      ConsumerUnit: The consumer unit.

    Raises:
      AssertionError: If no consumer unit is installed.
    '''
    assert self.consumer is not None
    return self.consumer

  def require_distribution(self) -> DistributionUnit:
    '''Return the installed distribution unit.

    Returns:
      DistributionUnit: The distribution unit.

    Raises:
      AssertionError: If no distribution unit is installed.
    '''
    assert self.distribution is not None
    return self.distribution

  def act(self, control: FacilityControl | None) -> BalanceSheet:
    '''Step every installed unit and deposit their sheets.

    Args:
      control: Facility-level control or ``None``.

    Returns:
      BalanceSheet: Aggregated monetary effect of this tick.
    '''
    sheets = [unit.act(control) for unit in self._units]
    return self.economy.deposit(sheets)


class RawMaterialsFactoryCell(FacilityCell):
  '''Extracts raw materials without any upstream suppliers.

  Attributes:
    storage: Storage unit installed by :meth:`_install_units`.
    manufacturing: Manufacturing unit installed.
    distribution: Distribution unit installed.
  '''

  def _install_units(self, config: FacilityConfig) -> None:
    '''Install storage, manufacturing, and distribution.

    Args:
      config: Assembly configuration.

    Returns:
      None: The facility units are mutated in place.
    '''
    storage_cfg = StorageConfig.from_facility_config(config)
    mfg_cfg = ManufacturingConfig.from_facility_config(config)
    dist_cfg = DistributionConfig.from_facility_config(config)
    self.storage = _storage_unit(storage_cfg)
    self.manufacturing = _manufacturing_unit(self, mfg_cfg)
    self.distribution = _distribution_unit(self, dist_cfg)


class SteelFactoryCell(RawMaterialsFactoryCell):
  '''Produces ``steel`` from nothing (extraction semantics).

  Attributes:
    storage: Storage unit installed by the parent class.
    manufacturing: Manufacturing unit installed by the parent.
    distribution: Distribution unit installed by the parent.
  '''


class LumberFactoryCell(RawMaterialsFactoryCell):
  '''Produces ``lumber`` from nothing (extraction semantics).

  Attributes:
    storage: Storage unit installed by the parent class.
    manufacturing: Manufacturing unit installed by the parent.
    distribution: Distribution unit installed by the parent.
  '''


class ValueAddFactoryCell(FacilityCell):
  '''Transforms purchased inputs into higher-value outputs.

  Attributes:
    storage: Storage unit installed by :meth:`_install_units`.
    consumer: Consumer unit installed.
    manufacturing: Manufacturing unit installed.
    distribution: Distribution unit installed.
  '''

  def _install_units(self, config: FacilityConfig) -> None:
    '''Install storage, consumer, manufacturing, distribution.

    Args:
      config: Assembly configuration.

    Returns:
      None: The facility units are mutated in place.
    '''
    storage_cfg = StorageConfig.from_facility_config(config)
    mfg_cfg = ManufacturingConfig.from_facility_config(config)
    dist_cfg = DistributionConfig.from_facility_config(config)
    self.storage = _storage_unit(storage_cfg)
    self.consumer = ConsumerUnit(self, list(config.sources or []))
    self.manufacturing = _manufacturing_unit(self, mfg_cfg)
    self.distribution = _distribution_unit(self, dist_cfg)


class ToyFactoryCell(ValueAddFactoryCell):
  '''Builds ``toy_car`` out of lumber and steel.

  Attributes:
    storage: Storage unit installed by the parent class.
    consumer: Consumer unit installed by the parent class.
    manufacturing: Manufacturing unit installed by the parent.
    distribution: Distribution unit installed by the parent.
  '''


class WarehouseCell(FacilityCell):
  '''Pure logistics node: buys, stores, and reships products.

  Attributes:
    storage: Storage unit installed by :meth:`_install_units`.
    consumer: Consumer unit installed.
    distribution: Distribution unit installed.
  '''

  def _install_units(self, config: FacilityConfig) -> None:
    '''Install storage, consumer, and distribution.

    Args:
      config: Assembly configuration.

    Returns:
      None: The facility units are mutated in place.
    '''
    storage_cfg = StorageConfig.from_facility_config(config)
    dist_cfg = DistributionConfig.from_facility_config(config)
    self.storage = _storage_unit(storage_cfg)
    self.consumer = ConsumerUnit(self, list(config.sources or []))
    self.distribution = _distribution_unit(self, dist_cfg)


class RetailerCell(FacilityCell):
  '''Final node selling inventory directly to market demand.

  Attributes:
    storage: Storage unit installed by :meth:`_install_units`.
    consumer: Consumer unit installed.
    seller: Seller unit installed (no trucks).
  '''

  def _install_units(self, config: FacilityConfig) -> None:
    '''Install storage, consumer, and seller (no trucks).

    Args:
      config: Assembly configuration.

    Returns:
      None: The facility units are mutated in place.
    '''
    storage_cfg = StorageConfig.from_facility_config(config)
    seller_cfg = SellerConfig.from_facility_config(config)
    self.storage = _storage_unit(storage_cfg)
    self.consumer = ConsumerUnit(self, list(config.sources or []))
    economy = SellerEconomy(
        price_demand_intercept=seller_cfg.price_demand_intercept,
        price_demand_slope=seller_cfg.price_demand_slope,
    )
    self.seller = SellerUnit(self, economy)
