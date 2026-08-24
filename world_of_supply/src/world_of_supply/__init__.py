#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''World of Supply: a multi-agent supply-chain simulation and RL sandbox.

The package is split into focused modules:

* ``economy`` / ``geography`` — money and grid primitives.
* ``storage`` / ``manufacturing`` / ``distribution`` / ``transport`` /
  ``consumer`` / ``seller`` — facility units.
* ``facility`` / ``world`` / ``scenario`` — agents, step engine, builder.
* ``policies`` — scripted control baselines.
* ``rendering`` — ASCII/PIL visualization and status formatting.
* ``analytics`` — experiment tracking and hardware probes.
* ``rl`` — Gymnasium/RLLib environment, encoders, PyTorch model, training.

Heavy RL dependencies (Ray) are imported lazily so the pure simulation runs
without them.
'''

from __future__ import annotations

from world_of_supply.economy import BalanceSheet
from world_of_supply.facility import (
    FacilityCell,
    FacilityConfig,
    FacilityControl,
    LumberFactoryCell,
    RawMaterialsFactoryCell,
    RetailerCell,
    SteelFactoryCell,
    ToyFactoryCell,
    ValueAddFactoryCell,
    WarehouseCell,
)
from world_of_supply.manufacturing import BillOfMaterials
from world_of_supply.policies import ControlPolicy, ScriptedSupplyChainPolicy
from world_of_supply.rendering.renderer import AsciiWorldRenderer
from world_of_supply.rendering.status import WorldStatusFormatter
from world_of_supply.scenario import PRODUCT_IDS, ScenarioConfig, WorldBuilder
from world_of_supply.world import Control, StepOutcome, World

__version__ = '1.0.0'

__all__ = [
    'BalanceSheet',
    'BillOfMaterials',
    'Control',
    'ControlPolicy',
    'FacilityCell',
    'FacilityConfig',
    'FacilityControl',
    'LumberFactoryCell',
    'PRODUCT_IDS',
    'RawMaterialsFactoryCell',
    'RetailerCell',
    'ScenarioConfig',
    'ScriptedSupplyChainPolicy',
    'SteelFactoryCell',
    'StepOutcome',
    'ToyFactoryCell',
    'ValueAddFactoryCell',
    'WarehouseCell',
    'World',
    'WorldBuilder',
    'WorldStatusFormatter',
    'AsciiWorldRenderer',
    '__version__',
]


def __getattr__(name: str):
  '''Lazily import the ``rl`` subpackage (pulls in Ray/Gymnasium).

  Args:
    name: Attribute name.

  Returns:
    object: The resolved attribute from :mod:`world_of_supply.rl`.

  Raises:
    AttributeError: If the attribute is unknown.
  '''
  if name.startswith('rl'):
    import world_of_supply.rl as rl_module

    return getattr(rl_module, name)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
