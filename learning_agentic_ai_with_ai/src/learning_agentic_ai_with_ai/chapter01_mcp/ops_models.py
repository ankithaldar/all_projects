#!/usr/bin/env python
# -- coding: utf-8 --

'''Pydantic input/output models for the retail & telecom MCP tools.

Every tool argument is a pydantic model: it yields the JSON Schema the LLM
sees, validation at the server boundary, and typed handlers in Python.
'''


from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Retail tools
# ---------------------------------------------------------------------------

class RetailLowStockInput(BaseModel):
  '''Input for retail_low_stock_report.'''

  model_config = ConfigDict(extra='ignore')

  store_id: Optional[str] = Field(
    default=None,
    description=(
      'Optional store id filter, e.g. S01. Omit for all stores.'
    ),
  )


class RetailSalesTrendInput(BaseModel):
  '''Input for retail_sales_trend.'''

  model_config = ConfigDict(extra='ignore')

  store_id: str = Field(description='Store id, e.g. S01')
  sku: str = Field(description='Product SKU, e.g. R-101')
  days: int = Field(
    default=7, ge=1, le=30, description='History window in days',
  )


class RetailSalesTrendPoint(BaseModel):
  '''One day of sales.'''

  model_config = ConfigDict(extra='ignore')

  day: str
  units: int


class RetailSalesTrendOutput(BaseModel):
  '''Output of retail_sales_trend.'''

  model_config = ConfigDict(extra='ignore')

  store_id: str
  sku: str
  points: List[RetailSalesTrendPoint] = Field(default_factory=list)
  avg_daily_units: float = 0.0


class RetailReorderSuggestion(BaseModel):
  '''One suggested reorder decision.'''

  model_config = ConfigDict(extra='ignore')

  store_id: str
  sku: str
  product: str = ''
  on_hand: int = 0
  reorder_point: int = 0
  avg_daily_units: float = 0.0
  lead_time_days: int = 0
  suggested_quantity: int = 0


class RetailReorderReport(BaseModel):
  '''Structured reorder suggestion report.'''

  model_config = ConfigDict(extra='ignore')

  items: List[RetailReorderSuggestion] = Field(default_factory=list)
  note: str = ''


class RetailRestockInput(BaseModel):
  '''Input for retail_restock_order (write tool).'''

  model_config = ConfigDict(extra='ignore')

  store_id: str = Field(description='Store id, e.g. S01')
  sku: str = Field(description='Product SKU, e.g. R-101')
  quantity: int = Field(ge=1, le=1000, description='Units to order')


class RetailRestockOutput(BaseModel):
  '''Output of retail_restock_order.'''

  model_config = ConfigDict(extra='ignore')

  order_id: int
  store_id: str
  sku: str
  quantity: int
  status: str
  created_at: str


# ---------------------------------------------------------------------------
# Telecom tools
# ---------------------------------------------------------------------------

class TelecomSiteStatusInput(BaseModel):
  '''Input for telecom_site_status.'''

  model_config = ConfigDict(extra='ignore')

  site_id: str = Field(description='Cell site id, e.g. CS-77')


class TelecomSiteStatus(BaseModel):
  '''Output of telecom_site_status.'''

  model_config = ConfigDict(extra='ignore')

  site_id: str
  name: str
  region: str
  status: str
  battery_backup_hours: float
  last_hour: Dict[str, float] = Field(default_factory=dict)


class TelecomDegradedInput(BaseModel):
  '''Input for telecom_degraded_sites.'''

  model_config = ConfigDict(extra='ignore')

  region: Optional[str] = Field(
    default=None, description='Optional region filter',
  )


class TelecomDispatchInput(BaseModel):
  '''Input for telecom_dispatch_technician (write tool).'''

  model_config = ConfigDict(extra='ignore')

  site_id: str = Field(description='Cell site id needing a technician')
  tech_id: str = Field(description='Technician id to dispatch')
  priority: Literal['low', 'medium', 'high'] = Field(
    description='Dispatch priority',
  )
  note: str = Field(
    default='',
    max_length=300,
    description='Short dispatch note',
  )


class TelecomDispatchOutput(BaseModel):
  '''Output of telecom_dispatch_technician.'''

  model_config = ConfigDict(extra='ignore')

  dispatch_id: int
  site_id: str
  tech_id: str
  priority: str
  status: str
  created_at: str
