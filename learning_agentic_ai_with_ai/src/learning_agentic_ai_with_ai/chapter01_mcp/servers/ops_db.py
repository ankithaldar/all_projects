#!/usr/bin/env python
# -- coding: utf-8 --

'''SQLite mock data for the retail & telecom operations use case.

The scenario: you are the operations AI for a company that runs
- a retail chain (stores, products, inventory, sales, restock orders), and
- a telecom network (cell sites, traffic metrics, incidents, field
  technicians, dispatch orders).

The agent must *make decisions*: e.g. "should we restock store S12's SKU
R-101, and how much?" or "site CS-77 is degraded - dispatch a technician?".
All data is deterministic (fixed seed) so demos, tests, and evals are stable.
'''


from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentic_common import paths
from agentic_common.logging import get_logger

logger = get_logger(__name__)


_SCHEMA = '''
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS products (
  sku TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  category TEXT NOT NULL,
  unit_cost REAL NOT NULL,
  reorder_point INTEGER NOT NULL,
  lead_time_days INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS stores (
  store_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  city TEXT NOT NULL,
  region TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
  store_id TEXT NOT NULL,
  sku TEXT NOT NULL,
  on_hand INTEGER NOT NULL,
  PRIMARY KEY (store_id, sku)
);

CREATE TABLE IF NOT EXISTS sales (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  store_id TEXT NOT NULL,
  sku TEXT NOT NULL,
  day TEXT NOT NULL,
  units INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS restock_orders (
  order_id INTEGER PRIMARY KEY AUTOINCREMENT,
  store_id TEXT NOT NULL,
  sku TEXT NOT NULL,
  quantity INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'created'
);

CREATE TABLE IF NOT EXISTS cell_sites (
  site_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  region TEXT NOT NULL,
  status TEXT NOT NULL,
  battery_backup_hours REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS site_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  site_id TEXT NOT NULL,
  hour TEXT NOT NULL,
  active_users INTEGER NOT NULL,
  latency_ms REAL NOT NULL,
  packet_loss_pct REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS technicians (
  tech_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  region TEXT NOT NULL,
  on_duty INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS dispatch_orders (
  dispatch_id INTEGER PRIMARY KEY AUTOINCREMENT,
  site_id TEXT NOT NULL,
  tech_id TEXT NOT NULL,
  priority TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'dispatched'
);
'''


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
  '''Open a connection to the ops mock database (creates file if needed).

  Args:
    db_path: Optional DB path override (default data/ops_mock.db).

  Returns:
    sqlite3 connection with row access by name.
  '''
  path = Path(db_path) if db_path else paths.OPS_MOCK_DB
  path.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(str(path))
  conn.row_factory = sqlite3.Row
  conn.executescript(_SCHEMA)
  conn.commit()
  return conn


def seed_if_empty(db_path: Path | str | None = None) -> None:
  '''Populate the mock database deterministically when it is empty.

  Args:
    db_path: Optional DB path override.
  '''
  conn = connect(db_path)
  try:
    count = conn.execute('SELECT COUNT(*) AS n FROM products').fetchone()['n']
    if count:
      logger.info('ops mock db already seeded')
      return

    rng = random.Random(42)
    today = datetime.now(timezone.utc).date()

    products = [
      ('R-101', 'Wireless Mouse Pro', 'electronics', 12.5, 40, 5),
      ('R-102', 'Mechanical Keyboard', 'electronics', 45.0, 25, 7),
      ('R-103', 'USB-C Hub 7in1', 'electronics', 22.0, 30, 6),
      ('R-201', 'Office Chair Basic', 'furniture', 85.0, 10, 14),
      ('R-202', 'Standing Desk 120cm', 'furniture', 199.0, 6, 21),
      ('R-301', 'Notebook A4 Pack', 'stationery', 4.5, 100, 3),
      ('R-302', 'Gel Pens 12pk', 'stationery', 6.0, 80, 3),
      ('R-401', 'Coffee Beans 1kg', 'pantry', 14.0, 50, 4),
    ]
    conn.executemany('INSERT INTO products VALUES (?,?,?,?,?,?)', products)

    stores = [
      ('S01', 'Central Store', 'Lisbon', 'north'),
      ('S02', 'Riverside Store', 'Porto', 'north'),
      ('S03', 'Beach Store', 'Faro', 'south'),
      ('S04', 'Mall Store', 'Lisbon', 'south'),
      ('S05', 'Tech Park Store', 'Braga', 'north'),
    ]
    conn.executemany('INSERT INTO stores VALUES (?,?,?,?)', stores)

    inventory_rows = []
    sales_rows = []
    for store_id, _, _, _ in stores:
      for sku, _, _, _, reorder_point, _ in products:
        # Some items deliberately start below their reorder point to give
        # the agent interesting decisions to make.
        if rng.random() < 0.18:
          on_hand = rng.randint(0, max(1, reorder_point // 2))
        else:
          on_hand = rng.randint(reorder_point, reorder_point * 4)
        inventory_rows.append((store_id, sku, on_hand))

        for day_offset in range(13, -1, -1):
          day = today - timedelta(days=day_offset)
          base = max(1, on_hand // 30)
          units = max(0, int(rng.gauss(base, max(1, base * 0.4))))
          sales_rows.append((store_id, sku, day.isoformat(), units))

    conn.executemany('INSERT INTO inventory VALUES (?,?,?)', inventory_rows)
    conn.executemany('INSERT INTO sales VALUES (NULL,?,?,?,?)', sales_rows)

    sites = [
      ('CS-11', 'Lisboa Centro', 'lisbon', 'healthy', 8.0),
      ('CS-22', 'Lisboa Parque', 'lisbon', 'healthy', 6.0),
      ('CS-33', 'Porto Boavista', 'porto', 'healthy', 9.0),
      ('CS-44', 'Porto Norte', 'porto', 'degraded', 4.0),
      ('CS-55', 'Faro Algarve', 'algarve', 'healthy', 5.0),
      ('CS-77', 'Braga Tecnopolo', 'braga', 'degraded', 2.5),
    ]
    conn.executemany('INSERT INTO cell_sites VALUES (?,?,?,?,?)', sites)

    metric_rows = []
    hour_now = datetime.now(timezone.utc).replace(
      minute=0, second=0, microsecond=0
    )
    for site_id, _, _, status, _ in sites:
      for hour_offset in range(23, -1, -1):
        hour = hour_now - timedelta(hours=hour_offset)
        if status == 'degraded':
          users = rng.randint(800, 2500)
          latency = rng.uniform(180.0, 420.0)
          loss = rng.uniform(2.5, 7.0)
        else:
          users = rng.randint(3000, 9000)
          latency = rng.uniform(15.0, 60.0)
          loss = rng.uniform(0.0, 0.6)
        metric_rows.append(
          (
            site_id,
            hour.isoformat(),
            users,
            round(latency, 2),
            round(loss, 2),
          )
        )

    conn.executemany(
      'INSERT INTO site_metrics VALUES (NULL,?,?,?,?,?)', metric_rows
    )

    technicians = [
      ('T-01', 'Ana Rocha', 'lisbon', 1),
      ('T-02', 'Bruno Silva', 'lisbon', 1),
      ('T-03', 'Carla Costa', 'porto', 1),
      ('T-04', 'Diogo Lima', 'porto', 0),
      ('T-05', 'Elsa Nunes', 'algarve', 1),
      ('T-06', 'Filipe Braga', 'braga', 1),
    ]
    conn.executemany('INSERT INTO technicians VALUES (?,?,?,?)', technicians)

    conn.commit()
    logger.info('ops mock db seeded')
  finally:
    conn.close()


def demo_summary(db_path: Path | str | None = None) -> dict:
  '''Return row counts per table (used by demos/tests).

  Args:
    db_path: Optional DB path override.

  Returns:
    Mapping of table name to row count.
  '''
  conn = connect(db_path)
  try:
    tables = [
      'products', 'stores', 'inventory', 'sales', 'restock_orders',
      'cell_sites', 'site_metrics', 'technicians', 'dispatch_orders',
    ]
    return {
      table: conn.execute(f'SELECT COUNT(*) AS n FROM {table}').fetchone()['n']
      for table in tables
    }
  finally:
    conn.close()
