#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The NBA player dataset, seeded into the local warehouse.

The source is a public binary-classification dataset of NBA players: 1,340 rows,
twenty per-game and career-rate numeric statistics, a player name, and a binary
label recording whether the player was still in the league five years later.
That makes it a genuine propensity problem -- predict a binary outcome from
numeric features -- which is exactly the task the framework is built for, and it
carries a real class imbalance rather than a synthetic one.

The label is already present, which is the one respect in which this is not the
production shape. Production derives the label in SQL, because the outcome is not
in the feature feed. To keep the local run honest, the label is *re-derived* into
the warehouse the way production derives it: the source rows are split into a
features table carrying no outcome and an events table carrying a deactivation
date, and the label is constructed by a query over the two. The catalog SQL then
derives the label itself, unmodified, so the local run exercises the real
label-derivation query rather than a shortcut around it.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting_ml_framework import constants
from forecasting_ml_framework.local.sqlite_warehouse import WRITE_TRUNCATE, SQLiteWarehouse
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: The public source of the dataset.
NBA_SOURCE_URL = (
  'https://raw.githubusercontent.com/readytensor/rt-datasets-binary-classification/'
  'refs/heads/main/datasets/raw/nba/raw/nba.csv'
)

#: The identifier column, which the contract routes to the identifier role.
IDENTIFIER_COLUMN = 'Name'

#: The outcome column in the source file.
LABEL_COLUMN = 'TARGET_5Yrs'

#: A source column name is not a valid column name in every sink, and ``%`` in a
#: column name requires quoting in SQL. The load normalises the names once, at
#: the boundary, so no query has to quote them.
_SOURCE_COLUMN_RENAMES = {
  'FG%': 'fg_pct',
  '3P%': 'three_p_pct',
  'FT%': 'ft_pct',
  '3P Made': 'three_p_made',
  '3PA': 'three_pa',
}

#: The feature columns in their source order. Recorded explicitly so a missing
#: or reordered column is a load-time error rather than a silently different
#: feature vector.
FEATURE_COLUMNS = (
  'GP', 'MIN', 'PTS', 'FGM', 'FGA', 'fg_pct', 'three_p_made', 'three_pa',
  'three_p_pct', 'FTM', 'FTA', 'ft_pct', 'OREB', 'DREB', 'REB', 'AST', 'STL',
  'BLK', 'TOV',
)


def download(url: str = NBA_SOURCE_URL, destination: str | Path | None = None) -> Path:
  """Fetch the dataset, reusing a cached copy when one is present.

  A local run should not depend on the network twice, and a repeated run should
  not silently pick up a different revision of the data, so the download is
  cached and the cache is the source of truth once it exists.

  Args:
    url: The dataset URL.
    destination: Where to place the file. Defaults to a file beside the
      warehouse directory.

  Returns:
    The path to the downloaded file.

  Raises:
    DatasetFetchError: If the download fails and no cached copy exists.
  """
  target = Path(destination) if destination is not None else default_data_dir() / 'nba.csv'
  if target.exists():
    LOGGER.info('Using the cached dataset', extra={'path': str(target), 'bytes': target.stat().st_size})
    return target

  target.parent.mkdir(parents=True, exist_ok=True)
  LOGGER.info('Downloading the dataset', extra={'url': url})
  try:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - a fixed https source
      payload = response.read()
  except (urllib.error.URLError, TimeoutError, OSError) as error:
    raise DatasetFetchError(
      f'Could not download the dataset from {url}: {error}. '
      f'Place a copy at {target} and re-run.'
    ) from error

  temporary = target.with_suffix('.partial')
  temporary.write_bytes(payload)
  temporary.replace(target)
  LOGGER.info('Downloaded the dataset', extra={'path': str(target), 'bytes': len(payload)})
  return target


def read_source(path: str | Path) -> pd.DataFrame:
  """Read the source CSV and normalise its column names.

  Args:
    path: The CSV path.

  Returns:
    A frame with snake_case columns, the identifier renamed, and numeric
    columns coerced.

  Raises:
    DatasetFetchError: If a required column is absent from the file.
  """
  frame = pd.read_csv(path)
  frame = frame.rename(columns=_SOURCE_COLUMN_RENAMES)

  missing = [column for column in (IDENTIFIER_COLUMN, LABEL_COLUMN, *FEATURE_COLUMNS) if column not in frame.columns]
  if missing:
    raise DatasetFetchError(
      f'The dataset at {path} is missing expected column(s) {missing}. '
      f'Present: {list(frame.columns)}'
    )

  for column in FEATURE_COLUMNS:
    frame[column] = pd.to_numeric(frame[column], errors='coerce')
  frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(int)
  frame[IDENTIFIER_COLUMN] = frame[IDENTIFIER_COLUMN].astype(str)
  return frame


def seed_warehouse(warehouse_path: str | Path, csv_path: str | Path | None = None) -> dict[str, int]:
  """Build the warehouse tables the catalog's queries expect.

  Three tables are created, matching the three the production catalog joins
  against:

  ``nba_features``
    The feature feed for one run date, carrying identifiers and statistics but
    **no outcome**. This is the production shape: the feature feed must not
    contain the label, or the model can be trained on its own answer.

  ``nba_deactivations``
    A snapshot of career-end events, carrying a date and a reason, so the label
    can be derived by a query rather than read.

  ``nba_customer_info``
    A snapshot of reference attributes as at the maturity horizon.

  Args:
    warehouse_path: The SQLite file to create.
    csv_path: The source CSV. Downloaded when omitted.

  Returns:
    A mapping of table name to row count.
  """
  source = read_source(csv_path if csv_path is not None else download())
  warehouse = SQLiteWarehouse(warehouse_path)
  try:
    # The identifier must be unique, and the source's ``Name`` is not: 29 names in
    # this dataset belong to more than one player, and "Charles Smith" appears
    # nine times. Using the name as the join key therefore multiplies rows -- a
    # 1,340-row feed becomes 1,526 -- and the run still trains, still scores, and
    # reports metrics computed over duplicated entities. Nothing errors, because
    # a fan-out join is a legitimate operation; it is only illegitimate against a
    # key that is supposed to identify one entity. So a synthetic unique key is
    # derived, and uniqueness is asserted rather than assumed.
    identifiers = pd.Series(
      [f'PLAYER-{index:05d}' for index in range(len(source))], index=source.index, dtype='string'
    )
    duplicates = int(identifiers.duplicated().sum())
    if duplicates:
      raise DatasetFetchError(
        f'The derived identifier is not unique: {duplicates} duplicate(s). A non-unique '
        f'join key inflates every downstream row count without raising.'
      )

    features = pd.DataFrame({
      'service_id': identifiers,
      'player_name': source[IDENTIFIER_COLUMN].to_numpy(),
      'run_date': '2024-01-01',
      **{column: source[column] for column in FEATURE_COLUMNS},
    })

    # A positive outcome is a career that ended. The date is spread
    # deterministically across a range that straddles the outcome window, so the
    # label query's date comparison genuinely decides the label rather than
    # always landing on one side of the boundary. A label that were simply a copy
    # of the source column would prove the date logic nothing.
    train_date = pd.Timestamp('2024-01-01')
    positives = source[LABEL_COLUMN] == 1
    offsets = source.index[source[LABEL_COLUMN] == 1] % 150 + 5
    deactivations = pd.DataFrame({
      'esn': identifiers[positives].to_numpy(),
      'deactivation_reason': ['VOLUNTARY'] * int(positives.sum()),
      'deactivation_date': [
        (train_date + pd.Timedelta(days=int(offset))).strftime('%Y-%m-%d') for offset in offsets
      ],
    })

    # The customer snapshot is dated at the label-maturity horizon, which is the
    # training date plus the outcome window and the settlement lag. The label
    # query filters on exactly that date, so a snapshot dated any other day joins
    # to nothing and every reference column comes back null -- a run that still
    # trains happily, on a frame that has silently lost half its schema.
    maturity_date = (train_date + pd.Timedelta(days=constants.MATURITY_WINDOW_DAYS)).strftime('%Y-%m-%d')
    customer_info = pd.DataFrame({
      'service_id': identifiers,
      'cust_id': [f'CUST-{index:05d}' for index in range(len(source))],
      'line_seq_id': '1',
      'account_ref': [f'ACCT-{index:05d}' for index in range(len(source))],
      'mobile_ref': [f'MOB-{index:05d}' for index in range(len(source))],
      'prcs_date': maturity_date,
    })

    counts = {
      'nba_features': warehouse.write(features, 'nba_features', disposition=WRITE_TRUNCATE),
      'nba_deactivations': warehouse.write(deactivations, 'nba_deactivations', disposition=WRITE_TRUNCATE),
      'nba_customer_info': warehouse.write(customer_info, 'nba_customer_info', disposition=WRITE_TRUNCATE),
    }
  finally:
    warehouse.close()

  LOGGER.info('Seeded the local warehouse', extra={'path': str(warehouse_path), 'tables': counts})
  return counts


def default_data_dir() -> Path:
  """Return the directory the local harness keeps its data in.

  Returns:
    The workspace root, from ``LOCAL_WORKSPACE`` or ``/tmp/forecasting_ml_local``.
  """
  import os  # noqa: PLC0415

  return Path(os.getenv('LOCAL_WORKSPACE', '/tmp/forecasting_ml_local'))


def reset_workspace() -> Path:
  """Remove and recreate the local workspace directory.

  A local run must start from a known state, and reusing a previous run's
  warehouse or object store is how a run appears to succeed while reading the
  last run's output.

  Returns:
    The recreated workspace root.
  """
  root = default_data_dir()
  if root.exists():
    shutil.rmtree(root)
  (root / 'data').mkdir(parents=True, exist_ok=True)
  (root / 'warehouse').mkdir(parents=True, exist_ok=True)
  (root / 'objects').mkdir(parents=True, exist_ok=True)
  LOGGER.info('Reset the local workspace', extra={'path': str(root)})
  return root


def workspace_paths() -> dict[str, str]:
  """Return the standard paths the local harness uses.

  Returns:
    A mapping of role to absolute path, covering the warehouse file, the object
    store and the downloaded dataset.
  """
  root = default_data_dir()
  return {
    'root': str(root),
    'warehouse': str(root / 'warehouse' / 'forecasting_ml.db'),
    'objects': str(root / 'objects'),
    'data': str(root / 'data'),
    'nba_csv': str(root / 'data' / 'nba.csv'),
  }


def describe_source(path: str | Path) -> dict[str, Any]:
  """Summarise the dataset for the run report.

  Args:
    path: The CSV path.

  Returns:
    A mapping of shape, class balance and feature list.
  """
  frame = read_source(path)
  positives = int(frame[LABEL_COLUMN].sum())
  total = len(frame)
  return {
    'rows': total,
    'columns': int(frame.shape[1]),
    'positives': positives,
    'negatives': total - positives,
    'positive_rate': round(positives / total, 4) if total else 0.0,
    'features': list(FEATURE_COLUMNS),
  }


class DatasetFetchError(RuntimeError):
  """Raised when the source dataset cannot be obtained or does not match."""
