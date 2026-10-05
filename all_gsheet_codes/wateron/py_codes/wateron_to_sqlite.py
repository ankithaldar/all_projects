#!/usr/bin/env python3
"""
Excel to SQLite ETL Pipeline

Reads water consumption data from Excel files (wide format with date columns),
transforms to long format, validates locations, parses dates flexibly,
and loads into a SQLite database using SQLAlchemy ORM.

If a file is processed again, all its previously loaded rows are removed first.
Handles missing apartment/owner values (propagates last valid value downwards)
and removes total rows/columns.
"""

import os
import sys
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql.expression import tuple_
from dateutil import parser as dateutil_parser

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Database Model
# -----------------------------------------------------------------------------
Base = declarative_base()

class ApartmentReading(Base):
  __tablename__ = "water_consumption"
  apartment_id = Column(String, nullable=False, primary_key=True)
  dates = Column(String, nullable=False, primary_key=True)
  meter_id = Column(String, nullable=False, primary_key=True)
  owner_tenent_name = Column(String)
  location = Column(String)
  water_consumption = Column(Float, nullable=True)
  source_file = Column(String, nullable=False)

# -----------------------------------------------------------------------------
# Date parsing strategies (Strategy Pattern)
# -----------------------------------------------------------------------------
class DateParseStrategy:
  def parse(self, date_str: str) -> Optional[str]:
    raise NotImplementedError

class ISOFormatStrategy(DateParseStrategy):
  def parse(self, date_str: str) -> Optional[str]:
    try:
      dt = datetime.strptime(date_str, "%Y-%m-%d")
      return dt.strftime("%Y-%m-%d")
    except ValueError:
      return None

class DDMMMYYYYStrategy(DateParseStrategy):
  def parse(self, date_str: str) -> Optional[str]:
    try:
      dt = datetime.strptime(date_str, "%d-%b-%Y")
      return dt.strftime("%Y-%m-%d")
    except ValueError:
      return None

class YYYYMMDDStrategy(DateParseStrategy):
  def parse(self, date_str: str) -> Optional[str]:
    try:
      dt = datetime.strptime(date_str, "%Y%m%d")
      return dt.strftime("%Y-%m-%d")
    except ValueError:
      return None

class FallbackDateutilStrategy(DateParseStrategy):
  def parse(self, date_str: str) -> Optional[str]:
    try:
      dt = dateutil_parser.parse(date_str, fuzzy=False)
      return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError, TypeError):
      return None

class CompositeDateParser(DateParseStrategy):
  def __init__(self, strategies: List[DateParseStrategy]):
    self.strategies = strategies

  def parse(self, date_str: str) -> Optional[str]:
    for strategy in self.strategies:
      result = strategy.parse(date_str)
      if result is not None:
        return result
    logger.warning(f"Unable to parse date: {date_str}")
    return None

# -----------------------------------------------------------------------------
# Location Validator
# -----------------------------------------------------------------------------
class LocationValidator:
  ALLOWED_LOCATIONS = {'K', 'B1', 'B2', 'U', 'S', 'K+B1', 'B3', 'CW', 'SP'}

  @classmethod
  def validate(cls, location: str) -> bool:
    if location is None:
      logger.warning("Location is None - skipping record")
      return False
    if '-' in location:
      logger.warning(f"Location contains '-' -> skipping record: {location}")
      return False
    if location not in cls.ALLOWED_LOCATIONS:
      logger.warning(f"Invalid location '{location}' - skipping record")
      return False
    return True

# -----------------------------------------------------------------------------
# Excel File Reader
# -----------------------------------------------------------------------------
class ExcelFileReader:
  @staticmethod
  def read(file_path: Path) -> Optional[pd.DataFrame]:
    try:
      df = pd.read_excel(file_path, dtype=str, header=0)
      logger.info(f"Read {file_path.name} - {len(df)} rows")
      return df
    except Exception as e:
      logger.error(f"Failed to read {file_path.name}: {e}")
      return None

# -----------------------------------------------------------------------------
# Data Cleaner (handles - propagation and total removal)
# -----------------------------------------------------------------------------

class DataCleaner:
  """Preprocesses the raw DataFrame before transformation."""

  FIXED_COLUMNS = ['Apartment', 'Owner/Tenent', 'Location', 'Meter No.']

  @staticmethod
  def _is_total_row(row: pd.Series) -> bool:
    apt = row.get('Apartment', '')
    if pd.isna(apt):
      return True
    apt_str = str(apt).strip().lower()
    return 'total' in apt_str or 'sum' in apt_str

  @staticmethod
  def _is_total_column(col_name: str) -> bool:
    col_lower = str(col_name).strip().lower()
    return 'total' in col_lower or 'sum' in col_lower

  def clean(self, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
      return df

    # 1. Remove total columns
    keep_cols = [c for c in df.columns if not self._is_total_column(c)]
    df = df[keep_cols].copy()
    logger.debug(f"After removing total columns: {len(df.columns)} columns")

    # 2. Remove total rows based on 'Apartment'
    if 'Apartment' in df.columns:
      mask = ~df['Apartment'].astype(str).str.lower().str.contains('total|sum', na=False)
      df = df[mask].copy()
      logger.debug(f"After removing total rows: {len(df)} rows")
    else:
      logger.warning("No 'Apartment' column, cannot remove total rows")

    # 3. Replace '-' and empty strings with NaN, then forward fill
    for col in ['Apartment', 'Owner/Tenent']:
      if col in df.columns:
        df[col] = df[col].replace(['-', ''], pd.NA)
        # Use ffill() instead of fillna(method='ffill')
        df[col] = df[col].ffill()
        # Drop any rows where Apartment is still missing after ffill
        if col == 'Apartment':
          df = df[df[col].notna()].copy()

    # 4. Strip whitespace and clean up
    for col in df.select_dtypes(include=['object']).columns:
      df[col] = df[col].astype(str).str.strip()
      df[col] = df[col].replace('nan', pd.NA)

    return df

# -----------------------------------------------------------------------------
# Wide to Long Transformer
# -----------------------------------------------------------------------------
class WideToLongTransformer:
  FIXED_COLUMNS = ['Apartment', 'Owner/Tenent', 'Location', 'Meter No.']

  def __init__(self, date_parser: DateParseStrategy):
    self.date_parser = date_parser

  def transform(self, df: pd.DataFrame, source_file: str) -> List[Dict[str, Any]]:
    # Check required fixed columns
    missing = [c for c in self.FIXED_COLUMNS if c not in df.columns]
    if missing:
      raise ValueError(f"Missing required columns: {missing}")

    date_cols = [col for col in df.columns if col not in self.FIXED_COLUMNS]
    if not date_cols:
      logger.warning("No date columns found after cleaning")
      return []

    # Melt
    id_vars = self.FIXED_COLUMNS
    df_long = pd.melt(
      df,
      id_vars=id_vars,
      value_vars=date_cols,
      var_name='date_str',
      value_name='water_consumption'
    )
    df_long = df_long.dropna(subset=['Apartment'])

    records = []
    for _, row in df_long.iterrows():
      # Parse date
      iso_date = self.date_parser.parse(row['date_str'])
      if iso_date is None:
        logger.debug(f"Skipping unparsable date: {row['date_str']}")
        continue

      # Validate location
      location = row['Location']
      if not LocationValidator.validate(location):
        continue

      # Meter ID must be present
      meter_id_raw = row['Meter No.']
      if pd.isna(meter_id_raw) or str(meter_id_raw).strip() == '':
        logger.warning(f"Skipping record: missing meter_id for apartment {row['Apartment']}")
        continue
      meter_id = str(meter_id_raw).strip()

      # Consumption
      consumption_raw = row['water_consumption']
      try:
        consumption = float(consumption_raw) if pd.notna(consumption_raw) else None
      except (ValueError, TypeError):
        consumption = None
        logger.warning(f"Non-numeric consumption '{consumption_raw}' for apt {row['Apartment']} on {iso_date}")

      record = {
        'apartment_id': str(row['Apartment']).strip(),
        'owner_tenent_name': str(row['Owner/Tenent']).strip() if pd.notna(row['Owner/Tenent']) else None,
        'location': location,
        'meter_id': meter_id,
        'dates': iso_date,
        'water_consumption': consumption,
        'source_file': source_file,
      }
      records.append(record)

    logger.info(f"Transformed {len(records)} valid records")
    return records

# -----------------------------------------------------------------------------
# Database Repository
# -----------------------------------------------------------------------------
class WaterConsumptionRepository:
  def __init__(self, session: Session, remove_old_file_data: bool = True):
    self.session = session
    self.remove_old_file_data = remove_old_file_data

  def delete_by_source_file(self, source_file: str) -> int:
    try:
      deleted = self.session.query(ApartmentReading).filter(
        ApartmentReading.source_file == source_file
      ).delete()
      self.session.commit()
      if deleted:
        logger.info(f"Removed {deleted} existing rows from file '{source_file}'")
      return deleted
    except Exception as e:
      self.session.rollback()
      logger.error(f"Failed to delete rows for file '{source_file}': {e}")
      raise

  def bulk_insert(self, records: List[Dict[str, Any]], batch_size: int = 1000) -> int:
    if not records:
      return 0
    inserted = 0
    for i in range(0, len(records), batch_size):
      batch = records[i:i+batch_size]
      try:
        self.session.bulk_insert_mappings(ApartmentReading, batch)
        self.session.commit()
        inserted += len(batch)
        logger.debug(f"Inserted batch of {len(batch)} records")
      except Exception as e:
        self.session.rollback()
        logger.error(f"Failed to insert batch: {e}")
    logger.info(f"Inserted {inserted} records")
    return inserted

  @staticmethod
  def create_tables(engine):
    Base.metadata.create_all(engine)

# -----------------------------------------------------------------------------
# Main ETL Processor
# -----------------------------------------------------------------------------
class ExcelToDatabaseProcessor:
  def __init__(
    self,
    reader: ExcelFileReader,
    cleaner: DataCleaner,
    transformer: WideToLongTransformer,
    repository: WaterConsumptionRepository
  ):
    self.reader = reader
    self.cleaner = cleaner
    self.transformer = transformer
    self.repository = repository

  def process_file(self, file_path: Path) -> int:
    source_file = file_path.name
    df = self.reader.read(file_path)
    if df is None:
      return 0

    try:
      cleaned_df = self.cleaner.clean(df)
      if cleaned_df.empty:
        logger.warning(f"No data left after cleaning in {source_file}")
        return 0
      records = self.transformer.transform(cleaned_df, source_file)
    except Exception as e:
      logger.error(f"Processing failed for {file_path.name}: {e}")
      return 0

    if not records:
      logger.info(f"No valid records to insert from {file_path.name}")
      return 0

    if self.repository.remove_old_file_data:
      self.repository.delete_by_source_file(source_file)

    return self.repository.bulk_insert(records)

  def process_folder(self, folder_path: Path) -> None:
    if not folder_path.exists() or not folder_path.is_dir():
      logger.error(f"Folder does not exist: {folder_path}")
      return

    xlsx_files = list(folder_path.glob("*.xlsx"))
    if not xlsx_files:
      logger.warning(f"No .xlsx files found in {folder_path}")
      return

    total_inserted = 0
    for file_path in xlsx_files:
      logger.info(f"Processing {file_path.name}...")
      inserted = self.process_file(file_path)
      total_inserted += inserted

    logger.info(f"Processing complete. Total inserted records: {total_inserted}")

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main(folder_path_str: str, db_path: str = "water_consumption.db", remove_old_data: bool = True):
  engine = create_engine(f'sqlite:///{db_path}', echo=False)
  WaterConsumptionRepository.create_tables(engine)
  SessionLocal = sessionmaker(bind=engine)
  session = SessionLocal()

  try:
    date_parser = CompositeDateParser([
      ISOFormatStrategy(),
      DDMMMYYYYStrategy(),
      YYYYMMDDStrategy(),
      FallbackDateutilStrategy()
    ])
    reader = ExcelFileReader()
    cleaner = DataCleaner()
    transformer = WideToLongTransformer(date_parser)
    repository = WaterConsumptionRepository(session, remove_old_file_data=remove_old_data)

    processor = ExcelToDatabaseProcessor(reader, cleaner, transformer, repository)
    processor.process_folder(Path(folder_path_str))

  except Exception as e:
    logger.exception(f"Unexpected error: {e}")
  finally:
    session.close()

if __name__ == "__main__":
  if len(sys.argv) > 1:
    folder = sys.argv[1]
  else:
    folder = input("Enter folder path containing Excel files: ").strip()
    if not folder:
      folder = "./data"
  db_path = sys.argv[2] if len(sys.argv) > 2 else "water_consumption.db"
  remove = sys.argv[3].lower() != "false" if len(sys.argv) > 3 else True

  main(folder, db_path, remove)
