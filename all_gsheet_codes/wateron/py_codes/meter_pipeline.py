#!/usr/bin/env python3
"""
Water Meter Consumption Pipeline
Processes versioned Excel files, computes deltas between monthly totals,
and generates billing adjustments for future months.

Usage:
python meter_pipeline.py --folder ./data --db ./billing.db --process full
python meter_pipeline.py --folder ./data --db ./billing.db --process load
python meter_pipeline.py --folder ./data --db ./billing.db --process compute_deltas

Tests:
pytest meter_pipeline.py
"""

import argparse
import logging
import re
import sys
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from calendar import monthrange
from collections import defaultdict

import pandas as pd
import numpy as np

from sqlalchemy import (
    create_engine, Column, Integer, String, Date, Float, Index, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dateutil.relativedelta import relativedelta


# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ------------------------------
# SQLAlchemy Models
# ------------------------------
Base = declarative_base()


class MeterReading(Base):
    __tablename__ = 'meter_readings'

    id = Column(Integer, primary_key=True)
    apartment_id = Column(String(50), nullable=False)
    owner_tenent_name = Column(String(200), nullable=False)
    location = Column(String(50), nullable=False)
    meter_id = Column(String(50), nullable=False)
    reading_date = Column(Date, nullable=False)
    daily_consumption = Column(Float, nullable=False)
    source_file = Column(String(255), nullable=False)
    target_month = Column(String(7), nullable=False)   # YYYY-MM
    download_date = Column(Date, nullable=False)
    version_sequence = Column(Integer, nullable=False)

    __table_args__ = (
        Index(
            'idx_reading_target_version',
            'target_month',
            'apartment_id',
            'meter_id',
            'version_sequence'
        ),
        Index('idx_source_file', 'source_file'),
    )


class MonthlyTotal(Base):
    """Materialized monthly totals per version (optional)."""

    __tablename__ = 'monthly_totals'

    id = Column(Integer, primary_key=True)
    apartment_id = Column(String(50), nullable=False)
    meter_id = Column(String(50), nullable=False)
    target_month = Column(String(7), nullable=False)
    version_sequence = Column(Integer, nullable=False)
    total_consumption = Column(Float, nullable=False)

    __table_args__ = (
        Index(
            'idx_monthly_unique',
            'target_month',
            'apartment_id',
            'meter_id',
            'version_sequence',
            unique=True
        ),
    )


class BillingAdjustment(Base):
    __tablename__ = 'billing_adjustments'

    id = Column(Integer, primary_key=True)
    apartment_id = Column(String(50), nullable=False)
    meter_id = Column(String(50), nullable=False)
    original_target_month = Column(String(7), nullable=False)
    adjustment_applied_to_month = Column(String(7), nullable=False)
    delta_consumption = Column(Float, nullable=False)
    source_version_pair = Column(String(255), nullable=False)

    __table_args__ = (
        Index('idx_adj_original', 'original_target_month'),
        Index('idx_adj_applied', 'adjustment_applied_to_month'),
    )


# ------------------------------
# File & Name Parsing
# ------------------------------
def parse_filename(filename: str) -> Tuple[date, str]:
    """
    Parse filename like '20260202_202601.xlsx'
    Returns (download_date, target_month YYYY-MM)
    """
    match = re.match(r'(\d{8})_(\d{6})\.xlsx$', filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")

    download_str, target_str = match.groups()

    download_date = datetime.strptime(download_str, '%Y%m%d').date()
    target_month = datetime.strptime(target_str, '%Y%m').strftime('%Y-%m')

    return download_date, target_month


# ------------------------------
# Date Parsing Strategies
# ------------------------------
class DateParser:
    """Parse dates, inferring year from target_month when missing."""

    MONTH_MAP = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    PATTERN_DAY_MONTH = re.compile(r'^(\d{1,2})\s+([A-Za-z]{3})$', re.IGNORECASE)
    PATTERN_MONTH_DAY = re.compile(r'^([A-Za-z]{3})\s+(\d{1,2})$', re.IGNORECASE)

    @classmethod
    def parse(cls, value: Any, target_year: int, target_month_num: int) -> Optional[date]:
        """
        Parse a date value. If year is missing, use target_year.
        Returns None if parsing fails or date does not match target month.
        """
        if pd.isna(value):
            return None

        # 1. Excel serial number
        if isinstance(value, (float, int)):
            try:
                dt = pd.to_datetime(value, unit='D', origin='1899-12-30')
                if dt.year == target_year and dt.month == target_month_num:
                    return dt.date()
                else:
                    logger.warning(
                        f"Excel date {value} -> {dt.date()} outside target month "
                        f"{target_year}-{target_month_num:02d}. Skipping."
                    )
                    return None
            except Exception:
                pass

        # 2. String parsing
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None

            # Try formats with explicit year
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']:
                try:
                    dt = datetime.strptime(value, fmt)
                    if dt.year == target_year and dt.month == target_month_num:
                        return dt.date()
                    else:
                        logger.warning(
                            f"Date '{value}' parsed as {dt.date()} but does not match "
                            f"target month {target_year}-{target_month_num:02d}. Skipping."
                        )
                        return None
                except ValueError:
                    continue

            # Try day-month without year (e.g., "11 Jan")
            match = cls.PATTERN_DAY_MONTH.match(value)
            if match:
                day = int(match.group(1))
                month_abbr = match.group(2).lower()
                month = cls.MONTH_MAP.get(month_abbr)

                if month and month == target_month_num:
                    return date(target_year, month, day)
                else:
                    if month and month != target_month_num:
                        logger.warning(
                            f"Date '{value}' month {month_abbr} does not match "
                            f"target month {target_month_num}. Skipping."
                        )
                    return None

            # Try month-day without year (e.g., "Jan 11")
            match = cls.PATTERN_MONTH_DAY.match(value)
            if match:
                month_abbr = match.group(1).lower()
                day = int(match.group(2))
                month = cls.MONTH_MAP.get(month_abbr)

                if month and month == target_month_num:
                    return date(target_year, month, day)
                else:
                    if month and month != target_month_num:
                        logger.warning(
                            f"Date '{value}' month {month_abbr} does not match "
                            f"target month {target_month_num}. Skipping."
                        )
                    return None

        # Fallback: pandas
        try:
            dt = pd.to_datetime(value)
            if dt.year == target_year and dt.month == target_month_num:
                return dt.date()
            else:
                logger.warning(
                    f"Date '{value}' parsed as {dt.date()} but does not match "
                    f"target month {target_year}-{target_month_num:02d}. Skipping."
                )
                return None
        except Exception:
            pass

        # Parsing failed completely
        logger.warning(
            f"Could not parse date '{value}' in target month "
            f"{target_year}-{target_month_num:02d}. Skipping."
        )
        return None


# ------------------------------
# Location Validator
# ------------------------------
class LocationValidator:
    ALLOWED_LOCATIONS: Set[str] = {'K', 'B1', 'B2', 'U', 'S', 'K+B1', 'B3', 'CW', 'SP'}

    @classmethod
    def is_valid(cls, location: str) -> bool:
        """Return True if location is allowed."""
        loc_clean = str(location).strip()
        return loc_clean in cls.ALLOWED_LOCATIONS

    @classmethod
    def validate_row(cls, location: str) -> bool:
        """Log warning if invalid, return True if allowed (we keep row)."""
        loc_clean = str(location).strip()
        if loc_clean not in cls.ALLOWED_LOCATIONS:
            logger.warning(
                f"Invalid location code '{loc_clean}' - not in allowed set. "
                f"Proceeding but check data."
            )
            return False
        return True


# ------------------------------
# Excel Processing & Inheritance
# ------------------------------
class ExcelLoader:
    """Loads a single Excel file, cleans totals, fills inheritance, returns meter reading rows."""

    # Canonical metadata columns used internally by the pipeline.
    # Both Set 1 and Set 2 headers are normalized into these names.
    META_COLS = [
        'Apartment',
        'Owner/Tenant',
        'Location',
        'Meter No.'
    ]

    @staticmethod
    def _canonical_header_name(col: str) -> Optional[str]:
        """
        Normalize incoming header names.

        Supported mappings:
            Apartment -> Apartment
            Owner -> Owner/Tenant
            Owner/Tenant -> Owner/Tenant
            Owner/Tenent -> Owner/Tenant
            Location -> Location
            Meter_No -> Meter No.
            Meter No. -> Meter No.
            Meter Number -> Meter No.
        """
        key = re.sub(r'[^a-z0-9]+', '', str(col).strip().lower())

        header_map = {
            'apartment': 'Apartment',

            'owner': 'Owner/Tenant',
            'ownertenant': 'Owner/Tenant',
            'ownertenent': 'Owner/Tenant',

            'location': 'Location',

            'meterno': 'Meter No.',
            'meternumber': 'Meter No.',
        }

        return header_map.get(key)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename incoming columns to canonical column names expected by the pipeline.
        This allows both Set 1 and Set 2 files to be appended into the same table.
        """
        rename_map = {}

        for col in df.columns:
            canonical = ExcelLoader._canonical_header_name(col)
            if canonical:
                rename_map[col] = canonical

        df = df.rename(columns=rename_map)

        # Detect multiple source columns mapped to the same canonical column.
        for canonical_col in ExcelLoader.META_COLS:
            count = list(df.columns).count(canonical_col)
            if count > 1:
                raise ValueError(
                    f"Multiple columns mapped to '{canonical_col}'. "
                    f"Please check the Excel headers."
                )

        missing = [col for col in ExcelLoader.META_COLS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns after normalization: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        return df

    @staticmethod
    def _forward_fill_apartment_owner(df: pd.DataFrame) -> pd.DataFrame:
        """Replace empty/'-'/' ' with forward fill from previous row."""
        df_clean = df.copy()

        for col in ['Apartment', 'Owner/Tenant']:
            df_clean[col] = df_clean[col].replace(['-', '—', '', ' '], np.nan)

        df_clean[['Apartment', 'Owner/Tenant']] = (
            df_clean[['Apartment', 'Owner/Tenant']].ffill()
        )

        return df_clean

    @staticmethod
    def _filter_total_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where Apartment or Owner/Tenant contains 'total' (case-insensitive)."""
        mask_apt = (
            df['Apartment']
            .astype(str)
            .str.lower()
            .str.contains('total', na=False)
        )

        mask_owner = (
            df['Owner/Tenant']
            .astype(str)
            .str.lower()
            .str.contains('total', na=False)
        )

        return df[~(mask_apt | mask_owner)]

    @staticmethod
    def _filter_total_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove any column whose name contains 'total' (case-insensitive)."""
        cols_to_drop = [col for col in df.columns if 'total' in col.lower()]
        return df.drop(columns=cols_to_drop)

    @staticmethod
    def _filter_dash_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where Location or Meter No. is '-' or space."""
        mask = (
            (df['Location'] == '-') |
            (df['Meter No.'] == '-') |
            (df['Location'] == ' ') |
            (df['Meter No.'] == ' ')
        )

        removed = mask.sum()

        if removed:
            logger.info(
                f"Removing {removed} row(s) where Location or Meter No. is '-' or space"
            )

        return df[~mask]

    @staticmethod
    def _validate_meter_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows missing Meter No. or Location (after inheritance and dash removal)."""
        # First remove dash rows (they would survive dropna because '-' is not NaN)
        df = ExcelLoader._filter_dash_rows(df)

        # Then drop rows where either column is empty/NaN
        df = df.dropna(subset=['Meter No.', 'Location'])

        return df

    def load_to_records(
        self,
        file_path: Path,
        target_month: str,
        download_date: date
    ) -> List[Dict[str, Any]]:
        """
        Read Excel, normalize headers, apply inheritance, remove totals,
        and return list of reading dicts.
        """
        logger.info(f"Processing file: {file_path.name}")

        df = pd.read_excel(file_path, dtype=str)

        # Normalize Set 1 and Set 2 headers into canonical metadata columns.
        df = self._normalize_columns(df)

        # Identify metadata columns
        meta_cols = self.META_COLS

        # Remove total columns first
        df = self._filter_total_columns(df)

        # Re-identify date columns after removal
        date_cols = [c for c in df.columns if c not in meta_cols]

        # Apply inheritance & remove total rows
        df = self._forward_fill_apartment_owner(df)
        df = self._filter_total_rows(df)
        df = self._validate_meter_rows(df)

        # Validate locations (warn only)
        df['_loc_valid'] = df['Location'].apply(
            lambda x: LocationValidator.validate_row(x)
        )

        # Melt date columns into rows
        id_vars = meta_cols
        value_vars = [c for c in date_cols if c in df.columns]

        if not value_vars:
            logger.warning(f"No date columns found in {file_path.name}")
            return []

        df_melted = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='reading_date_str',
            value_name='daily_consumption'
        )

        # Clean consumption values
        df_melted['daily_consumption'] = (
            pd.to_numeric(df_melted['daily_consumption'], errors='coerce')
            .fillna(0.0)
        )

        # Extract target year and month for date validation
        target_year, target_month_num = map(int, target_month.split('-'))

        # Build records
        records = []

        for _, row in df_melted.iterrows():
            # Parse date using our intelligent parser
            reading_date = DateParser.parse(
                row['reading_date_str'],
                target_year=target_year,
                target_month_num=target_month_num
            )

            if reading_date is None:
                continue   # warning already logged inside parse()

            records.append({
                'apartment_id': str(row['Apartment']).strip(),
                'owner_tenent_name': str(row['Owner/Tenant']).strip(),
                'location': str(row['Location']).strip(),
                'meter_id': str(row['Meter No.']).strip(),
                'reading_date': reading_date,
                'daily_consumption': row['daily_consumption'],
                'source_file': file_path.name,
                'target_month': target_month,
                'download_date': download_date,
                'version_sequence': 0,   # <-- FIX
            })

        logger.info(f"Extracted {len(records)} daily readings from {file_path.name}")

        return records


# ------------------------------
# Database Repository
# ------------------------------
class DatabaseRepository:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def file_already_loaded(self, source_file: str) -> bool:
        with self.get_session() as sess:
            return sess.query(MeterReading).filter(
                MeterReading.source_file == source_file
            ).first() is not None

    def insert_readings(self, readings: List[Dict[str, Any]]) -> int:
        with self.get_session() as sess:
            objects = [MeterReading(**r) for r in readings]
            sess.bulk_save_objects(objects)
            sess.commit()
            return len(objects)

    def reassign_version_sequences(self, target_month: Optional[str] = None):
        """Recompute version_sequence per target_month based on download_date order."""
        with self.get_session() as sess:
            # Get unique (target_month, download_date) groups
            query = sess.query(
                MeterReading.target_month,
                MeterReading.download_date,
                func.min(MeterReading.id).label('min_id')
            ).group_by(
                MeterReading.target_month,
                MeterReading.download_date
            )

            if target_month:
                query = query.filter(MeterReading.target_month == target_month)

            groups = query.all()

            month_groups = defaultdict(set)

            for tm, dd, _ in groups:
                month_groups[tm].add(dd)

            for tm, dates in month_groups.items():
                sorted_dates = sorted(dates)
                date_to_seq = {d: i + 1 for i, d in enumerate(sorted_dates)}

                for dd, seq in date_to_seq.items():
                    sess.query(MeterReading).filter(
                        MeterReading.target_month == tm,
                        MeterReading.download_date == dd
                    ).update(
                        {MeterReading.version_sequence: seq},
                        synchronize_session=False
                    )

            sess.commit()

        logger.info(
            f"Reassigned version sequences for {len(month_groups)} target month(s)"
        )

    def get_monthly_totals_by_version(self, target_month: str) -> List[Dict]:
        """Return monthly totals per (apartment, meter, version) for a target month."""
        with self.get_session() as sess:
            results = sess.query(
                MeterReading.apartment_id,
                MeterReading.meter_id,
                MeterReading.version_sequence,
                func.sum(MeterReading.daily_consumption).label('total_consumption')
            ).filter(
                MeterReading.target_month == target_month
            ).group_by(
                MeterReading.apartment_id,
                MeterReading.meter_id,
                MeterReading.version_sequence
            ).order_by(
                MeterReading.apartment_id,
                MeterReading.meter_id,
                MeterReading.version_sequence
            ).all()

            return [
                {
                    'apartment_id': r.apartment_id,
                    'meter_id': r.meter_id,
                    'version_sequence': r.version_sequence,
                    'total_consumption': float(r.total_consumption)
                }
                for r in results
            ]

    def clear_adjustments(self, original_target_month: str):
        with self.get_session() as sess:
            sess.query(BillingAdjustment).filter(
                BillingAdjustment.original_target_month == original_target_month
            ).delete()
            sess.commit()

    def insert_adjustments(self, adjustments: List[Dict[str, Any]]):
        with self.get_session() as sess:
            objs = [BillingAdjustment(**adj) for adj in adjustments]
            sess.bulk_save_objects(objs)
            sess.commit()

        logger.info(f"Inserted {len(adjustments)} billing adjustments")

    def get_existing_target_months(self) -> List[str]:
        with self.get_session() as sess:
            return [
                r[0]
                for r in sess.query(MeterReading.target_month).distinct().all()
            ]


# ------------------------------
# Delta Engine
# ------------------------------
class DeltaCalculator:
    def __init__(self, repo: DatabaseRepository):
        self.repo = repo

    @staticmethod
    def add_months(month_str: str, months_to_add: int) -> str:
        dt = datetime.strptime(month_str, '%Y-%m')
        new_dt = dt + relativedelta(months=months_to_add)
        return new_dt.strftime('%Y-%m')

    def compute_deltas_for_month(self, target_month: str):
        logger.info(f"Computing deltas for {target_month}")

        totals_by_pair = self.repo.get_monthly_totals_by_version(target_month)

        grouped = defaultdict(list)

        for rec in totals_by_pair:
            key = (rec['apartment_id'], rec['meter_id'])
            grouped[key].append(rec)

        adjustments = []

        for (apartment_id, meter_id), versions in grouped.items():
            totals = [v['total_consumption'] for v in versions]

            if len(totals) < 2:
                continue

            for idx in range(len(totals) - 1):
                delta = totals[idx + 1] - totals[idx]

                if abs(delta) < 1e-6:
                    continue

                applied_month = self.add_months(target_month, idx + 1)
                source_pair = f"v{idx + 1} -> v{idx + 2}"

                adjustments.append({
                    'apartment_id': apartment_id,
                    'meter_id': meter_id,
                    'original_target_month': target_month,
                    'adjustment_applied_to_month': applied_month,
                    'delta_consumption': delta,
                    'source_version_pair': source_pair
                })

        self.repo.clear_adjustments(target_month)

        if adjustments:
            self.repo.insert_adjustments(adjustments)
        else:
            logger.info(f"No non-zero deltas for {target_month}")

    def run_for_all_months(self):
        months = self.repo.get_existing_target_months()

        for month in months:
            self.compute_deltas_for_month(month)


# ------------------------------
# Main Pipeline
# ------------------------------
class Pipeline:
    def __init__(self, folder_path: Path, db_url: str):
        self.folder = folder_path
        self.repo = DatabaseRepository(db_url)
        self.loader = ExcelLoader()
        self.delta_calc = DeltaCalculator(self.repo)

    def load_all_files(self):
        excel_files = list(self.folder.glob("*.xlsx"))

        if not excel_files:
            logger.warning(f"No Excel files found in {self.folder}")
            return

        total_records = 0

        for file_path in excel_files:
            if self.repo.file_already_loaded(file_path.name):
                logger.info(f"Skipping already loaded file: {file_path.name}")
                continue

            try:
                download_date, target_month = parse_filename(file_path.name)

                records = self.loader.load_to_records(
                    file_path,
                    target_month,
                    download_date
                )

                if records:
                    inserted = self.repo.insert_readings(records)
                    total_records += inserted

            except Exception as e:
                logger.exception(f"Failed to process {file_path.name}: {e}")

        self.repo.reassign_version_sequences()

        logger.info(f"Total new records inserted: {total_records}")

    def run_full(self):
        self.load_all_files()
        self.delta_calc.run_for_all_months()
        logger.info("Full pipeline finished.")


# ------------------------------
# CLI Entry Point
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Water meter billing adjustment pipeline"
    )

    parser.add_argument(
        '--folder',
        required=True,
        help="Path to folder containing Excel files"
    )

    parser.add_argument(
        '--db',
        required=True,
        help="SQLite database file path (e.g., billing.db)"
    )

    parser.add_argument(
        '--process',
        choices=['load', 'compute_deltas', 'full'],
        required=True,
        help="Pipeline step to execute"
    )

    args = parser.parse_args()

    folder_path = Path(args.folder)

    if not folder_path.exists() or not folder_path.is_dir():
        logger.error(f"Folder does not exist: {folder_path}")
        sys.exit(1)

    db_url = f"sqlite:///{args.db}"

    pipeline = Pipeline(folder_path, db_url)

    if args.process == 'load':
        pipeline.load_all_files()
    elif args.process == 'compute_deltas':
        pipeline.delta_calc.run_for_all_months()
    elif args.process == 'full':
        pipeline.run_full()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] not in ['--folder', '--db', '--process']:
        sys.exit(pytest.main([__file__] + sys.argv[1:]))
    else:
        main()
