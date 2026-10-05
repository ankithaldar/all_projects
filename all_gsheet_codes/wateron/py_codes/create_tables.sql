-- ============================================================================
-- Tables for versioned cumulative readings and billing tracking
-- ============================================================================

-- Stores every version of cumulative reading per meter per month
CREATE TABLE IF NOT EXISTS meter_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meter_id TEXT NOT NULL,
    apartment_id TEXT NOT NULL,
    owner_tenent_name TEXT,
    location TEXT,
    reading_date TEXT NOT NULL,     -- ISO date YYYY-MM-DD (first of month)
    cumulative_value REAL,          -- cumulative meter reading at end of month
    source_file TEXT NOT NULL,      -- original Excel file name
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(meter_id, reading_date, source_file)  -- same file version once
);

-- Latest cumulative reading per meter & month (view)
CREATE VIEW IF NOT EXISTS latest_cumulative AS
SELECT
    meter_id,
    apartment_id,
    owner_tenent_name,
    location,
    reading_date,
    cumulative_value,
    MAX(created_at) AS latest_version_time
FROM meter_readings
GROUP BY meter_id, reading_date;

-- Monthly consumption derived from latest cumulative readings
CREATE VIEW IF NOT EXISTS current_consumption AS
WITH ranked AS (
    SELECT
        meter_id,
        apartment_id,
        owner_tenent_name,
        location,
        reading_date,
        cumulative_value,
        LAG(cumulative_value) OVER (PARTITION BY meter_id ORDER BY reading_date) AS prev_cumulative
    FROM latest_cumulative
)
SELECT
    meter_id,
    apartment_id,
    owner_tenent_name,
    location,
    reading_date,
    cumulative_value - COALESCE(prev_cumulative, 0) AS monthly_consumption
FROM ranked;

-- Stores how much monthly consumption has already been billed
CREATE TABLE IF NOT EXISTS billed_consumption (
    apartment_id TEXT NOT NULL,
    reading_date TEXT NOT NULL,   -- month of consumption
    billed_amount REAL NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (apartment_id, reading_date)
);

-- Pending adjustments (output of each ETL run)
CREATE TABLE IF NOT EXISTS pending_adjustments (
    apartment_id TEXT NOT NULL,
    adjustment_amount REAL NOT NULL,
    reason TEXT,                    -- e.g., 'correction for Jan 2025'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
