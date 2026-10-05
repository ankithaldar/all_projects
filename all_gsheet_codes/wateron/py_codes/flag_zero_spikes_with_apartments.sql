DROP VIEW IF EXISTS water_consumption;
CREATE VIEW IF NOT EXISTS water_consumption AS
SELECT
  apartment_id,
  reading_dates,
  meter_id,
  owner_tenent_name,
  location,
  water_consumption,
  source_file
FROM (
  SELECT
    apartment_id,
    reading_date AS reading_dates,
    meter_id,
    owner_tenent_name,
    location,
    CASE WHEN daily_consumption <= 0.5 THEN 0 ELSE daily_consumption END AS water_consumption,
    source_file,
    ROW_NUMBER() OVER(PARTITION BY apartment_id, location, reading_date ORDER BY version_sequence DESC) AS rn_1
  FROM meter_readings
  WHERE 1=1
)
WHERE rn_1=1
;

-- ============================================================================
-- Flag meters with long zero-consumption streaks followed by a spike
-- Thresholds are defined as variables at the top for easy adjustment
-- ============================================================================
DROP VIEW IF EXISTS flag_meters_with_long_zero_consumption;
CREATE VIEW IF NOT EXISTS flag_meters_with_long_zero_consumption AS
WITH
-- Define thresholds as variables
params AS (
    SELECT
        30 AS min_zero_days,      -- minimum consecutive zero days to consider a streak
        3000 AS spike_threshold   -- minimum consumption value to consider a spike
),

-- Add previous row info for each reading
lagged AS (
    SELECT
        meter_id,
        apartment_id,
        owner_tenent_name,
        location,
        reading_dates,
        water_consumption,
        LAG(reading_dates) OVER (PARTITION BY apartment_id, location ORDER BY reading_dates) AS prev_dates,
        LAG(water_consumption) OVER (PARTITION BY apartment_id, location ORDER BY reading_dates) AS prev_consumption
    FROM water_consumption
    WHERE water_consumption IS NOT NULL
),

-- Identify start of each zero streak
streak_starts AS (
    SELECT
        apartment_id,
        location,
        meter_id,
        reading_dates AS streak_start,
        ROW_NUMBER() OVER (PARTITION BY apartment_id, location ORDER BY reading_dates) AS start_rn
    FROM lagged
    WHERE water_consumption = 0
      AND (prev_consumption IS NULL OR prev_consumption != 0)
),

-- Find end of each streak
streak_ends AS (
    SELECT
        s.meter_id,
        s.apartment_id,
        s.location,
        s.streak_start,
        MAX(l.reading_dates) AS streak_end,
        COUNT(*) AS zero_days,
        (JULIANDAY(MAX(l.reading_dates)) - JULIANDAY(s.streak_start) + 1) AS streak_duration_days
    FROM streak_starts s
    JOIN lagged l
    ON 1=1
    -- AND l.meter_id = s.meter_id
    AND l.apartment_id = s.apartment_id
    AND l.location = s.location
    AND l.reading_dates >= s.streak_start
    AND l.water_consumption = 0
    AND NOT EXISTS (
        SELECT 1 FROM lagged l2
        WHERE 1=1
          -- AND l2.meter_id = s.meter_id
          AND l2.apartment_id = s.apartment_id
          AND l2.location = s.location
          AND l2.reading_dates BETWEEN s.streak_start AND l.reading_dates
          AND l2.water_consumption != 0
    )
    GROUP BY s.meter_id, s.apartment_id, s.location, s.streak_start
    -- Use the threshold from params
    HAVING streak_duration_days >= (SELECT min_zero_days FROM params)
),

-- Get first non-zero reading after each streak
streak_with_spike AS (
    SELECT
        e.meter_id,
        e.apartment_id,
        e.location,
        e.streak_start,
        e.streak_end,
        e.streak_duration_days,
        (SELECT water_consumption
         FROM water_consumption w
         WHERE 1=1
           -- AND w.meter_id = e.meter_id
           AND w.apartment_id = e.apartment_id
           AND w.location = e.location
           AND w.reading_dates > e.streak_end
           AND w.water_consumption != 0
         ORDER BY w.reading_dates
         LIMIT 1) AS spike_consumption,
        (SELECT reading_dates
         FROM water_consumption w
         WHERE 1=1
           -- AND w.meter_id = e.meter_id
           AND w.apartment_id = e.apartment_id
           AND w.location = e.location
           AND w.reading_dates > e.streak_end
           AND w.water_consumption != 0
         ORDER BY w.reading_dates
         LIMIT 1) AS spike_dates
    FROM streak_ends e
)

-- Final output
SELECT
    s.meter_id,
    w.apartment_id,
    w.owner_tenent_name,
    w.location,
    s.streak_start,
    s.streak_end,
    s.streak_duration_days AS zero_days,
    s.spike_dates,
    s.spike_consumption
FROM streak_with_spike s
JOIN water_consumption w
    ON 1=1
    -- AND w.meter_id = s.meter_id
    AND w.apartment_id = s.apartment_id
    AND w.location = s.location
    AND w.reading_dates = s.spike_dates
WHERE s.spike_consumption >= (SELECT spike_threshold FROM params)
ORDER BY s.spike_dates DESC, s.apartment_id;



-- ============================================================================
-- Differentiate logic between
  -- LIKELY_CONNECTIVITY
  -- LIKELY_ABSENCE
  -- POSSIBLE_SHORT_CONNECTIVITY
  -- REVIEW
-- ============================================================================
DROP VIEW IF EXISTS reason_with_long_zero_consumption;
CREATE VIEW IF NOT EXISTS reason_with_long_zero_consumption AS
WITH
baseline AS (
SELECT
  apartment_id,
  location,
  meter_id,
  AVG(CASE WHEN COALESCE(water_consumption, 0) > 0 THEN water_consumption END) AS avg_positive_consumption,
  COUNT(*) AS total_days,
  SUM(CASE WHEN COALESCE(water_consumption, 0) = 0 THEN 1 ELSE 0 END) AS total_zero_days
FROM water_consumption
GROUP BY
  apartment_id,
  meter_id
),

ordered AS (
SELECT
  f.*,
  SUM(
    CASE WHEN COALESCE(f.water_consumption, 0) > 0 THEN 1 ELSE 0 END
    ) OVER (
    PARTITION BY
      f.apartment_id,
      f.meter_id
    ORDER BY
      f.reading_dates
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS positive_group
FROM water_consumption f
),

zero_runs AS (
SELECT
  apartment_id,
  location,
  meter_id,
  positive_group,
  MIN(reading_dates) AS zero_start_date,
  MAX(reading_dates) AS zero_end_date,
  COUNT(*) AS zero_days,
  SUM(COALESCE(water_consumption, 0)) AS zero_consumption
FROM ordered
WHERE COALESCE(water_consumption, 0) = 0
GROUP BY
  apartment_id,
  location,
  positive_group
),

zero_runs_context AS (
SELECT
  z.apartment_id,
  z.location,
  z.meter_id,
  z.zero_start_date,
  z.zero_end_date,
  z.zero_days,
  b.avg_positive_consumption,
  b.total_days,
  b.total_zero_days,

  (
    SELECT o.reading_dates
    FROM ordered o
    WHERE o.apartment_id = z.apartment_id
      AND o.location = z.location
      AND o.reading_dates > z.zero_end_date
    ORDER BY o.reading_dates
    LIMIT 1
  ) AS next_reading_dates,

  (
    SELECT o.water_consumption
    FROM ordered o
    WHERE o.apartment_id = z.apartment_id
      AND o.location = z.location
      AND o.reading_dates > z.zero_end_date
    ORDER BY o.reading_dates
    LIMIT 1
  ) AS next_consumption,

  (
    SELECT o.water_consumption
    FROM ordered o
    WHERE o.apartment_id = z.apartment_id
      AND o.location = z.location
      AND o.reading_dates < z.zero_start_date
    ORDER BY o.reading_dates DESC
    LIMIT 1
  ) AS previous_consumption

FROM zero_runs z
JOIN baseline b
  ON b.apartment_id = z.apartment_id
 AND b.location = z.location
)

SELECT
apartment_id,
location,
meter_id,
zero_start_date,
zero_end_date,
zero_days,

previous_consumption,
next_reading_dates,
next_consumption,

avg_positive_consumption,
total_days,
total_zero_days,

CASE
  WHEN next_consumption IS NULL
    OR avg_positive_consumption IS NULL
    OR avg_positive_consumption <= 0
  THEN NULL
  ELSE CAST(next_consumption AS REAL) / avg_positive_consumption
END AS spike_ratio,

CASE
  WHEN avg_positive_consumption IS NULL THEN NULL
  ELSE avg_positive_consumption * zero_days
END AS expected_missed_consumption,

CASE
  WHEN zero_days >= 2
   AND next_consumption IS NOT NULL
   AND avg_positive_consumption > 0
   AND next_consumption >= 3 * avg_positive_consumption
   AND next_consumption >= 0.5 * avg_positive_consumption * zero_days
   AND next_consumption <= 2.0 * avg_positive_consumption * zero_days
  THEN 'LIKELY_CONNECTIVITY'

  WHEN zero_days >= 2
   AND next_consumption IS NOT NULL
   AND avg_positive_consumption > 0
   AND next_consumption <= 1.5 * avg_positive_consumption
  THEN 'LIKELY_ABSENCE'

  WHEN zero_days = 1
   AND next_consumption IS NOT NULL
   AND avg_positive_consumption > 0
   AND next_consumption >= 2 * avg_positive_consumption
  THEN 'POSSIBLE_SHORT_CONNECTIVITY'

  ELSE 'REVIEW'
END AS likely_reason

FROM zero_runs_context
ORDER BY
apartment_id,
meter_id,
zero_start_date;



-- SELECT *
-- FROM reason_with_long_zero_consumption
-- WHERE likely_reason IN (
--   'LIKELY_CONNECTIVITY',
--   'POSSIBLE_SHORT_CONNECTIVITY'
-- )
-- ORDER BY
--   apartment_id,
--   meter_id,
--   zero_start_date;
