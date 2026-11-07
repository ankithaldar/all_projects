-- 0003_indexes.sql: query-path indexes, runs.created_at, source seeds.
--
-- NON-unique indexes only (ADR D4): existing databases may contain
-- duplicate values, so a UNIQUE index could fail to build. The runs
-- table had no created_at column in 0001, which both the index below
-- and orphan age-gating need; ADD COLUMN with a non-constant default
-- is rejected by SQLite, so the column is nullable, backfilled to
-- now, and RunsRepository.create() now sets it explicitly.
-- SQLite cannot alter a CHECK constraint (runs.status excludes
-- 'cancelled'); that change is intentionally skipped here.
-- The llm_calls table lives in the gateway database, not the app
-- database, so its indexes cannot be created here.

ALTER TABLE runs ADD COLUMN created_at TEXT;
UPDATE runs SET created_at = COALESCE(started_at, datetime('now'))
WHERE created_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_jobs_canonical_url
ON jobs(canonical_url);
CREATE INDEX IF NOT EXISTS idx_jobs_source_external
ON jobs(source_key, external_id);
CREATE INDEX IF NOT EXISTS idx_candidate_profiles_candidate_version
ON candidate_profiles(candidate_id, version);
CREATE INDEX IF NOT EXISTS idx_companies_status_vertical_priority
ON companies(status, vertical, priority);
CREATE INDEX IF NOT EXISTS idx_runs_status_created
ON runs(status, created_at);

INSERT OR IGNORE INTO sources (key, kind) VALUES
  ('workday', 'ats'), ('himalayas', 'aggregator'),
  ('arbeitnow', 'aggregator'), ('jobicy', 'aggregator'),
  ('cutshort', 'aggregator');
