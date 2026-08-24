-- 0002_perf.sql: query-path indexes added before v0.1.0.

CREATE INDEX IF NOT EXISTS idx_jobs_city_status ON jobs(city, status);
CREATE INDEX IF NOT EXISTS idx_jobs_mode_status ON jobs(work_mode, status);
CREATE INDEX IF NOT EXISTS idx_recs_status_score ON recommendations(candidate_id, status, total_score DESC);
CREATE INDEX IF NOT EXISTS idx_companies_ats ON companies(ats_provider) WHERE ats_provider IS NOT NULL;
