-- 0001_init.sql: full v1 schema.

CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL DEFAULT 'candidate',
  email TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO candidates (id, name) VALUES (1, 'me');

CREATE TABLE IF NOT EXISTS resumes (
  id INTEGER PRIMARY KEY,
  candidate_id INTEGER NOT NULL REFERENCES candidates(id),
  file_path TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE,
  mime TEXT,
  uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
  parsed_ok INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS candidate_profiles (
  id INTEGER PRIMARY KEY,
  candidate_id INTEGER NOT NULL REFERENCES candidates(id),
  version INTEGER NOT NULL DEFAULT 1,
  target_roles TEXT NOT NULL DEFAULT '[]',
  seniority_keywords TEXT NOT NULL DEFAULT '[]',
  target_verticals TEXT NOT NULL DEFAULT '[]',
  blocked_verticals TEXT NOT NULL DEFAULT '[]',
  cities TEXT NOT NULL DEFAULT '[]',
  relocate_ok INTEGER NOT NULL DEFAULT 0,
  remote_pref TEXT NOT NULL DEFAULT 'any'
    CHECK (remote_pref IN ('remote','hybrid','onsite','any')),
  salary_floor_lpa REAL NOT NULL DEFAULT 45.0,
  experience_years REAL NOT NULL DEFAULT 0,
  employment_types TEXT NOT NULL DEFAULT '["full_time"]',
  summary TEXT NOT NULL DEFAULT '',
  confidence REAL,
  resume_id INTEGER REFERENCES resumes(id),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS skills (
  id INTEGER PRIMARY KEY,
  canonical_name TEXT NOT NULL UNIQUE,
  category TEXT
);
CREATE TABLE IF NOT EXISTS skill_aliases (
  alias TEXT PRIMARY KEY,
  skill_id INTEGER NOT NULL REFERENCES skills(id)
);
CREATE TABLE IF NOT EXISTS candidate_skills (
  candidate_id INTEGER NOT NULL REFERENCES candidates(id),
  skill_id INTEGER NOT NULL REFERENCES skills(id),
  weight REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (candidate_id, skill_id)
);

CREATE TABLE IF NOT EXISTS companies (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  normalized_name TEXT NOT NULL UNIQUE,
  domain TEXT UNIQUE,
  careers_url TEXT,
  ats_provider TEXT,
  board_ref TEXT,
  vertical TEXT,
  sub_vertical TEXT,
  vertical_confidence REAL,
  hq_city TEXT,
  india_presence INTEGER NOT NULL DEFAULT 1,
  size_band TEXT,
  funding_stage TEXT,
  priority INTEGER NOT NULL DEFAULT 3,
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','paused','needs_review','blacklisted','merged')),
  discovered_via TEXT,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  last_checked_at TEXT
);
CREATE TABLE IF NOT EXISTS company_aliases (
  alias TEXT PRIMARY KEY,
  company_id INTEGER NOT NULL REFERENCES companies(id)
);

CREATE TABLE IF NOT EXISTS sources (
  key TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  rate_limit_rpm INTEGER NOT NULL DEFAULT 30,
  config_json TEXT NOT NULL DEFAULT '{}'
);

INSERT OR IGNORE INTO sources (key, kind) VALUES
  ('greenhouse', 'ats'), ('lever', 'ats'), ('ashby', 'ats'),
  ('workable', 'ats'), ('smartrecruiters', 'ats'), ('recruitee', 'ats'),
  ('personio', 'ats'), ('remotive', 'aggregator'), ('remoteok', 'aggregator'),
  ('weworkremotely', 'aggregator'), ('manual', 'manual'), ('career_page', 'career');

CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY,
  source_key TEXT NOT NULL REFERENCES sources(key),
  external_id TEXT,
  url TEXT NOT NULL,
  canonical_url TEXT NOT NULL,
  company_id INTEGER REFERENCES companies(id),
  company_raw_name TEXT,
  title TEXT NOT NULL,
  location_text TEXT,
  city TEXT,
  region TEXT,
  country TEXT NOT NULL DEFAULT 'IN',
  work_mode TEXT CHECK (work_mode IN ('remote','hybrid','onsite','unknown')),
  employment_type TEXT,
  salary_min_lpa REAL,
  salary_max_lpa REAL,
  salary_raw TEXT,
  experience_min_yrs REAL,
  experience_max_yrs REAL,
  posted_at TEXT,
  first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
  description_text TEXT,
  raw_json TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','stale','closed','error')),
  quality_score REAL NOT NULL DEFAULT 0.5,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_jobs_company_posted ON jobs(company_id, posted_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(content_hash);
CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
  title, description_text, content='jobs', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS jobs_ai AFTER INSERT ON jobs BEGIN
  INSERT INTO jobs_fts(rowid, title, description_text)
  VALUES (new.id, new.title, new.description_text);
END;
CREATE TRIGGER IF NOT EXISTS jobs_ad AFTER DELETE ON jobs BEGIN
  INSERT INTO jobs_fts(jobs_fts, rowid, title, description_text)
  VALUES ('delete', old.id, old.title, old.description_text);
END;
CREATE TRIGGER IF NOT EXISTS jobs_au AFTER UPDATE ON jobs BEGIN
  INSERT INTO jobs_fts(jobs_fts, rowid, title, description_text)
  VALUES ('delete', old.id, old.title, old.description_text);
  INSERT INTO jobs_fts(rowid, title, description_text)
  VALUES (new.id, new.title, new.description_text);
END;

CREATE TABLE IF NOT EXISTS job_skills (
  job_id INTEGER NOT NULL REFERENCES jobs(id),
  skill_id INTEGER NOT NULL REFERENCES skills(id),
  kind TEXT NOT NULL CHECK (kind IN ('must_have','nice_to_have')),
  confidence REAL NOT NULL DEFAULT 0.8,
  PRIMARY KEY (job_id, skill_id)
);

CREATE TABLE IF NOT EXISTS job_embeddings (
  job_id INTEGER NOT NULL REFERENCES jobs(id),
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (job_id, model)
);

CREATE TABLE IF NOT EXISTS recommendations (
  id INTEGER PRIMARY KEY,
  job_id INTEGER NOT NULL REFERENCES jobs(id),
  candidate_id INTEGER NOT NULL REFERENCES candidates(id),
  run_id INTEGER NOT NULL REFERENCES runs(id),
  total_score REAL NOT NULL,
  rank INTEGER,
  gate_pass INTEGER NOT NULL,
  gate_failures TEXT,
  score_breakdown_json TEXT NOT NULL,
  rationale TEXT,
  status TEXT NOT NULL DEFAULT 'new'
    CHECK (status IN ('new','saved','dismissed','applied','expired')),
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  reviewed_at TEXT,
  UNIQUE (job_id, candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_recs_score ON recommendations(candidate_id, total_score DESC);

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL
    CHECK (kind IN ('discovery','refresh','on_demand','maintenance')),
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending','running','success','partial','failed')),
  triggered_by TEXT,
  started_at TEXT,
  finished_at TEXT,
  stats_json TEXT,
  error_text TEXT
);

CREATE TABLE IF NOT EXISTS run_events (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  level TEXT NOT NULL,
  node TEXT,
  message TEXT NOT NULL,
  data_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON run_events(run_id, id);

CREATE TABLE IF NOT EXISTS crawl_state (
  id INTEGER PRIMARY KEY,
  scope TEXT NOT NULL UNIQUE,
  cursor TEXT,
  etag TEXT,
  last_success_at TEXT,
  consecutive_failures INTEGER NOT NULL DEFAULT 0,
  cooldown_until TEXT
);

CREATE TABLE IF NOT EXISTS notifications (
  id INTEGER PRIMARY KEY,
  channel TEXT,
  payload_json TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT,
  sent_at TEXT
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
