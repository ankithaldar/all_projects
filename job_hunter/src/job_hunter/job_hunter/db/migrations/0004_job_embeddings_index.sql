-- 0004_job_embeddings_index.sql: covering index for unfiltered
-- (model-only) embedding loads. Non-unique (ADR D4): the table
-- already has PRIMARY KEY (job_id, model), which cannot serve a
-- model-first lookup; existing rows may not be unique on
-- (model, job_id) beyond that key, so a UNIQUE index could fail.

CREATE INDEX IF NOT EXISTS idx_job_embeddings_model_job ON job_embeddings(model, job_id);
