-- ETL build step (level 1).
--
-- Placeholders here are a name wrapped in braces, filled from the merged config
-- before the job is submitted. project_id and dataset_id come from
-- base_config; key_01 and key_02 come from each entry in run_configs. Because
-- run_configs is a list, this same file runs once per slice, each time with
-- different values.
--
-- A placeholder with no matching key makes the runner exit 1 before any job is
-- submitted, so a typo costs no BigQuery slots. Check the contract cheaply
-- first:  python run_etl_in_levels.py --all_configs config.yml --dry_run
--
-- Note: the placeholder scanner is a plain regex for "brace, word, brace", so
-- a lone "var" inside braces in a comment is treated as a placeholder. This
-- file deliberately contains no literal braces outside the placeholders below.

CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{key_01}_events` AS
SELECT
  event_id,
  user_id,
  event_type,
  event_ts,
  region,
  DATE(event_ts) AS event_date
FROM
  `{project_id}.{dataset_id}.staging_events`
WHERE
  DATE(event_ts) = DATE('{key_02}')
  AND region = '{key_01}';
