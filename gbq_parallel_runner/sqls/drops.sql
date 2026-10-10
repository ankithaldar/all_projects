-- Cleanup step (levels 0 and 2 in the shipped config.yml).
--
-- Placeholders resolve exactly as in etl.sql: project_id and dataset_id from
-- base_config, key_01 from the run config.
--
-- IF EXISTS keeps the level 0 pass from failing when the target has not been
-- built yet. Note the shipped config runs this at level 2 as well, which drops
-- the table etl.sql just built -- point level 2 at a different file if you
-- want the result kept.

DROP TABLE IF EXISTS `{project_id}.{dataset_id}.{key_01}_events`;
