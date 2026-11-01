INSERT INTO {forecasting_project_id}.{forecasting_curated_dataset}.forecasting_etl_{forecasting_score_model_id}_{forecasting_table_suffix}
SELECT * FROM {forecasting_project_id}.{forecasting_staging_dataset}.forecasting_etl_{forecasting_score_model_id}_{forecasting_table_suffix}
WHERE forecasting_run_date = '{forecasting_load_date}';
