INSERT INTO {forecasting_data_project}.{forecasting_db_score_data}.forecasting_gcr_load_status_table
(
  forecasting_score_model_id,
  forecasting_select_query,
  forecasting_source_table_name,
  forecasting_filter_column_name,
  forecasting_filter_column_value,
  forecasting_target_table_name,
  forecasting_target_location,
  forecasting_file_format,
  forecasting_columns_to_select,
  forecasting_run_date,
  forecasting_insert_ts,
  forecasting_load_status
)
SELECT '{forecasting_score_model_id}' AS forecasting_score_model_id,
  'SELECT * FROM {forecasting_data_project}.{forecasting_db_score_data}.{forecasting_tbl_final_score} WHERE forecasting_rpt_mth = "{forecasting_run_date}" AND forecasting_score_model_id = "{forecasting_score_model_id}"' AS forecasting_select_query,
  '{forecasting_data_project}.{forecasting_db_score_data}.{forecasting_tbl_final_score}' AS forecasting_source_table_name,
  'forecasting_score_model_id,forecasting_rpt_mth' AS forecasting_filter_column_name,
  '{forecasting_score_model_id},{forecasting_run_date}' AS forecasting_filter_column_value,
  '' AS forecasting_target_table_name,
  'forecasting_egress_location' AS forecasting_target_location,
  '' AS forecasting_file_format,
  '' AS forecasting_columns_to_select,
  DATE('{forecasting_run_date}') AS forecasting_run_date,
  CURRENT_TIMESTAMP() AS forecasting_insert_ts,
  'READY' AS forecasting_load_status
FROM {forecasting_data_project}.{forecasting_db_score_data}.{forecasting_tbl_final_score}
WHERE forecasting_rpt_mth = '{forecasting_run_date}'
  AND forecasting_score_model_id = '{forecasting_score_model_id}'
LIMIT 1;
