INSERT INTO {forecasting_data_project}.{forecasting_db_internal_data}.forecasting_score_cntrl_tbl
(
  forecasting_score_table_nm,
  forecasting_insert_ts,
  forecasting_process_dt,
  forecasting_score_model_id,
  forecasting_rpt_dt,
  forecasting_status
)
SELECT 'forecasting_cust_acct_line_score' AS forecasting_score_table_nm,
  CAST(CURRENT_TIMESTAMP() AS DATE) AS forecasting_insert_ts,
  CAST(CURRENT_DATE() AS STRING) AS forecasting_process_dt,
  '{forecasting_score_model_id}' AS forecasting_score_model_id,
  DATE('{forecasting_run_date}') AS forecasting_rpt_dt,
  '{forecasting_score_cntrl_tbl_status}' AS forecasting_status
FROM {forecasting_data_project}.{forecasting_db_score_data}.{forecasting_tbl_final_score}
WHERE forecasting_score_model_id = '{forecasting_score_model_id}'
  AND forecasting_rpt_mth = '{forecasting_run_date}'
LIMIT 1;
