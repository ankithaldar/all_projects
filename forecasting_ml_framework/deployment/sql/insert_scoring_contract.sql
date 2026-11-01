DELETE FROM {forecasting_data_project}.{forecasting_db_score_data}.{forecasting_tbl_final_score}
WHERE forecasting_score_model_id IN ('{forecasting_score_model_id}')
AND forecasting_rpt_mth IN (
  SELECT DISTINCT CAST(forecasting_rpt_dt AS DATE)
  FROM {forecasting_data_project}.{forecasting_db_staging_data}.forecasting_score_{forecasting_model_id}
);

INSERT INTO {forecasting_data_project}.{forecasting_db_score_data}.{forecasting_tbl_final_score}
(
  forecasting_sor_id,
  forecasting_cust_id,
  forecasting_cust_line_seq_id,
  forecasting_acct_num,
  forecasting_mtn,
  forecasting_mkt_cd,
  forecasting_area_cd,
  forecasting_score_type,
  forecasting_score_value,
  forecasting_score_decile,
  forecasting_score_centile,
  forecasting_orig_score_value,
  forecasting_high_score_ind,
  forecasting_last_upd_dt,
  forecasting_score_dt,
  forecasting_rpt_mth,
  forecasting_score_model_id
)
SELECT
  '{forecasting_source_id}' AS forecasting_sor_id,
  CAST(forecasting_cust_id AS STRING) AS forecasting_cust_id,
  forecasting_cust_line_seq_id,
  forecasting_acct_num,
  forecasting_mtn,
  '' AS forecasting_mkt_cd,
  '' AS forecasting_area_cd,
  '{forecasting_score_type}' AS forecasting_score_type,
  CAST(forecasting_score_value AS NUMERIC) AS forecasting_score_value,
  CAST(forecasting_score_decile AS INT64) AS forecasting_score_decile,
  CAST(forecasting_score_centile AS INT64) AS forecasting_score_centile,
  CAST(forecasting_orig_score_value AS NUMERIC) AS forecasting_orig_score_value,
  forecasting_high_score_ind,
  CAST(CURRENT_TIMESTAMP() AS DATE) AS forecasting_last_upd_dt,
  CAST('{forecasting_run_date}' AS DATE) AS forecasting_score_dt,
  CAST(forecasting_rpt_dt AS DATE) AS forecasting_rpt_mth,
  '{forecasting_score_model_id}' AS forecasting_score_model_id
FROM {forecasting_data_project}.{forecasting_db_staging_data}.forecasting_score_{forecasting_model_id};
