DECLARE forecasting_var_run_date DATE;
DECLARE forecasting_var_input_data_sla TIMESTAMP;
DECLARE forecasting_var_model_run_sla TIMESTAMP;
DECLARE forecasting_var_egress_data_sla TIMESTAMP;

SET forecasting_var_run_date = DATE('{forecasting_run_date}');
SET forecasting_var_input_data_sla = CAST(CONCAT("{forecasting_run_date} 20:30:00") AS TIMESTAMP);
SET forecasting_var_model_run_sla = CAST(CONCAT("{forecasting_run_date} 21:30:00") AS TIMESTAMP);
SET forecasting_var_egress_data_sla = CAST(CONCAT("{forecasting_run_date} 22:00:00") AS TIMESTAMP);

CREATE TEMP TABLE forecasting_lift_model_score_analysis AS (
  SELECT forecasting_score_model_id,
         COUNT(*) AS forecasting_total_customers,
         COUNTIF(forecasting_score_decile = 10) AS forecasting_top_decile_customers,
         ROUND(MIN(IF(forecasting_score_decile = 10, forecasting_orig_score_value, NULL)), 4) AS forecasting_top_decile_min_score
  FROM {forecasting_data_project}.{forecasting_db_score_data}.{forecasting_tbl_final_score}
  WHERE forecasting_rpt_mth = forecasting_var_run_date
    AND forecasting_score_model_id IN UNNEST([
      STRUCT('{forecasting_score_model_id}' AS forecasting_score_model_id)
    ])
  GROUP BY forecasting_score_model_id
);

SELECT forecasting_score_model_id,
       forecasting_run_date,
       IFNULL(forecasting_etl_status, 'Not Available') AS forecasting_etl_status,
       IFNULL(forecasting_model_status, 'Not Available') AS forecasting_model_status,
       IFNULL(forecasting_egress_status, 'Not Available') AS forecasting_egress_status,
       forecasting_total_customers,
       forecasting_top_decile_customers,
       forecasting_top_decile_min_score,
       ROUND(
         ((forecasting_top_decile_min_score - forecasting_avg_min_score) / forecasting_avg_min_score) * 100, 2
       ) AS forecasting_score_deviation_percent,
       CASE
         WHEN forecasting_egress_start_timestamp < forecasting_var_egress_data_sla
           AND forecasting_egress_status = 'COMPLETE' THEN 'ON TIME'
         WHEN forecasting_egress_start_timestamp < forecasting_var_egress_data_sla
           AND forecasting_egress_status = 'new' THEN 'EGRESS STILL RUNNING'
         WHEN forecasting_model_run_timestamp < forecasting_var_model_run_sla
           AND forecasting_egress_start_timestamp IS NULL THEN 'EGRESS YET TO START'
         WHEN (forecasting_top_decile_min_score_deviation <= -10)
           OR (forecasting_top_decile_min_score_deviation >= 10) THEN 'OUTPUT DQ CHECK FAILED'
         WHEN forecasting_model_status = 'FAILED' THEN 'MODEL FAILED'
         WHEN forecasting_etl_run_timestamp < forecasting_var_etl_run_sla
           AND forecasting_etl_status LIKE 'SUCCESS%'
           AND forecasting_model_run_timestamp IS NULL THEN 'MODEL STILL RUNNING'
         ELSE 'INPUT DQ CHECK FAILED'
       END AS forecasting_final_status
FROM forecasting_lift_model_score_analysis;
