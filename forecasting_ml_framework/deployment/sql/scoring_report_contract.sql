DECLARE forecasting_var_run_date DATE;
DECLARE forecasting_var_input_data_sla TIMESTAMP;
DECLARE forecasting_var_etl_run_sla TIMESTAMP;
DECLARE forecasting_var_model_run_sla TIMESTAMP;
DECLARE forecasting_var_egress_data_sla TIMESTAMP;

SET forecasting_var_run_date = DATE('{forecasting_run_date}');
SET forecasting_var_input_data_sla = CAST(CONCAT("{forecasting_run_date} 20:30:00") AS TIMESTAMP);
SET forecasting_var_etl_run_sla = CAST(CONCAT("{forecasting_run_date} 21:00:00") AS TIMESTAMP);
SET forecasting_var_model_run_sla = CAST(CONCAT("{forecasting_run_date} 21:30:00") AS TIMESTAMP);
SET forecasting_var_egress_data_sla = CAST(CONCAT("{forecasting_run_date} 22:00:00") AS TIMESTAMP);

CREATE OR REPLACE TABLE {forecasting_data_project}.{forecasting_staging_dataset}.forecasting_scoring_status_report AS (
  SELECT forecasting_score_model_id,
         forecasting_run_date,
         IFNULL(forecasting_etl_run_status, 'Not Available') AS forecasting_etl_run_status,
         IFNULL(forecasting_model_run_status, 'Not Available') AS forecasting_model_run_status,
         IFNULL(forecasting_egress_status, 'Not Available') AS forecasting_egress_status,
         IFNULL(forecasting_total_customers, 0) AS forecasting_total_customers,
         IFNULL(forecasting_top_decile_customers, 0) AS forecasting_top_decile_customers,
         IFNULL(forecasting_top_decile_min_score, 0.0) AS forecasting_top_decile_min_score,
         IFNULL(forecasting_avg_min_score, 0.0) AS forecasting_avg_min_score,
         IFNULL(forecasting_score_deviation_percent, 0.0) AS forecasting_score_deviation_percent,
         CASE
           WHEN forecasting_egress_start_timestamp < forecasting_var_egress_data_sla
             AND forecasting_egress_status = 'COMPLETE' THEN 'ON TIME'
           WHEN forecasting_model_run_timestamp < forecasting_var_model_run_sla
             AND forecasting_egress_start_timestamp IS NULL THEN 'EGRESS YET TO START'
           WHEN forecasting_top_decile_min_score_deviation <= -10
             OR forecasting_top_decile_min_score_deviation >= 10 THEN 'OUTPUT DQ CHECK FAILED'
           WHEN forecasting_model_status = 'FAILED' THEN 'MODEL FAILED'
           WHEN forecasting_etl_run_timestamp < forecasting_var_etl_run_sla
             AND forecasting_etl_status LIKE 'SUCCESS%'
             AND forecasting_model_run_timestamp IS NULL THEN 'MODEL STILL RUNNING'
           WHEN forecasting_egress_start_timestamp < forecasting_var_egress_data_sla
             AND forecasting_egress_status = 'new' THEN 'EGRESS STILL RUNNING'
           ELSE 'INPUT DQ CHECK FAILED'
         END AS forecasting_final_status
  FROM forecasting_lift_deviation_table
);
