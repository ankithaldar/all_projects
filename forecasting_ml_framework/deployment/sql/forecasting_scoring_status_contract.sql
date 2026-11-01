INSERT INTO {forecasting_data_project}.{forecasting_db_internal_data}.forecasting_scoring_status_contract
(
  forecasting_contract_name,
  forecasting_scoring_stage,
  forecasting_run_date,
  forecasting_contract_status,
  forecasting_contract_timestamp
)
SELECT 'forecasting_scoring_contract' AS forecasting_contract_name,
  'forecasting_scoring_stage' AS forecasting_scoring_stage,
  DATE('{forecasting_run_date}') AS forecasting_run_date,
  '{forecasting_contract_status}' AS forecasting_contract_status,
  CURRENT_TIMESTAMP() AS forecasting_contract_timestamp;
