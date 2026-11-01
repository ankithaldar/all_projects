#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Forecasting ETL workflow definition for forecasting pipeline stages.'''

from airflow.models import DAG
from airflow.operators.dummy import DummyOperator


class ForecastingETLWorkflow:
  '''Forecasting ETL workflow using forecasting pipeline stage naming.'''

  def __init__(self, forecasting_dag_id='forecasting_etl_stage', forecasting_schedule='@once'):
    '''Initialize the forecasting ETL workflow with forecasting naming.

    Args:
      forecasting_dag_id: The forecasting pipeline stage identifier.
      forecasting_schedule: The forecasting pipeline scheduling interval.
    '''
    self.forecasting_dag_id = forecasting_dag_id
    self.forecasting_schedule = forecasting_schedule
    self.forecasting_dag = self._build_forecasting_dag()

  def _build_forecasting_dag(self):
    '''Build the forecasting ETL DAG using forecasting constants.

    Returns:
      A forecasting pipeline stage DAG.
    '''
    forecasting_default_args = {
      'email_on_failure': False,
      'email_on_retry': False,
      'owner': 'forecasting_pipeline',
      'retries': 0,
      'dags_are_paused_at_creation': True,
    }
    forecasting_params = {
      'forecasting_run_date': '',
      'forecasting_stage': 'forecasting_etl',
    }
    forecasting_dag = DAG(
      dag_id=self.forecasting_dag_id,
      catchup=False,
      default_args=forecasting_default_args,
      schedule_interval=self.forecasting_schedule,
      max_active_runs=1,
      start_date=None,
      description='Forecasting ETL pipeline stage',
      concurrency=1,
      params=forecasting_params,
      tags=['forecasting', 'forecasting_pipeline_stage'],
    )
    forecasting_etl_start = DummyOperator(
      task_id='forecasting_etl_start',
      dag=forecasting_dag,
    )
    forecasting_etl_end = DummyOperator(
      task_id='forecasting_etl_end',
      dag=forecasting_dag,
    )
    forecasting_etl_start >> forecasting_etl_end  # pylint: disable=pointless-statement  # pylint: disable=pointless-statement
    return forecasting_dag
