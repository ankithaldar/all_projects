#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Forecasting pipeline workflow definition using forecasting naming and constants.'''

from airflow.models import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator

from forecasting_ml_framework.constants import (
  PIPELINE_FE_SCORING,
  PIPELINE_FE_TRAINING,
  PIPELINE_MODEL_SCORING,
  PIPELINE_MODEL_TRAINING,
)


class ForecastingPipelineWorkflow:
  '''Forecasting pipeline workflow using forecasting pipeline stage naming.'''

  def __init__(self, forecasting_dag_id='forecasting_pipeline_stage', forecasting_schedule='@once'):
    '''Initialize forecasting pipeline workflow.

    Args:
      forecasting_dag_id: Forecasting pipeline stage identifier.
      forecasting_schedule: Forecasting scheduling interval.
    '''
    self.forecasting_dag_id = forecasting_dag_id
    self.forecasting_schedule = forecasting_schedule
    self.forecasting_pipeline_stages = [
      PIPELINE_FE_TRAINING,
      PIPELINE_MODEL_TRAINING,
      PIPELINE_FE_SCORING,
      PIPELINE_MODEL_SCORING,
    ]
    self.forecasting_dag = self._build_forecasting_dag()

  def _get_forecasting_run_key(self, **forecasting_context):
    '''Retrieve forecasting pipeline run key.

    Args:
      forecasting_context: Airflow forecasting pipeline context.

    Returns:
      Forecasting run date dictionary.
    '''
    return {'forecasting_run_date': forecasting_context.get('forecasting_run_date', '')}

  def _build_forecasting_dag(self):
    '''Build forecasting pipeline stage DAG.

    Returns:
      Forecasting pipeline DAG.
    '''
    forecasting_default_args = {
      'email_on_failure': False,
      'email_on_retry': False,
      'owner': 'forecasting_pipeline',
      'retries': 0,
    }
    forecasting_params = {
      'forecasting_run_date': '',
    }
    forecasting_dag = DAG(
      dag_id=self.forecasting_dag_id,
      catchup=False,
      default_args=forecasting_default_args,
      schedule_interval=self.forecasting_schedule,
      max_active_runs=1,
      start_date=None,
      description='Forecasting pipeline stage',
      concurrency=1,
      params=forecasting_params,
      tags=['forecasting', 'forecasting_pipeline_stage'],
    )
    forecasting_get_run_key = PythonOperator(
      task_id='forecasting_get_run_key',
      python_callable=self._get_forecasting_run_key,
      provide_context=True,
      dag=forecasting_dag,
    )
    forecasting_start = DummyOperator(
      task_id='forecasting_start',
      dag=forecasting_dag,
    )
    forecasting_start >> forecasting_get_run_key  # pylint: disable=pointless-statement
    return forecasting_dag
