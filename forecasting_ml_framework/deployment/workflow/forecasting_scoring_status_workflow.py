#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Forecasting scoring status workflow using forecasting pipeline naming.'''

from airflow.models import DAG
from airflow.operators.dummy import DummyOperator


class ForecastingScoringStatusWorkflow:
  '''Forecasting scoring status workflow for forecasting pipeline stages.'''

  def __init__(self, forecasting_dag_id='forecasting_scoring_status_stage', forecasting_schedule='@once'):
    '''Initialize forecasting scoring status workflow.

    Args:
      forecasting_dag_id: Forecasting pipeline stage identifier.
      forecasting_schedule: Forecasting scheduling interval.
    '''
    self.forecasting_dag_id = forecasting_dag_id
    self.forecasting_schedule = forecasting_schedule
    self.forecasting_dag = self._build_forecasting_dag()

  def _build_forecasting_dag(self):
    '''Build forecasting scoring status pipeline stage DAG.

    Returns:
      Forecasting scoring status pipeline DAG.
    '''
    forecasting_default_args = {
      'email_on_failure': True,
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
      description='Forecasting scoring status pipeline stage',
      concurrency=1,
      params=forecasting_params,
      tags=['forecasting', 'forecasting_scoring_status_stage'],
    )
    forecasting_start = DummyOperator(
      task_id='forecasting_start',
      dag=forecasting_dag,
    )
    forecasting_status = DummyOperator(
      task_id='forecasting_scoring_status',
      dag=forecasting_dag,
    )
    forecasting_end = DummyOperator(
      task_id='forecasting_end',
      dag=forecasting_dag,
    )
    forecasting_start >> forecasting_status >> forecasting_end  # pylint: disable=pointless-statement
    return forecasting_dag
