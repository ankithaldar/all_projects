#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Forecasting scoring report workflow using forecasting pipeline naming.'''

from airflow.models import DAG
from airflow.operators.dummy import DummyOperator


class ForecastingScoringReportWorkflow:
  '''Forecasting scoring report workflow for forecasting pipeline stages.'''

  def __init__(self, forecasting_dag_id='forecasting_scoring_report', forecasting_schedule='@once'):
    '''Initialize forecasting scoring report workflow.

    Args:
      forecasting_dag_id: Forecasting pipeline stage identifier.
      forecasting_schedule: Forecasting scheduling interval.
    '''
    self.forecasting_dag_id = forecasting_dag_id
    self.forecasting_schedule = forecasting_schedule
    self.forecasting_dag = self._build_forecasting_dag()

  def _build_forecasting_dag(self):
    '''Build forecasting scoring report DAG.

    Returns:
      Forecasting pipeline stage DAG.
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
      description='Forecasting scoring report pipeline stage',
      concurrency=1,
      params=forecasting_params,
      tags=['forecasting', 'forecasting_scoring_report'],
    )
    forecasting_start = DummyOperator(
      task_id='forecasting_start',
      dag=forecasting_dag,
    )
    forecasting_report = DummyOperator(
      task_id='forecasting_scoring_report',
      dag=forecasting_dag,
    )
    forecasting_end = DummyOperator(
      task_id='forecasting_end',
      dag=forecasting_dag,
    )
    forecasting_start >> forecasting_report >> forecasting_end  # pylint: disable=pointless-statement
    return forecasting_dag
