#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Forecasting scoring workflow for scoring acquisition pipeline stages.'''

from airflow.models import DAG
from airflow.operators.dummy import DummyOperator

from forecasting_ml_framework.constants import PIPELINE_MODEL_SCORING


class ForecastingScoringWorkflow:
  '''Forecasting scoring workflow using forecasting pipeline naming.'''

  def __init__(self, forecasting_dag_id='forecasting_scoring_stage', forecasting_schedule='@once'):
    '''Initialize forecasting scoring workflow.

    Args:
      forecasting_dag_id: Forecasting pipeline stage identifier.
      forecasting_schedule: Forecasting scheduling interval.
    '''
    self.forecasting_dag_id = forecasting_dag_id
    self.forecasting_schedule = forecasting_schedule
    self.forecasting_stage = PIPELINE_MODEL_SCORING
    self.forecasting_dag = self._build_forecasting_dag()

  def _build_forecasting_dag(self):
    '''Build forecasting scoring pipeline stage DAG.

    Returns:
      Forecasting scoring pipeline DAG.
    '''
    forecasting_default_args = {
      'email_on_failure': False,
      'email_on_retry': False,
      'owner': 'forecasting_pipeline',
      'retries': 0,
    }
    forecasting_params = {
      'forecasting_run_date': '',
      'forecasting_stage': self.forecasting_stage,
    }
    forecasting_dag = DAG(
      dag_id=self.forecasting_dag_id,
      catchup=False,
      default_args=forecasting_default_args,
      schedule_interval=self.forecasting_schedule,
      max_active_runs=1,
      start_date=None,
      description='Forecasting scoring pipeline stage',
      concurrency=1,
      params=forecasting_params,
      tags=['forecasting', 'forecasting_scoring_stage'],
    )
    forecasting_scoring_task = DummyOperator(
      task_id='forecasting_scoring_task',
      dag=forecasting_dag,
    )
    forecasting_start = DummyOperator(
      task_id='forecasting_start',
      dag=forecasting_dag,
    )
    forecasting_start >> forecasting_scoring_task  # pylint: disable=pointless-statement
    return forecasting_dag
