#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Extending the Neptune Logger class of Pytorch Lightning to use the ENVIRONMENT variable assigned for neptune logger'''


# imports
import os

from pytorch_lightning.loggers import NeptuneLogger

#    script imports
# imports


# constants
# constants


# classes
class CustomNeptuneLogger(NeptuneLogger):
  '''Extending the Neptune Logger class of Pytorch Lightning to use the ENVIRONMENT variable assigned for neptune logger'''

  def __init__(self, **kwargs):
    api_token_env_var = kwargs.get('api_token_env_var', None) or 'NEPTUNE_API_TOKEN'
    api_key = os.getenv(api_token_env_var).strip().strip('"').strip("'")

    if api_key is None:
      raise ValueError(f"Environment variable '{api_token_env_var}' not found")

    kwargs.pop('api_token_env_var')

    # Initialize the parent NeptuneLogger with the retrieved token
    super().__init__(api_key=api_key, **kwargs)

# classes
