#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Load Config File"""


# imports
import yaml
from typing import Dict, Any
import logging
#    script imports
# imports


# constants
# constants


# classes
class ConfigLoader:
  """Load Config File"""

  def __init__(self, config_path: str):
    self.config_path = config_path
    self.logger = logging.getLogger(self.__class__.__name__)

  def load(self) -> Dict[str, Any]:
    try:
      with open(self.config_path, 'r') as f:
        config = yaml.safe_load(f)
      self.logger.info("Configuration loaded successfully.")
      return config
    except Exception as e:
      self.logger.error(f"Failed to load config: {e}")
      raise

# classes
