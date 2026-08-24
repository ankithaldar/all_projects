#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Abstract base class for all data loaders.
Enforces a consistent interface for loading data and extracting features.
"""


# imports
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
#    script imports
# imports


# constants
# constants


# classes
class DataLoader(ABC):
    """
    Abstract base class for all data loaders.
    Enforces a consistent interface for loading data and extracting features.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger_name = self.__class__.__name__

    @property
    def logger_name(self) -> str:
        return self._logger_name

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data into a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        pass

    def get_features(self, df: pd.DataFrame) -> List[str]:
        """
        Extract feature column names based on config.

        Args:
            df (pd.DataFrame): Loaded dataframe.

        Returns:
            List[str]: List of feature column names.
        """
        target = self.config['target_column']
        score = self.config.get('score_column')
        exclude = {target}
        if score:
            exclude.add(score)

        if self.config.get('features'):
            return [f for f in self.config['features'] if f in df.columns and f not in exclude]
        else:
            return [col for col in df.columns if col not in exclude]
# classes
