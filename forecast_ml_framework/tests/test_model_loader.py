#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for model loaders.'''

import pickle

import pytest


class Dummy:
  '''Dummy picklable model.'''

  def predict(self, rows):
    '''Predict sum.'''
    return [sum(row) for row in rows]


def test_file_loader_loads_pickle(tmp_path):
  '''File loader loads pickled model lazily.'''
  from forecast_ml_framework.serving.model_loader import (
    FileModelLoader,
  )
  from sklearn.linear_model import LinearRegression
  model = LinearRegression().fit([[1.0, 2.0]], [3.0])
  path = tmp_path / 'model.pkl'
  path.write_bytes(pickle.dumps(model))
  loader = FileModelLoader(str(path))
  loaded = loader()
  assert loaded.predict([[1.0, 2.0]])[0] == pytest.approx(3.0)


def test_registry_loader_not_implemented():
  '''Registry loader raises NotImplementedError.'''
  import pytest
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  loader = RegistryModelLoader('model-name', alias='latest')
  with pytest.raises(NotImplementedError):
    loader()


def test_file_loader_evil_pickle_rejected(tmp_path):
  import os
  import pickle
  import pytest
  from forecast_ml_framework.serving.model_loader import FileModelLoader

  class Evil:
    def __reduce__(self):
      return (os.system, ('echo pwned',))

  path = tmp_path / 'evil.pkl'
  path.write_bytes(pickle.dumps(Evil()))
  with pytest.raises(ValueError):
    FileModelLoader(str(path)).load()


def test_file_loader_missing_no_path_disclosure(tmp_path):
  import pytest
  from forecast_ml_framework.serving.model_loader import FileModelLoader
  secret = str(tmp_path / 'secret-dir' / 'model.pkl')
  with pytest.raises(FileNotFoundError) as excinfo:
    FileModelLoader(secret).load()
  assert secret not in str(excinfo.value)
  assert str(tmp_path) not in str(excinfo.value)


def test_file_loader_chain_read_pickle_blocked(tmp_path):
  import os
  import pickle
  import pandas as pd
  import pytest
  from forecast_ml_framework.serving.model_loader import FileModelLoader
  marker = tmp_path / 'pwned'

  class Evil:
    def __reduce__(self):
      return (os.system, (f'touch {marker}',))

  inner = pickle.dumps(Evil())

  class Outer:
    def __reduce__(self):
      return (pd.read_pickle, (inner,))

  path = tmp_path / 'chain.pkl'
  path.write_bytes(pickle.dumps(Outer()))
  with pytest.raises(ValueError):
    FileModelLoader(str(path)).load()
  assert not marker.exists()


def test_file_loader_read_csv_blocked(tmp_path):
  import pickle
  import pandas as pd
  import pytest
  from forecast_ml_framework.serving.model_loader import FileModelLoader
  secret = tmp_path / 'secret.csv'
  secret.write_text('secret_col\n42\n', encoding='utf-8')

  class Outer:
    def __reduce__(self):
      return (pd.read_csv, (str(secret),))

  path = tmp_path / 'csv.pkl'
  path.write_bytes(pickle.dumps(Outer()))
  with pytest.raises(ValueError):
    FileModelLoader(str(path)).load()
