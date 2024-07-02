#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Train the model base script'''

from argparse import ArgumentParser
from engines.binary_classification_engine import BinaryClassificationEngine
from config_params.params_reader import Parameters


def main(module_params=None):
  '''main function'''
  pe = BinaryClassificationEngine(module_params)
  pe.train(module_params.run_params)


if __name__ == '__main__':

  parser = ArgumentParser(parents=[])
  parser.add_argument('--params_yml', type=str)
  params_file_path = parser.parse_args()

  hparams = Parameters(params_file_path.params_yml)

  main(module_params=hparams)
