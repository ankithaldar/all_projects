#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Train the model base script'''

from argparse import ArgumentParser
from engine.classification_engine import ClassificationEngine
from config_params.params_reader import Parameters


def main(module_params=None):
  '''main function'''
  pe = ClassificationEngine(module_params)
  pe.train()


if __name__ == '__main__':

  parser = ArgumentParser(parents=[])
  parser.add_argument('--params_yml', type=str)
  params_file_path = parser.parse_args()

  hparams = Parameters(params_file_path.params_yml)

  main(module_params=hparams)
