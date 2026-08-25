#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Hardware inspection helpers (CPU, memory, GPU visibility).'''

from __future__ import annotations

import multiprocessing
import os
import shutil
import subprocess


def _print_lspci_gpus() -> None:
  '''Print NVIDIA devices via lspci when the tool is available.'''
  if not shutil.which('lspci'):
    return
  try:
    result = subprocess.run(
        ['lspci'],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
  except (OSError, subprocess.TimeoutExpired):
    return
  gpu_lines = [line for line in result.stdout.splitlines() if 'nvidia' in line.lower()]
  print(f'GPU status: {chr(10).join(gpu_lines)}' if gpu_lines else 'GPU status: none found')


def print_hardware_status() -> None:
  '''Print CPU, memory, and GPU availability information.

  GPU detection uses lspci plus PyTorch when installed; Ray GPU visibility
  is reported when a Ray runtime is present. Missing components degrade to
  a note.
  '''
  print('Number of CPU cores:', multiprocessing.cpu_count())
  if os.path.exists('/proc/meminfo'):
    with open('/proc/meminfo', encoding='utf-8') as meminfo:
      for line in meminfo:
        if line.startswith('Mem'):
          print(line.strip())
  _print_lspci_gpus()

  try:
    import torch

    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    for index in range(torch.cuda.device_count()):
      print(f'GPU {index}: {torch.cuda.get_device_name(index)}')
  except ImportError:
    print('PyTorch not installed; skipping CUDA probe')

  try:
    import ray

    ray.init(ignore_reinit_error=True)
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f'ray.get_gpu_ids(): {ray.get_gpu_ids()}')
    print(f'CUDA_VISIBLE_DEVICES: {cuda_visible}')
  except ImportError:
    print('Ray not installed; skipping Ray GPU probe')
