#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Hardware inspection helpers (CPU, memory, GPU visibility).'''

from __future__ import annotations

import multiprocessing
import os
import shutil
import subprocess


def _gpu_lines(stdout: str) -> list[str]:
  '''Select NVIDIA lines from ``lspci`` output.

  Args:
    stdout: Raw stdout of the ``lspci`` command.

  Returns:
    list[str]: Lines mentioning NVIDIA, in output order.
  '''
  return [
      line for line in stdout.splitlines() if 'nvidia' in line.lower()
  ]


def _format_gpu_status(gpu_lines: list[str]) -> str:
  '''Format the GPU status line for display.

  Args:
    gpu_lines: NVIDIA device lines reported by ``lspci``.

  Returns:
    str: Status text, joining lines with newlines or reporting
      that no GPU was found.
  '''
  if not gpu_lines:
    return 'GPU status: none found'
  return f'GPU status: {chr(10).join(gpu_lines)}'


def _print_lspci_gpus() -> None:
  '''Print NVIDIA devices via lspci when the tool is available.

  Runs ``lspci`` with a short timeout and degrades silently when
  the tool is missing or fails. Detection itself is delegated to
  :func:`_gpu_lines` and :func:`_format_gpu_status`.

  Returns:
    None: Output is printed; no value is returned.
  '''
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
  print(_format_gpu_status(_gpu_lines(result.stdout)))


def _print_cpu_status() -> None:
  '''Print the number of available CPU cores.

  Uses :mod:`multiprocessing` so the reported count matches the
  interpreter's view of the machine.

  Returns:
    None: Output is printed; no value is returned.
  '''
  print('Number of CPU cores:', multiprocessing.cpu_count())


def _print_memory_status() -> None:
  '''Print memory totals from ``/proc/meminfo`` when present.

  Skips silently on platforms without ``/proc/meminfo`` so the
  remaining probes still run.

  Returns:
    None: Output is printed; no value is returned.
  '''
  if os.path.exists('/proc/meminfo'):
    with open('/proc/meminfo', encoding='utf-8') as meminfo:
      for line in meminfo:
        if line.startswith('Mem'):
          print(line.strip())


def _print_torch_status() -> None:
  '''Print PyTorch CUDA availability when PyTorch is installed.

  Degrades to a note when PyTorch is not installed.

  Returns:
    None: Output is printed; no value is returned.
  '''
  try:
    import torch

    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    for index in range(torch.cuda.device_count()):
      print(f'GPU {index}: {torch.cuda.get_device_name(index)}')
  except ImportError:
    print('PyTorch not installed; skipping CUDA probe')


def _init_ray(ray_module) -> None:
  '''Initialize the Ray runtime, tolerating re-initialization.

  Args:
    ray_module: Imported ``ray`` module to initialize.

  Returns:
    None: The Ray runtime is mutated in place.
  '''
  ray_module.init(ignore_reinit_error=True)


def _report_ray_gpu(ray_module) -> None:
  '''Print Ray GPU visibility after initializing Ray.

  Args:
    ray_module: Imported ``ray`` module to query.

  Returns:
    None: Output is printed; no value is returned.
  '''
  _init_ray(ray_module)
  cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
  print(f'ray.get_gpu_ids(): {ray_module.get_gpu_ids()}')
  print(f'CUDA_VISIBLE_DEVICES: {cuda_visible}')


def _print_ray_status() -> None:
  '''Print Ray GPU visibility when a Ray runtime is present.

  Degrades to a note when Ray is not installed.

  Returns:
    None: Output is printed; no value is returned.
  '''
  try:
    import ray

    _report_ray_gpu(ray)
  except ImportError:
    print('Ray not installed; skipping Ray GPU probe')


def print_hardware_status() -> None:
  '''Print CPU, memory, and GPU availability information.

  GPU detection uses lspci plus PyTorch when installed; Ray GPU
  visibility is reported when a Ray runtime is present. Missing
  components degrade to a note.

  Returns:
    None: Output is printed; no value is returned.
  '''
  _print_cpu_status()
  _print_memory_status()
  _print_lspci_gpus()
  _print_torch_status()
  _print_ray_status()
