#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Example CLI entry point for the LLM gateway.'''


from __future__ import annotations

import argparse
from pathlib import Path

from llm_gateway.app import LLMGateway
from llm_gateway.schemas import GatewayRequest

PACKAGE_ROOT = Path(__file__).resolve().parent


def main() -> None:
  '''Run a simple synchronous gateway request.'''
  parser = argparse.ArgumentParser(description='LLM gateway demo')
  parser.add_argument(
    '--prompt',
    default='Explain what an LLM gateway does in one sentence.',
    help='Prompt to send to the gateway',
  )
  parser.add_argument(
    '--session-id',
    default='demo-session',
    help='Session identifier',
  )

  args = parser.parse_args()

  gateway = LLMGateway(
    config_path=str(PACKAGE_ROOT / 'config' / 'gateway.yaml'),
    env_path=str(PACKAGE_ROOT / '.env'),
  )

  request = GatewayRequest(
    prompt=args.prompt,
    session_id=args.session_id,
  )

  try:
    response = gateway.complete(request)
    print(response.content)
    print(f'provider={response.provider}')
    print(f'model={response.model}')
    print(f'cached={response.cached}')
    print(f'latency_ms={response.latency_ms:.2f}')
    print(f'cost={response.cost:.6f}')
  finally:
    gateway.close()


if __name__ == '__main__':
  main()
