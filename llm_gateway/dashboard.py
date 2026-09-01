#!/usr/bin/env python
# -- coding: utf-8 --

'''Local HTML dashboard and JSON stats API.'''


from __future__ import annotations

import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

TEMPLATE_PATH = Path(__file__).parent / 'templates' / 'dashboard.html'
DEFAULT_HTML = '''<!doctype html>
<html>
  <head>
    <meta charset='utf-8'>
    <title>LLM Gateway Dashboard</title>
  </head>
  <body>
    <h1>LLM Gateway Dashboard</h1>
    <p>Missing dashboard template.</p>
  </body>
</html>
'''


class _DashboardHandler(BaseHTTPRequestHandler):
  '''HTTP handler for dashboard HTML and stats API.'''

  db_path = './data/gateway.db'

  def do_GET(self) -> None:
    '''Handle GET requests.'''
    path = self.path.split('?')[0]

    if path == '/':
      self._send_html()
    elif path == '/api/stats':
      self._send_stats()
    else:
      self.send_error(404, 'Not Found')

  def _send_html(self) -> None:
    '''Serve dashboard HTML.'''
    if TEMPLATE_PATH.exists():
      body = TEMPLATE_PATH.read_text(encoding='utf-8').encode('utf-8')
    else:
      body = DEFAULT_HTML.encode('utf-8')

    self._send(200, body, 'text/html; charset=utf-8')

  def _send_stats(self) -> None:
    '''Serve JSON stats.'''
    stats = self._build_stats()
    body = json.dumps(stats, default=str).encode('utf-8')
    self._send(200, body, 'application/json; charset=utf-8')

  def _send(self, status: int, body: bytes, content_type: str) -> None:
    '''Send HTTP response.

    Args:
      status: HTTP status code.
      body: Response body.
      content_type: Content type header.
    '''
    self.send_response(status)
    self.send_header('Content-Type', content_type)
    self.send_header('Content-Length', str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def _build_stats(self) -> Dict[str, Any]:
    '''Build usage stats from SQLite.

    Returns:
      Stats dictionary.
    '''
    try:
      conn = sqlite3.connect(self.db_path)
      conn.row_factory = sqlite3.Row
      cur = conn.cursor()

      cur.execute(
        '''
        SELECT
          COUNT(*) AS total_calls,
          COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
          COALESCE(SUM(input_tokens), 0) AS input_tokens,
          COALESCE(SUM(output_tokens), 0) AS output_tokens,
          COALESCE(SUM(cost), 0) AS total_cost
        FROM llm_calls
        '''
      )
      summary = dict(cur.fetchone())

      cur.execute(
        '''
        SELECT status, COUNT(*) AS count
        FROM llm_calls
        GROUP BY status
        ORDER BY count DESC
        '''
      )
      statuses = [dict(row) for row in cur.fetchall()]

      total_calls = int(summary.get('total_calls') or 0)
      failed_calls = sum(
        int(item.get('count') or 0)
        for item in statuses
        if str(item.get('status') or '').lower() != 'success'
      )
      error_rate = failed_calls / max(total_calls, 1)

      cur.execute(
        '''
        SELECT
          provider,
          COUNT(*) AS calls,
          COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
          COALESCE(SUM(input_tokens), 0) AS input_tokens,
          COALESCE(SUM(output_tokens), 0) AS output_tokens,
          COALESCE(SUM(cost), 0) AS total_cost
        FROM llm_calls
        GROUP BY provider
        ORDER BY calls DESC
        '''
      )
      per_provider = [dict(row) for row in cur.fetchall()]

      cur.execute(
        '''
        SELECT
          COALESCE(session_id, 'unknown') AS session_id,
          COUNT(*) AS calls,
          COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
          COALESCE(SUM(input_tokens), 0) AS input_tokens,
          COALESCE(SUM(output_tokens), 0) AS output_tokens,
          COALESCE(SUM(cost), 0) AS total_cost
        FROM llm_calls
        GROUP BY session_id
        ORDER BY calls DESC
        '''
      )
      per_session = [dict(row) for row in cur.fetchall()]

      cur.execute(
        '''
        SELECT
          timestamp,
          provider,
          model_used,
          status,
          latency_ms,
          input_tokens,
          output_tokens,
          cost
        FROM llm_calls
        ORDER BY id DESC
        LIMIT 100
        '''
      )
      recent_calls = [dict(row) for row in cur.fetchall()]

      conn.close()

      return {
        'summary': summary,
        'error_rate': error_rate,
        'statuses': statuses,
        'per_provider': per_provider,
        'per_session': per_session,
        'recent_calls': recent_calls,
      }
    except sqlite3.Error:
      return {
        'summary': {
          'total_calls': 0,
          'avg_latency_ms': 0.0,
          'input_tokens': 0,
          'output_tokens': 0,
          'total_cost': 0.0,
        },
        'error_rate': 0.0,
        'statuses': [],
        'per_provider': [],
        'per_session': [],
        'recent_calls': [],
      }


def run_dashboard(
  db_path: str | Path,
  host: str = '127.0.0.1',
  port: int = 8099,
) -> None:
  '''Run dashboard HTTP server.

  Args:
    db_path: Path to gateway logging database.
    host: Bind host.
    port: Bind port.
  '''
  _DashboardHandler.db_path = str(db_path)
  server = ThreadingHTTPServer((host, port), _DashboardHandler)
  print(f'Dashboard available at http://{host}:{port}/')
  server.serve_forever()


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='LLM gateway dashboard')
  parser.add_argument(
    '--db',
    default=os.getenv('GATEWAY_DB_PATH', './data/gateway.db'),
    help='SQLite logging database path',
  )
  parser.add_argument('--host', default='127.0.0.1', help='Bind host')
  parser.add_argument('--port', type=int, default=8099, help='Bind port')

  args = parser.parse_args()
  run_dashboard(args.db, args.host, args.port)
