#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Discord webhook notifier -- fire-and-forget transport for training updates.

Secret resolution follows knee.helpers.env.get_secret():
os.environ -> .env file -> Kaggle Secrets. If DISCORD_WEBHOOK_URL is
absent every send is a silent no-op, so the callback can stay wired in
base.yaml for local runs without a channel.

Design guarantees:
- NEVER raises: a Discord outage must never kill a 9-hour kernel run.
- Rate-limit aware: drops messages inside ``min_interval`` instead of
  queueing (stale progress spam is worthless anyway).
- Respects Discord's 2000-char content limit.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import requests

from knee.helpers.env import get_secret
from knee.helpers.logging_utils import get_logger

WEBHOOK_SECRET_KEY = 'DISCORD_WEBHOOK_URL'
_DISCORD_CONTENT_LIMIT = 1900


class DiscordNotifier:
  """Fire-and-forget Discord webhook transport.

  Never raises; rate-limited; silently disabled without a webhook URL.

  Args:
      webhook_url: Explicit URL override; defaults to secret lookup.
      username: Display name posted with each message.
      min_interval: Minimum seconds between non-forced sends.
      timeout: HTTP timeout per send attempt.
      verbose: Emit local warnings on failed deliveries.
  """

  def __init__(
    self,
    webhook_url: str | None = None,
    username: str = 'rsna-knee-bot',
    min_interval: float = 1.0,
    timeout: float = 10.0,
    verbose: bool = True,
  ) -> None:
    self.webhook_url = webhook_url or get_secret(WEBHOOK_SECRET_KEY)
    self.username = username
    self.min_interval = min_interval
    self.timeout = timeout
    self._last_send = 0.0
    self._log = get_logger('knee.discord') if verbose else None

  @property
  def enabled(self) -> bool:
    return bool(self.webhook_url)

  def send(self, message: str, force: bool = False, **fields: Any) -> bool:
    """POST ``message`` to the webhook; returns success flag.

    False means skipped or failed.

    ``force=True`` skips rate limiting (crash reports must always land).
    """
    if not self.enabled:
      return False
    now = time.monotonic()
    if not force and now - self._last_send < self.min_interval:
      return False
    payload = {
      'username': self.username,
      'content': self._clip(message),
      **({'embeds': [fields]} if fields else {}),
    }
    try:
      resp = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
      ok = resp.status_code < 400
      if not ok and self._log:
        self._log.warning('discord webhook returned %s', resp.status_code)
      self._last_send = now
      return ok
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if self._log:
        self._log.warning('discord notification failed: %s', exc)
      return False

  @staticmethod
  def _clip(text: str) -> str:
    return (
      text
      if len(text) <= _DISCORD_CONTENT_LIMIT
      else text[: _DISCORD_CONTENT_LIMIT - 3] + '...'
    )

  # ------------------------------------------------------------------ #
  @staticmethod
  def fmt_metrics(metrics: Mapping[str, Any], decimals: int = 4) -> str:
    """Render a metric dict as aligned ``name=value`` lines."""

    def render(v: Any) -> str:
      try:
        return f'{float(v):.{decimals}f}'
      except (TypeError, ValueError):
        return str(v)

    width = max((len(k) for k in metrics), default=0)
    return '\n'.join(
      f'`{k.ljust(width)}` = {render(v)}' for k, v in sorted(metrics.items())
    )
