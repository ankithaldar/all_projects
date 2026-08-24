#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier-2 alternative: LLM report labeling via the OpenRouter API.

Reads each multilingual radiology report zero-shot and returns the 12
finding probabilities -- replacing/complementing the XLM-R text teacher
where no GPU budget or training time is available (BLUEPRINT section 3,
Tier 2). Design notes:

* SRP/DIP: this module owns *transport + schema* (one report -> one
  labeled vector); orchestration/caching lives with callers through an
  injectable ``label_fn`` so kernels stay testable without network.
* Robustness: API-level ``response_format: json_object`` constraint
  with automatic per-provider fallback when rejected, strict JSON
  contract in the prompt, fence/prose-tolerant parsing, retries with
  exponential backoff on 429/5xx/timeouts, and a hard failure signal
  (``LLMLabelError``) that callers may route to Tier-1 rules.
* Cost: one call per study at roughly 500-1500 tokens; flash-class
  models keep full-train labeling in the few-dollar range.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from knee.config_params.schema import TARGETS
from knee.helpers.logging_utils import get_logger

_OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'

_SYSTEM_PROMPT = (
  'You are an expert musculoskeletal radiologist reading knee MRI '
  'reports written in any language. For EACH finding below, decide '
  'from the report alone:\n'
  '  1.0 = definitely present, 0.0 = definitely absent/negated,\n'
  '  0.5 = not mentioned or too uncertain.\n'
  'Findings: ACL (anterior cruciate ligament tear), MCL (medial '
  'collateral ligament tear), Medial Meniscus tear, Lateral Meniscus '
  'tear, Medial OA (medial compartment osteoarthritis), Lateral OA, '
  'PF OA (patellofemoral osteoarthritis), Effusion (joint effusion), '
  "Synovitis, Baker's cyst (popliteal cyst), Contusion (bone bruise), "
  'Fracture.\n'
  'Respect negation ("intact", "no evidence of", "kein", "sans", '
  '"нет", "无"). Ignore findings of other joints. Answer with ONLY a '
  'JSON object whose keys are exactly the finding names above and '
  'whose values are 0, 0.5 or 1.'
)


class LLMLabelError(RuntimeError):
  """Raised when a report cannot be labeled after all retries."""


def parse_llm_json(text: str) -> dict:
  """Extract a JSON object from a model reply.

  Tolerates markdown fences and surrounding prose.

  Args:
      text: Raw assistant message content.

  Returns:
      Parsed dict.

  Raises:
      LLMLabelError: If no JSON object can be recovered.
  """
  cleaned = text.strip()
  if '```' in cleaned:
    for chunk in cleaned.split('```'):
      candidate = chunk.strip()
      if candidate.startswith('json'):
        candidate = candidate[4:]
      if candidate.startswith('{'):
        cleaned = candidate
        break
  start, end = cleaned.find('{'), cleaned.rfind('}')
  if start < 0 or end <= start:
    raise LLMLabelError(f'no JSON object in reply: {text[:200]!r}')
  try:
    parsed = json.loads(cleaned[start : end + 1])
  except json.JSONDecodeError as exc:
    raise LLMLabelError(f'unparsable JSON: {exc}: {text[:200]!r}') from exc
  return parsed if isinstance(parsed, dict) else {}


def strip_to_findings(mapping: dict) -> dict:
  """Keep ONLY the canonical finding keys from a model reply.

  Everything else -- rationales, summaries, confidence notes, wrapper
  objects like ``{"findings": {...}}`` -- is stripped so downstream
  fusion sees exactly the 12 competition targets and nothing more.

  Args:
      mapping: Parsed JSON object from the model reply.

  Returns:
      Dict restricted to the TARGETS keys that are present.
  """
  known = {target: mapping[target] for target in TARGETS if target in mapping}
  if not known:
    # Some models nest the findings under a single wrapper key
    # (e.g. {"findings": {...}}); descend once before giving up.
    nested = [v for v in mapping.values() if isinstance(v, dict)]
    if len(nested) == 1:
      inner = nested[0]
      known = {target: inner[target] for target in TARGETS if target in inner}
  return known


def _coerce(value) -> float:  # noqa: ANN001 - model output is untyped
  """Convert one model value into a [0, 1] probability.

  Args:
      value: Number, numeric string, or anything unparsable.

  Returns:
      Clamped float; 0.5 (unknown) when the value cannot be parsed.
  """
  try:
    result = float(value)
  except (TypeError, ValueError):
    return 0.5
  return min(max(result, 0.0), 1.0)


def apply_schema(mapping: dict) -> tuple[np.ndarray, np.ndarray]:
  """Project a model reply onto the canonical 12-target vectors.

  Non-finding keys are stripped first (:func:`strip_to_findings`), so
  extra model chatter can never influence the label vectors.

  Args:
      mapping: Finding-name -> {0, 0.5, 1} mapping from the model.

  Returns:
      ``(probs, mask)`` where probs is float32(12,) clamped to [0, 1]
      (missing keys -> 0.5) and mask marks definite answers (values
      other than 0.5) that downstream fusion may trust.
  """
  findings = strip_to_findings(mapping)
  probs = np.array(
    [_coerce(findings.get(target, 0.5)) for target in TARGETS],
    dtype=np.float32,
  )
  np.clip(probs, 0.0, 1.0, out=probs)
  mask = (~np.isclose(probs, 0.5)).astype(bool)
  return probs, mask


class OpenRouterLabeler:
  """Labels single reports through OpenRouter chat completions.

  Args:
      api_key: OpenRouter API key (Bearer token).
      model: OpenRouter model slug, e.g.
          'stealth/ox-alpha'.
      temperature: Sampling temperature (0 for deterministic labels).
      max_tokens: Completion cap; JSON replies are short.
      max_retries: Attempts before raising LLMLabelError.
      timeout: Per-request timeout in seconds.
  """

  def __init__(
    self,
    api_key: str,
    model: str = 'stealth/ox-alpha',
    temperature: float = 0.0,
    max_tokens: int = 256,
    max_retries: int = 3,
    timeout: int = 90,
    force_json: bool = True,
  ) -> None:
    """Store request configuration.

    Args:
        api_key: OpenRouter Bearer key.
        model: Model slug on openrouter.ai.
        temperature: Generation temperature.
        max_tokens: Reply token cap.
        max_retries: Total attempts per report.
        timeout: Request timeout seconds.
        force_json: Send ``response_format: json_object`` so compliant
            providers hard-constrain output to JSON. Models that reject
            the flag are auto-detected and fall back to prompt-only
            enforcement (the parser tolerates decorations either way).
    """
    if not api_key:
      raise ValueError('api_key must be non-empty')
    self.api_key = api_key
    self.model = model
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.max_retries = max(1, int(max_retries))
    self.timeout = timeout
    self.force_json = force_json
    self._json_mode_ok = True

  def _build_payload(self, report: str) -> dict:
    """Assemble the chat-completions request body for one report.

    Args:
        report: Verbatim report text (any language).

    Returns:
        Request payload dict; includes the JSON response format while
        it is known to be supported by the routed provider.
    """
    payload = {
      'model': self.model,
      'temperature': self.temperature,
      'max_tokens': self.max_tokens,
      'messages': [
        {'role': 'system', 'content': _SYSTEM_PROMPT},
        {'role': 'user', 'content': report},
      ],
    }
    if self.force_json and self._json_mode_ok:
      payload['response_format'] = {'type': 'json_object'}
    return payload

  @staticmethod
  def _rejects_json_mode(status_code: int, body: str) -> bool:
    """Detect providers that refuse ``response_format`` outright.

    Args:
        status_code: HTTP status of the failed call.
        body: Response body text.

    Returns:
        True when the failure is a JSON-mode capability rejection
        (as opposed to auth/billing/schema errors).
    """
    if status_code != 400:
      return False
    lowered = body.lower()
    return any(
      token in lowered
      for token in (
        'response_format',
        'json_object',
        'json mode',
        'not supported',
        'unsupported',
      )
    )

  def _post(self, report: str) -> str:
    """Call the chat endpoint once, retrying transient failures.

    Args:
        report: Verbatim report text (any language).

    Returns:
        Assistant message content.

    Raises:
        LLMLabelError: After exhausting retries.
    """
    log = get_logger('llm_labeler')
    headers = {
      'Authorization': f'Bearer {self.api_key}',
      'Content-Type': 'application/json',
      'X-Title': 'rsna-knee-weak-labels',
    }
    last_error = ''
    for attempt in range(self.max_retries):
      try:
        response = requests.post(
          _OPENROUTER_URL,
          json=self._build_payload(report),
          headers=headers,
          timeout=self.timeout,
        )
        if response.status_code == 200:
          return response.json()['choices'][0]['message']['content']
        last_error = f'{response.status_code}: {response.text[:300]}'
        if self._rejects_json_mode(response.status_code, response.text):
          # Provider cannot honor response_format; degrade gracefully
          # to prompt-only enforcement and retry immediately.
          self._json_mode_ok = False
          log.warning(
            'model %r rejected json response_format; '
            'falling back to prompt-only JSON contract',
            self.model,
          )
          continue
        if response.status_code not in (429, 500, 502, 503, 504):
          break  # permanent (auth/billing/schema): do not retry
        wait = float(response.headers.get('Retry-After', 2**attempt))
      except requests.RequestException as exc:
        last_error = repr(exc)
        wait = 2**attempt
      log.warning(
        'attempt %d/%d failed (%s); backing off %.1fs',
        attempt + 1,
        self.max_retries,
        last_error,
        wait,
      )
      time.sleep(min(wait, 60))
    raise LLMLabelError(
      f'OpenRouter labeling failed after {self.max_retries} attempts: '
      f'{last_error}'
    )

  def label_report(self, report: str) -> tuple[np.ndarray, np.ndarray]:
    """Label one report into canonical probabilities and trust mask.

    Args:
        report: Verbatim multilingual report text.

    Returns:
        ``(probs, mask)`` as produced by :func:`apply_schema`.
    """
    content = self._post(report.strip())
    return apply_schema(parse_llm_json(content))


def report_key(report: str) -> str:
  """Stable cache key for a report body.

  Args:
      report: Report text.

  Returns:
      SHA-1 hex digest.
  """
  return hashlib.sha1(report.encode('utf-8')).hexdigest()


def label_many(
  uids: pd.Series,
  texts: pd.Series,
  label_fn,
  cache_path: str | None = None,
  concurrency: int = 2,
  flush_every: int = 25,
) -> pd.DataFrame:
  """Label many reports concurrently behind a persistent disk cache.

  The cache makes the 12 h kernel limit survivable: completed reports
  are never re-billed, and partial progress is flushed periodically
  (and at shutdown) so a fresh kernel resumes where this one stopped.

  Args:
      uids: StudyInstanceUID series aligned with ``texts``.
      texts: Report texts (NaN-safe; empties yield all-unknown rows).
      label_fn: Callable text -> (probs, mask); exceptions are caught
          and become all-unknown rows so one bad report cannot kill a
          long run.
      cache_path: Optional parquet keyed by ``report_key(text)`` with
          columns [key, probs_json, mask_json].
      concurrency: Worker threads (the workload is I/O bound).
      flush_every: Cache flush interval in newly labeled reports.

  Returns:
      Frame indexed like ``uids`` with columns
      ``[StudyInstanceUID, *TARGETS]`` holding the probabilities;
      trust masks ride along in ``_mask_<target>`` columns for fusion
      scripts that need them.
  """
  log = get_logger('llm_labeler')
  cache: dict[str, tuple[list, list]] = {}
  if cache_path and Path(cache_path).exists():
    stored = pd.read_parquet(cache_path)
    cache = {
      row.key: (json.loads(row.probs_json), json.loads(row.mask_json))
      for row in stored.itertuples()
    }
    log.info('loaded %d cached labels', len(cache))

  def _encode(probs: np.ndarray, mask: np.ndarray) -> tuple[list, list]:
    """Serialize one label pair for cache storage.

    Args:
        probs: Probability vector.
        mask: Trust-mask vector.

    Returns:
        JSON-ready (probs_list, mask_list) tuple.
    """
    return [float(p) for p in probs], [bool(m) for m in mask]

  keys = [report_key(str(text)) if str(text).strip() else '' for text in texts]
  pending = [
    index for index, key in enumerate(keys) if key and key not in cache
  ]
  done_since_flush = 0

  def _flush() -> None:
    """Persist the cache snapshot to disk."""
    if not cache_path or not cache:
      return
    frame = pd.DataFrame(
      [
        {
          'key': key,
          'probs_json': json.dumps(v[0]),
          'mask_json': json.dumps(v[1]),
        }
        for key, v in cache.items()
      ]
    )
    frame.to_parquet(cache_path)

  with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
    futures = {}
    for index in pending:
      text = str(texts.iloc[index])
      futures[pool.submit(label_fn, text)] = index
    for future in as_completed(futures):
      index = futures[future]
      key = keys[index]
      try:
        cache[key] = _encode(*future.result())
      except Exception as exc:  # pylint: disable=broad-exception-caught
        log.warning(
          'study %s unlabeled (%s); marking unknown', uids.iloc[index], exc
        )
        cache[key] = _encode(
          np.full(len(TARGETS), 0.5, np.float32),
          np.zeros(len(TARGETS), bool),
        )
      done_since_flush += 1
      if done_since_flush % flush_every == 0:
        _flush()
        log.info('labeled %d/%d reports', done_since_flush, len(pending))
  _flush()

  unknown_probs = [0.5] * len(TARGETS)
  unknown_mask = [False] * len(TARGETS)
  prob_rows, mask_rows = [], []
  for key in keys:
    entry = cache.get(key, (unknown_probs, unknown_mask))
    prob_rows.append(entry[0])
    mask_rows.append(entry[1])
  result = pd.DataFrame(prob_rows, columns=list(TARGETS))
  result.insert(0, 'StudyInstanceUID', uids.to_numpy())
  for position, target in enumerate(TARGETS):
    result[f'_mask_{target}'] = [row[position] for row in mask_rows]
  return result
