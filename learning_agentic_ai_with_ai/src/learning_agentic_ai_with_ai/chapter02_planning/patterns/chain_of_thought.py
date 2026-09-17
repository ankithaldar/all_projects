#!/usr/bin/env python
# -- coding: utf-8 --

'''Chain-of-thought reasoning with optional self-consistency.

Pattern: STEP-LABELED REASONING (a.k.a. chain-of-thought)

    task ─▶ CoT prompt ─▶ LLM ─▶ "Reasoning: 1.. 2.. Answer: X"
                         │
                         ├─ parse steps + answer (defensive)
                         └─ optional: sample N times, majority-vote answers

Why it works: autoregressive models spend compute per generated token.
Forcing intermediate steps gives the model more tokens to compute with -
"latent reasoning circuits" get activated - which measurably improves
multi-step arithmetic and constraint satisfaction. The engineering catch is
that free-form reasoning must not leak into machine-consumed outputs, so
the final answer is delimited and parsed structurally.

Self-consistency: sample several reasoning traces at a higher temperature
and keep the majority answer. Agreement across independent traces is a
usable confidence signal; low agreement marks the run `needs_review`.
'''


from __future__ import annotations

import re
from typing import Dict, List, Optional

from agentic_common.logging import get_logger, log_event
from agentic_common.tracing import TokenUsage
from chapter02_planning.config import PlanningConfig
from chapter02_planning.llm import ReasoningLLM, usage_delta
from chapter02_planning.parsing import (
  CoTParsed,
  extract_numbers,
  parse_cot_output,
)
from chapter02_planning.prompting import (
  COT_EXEMPLAR,
  COT_PROMPT,
  COT_SYSTEM,
)
from chapter02_planning.schemas import (
  ChainOfThoughtResult,
  ChainOfThoughtSample,
)

logger = get_logger(__name__)

_PUNCT_RE = re.compile(r'[^\w\s.%/-]')


def _normalize_answer(text: str) -> str:
  '''Normalize an answer for majority voting.

  Args:
    text: Raw answer text.

  Returns:
    Lowercased, punctuation-stripped, whitespace-collapsed text.
  '''
  lowered = (text or '').lower().strip()
  return re.sub(r'\s+', ' ', _PUNCT_RE.sub('', lowered)).strip()


class ChainOfThoughtReasoner:
  '''Runs step-labeled reasoning, optionally with self-consistency.'''

  def __init__(
    self,
    llm: ReasoningLLM,
    config: PlanningConfig,
  ) -> None:
    '''Initialize the reasoner.

    Args:
      llm: Gateway-backed reasoning LLM wrapper.
      config: Chapter 2 configuration.
    '''
    self._llm = llm
    self._config = config

  def reason(
    self,
    task: str,
    context: str = '',
    samples: Optional[int] = None,
  ) -> ChainOfThoughtResult:
    '''Reason about a task and return steps plus a final answer.

    Args:
      task: The question.
      context: Optional sanitized context (treated as untrusted data).
      samples: Override for the number of self-consistency samples.

    Returns:
      A ChainOfThoughtResult with the selected answer, all samples, and the
      agreement fraction.
    '''
    sample_count = max(1, samples or self._config.cot_samples)
    temperature = (
      self._config.cot_temperature if sample_count > 1
      else self._config.base_temperature
    )
    before = TokenUsage(
      input_tokens=self._llm.budget.spent.input_tokens,
      output_tokens=self._llm.budget.spent.output_tokens,
      total_tokens=self._llm.budget.spent.total_tokens,
    )
    errors: List[str] = []
    collected: List[CoTParsed] = []

    for index in range(1, sample_count + 1):
      if self._llm.budget.exceeded:
        errors.append('token budget exhausted during CoT sampling')
        break
      parsed = self._sample(task, context, temperature, index, errors)
      if parsed is not None:
        collected.append(parsed)

    after = self._llm.budget.spent
    usage = usage_delta(before, after)

    if not collected:
      log_event(logger, 40, 'cot_failed', task_preview=task[:120])
      return ChainOfThoughtResult(
        task=task, answer='', samples=[], agreement=0.0,
        status='error', usage=usage,
        errors=errors or ['no valid chain-of-thought samples'],
      )

    winner_index, agreement = self._select_sample(collected)
    winner = collected[winner_index]
    status = 'completed' if winner.ok and agreement >= 0.5 else 'needs_review'

    log_event(
      logger, 20, 'cot_done',
      samples=len(collected), agreement=round(agreement, 2),
      steps=len(winner.steps), status=status,
    )

    return ChainOfThoughtResult(
      task=task,
      steps=winner.steps,
      answer=winner.answer,
      samples=[
        ChainOfThoughtSample(
          steps=sample.steps, answer=sample.answer,
          parse_ok=sample.ok, error=sample.error,
        )
        for sample in collected
      ],
      agreement=round(agreement, 3),
      status=status,
      usage=usage,
      errors=errors,
    )

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _sample(
    self,
    task: str,
    context: str,
    temperature: float,
    index: int,
    errors: List[str],
  ) -> Optional[CoTParsed]:
    '''Produce and parse one reasoning sample.

    A single repair round is attempted when the answer marker is missing.

    Args:
      task: The question.
      context: Sanitized context string.
      temperature: Sampling temperature.
      index: 1-based sample index (for labels).
      errors: Mutable error collector.

    Returns:
      Parsed sample, or None when the call itself failed.
    '''
    prompt = COT_PROMPT.render(task=task, context=context or '(none)')
    result = self._llm.complete(
      prompt,
      system=f'{COT_SYSTEM}\n\n{COT_EXEMPLAR}',
      temperature=temperature,
      label=f'cot.sample{index}',
    )
    if result is None:
      errors.append(f'cot sample {index}: gateway call failed')
      return None

    parsed = parse_cot_output(result.content)
    if parsed.ok:
      return parsed

    errors.append(f'cot sample {index}: {parsed.error}')
    repair = self._llm.complete(
      f'{prompt}\n\nYour previous answer was malformed: {parsed.error}\n'
      'Rewrite it in the exact required format. Include the "Answer:" line.',
      system=COT_SYSTEM,
      temperature=0.0,
      label=f'cot.sample{index}.repair',
    )
    if repair is None:
      return parsed
    repaired = parse_cot_output(repair.content)
    if not repaired.ok:
      errors.append(f'cot sample {index}: repair failed ({repaired.error})')
    return repaired

  def _select_sample(
    self,
    samples: List[CoTParsed],
  ) -> tuple[int, float]:
    '''Majority-vote across samples on normalized answers.

    First groups by normalized text; if every group is size one, regroups by
    numeric content so "190 units" and "quantity: 190" agree.

    Args:
      samples: Parsed samples.

    Returns:
      (winning index, agreement fraction) pair.
    '''
    text_groups: Dict[str, List[int]] = {}
    for index, sample in enumerate(samples):
      text_groups.setdefault(_normalize_answer(sample.answer), []).append(index)

    groups = text_groups
    largest_group = max(len(indices) for indices in groups.values())
    if largest_group == 1 and len(samples) > 1:
      number_groups: Dict[tuple, List[int]] = {}
      for index, sample in enumerate(samples):
        numbers = tuple(sorted(extract_numbers(sample.answer)))
        key = numbers if numbers else (
          'text', _normalize_answer(sample.answer),
        )
        number_groups.setdefault(key, []).append(index)
      groups = number_groups

    best_key = max(groups, key=lambda key: len(groups[key]))
    winner = groups[best_key][0]
    agreement = len(groups[best_key]) / len(samples)
    return winner, agreement
