#!/usr/bin/env python
# -- coding: utf-8 --

'''ReAct: interleaving reasoning with tool actions.

Pattern: REASON -> ACT -> OBSERVE LOOP (a.k.a. ReAct)

    ┌─────────────────────────────────────────────────────────────┐
    │ for step in 1..max_steps:                                   │
    │   LLM(prompt + scratchpad)                                  │
    │     ├─ "Final Answer: ..."  ─────────────────▶ DONE         │
    │     └─ "Action: tool / Action Input: {...}"                 │
    │           └─ GATED TOOL EXECUTION ─▶ sanitized Observation  │
    │                 └─ append to scratchpad ─▶ next iteration   │
    └─────────────────────────────────────────────────────────────┘

Why it works: the model externalizes a thought before every action, so each
tool call is *justified* in context, and each observation updates the
reasoning state for the next turn. Unlike raw function calling (Chapter 1),
the reasoning trace is explicit and auditable.

Production guardrails implemented here:
  - strict protocol parsing with a one-turn format reminder,
  - bounded steps and token budget,
  - repeated-action detection (loop breaking) after 2 identical calls,
  - every observation is untrusted data wrapped by the tool gate,
  - stop reasons are machine-readable so callers can degrade gracefully.
'''


from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from agentic_common.logging import get_logger, log_event
from chapter02_planning.config import PlanningConfig
from chapter02_planning.llm import ReasoningLLM
from chapter02_planning.parsing import parse_react_turn
from chapter02_planning.prompting import (
  REACT_PROMPT,
  REACT_SYSTEM_TEMPLATE,
  render_tool_catalog,
)
from chapter02_planning.schemas import ReActStep, ReActTrace
from chapter02_planning.tools.provider import ToolProvider

logger = get_logger(__name__)

_FORMAT_REMINDER = (
  'FORMAT ERROR: {error}. Reply with exactly:\n'
  'Thought: <reasoning>\n'
  'Action: <tool name>\n'
  'Action Input: <JSON object>\n'
  'or finish with "Thought: ..." followed by "Final Answer: ...".'
)


class ReActAgent:
  '''A bounded ReAct loop over a (gated) tool provider.'''

  def __init__(
    self,
    llm: ReasoningLLM,
    tools: ToolProvider,
    config: PlanningConfig,
  ) -> None:
    '''Initialize the agent.

    Args:
      llm: Gateway-backed reasoning LLM wrapper.
      tools: Tool provider (wrap with GatedToolProvider for safety).
      config: Chapter 2 configuration.
    '''
    self._llm = llm
    self._tools = tools
    self._config = config

  def run(
    self,
    task: str,
    max_steps: Optional[int] = None,
  ) -> ReActTrace:
    '''Execute the ReAct loop for one task.

    Args:
      task: Natural-language task.
      max_steps: Optional step cap override.

    Returns:
      ReActTrace with every Thought/Action/Observation cycle and a
      machine-readable stop reason.
    '''
    steps_cap = max(1, min(
      max_steps or self._config.max_steps, self._config.max_steps * 2,
    ))
    tool_specs = self._tools.list_tools()
    system = REACT_SYSTEM_TEMPLATE.render(
      tools=render_tool_catalog(tool_specs),
    )
    tool_names = {tool.name for tool in tool_specs}

    trace = ReActTrace(task=task)
    scratchpad = ''
    repeat_counts: Dict[str, int] = {}
    format_failures = 0

    for index in range(1, steps_cap + 1):
      if self._llm.budget.exceeded:
        trace.status = 'error'
        trace.stop_reason = 'token_budget'
        break

      result = self._llm.complete(
        REACT_PROMPT.render(task=task, scratchpad=scratchpad or '(empty)'),
        system=system,
        temperature=self._config.react_temperature,
        label=f'react.step{index}',
      )
      if result is None:
        trace.status = 'error'
        trace.stop_reason = 'gateway_unavailable'
        break

      parsed = parse_react_turn(result.content)
      if parsed.is_final:
        trace.steps.append(
          ReActStep(index=index, thought=parsed.thought),
        )
        trace.final_answer = parsed.final_answer
        trace.status = 'completed'
        trace.stop_reason = 'final_answer'
        log_event(
          logger, 20, 'react_done',
          steps=index, stop_reason=trace.stop_reason,
        )
        return trace

      if not parsed.ok:
        format_failures += 1
        reminder = _FORMAT_REMINDER.format(error=parsed.error)
        trace.steps.append(
          ReActStep(
            index=index, thought=parsed.thought, error=parsed.error,
          ),
        )
        scratchpad += f'Observation: {reminder}\n'
        log_event(
          logger, 30, 'react_format_error',
          step=index, error=parsed.error,
        )
        if format_failures >= 2:
          trace.status = 'error'
          trace.stop_reason = 'format_failed'
          break
        continue

      format_failures = 0
      observation, step = self._act(index, parsed, tool_names, repeat_counts)
      trace.steps.append(step)
      scratchpad = self._append_scratchpad(
        scratchpad, parsed.thought, parsed.action,
        parsed.action_input, observation,
      )

      if step.error == 'loop_detected':
        trace.status = 'needs_review'
        trace.stop_reason = 'loop_detected'
        break
    else:
      trace.status = 'max_iterations'
      trace.stop_reason = 'max_steps'

    log_event(
      logger, 30, 'react_stopped',
      status=trace.status, stop_reason=trace.stop_reason,
      steps=len(trace.steps),
    )
    return trace

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _act(
    self,
    index: int,
    parsed: Any,
    tool_names: set,
    repeat_counts: Dict[str, int],
  ) -> tuple[str, ReActStep]:
    '''Validate, deduplicate, and execute one proposed action.

    Args:
      index: 1-based step index.
      parsed: Parsed ReAct turn with action + action_input.
      tool_names: Known tool names.
      repeat_counts: Mutable map of action signature to occurrence count.

    Returns:
      (observation text, ReActStep) pair.
    '''
    started = time.perf_counter()
    action = parsed.action
    signature = f'{action}:{json.dumps(parsed.action_input, sort_keys=True)}'
    repeat_counts[signature] = repeat_counts.get(signature, 0) + 1
    repeats = repeat_counts[signature]

    if action not in tool_names:
      available = ', '.join(sorted(tool_names)) or 'none'
      observation = (
        f'Unknown tool {action!r}. Available tools: '
        f'{available}. Pick one of these or give a Final Answer.'
      )
      return observation, ReActStep(
        index=index,
        thought=parsed.thought,
        action=action,
        action_input=parsed.action_input,
        observation=observation,
        tool_ok=False,
        error='unknown_tool',
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )

    if repeats >= 3:
      observation = (
        'Loop detected: you proposed the same action three times. '
        'Stop and give a Final Answer with what you know.'
      )
      return observation, ReActStep(
        index=index,
        thought=parsed.thought,
        action=action,
        action_input=parsed.action_input,
        observation=observation,
        tool_ok=False,
        error='loop_detected',
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )

    result = self._tools.execute(action, parsed.action_input)
    observation = result.text or '(empty observation)'
    if repeats == 2:
      observation = (
        'Note: this exact action was already tried once; its result is '
        f'repeated below.\n{observation}'
      )

    return observation, ReActStep(
      index=index,
      thought=parsed.thought,
      action=action,
      action_input=parsed.action_input,
      observation=observation,
      tool_ok=result.ok,
      error=result.error,
      latency_ms=result.latency_ms or (
        (time.perf_counter() - started) * 1000.0
      ),
    )

  def _append_scratchpad(
    self,
    scratchpad: str,
    thought: str,
    action: str,
    action_input: Dict[str, Any],
    observation: str,
  ) -> str:
    '''Append one cycle to the scratchpad and cap its size.

    Args:
      scratchpad: Current scratchpad text.
      thought: Model thought.
      action: Tool name.
      action_input: Tool arguments.
      observation: Sanitized tool observation.

    Returns:
      Updated scratchpad text.
    '''
    entry = (
      f'Thought: {thought}\n'
      f'Action: {action}\n'
      f'Action Input: {json.dumps(action_input, default=str)}\n'
      f'Observation: {observation}\n'
    )
    combined = scratchpad + entry
    cap = max(4000, self._config.max_result_chars * 4)
    if len(combined) > cap:
      combined = '... [earlier history truncated] ...\n' + combined[-cap:]
    return combined
