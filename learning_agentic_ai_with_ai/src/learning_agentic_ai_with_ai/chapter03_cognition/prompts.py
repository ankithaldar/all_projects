#!/usr/bin/env python
# -- coding: utf-8 --

'''Prompt templates and output contracts for the cognitive pipeline.

Prompts are interfaces, not strings. Every prompt here has named `$slots`
rendered with `string.Template` (so JSON examples need no brace escaping),
a declared output format, and - for machine-consumed outputs - a JSON
contract generated from a pydantic model so the model sees exact fields.

The chapter uses four prompt families:

  - DAG planning: structured JSON plan validated by pydantic/networkx,
  - code planning: an agent-written `solve(context)` Python function run in
    a sandbox,
  - reason/synthesis: bounded text generation for individual steps,
  - fallback: memory-and-reason-only degraded answers with caveats.
'''


from __future__ import annotations

import json
import re
from string import Template
from typing import Any, Dict, Iterable, List, Set, Type

from pydantic import BaseModel

from chapter03_cognition.schemas import ToolSpec

_SLOT_RE = re.compile(r'\$(?:\{(\w+)\}|(\w+))')


class PromptTemplate:
  '''A named `$slot` template with strict rendering.'''

  def __init__(self, name: str, template: str) -> None:
    '''Initialize the template.

    Args:
      name: Template name for logs/errors.
      template: `$slot`-style template text.
    '''
    self.name = name
    self.template = template
    self._template = Template(template)

  def slots(self) -> Set[str]:
    '''List placeholder names used by the template.

    Returns:
      Set of slot names.
    '''
    names: Set[str] = set()
    for match in _SLOT_RE.finditer(self.template):
      names.add(match.group(1) or match.group(2))
    return names

  def render(self, **values: Any) -> str:
    '''Render the template with the given slot values.

    Args:
      **values: Slot values; values are stringified.

    Returns:
      Rendered prompt text.

    Raises:
      ValueError: When required slots are missing.
    '''
    missing = sorted(self.slots() - set(values))
    if missing:
      raise ValueError(
        f'template {self.name!r} missing slots: {missing}'
      )
    return self._template.substitute(
      {key: str(value) for key, value in values.items()}
    )


def output_contract(model: Type[BaseModel], purpose: str = '') -> str:
  '''Render JSON output instructions from a pydantic model.

  Args:
    model: Pydantic model the output must validate against.
    purpose: Optional purpose sentence.

  Returns:
    Instruction text to append to the prompt.
  '''
  schema = json.dumps(model.model_json_schema(), indent=2)
  header = purpose or f'Return a JSON object matching {model.__name__}.'
  return (
    f'{header}\n'
    'Rules:\n'
    '- Output exactly one JSON object, no prose, no markdown fences.\n'
    '- Use only the fields in the schema; omit fields you cannot fill.\n'
    'JSON schema:\n'
    f'{schema}'
  )


def render_tool_catalog(tools: Iterable[ToolSpec]) -> str:
  '''Render a textual tool catalog for prompts.

  Args:
    tools: Tool specs available to the agent.

  Returns:
    Multi-line catalog text.
  '''
  lines: List[str] = []
  for tool in tools:
    args = json.dumps(tool.parameters or {}, default=str)
    if len(args) > 400:
      args = args[:397] + '...'
    kind = 'read' if tool.read_only else 'write'
    lines.append(f'- {tool.name} ({kind}): {tool.description}')
    lines.append(f'  arguments schema: {args}')
  return '\n'.join(lines) if lines else '(no tools available)'


def format_memory_notes(notes: Iterable[str]) -> str:
  '''Render memory notes as a bullet list.

  Args:
    notes: Note lines.

  Returns:
    Bullet list text.
  '''
  rendered = [f'- {note}' for note in notes]
  return '\n'.join(rendered) if rendered else '- no relevant memory'


# ---------------------------------------------------------------------------
# DAG planning
# ---------------------------------------------------------------------------

DAG_PLAN_SYSTEM = (
  'You are a planning agent for retail and telecom operations. Decompose '
  'the goal into the smallest set of verifiable steps. Use only the tools '
  'listed; never invent systems. Prefer read-only steps before writes. '
  'Step ids are short lowercase identifiers. depends_on must reference ids '
  'defined in the same plan.\n'
  'Rules:\n'
  '- Arguments must be concrete values (ids, numbers) taken from the goal.\n'
  '- Never use placeholders like "<sku>" or "the worst item" as argument '
  'values; tool arguments are sent as-is.\n'
  '- If a step needs a value produced by an earlier step, set kind="reason" '
  'and describe the dependency in the objective instead.\n'
  '- Use kind="write" only for state-changing tools.'
)

DAG_PLAN_PROMPT = PromptTemplate(
  'dag_plan',
  'Goal:\n$goal\n\n'
  'Perception:\n'
  '- domain: $domain\n'
  '- risk: $risk   data_need: $data_need   multi_step: $multi_step\n'
  '- entities: $entities\n\n'
  'Memory notes:\n$memory\n\n'
  'Available tools:\n$tools\n\n'
  'Produce at most $max_steps steps.\n\n'
  '$contract',
)


# ---------------------------------------------------------------------------
# Agent-written Python plans
# ---------------------------------------------------------------------------

CODE_PLAN_SYSTEM = (
  'You write small, safe Python functions for operations planning. '
  'The function must be pure except for tool calls through the provided '
  '`tools` object. Never import anything. Never touch files, network, or '
  'the filesystem. Return a JSON-serializable dict.'
)

CODE_PLAN_PROMPT = PromptTemplate(
  'code_plan',
  'Goal:\n$goal\n\n'
  'Known facts (from perception and memory):\n$facts\n\n'
  'Available tools (call them only through context["tools"]):\n$tools\n\n'
  'Write exactly one function with this interface:\n'
  '\n'
  'def solve(context):\n'
  '    """context is a dict with keys: task, facts, tools."""\n'
  '    # context["tools"].call("server.tool", {...}) -> dict\n'
  '    #            .call returns {"ok": bool, "data": ..., "error": ...}\n'
  '    #            raises on policy blocks or budget exhaustion\n'
  '    # Use plain Python for arithmetic; do not re-derive data you can\n'
  '    # read from facts.\n'
  '    return {\n'
  '        "answer": "concise final answer with concrete numbers",\n'
  '        "used_tools": ["server.tool", ...],\n'
  '        "notes": ["short note", ...],\n'
  '    }\n'
  '\n'
  'Constraints:\n'
  '- No imports, no classes, no global state, no dunder access.\n'
  '- At most $call_budget tool calls.\n'
  '- Handle missing data by returning an explicit caveat in "answer".\n'
  '- Output only the Python code, no explanation.',
)


# ---------------------------------------------------------------------------
# Step reasoning and final synthesis
# ---------------------------------------------------------------------------

REASON_SYSTEM = (
  'You execute one step of a larger operations plan. Stay strictly within '
  'the objective, use the provided context, and answer concisely with '
  'concrete numbers and identifiers.'
)

REASON_PROMPT = PromptTemplate(
  'reason',
  'Overall goal:\n$goal\n\n'
  'Your step objective:\n$objective\n\n'
  'Expected output:\n$expected\n\n'
  'Context from completed steps:\n$context',
)

SYNTHESIS_SYSTEM = (
  'You are an operations lead. Combine completed plan steps into one '
  'decisive answer for the goal. Cite concrete numbers, mention failed or '
  'skipped steps honestly, and keep it under 200 words.'
)

SYNTHESIS_PROMPT = PromptTemplate(
  'synthesis',
  'Goal:\n$goal\n\n'
  'Completed steps:\n$steps\n\n'
  'Unresolved errors:\n$errors\n\n'
  'Write the final answer now.',
)


# ---------------------------------------------------------------------------
# Fallback (degraded) answers
# ---------------------------------------------------------------------------

FALLBACK_SYSTEM = (
  'You are operating in degraded mode: live tools and code execution are '
  'unavailable. Answer from the task statement and remembered facts only. '
  'State clearly what could not be verified and what a follow-up run '
  'should check. Never invent numbers.'
)

FALLBACK_PROMPT = PromptTemplate(
  'fallback',
  'Task:\n$task\n\n'
  'Remembered facts:\n$facts\n\n'
  'Memory notes:\n$memory\n\n'
  'Write a short answer with an explicit verification caveat.',
)


def compact_json(payload: Dict[str, Any], max_chars: int = 800) -> str:
  '''Serialize a payload compactly for prompts.

  Args:
    payload: JSON-serializable mapping.
    max_chars: Maximum length before truncation.

  Returns:
    JSON string (possibly truncated).
  '''
  text = json.dumps(payload, default=str)
  if len(text) <= max_chars:
    return text
  return text[: max(0, max_chars - 3)] + '...'
