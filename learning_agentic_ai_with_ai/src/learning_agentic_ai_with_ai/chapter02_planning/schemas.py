#!/usr/bin/env python
# -- coding: utf-8 --

'''Pydantic schemas for Chapter 2: planning, reasoning, structured prompting.

Everything that crosses a boundary between components - the router decision,
the chain-of-thought result, the ReAct trace, the task plan DAG, tool
specs/results, validation reports, and the final agent input/output - is
modeled here. Validation at the boundary keeps the reasoning loops simple:
each pattern receives typed inputs and returns typed outputs, and the
orchestrator composes patterns without guessing at their shapes.

Design notes:
  - `ReasoningRoute` is the central enum of the chapter: `direct` (answer
    with no extra reasoning), `cot` (single model call with step-labeled
    reasoning), `react` (interleaved reason/act/observe loop with tools),
    and `plan` (decompose into a dependency-aware DAG, then execute).
  - `Plan` structural validation (unique ids, known dependencies, no
    self-references) lives here; cycle detection lives in the decomposition
    module next to the networkx graph code, so this module stays graph-free.
  - All usage fields are plain dicts so outputs serialize cleanly to JSON
    and SQLite without importing the tracing layer.
'''


from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
  BaseModel,
  ConfigDict,
  Field,
  field_validator,
  model_validator,
)

ReasoningRoute = Literal['direct', 'cot', 'react', 'plan']
RunStatus = Literal[
  'completed', 'max_iterations', 'error', 'needs_review', 'blocked',
]
PlanNodeStatus = Literal['pending', 'completed', 'failed', 'skipped']
Severity = Literal['info', 'warning', 'error']
CritiqueVerdict = Literal['pass', 'fail']

_NODE_ID_RE = re.compile(r'^[a-z][a-z0-9_]{0,31}$')


def _string_list(value: Any) -> List[str]:
  '''Coerce a value into a list of strings.

  Args:
    value: Candidate list or scalar.

  Returns:
    List of non-empty strings.
  '''
  if isinstance(value, str):
    return [value] if value else []
  if isinstance(value, list):
    return [str(item) for item in value if str(item).strip()]
  return []


# ---------------------------------------------------------------------------
# Tool layer (provider-agnostic: in-process or MCP both produce these)
# ---------------------------------------------------------------------------

class ToolSpec(BaseModel):
  '''Provider-agnostic description of one callable tool.'''

  model_config = ConfigDict(extra='ignore')

  name: str
  description: str = ''
  parameters: Dict[str, Any] = Field(default_factory=dict)
  read_only: bool = True
  server: str = ''


class ToolResult(BaseModel):
  '''Normalized result of one tool execution.

  `text` is the observation handed back to the model. It is always
  sanitized, capped, and marked as untrusted data by the provider layer.
  '''

  model_config = ConfigDict(extra='ignore')

  tool: str
  args: Dict[str, Any] = Field(default_factory=dict)
  ok: bool = True
  text: str = ''
  error: Optional[str] = None
  latency_ms: float = 0.0
  truncated: bool = False


class ToolExecutionAudit(BaseModel):
  '''Audit record of one tool call made during a reasoning run.

  Attributes:
    tool: Qualified tool name (`server.tool` or provider alias).
    args: Arguments proposed by the model.
    ok: Whether the tool executed successfully.
    approved: Whether the safety gate approved the call.
    blocked: True when the call was refused before execution.
    error: Failure or block reason.
    latency_ms: Wall-clock duration.
    result_preview: Short preview of the sanitized observation.
  '''

  model_config = ConfigDict(extra='ignore')

  tool: str
  args: Dict[str, Any] = Field(default_factory=dict)
  ok: bool = True
  approved: bool = True
  blocked: bool = False
  error: Optional[str] = None
  latency_ms: float = 0.0
  result_preview: str = ''


# ---------------------------------------------------------------------------
# Routing policy
# ---------------------------------------------------------------------------

class PolicyDecision(BaseModel):
  '''Transparent, explainable routing decision for one task.

  Attributes:
    route: Chosen reasoning route.
    rationale: Human-readable reason for the route.
    scores: Feature scores that produced the decision (observability).
    confidence: 0..1 confidence derived from the winning score margin.
    forced: True when a caller hint overrode the rule-based scoring.
  '''

  model_config = ConfigDict(extra='ignore')

  route: ReasoningRoute
  rationale: str = ''
  scores: Dict[str, float] = Field(default_factory=dict)
  confidence: float = Field(default=0.5, ge=0.0, le=1.0)
  forced: bool = False


# ---------------------------------------------------------------------------
# Chain-of-thought
# ---------------------------------------------------------------------------

class ChainOfThoughtSample(BaseModel):
  '''One sampled reasoning trace (self-consistency building block).'''

  model_config = ConfigDict(extra='ignore')

  steps: List[str] = Field(default_factory=list)
  answer: str = ''
  parse_ok: bool = True
  error: Optional[str] = None


class ChainOfThoughtResult(BaseModel):
  '''Outcome of chain-of-thought reasoning (possibly multi-sample).

  Attributes:
    task: Original question.
    steps: Steps of the selected (winning) sample.
    answer: Selected final answer.
    samples: All sampled traces.
    agreement: Fraction of samples agreeing with the selected answer.
    status: Run status (`completed` or `needs_review` on weak agreement).
    usage: Aggregated token usage.
    errors: Non-fatal issues collected along the way.
  '''

  model_config = ConfigDict(extra='ignore')

  task: str
  steps: List[str] = Field(default_factory=list)
  answer: str = ''
  samples: List[ChainOfThoughtSample] = Field(default_factory=list)
  agreement: float = 1.0
  status: RunStatus = 'completed'
  usage: Dict[str, Any] = Field(default_factory=dict)
  errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ReAct
# ---------------------------------------------------------------------------

class ReActStep(BaseModel):
  '''One Thought -> Action -> Observation cycle (or the final answer).'''

  model_config = ConfigDict(extra='ignore')

  index: int
  thought: str = ''
  action: str = ''
  action_input: Dict[str, Any] = Field(default_factory=dict)
  observation: str = ''
  tool_ok: Optional[bool] = None
  error: Optional[str] = None
  latency_ms: float = 0.0


class ReActTrace(BaseModel):
  '''Full transcript of a ReAct run.

  Attributes:
    task: Task given to the agent.
    steps: Ordered Thought/Action/Observation cycles.
    final_answer: Answer emitted with `Final Answer:` (or synthesized).
    status: Run status.
    stop_reason: Why the loop ended, e.g. `final_answer`, `max_steps`,
      `loop_detected`, `gateway_unavailable`.
  '''

  model_config = ConfigDict(extra='ignore')

  task: str
  steps: List[ReActStep] = Field(default_factory=list)
  final_answer: str = ''
  status: RunStatus = 'completed'
  stop_reason: str = ''


# ---------------------------------------------------------------------------
# Task decomposition and dependency-aware planning
# ---------------------------------------------------------------------------

class PlanNode(BaseModel):
  '''One executable node of a task plan.

  Attributes:
    id: Stable lowercase identifier used by `depends_on`.
    objective: What this node must accomplish (becomes the node task).
    depends_on: Node ids that must complete first.
    suggested_tools: Tool names the planner believes are relevant.
    success_criteria: Machine/human checkable completion statement.
    route: Optional per-node reasoning route; None lets the executor choose.
  '''

  model_config = ConfigDict(extra='ignore')

  id: str
  objective: str = Field(min_length=1, max_length=600)
  depends_on: List[str] = Field(default_factory=list)
  suggested_tools: List[str] = Field(default_factory=list)
  success_criteria: str = ''
  route: Optional[ReasoningRoute] = None

  @field_validator('id')
  @classmethod
  def _validate_id(cls, value: str) -> str:
    '''Normalize and validate node ids.

    Args:
      value: Raw node id.

    Returns:
      Normalized id.

    Raises:
      ValueError: When the id does not match the required pattern.
    '''
    normalized = re.sub(r'[^a-z0-9_]+', '_', value.strip().lower()).strip('_')
    if not _NODE_ID_RE.match(normalized):
      raise ValueError(
        f'node id must match {_NODE_ID_RE.pattern!r}: {value!r}'
      )
    return normalized


class Plan(BaseModel):
  '''A dependency-aware plan: a DAG of PlanNode objectives.

  Structural validation only - cycle detection and topological ordering are
  in `chapter02_planning.patterns.decomposition` (graph layer).
  '''

  model_config = ConfigDict(extra='ignore')

  goal: str = ''
  nodes: List[PlanNode] = Field(default_factory=list)
  assumptions: List[str] = Field(default_factory=list)
  notes: str = ''

  @model_validator(mode='after')
  def _require_nodes(self) -> 'Plan':
    '''Require at least one executable node.

    Returns:
      The validated plan.

    Raises:
      ValueError: When the plan has no nodes.
    '''
    if not self.nodes:
      raise ValueError('plan must contain at least one node')
    return self

  @classmethod
  def from_llm_payload(cls, payload: Any) -> Dict[str, Any]:
    '''Normalize common model-emitted plan shapes into schema fields.

    Real models wrap the list differently (`nodes`, `plan`, `steps`,
    `tasks`) and rename node fields (`description`, `dependencies`). This
    classmethod maps those synonyms before pydantic validation, which keeps
    the planner working across model families without prompt surgery.

    Args:
      payload: Decoded JSON value from the model.

    Returns:
      Dict ready for `Plan.model_validate`.

    Raises:
      ValueError: When no usable node list is present.
    '''
    if isinstance(payload, list):
      payload = {'nodes': payload}
    if not isinstance(payload, dict):
      raise ValueError('plan payload must be a JSON object or list')

    nodes = payload.get('nodes')
    if nodes is None:
      for key in ('plan', 'steps', 'tasks', 'schedule'):
        candidate = payload.get(key)
        if isinstance(candidate, list):
          nodes = candidate
          break
    if not isinstance(nodes, list) or not nodes:
      raise ValueError('plan payload must contain a non-empty node list')

    normalized_nodes: List[Dict[str, Any]] = []
    rename: Dict[str, str] = {}

    prepared: List[tuple] = []
    for node in nodes:
      if not isinstance(node, dict):
        continue
      raw_id = str(
        node.get('id')
        or node.get('node_id')
        or node.get('name')
        or node.get('step')
        or ''
      ).strip()
      normalized_id = (
        f'step_{raw_id}' if raw_id.isdigit() else raw_id
      )
      if raw_id and normalized_id:
        rename[raw_id] = normalized_id
      prepared.append((node, normalized_id))

    for node, normalized_id in prepared:
      dependencies = _string_list(
        node.get('depends_on')
        or node.get('dependencies')
        or node.get('depends')
        or node.get('requires')
        or []
      )
      normalized: Dict[str, Any] = {
        'id': normalized_id,
        'objective': str(
          node.get('objective')
          or node.get('task')
          or node.get('description')
          or node.get('action')
          or node.get('title')
          or ''
        ).strip(),
        'depends_on': [rename.get(dep, dep) for dep in dependencies],
        'suggested_tools': _string_list(
          node.get('suggested_tools')
          or node.get('tools')
          or node.get('tool')
          or []
        ),
        'success_criteria': str(
          node.get('success_criteria')
          or node.get('done_when')
          or node.get('criteria')
          or ''
        ).strip(),
      }
      if node.get('route') in ('direct', 'cot', 'react', 'plan'):
        normalized['route'] = node['route']
      if normalized['id'] and normalized['objective']:
        normalized_nodes.append(normalized)

    if not normalized_nodes:
      raise ValueError('plan contained no usable nodes')

    return {
      'goal': str(payload.get('goal', '')).strip(),
      'nodes': normalized_nodes,
      'assumptions': _string_list(payload.get('assumptions') or []),
      'notes': str(payload.get('notes', '')).strip(),
    }

  @field_validator('nodes')
  @classmethod
  def _validate_graph_shape(cls, nodes: List[PlanNode]) -> List[PlanNode]:
    '''Check id uniqueness, self-dependencies, and known dependencies.

    Args:
      nodes: Proposed plan nodes.

    Returns:
      The validated node list.

    Raises:
      ValueError: On duplicate ids, self-dependency, or unknown dependency.
    '''
    ids = [node.id for node in nodes]
    duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
    if duplicates:
      raise ValueError(f'duplicate node ids: {duplicates}')

    known = set(ids)
    for node in nodes:
      if node.id in node.depends_on:
        raise ValueError(f'node {node.id!r} depends on itself')
      unknown = [dep for dep in node.depends_on if dep not in known]
      if unknown:
        raise ValueError(
          f'node {node.id!r} depends on unknown nodes: {unknown}'
        )

    return nodes

  def node_map(self) -> Dict[str, PlanNode]:
    '''Map node id to node.

    Returns:
      Dictionary keyed by node id.
    '''
    return {node.id: node for node in self.nodes}


class PlanRevision(BaseModel):
  '''A planner's adjustment after a failed or invalidated node.'''

  model_config = ConfigDict(extra='ignore')

  reason: str = ''
  dropped: List[str] = Field(default_factory=list)
  added: List[PlanNode] = Field(default_factory=list)
  notes: str = ''


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class NodeResult(BaseModel):
  '''Result of executing one plan node.'''

  model_config = ConfigDict(extra='ignore')

  node_id: str
  status: PlanNodeStatus = 'pending'
  output: str = ''
  tool_calls: List[ToolExecutionAudit] = Field(default_factory=list)
  errors: List[str] = Field(default_factory=list)
  usage: Dict[str, Any] = Field(default_factory=dict)
  duration_ms: float = 0.0
  skipped_reason: str = ''


class ExecutionReport(BaseModel):
  '''Result of executing a full plan.

  Attributes:
    goal: Plan goal.
    waves: Topological generations (nodes that can run together).
    results: One NodeResult per node, in execution order.
    status: Aggregate status.
    errors: Aggregate errors.
    usage: Aggregated token usage.
  '''

  model_config = ConfigDict(extra='ignore')

  goal: str = ''
  waves: List[List[str]] = Field(default_factory=list)
  results: List[NodeResult] = Field(default_factory=list)
  status: RunStatus = 'completed'
  errors: List[str] = Field(default_factory=list)
  usage: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Self-validation
# ---------------------------------------------------------------------------

class ValidationIssue(BaseModel):
  '''One finding from a deterministic check or the LLM critic.'''

  model_config = ConfigDict(extra='ignore')

  check: str
  severity: Severity = 'warning'
  detail: str = ''
  suggestion: str = ''


class Critique(BaseModel):
  '''Structured LLM critic output.'''

  model_config = ConfigDict(extra='ignore')

  verdict: CritiqueVerdict = 'pass'
  issues: List[ValidationIssue] = Field(default_factory=list)
  revised_answer: str = ''


class ValidationReport(BaseModel):
  '''Outcome of validating (and possibly revising) a candidate answer.

  Attributes:
    passed: True when no `error` severity issues remain after revision.
    issues: All issues found across deterministic checks and the critic.
    checks_run: Names of the checks that executed.
    rounds: Revision rounds used.
    revised: True when the answer was replaced by a revision.
    revision_rejected: True when a revision was attempted but discarded
      because it failed the post-revision checks.
    final_answer: The answer after validation/repair.
    critic_used: True when the LLM critic ran.
    critic_unavailable: True when the critic was needed but failed.
    usage: Token usage of critic/revision calls.
    errors: Transport-level errors (critic unavailable, parse issues).
  '''

  model_config = ConfigDict(extra='ignore')

  passed: bool = True
  issues: List[ValidationIssue] = Field(default_factory=list)
  checks_run: List[str] = Field(default_factory=list)
  rounds: int = 0
  revised: bool = False
  revision_rejected: bool = False
  final_answer: str = ''
  critic_used: bool = False
  critic_unavailable: bool = False
  usage: Dict[str, Any] = Field(default_factory=dict)
  errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent task input/output
# ---------------------------------------------------------------------------

class PlanningTaskInput(BaseModel):
  '''Validated input for one Chapter 2 agent run.

  Attributes:
    task: Natural-language task.
    session_id: Session identifier for persistence.
    max_steps: Hard cap on ReAct steps / plan nodes.
    allow_writes: Whether state-mutating tools may execute this run.
    cot_samples: Self-consistency samples for the `cot` route.
    prefer_route: Optional caller override of the routing policy.
    run_validation: Whether the self-validation stage runs.
  '''

  model_config = ConfigDict(extra='ignore')

  task: str = Field(min_length=1, max_length=4000)
  session_id: str = Field(min_length=1, max_length=128)
  max_steps: int = Field(default=8, ge=1, le=32)
  allow_writes: bool = False
  cot_samples: int = Field(default=1, ge=1, le=5)
  prefer_route: Optional[ReasoningRoute] = None
  run_validation: bool = True


class PlanningTaskOutput(BaseModel):
  '''Validated output of one Chapter 2 agent run.

  Attributes:
    session_id: Session identifier.
    answer: Final answer after validation/repair.
    status: Run status.
    route: Reasoning route actually used.
    policy: The routing decision and its rationale.
    reasoning_steps: CoT steps for the selected sample (may be empty).
    react: ReAct transcript when the ReAct route ran.
    plan: The task plan when the plan route ran.
    execution: Plan execution report when the plan route ran.
    validation: Validation report when validation ran.
    tool_calls: Every tool call audit across all stages.
    usage: Aggregated token usage.
    errors: Non-fatal and fatal error strings.
    duration_ms: End-to-end wall duration.
  '''

  model_config = ConfigDict(extra='ignore')

  session_id: str
  answer: str = ''
  status: RunStatus = 'completed'
  route: ReasoningRoute = 'direct'
  policy: Optional[PolicyDecision] = None
  reasoning_steps: List[str] = Field(default_factory=list)
  react: Optional[ReActTrace] = None
  plan: Optional[Plan] = None
  execution: Optional[ExecutionReport] = None
  validation: Optional[ValidationReport] = None
  tool_calls: List[ToolExecutionAudit] = Field(default_factory=list)
  usage: Dict[str, Any] = Field(default_factory=dict)
  errors: List[str] = Field(default_factory=list)
  duration_ms: float = 0.0
