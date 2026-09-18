#!/usr/bin/env python
# -- coding: utf-8 --

'''Pydantic schemas for Chapter 3: the cognitive pipeline.

Chapter 3 organizes an agent as four cooperating layers, each with a typed
input and output. Typing the seams between layers is the architectural
point of the chapter:

    task ─▶ Perception ─▶ Memory ─▶ Decision ─▶ Action ─▶ answer
              Percept   MemoryCtx  Decision   ActionResult

Every structure that crosses a layer boundary lives here, plus the
strategy profiles, plan/step models (including agent-written Python plans),
adaptive-loop records, and the task input/output. Keeping all of it in one
module makes the data flow reviewable at a glance and prevents layers from
smuggling ad-hoc dicts across boundaries.

Design notes:
  - `StrategyName` is the chapter's central enum: conservative,
    exploratory, fallback.
  - `CognitivePlan.source` distinguishes planner outputs: a structured DAG
    (`dag`), an LLM-written Python `solve()` function (`code`), or a
    single-step safe degradation (`fallback`).
  - Plan structural validation (unique ids, known dependencies, no
    self-references) lives here; cycle detection and static risk analysis
    live in `planner.py` next to the networkx/AST code.
  - Usage fields are plain dicts so outputs serialize cleanly to JSON and
    SQLite without importing the tracing layer.
'''


from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

StrategyName = Literal['conservative', 'exploratory', 'fallback']
DomainName = Literal['retail', 'telecom', 'unknown']
RunStatus = Literal[
  'completed', 'partial', 'needs_review', 'error', 'blocked',
]
StepKind = Literal['tool', 'code', 'reason', 'write']
StepStatus = Literal['pending', 'completed', 'failed', 'skipped']
PlanSource = Literal['dag', 'code', 'fallback']
RiskLevel = Literal['low', 'medium', 'high']
ErrorClass = Literal[
  'none', 'transient', 'unknown_tool', 'invalid_arguments', 'blocked',
  'tool_error', 'empty_output', 'timeout', 'code_error', 'gateway_error',
]
AdaptationKind = Literal[
  'retry', 'tool_switch', 'replan', 'strategy_escalation', 'fallback',
]
CodeRunnerName = Literal['restricted', 'subprocess']

_STEP_ID_RE = re.compile(r'^[a-z][a-z0-9_]{0,31}$')
_TOOL_NAME_RE = re.compile(r'^[a-z0-9-]+\.[a-z][a-z0-9_]{0,63}$')


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
  '''Normalized result of one tool execution.'''

  model_config = ConfigDict(extra='ignore')

  tool: str
  args: Dict[str, Any] = Field(default_factory=dict)
  ok: bool = True
  text: str = ''
  error: Optional[str] = None
  latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Layer 1 - Perception
# ---------------------------------------------------------------------------

class PerceptionSignals(BaseModel):
  '''Numeric/boolean features extracted from a task before any LLM call.

  Attributes:
    domain: Best-guess business domain.
    risk: 0..1 likelihood the task mutates state or carries compliance risk.
    ambiguity: 0..1 how much diagnosis/recommendation the task needs.
    multi_step: 0..1 how many objectives the task bundles.
    data_need: 0..1 how much the answer depends on live data.
    complexity: 0..1 weighted summary used for strategy selection.
    write_intent: True when a state-changing verb was detected.
    keywords: Matched signal phrases (for logs/introspection).
  '''

  model_config = ConfigDict(extra='ignore')

  domain: DomainName = 'unknown'
  risk: float = Field(default=0.0, ge=0.0, le=1.0)
  ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)
  multi_step: float = Field(default=0.0, ge=0.0, le=1.0)
  data_need: float = Field(default=0.0, ge=0.0, le=1.0)
  complexity: float = Field(default=0.0, ge=0.0, le=1.0)
  write_intent: bool = False
  keywords: List[str] = Field(default_factory=list)


class Percept(BaseModel):
  '''Output of the Perception layer (typed input to Memory/Decision).

  Attributes:
    task: Sanitized, whitespace-normalized task text.
    session_id: Owning session.
    signals: Extracted numeric signals.
    entities: Recognized identifiers by kind (stores, skus, sites, ...).
    required_capabilities: Capability names the task appears to need, e.g.
      `inventory_read`, `restock_write`, `site_read`, `dispatch_write`.
  '''

  model_config = ConfigDict(extra='ignore')

  task: str
  session_id: str
  signals: PerceptionSignals = Field(default_factory=PerceptionSignals)
  entities: Dict[str, List[str]] = Field(default_factory=dict)
  required_capabilities: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 2 - Memory
# ---------------------------------------------------------------------------

class Episode(BaseModel):
  '''One remembered run: the unit of experiential memory.

  Attributes:
    episode_id: Unique id.
    session_id: Session that produced the run.
    task: Task text (sanitized).
    domain: Detected domain.
    strategy: Strategy used.
    status: Final run status.
    success: Whether the run completed cleanly.
    answer_preview: First ~300 chars of the answer.
    tool_sequence: Tools called, in order.
    error_classes: Distinct error classes observed.
    adaptations: Adaptation kinds applied during the run.
    duration_ms: Wall duration.
    total_tokens: Token spend.
    tags: Free-form retrieval tags (domain, entities, keywords).
    created_at: ISO timestamp.
  '''

  model_config = ConfigDict(extra='ignore')

  episode_id: str
  session_id: str
  task: str
  domain: DomainName = 'unknown'
  strategy: StrategyName = 'conservative'
  status: RunStatus = 'completed'
  success: bool = True
  answer_preview: str = ''
  tool_sequence: List[str] = Field(default_factory=list)
  error_classes: List[str] = Field(default_factory=list)
  adaptations: List[str] = Field(default_factory=list)
  duration_ms: float = 0.0
  total_tokens: int = 0
  tags: List[str] = Field(default_factory=list)
  created_at: str = ''


class StrategyStats(BaseModel):
  '''Aggregate performance of one strategy over remembered episodes.'''

  model_config = ConfigDict(extra='ignore')

  strategy: StrategyName
  runs: int = 0
  successes: int = 0
  failures: int = 0
  avg_duration_ms: float = 0.0
  updated_at: str = ''

  @property
  def success_rate(self) -> float:
    '''Success fraction over recorded runs.

    Returns:
      Success rate in 0..1 (0 when no runs).
    '''
    return (self.successes / self.runs) if self.runs else 0.0


class MemoryContext(BaseModel):
  '''Output of the Memory layer (typed input to Decision).

  Attributes:
    relevant_episodes: Episodes recalled for this task (best first).
    facts: Durable key/value facts relevant to the task.
    strategy_stats: Per-strategy aggregates.
    recall_notes: Short human-readable notes handed to the planner prompt.
    recommended_strategy: Memory's suggestion before the selector runs.
  '''

  model_config = ConfigDict(extra='ignore')

  relevant_episodes: List[Episode] = Field(default_factory=list)
  facts: Dict[str, Any] = Field(default_factory=dict)
  strategy_stats: List[StrategyStats] = Field(default_factory=list)
  recall_notes: List[str] = Field(default_factory=list)
  recommended_strategy: Optional[StrategyName] = None


# ---------------------------------------------------------------------------
# Strategy profiles
# ---------------------------------------------------------------------------

class RetryPolicy(BaseModel):
  '''Bounded adaptive behavior for one strategy profile.

  Attributes:
    max_attempts_per_step: Attempts before a step is considered failed.
    max_replans: Whole-plan replans allowed after failures.
    tool_switch: Whether the executor may use a step's fallback tool.
    escalate_on_failure: Whether repeated failure may change strategy.
    verify_each_step: Whether every step output is checked before dependents.
    allow_writes: Whether state-changing tools may run under this strategy.
    allow_code_execution: Whether agent-written code plans may run.
  '''

  model_config = ConfigDict(extra='ignore')

  max_attempts_per_step: int = Field(default=2, ge=1, le=5)
  max_replans: int = Field(default=0, ge=0, le=3)
  tool_switch: bool = False
  escalate_on_failure: bool = False
  verify_each_step: bool = True
  allow_writes: bool = False
  allow_code_execution: bool = True


class StrategyProfile(BaseModel):
  '''A named operating policy: how boldly the agent plans and adapts.

  Attributes:
    name: Profile name.
    description: Human explanation used in docs/logs.
    retry: Bounded retry/adaptation policy.
    prefer_code_plans: Whether computation-heavy tasks should use
      agent-written Python instead of a JSON DAG.
    temperature: Planner/codegen sampling temperature.
    max_steps: Step cap for plans produced under this profile.
  '''

  model_config = ConfigDict(extra='ignore')

  name: StrategyName
  description: str = ''
  retry: RetryPolicy = Field(default_factory=RetryPolicy)
  prefer_code_plans: bool = False
  temperature: float = Field(default=0.0, ge=0.0, le=1.0)
  max_steps: int = Field(default=8, ge=1, le=32)


# ---------------------------------------------------------------------------
# Layer 3 - Decision (planning)
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
  '''One executable step of a DAG plan.

  Attributes:
    id: Stable lowercase identifier.
    objective: What the step must accomplish.
    kind: Execution kind: tool, code, reason, or write.
    tool: Qualified tool name for tool/write steps.
    arguments: Static arguments proposed by the planner.
    fallback_tool: Alternative tool for adaptive tool switching.
    fallback_arguments: Arguments used with the fallback tool.
    depends_on: Step ids that must complete first.
    expected_output: Completion criterion (human/model checkable).
    max_attempts: Optional per-step override of the strategy retry cap.
  '''

  model_config = ConfigDict(extra='ignore')

  id: str
  objective: str = Field(min_length=1, max_length=600)
  kind: StepKind = 'tool'
  tool: str = ''
  arguments: Dict[str, Any] = Field(default_factory=dict)
  fallback_tool: str = ''
  fallback_arguments: Dict[str, Any] = Field(default_factory=dict)
  depends_on: List[str] = Field(default_factory=list)
  expected_output: str = ''
  max_attempts: Optional[int] = Field(default=None, ge=1, le=5)

  @field_validator('id')
  @classmethod
  def _validate_id(cls, value: str) -> str:
    '''Normalize and validate step ids.

    Args:
      value: Raw step id.

    Returns:
      Normalized id.

    Raises:
      ValueError: When the id does not match the required pattern.
    '''
    normalized = re.sub(r'[^a-z0-9_]+', '_', value.strip().lower()).strip('_')
    if not _STEP_ID_RE.match(normalized):
      raise ValueError(
        f'step id must match {_STEP_ID_RE.pattern!r}: {value!r}'
      )
    return normalized

  @field_validator('tool', 'fallback_tool')
  @classmethod
  def _validate_tool(cls, value: str) -> str:
    '''Validate qualified tool names (empty allowed).

    Args:
      value: Raw tool name.

    Returns:
      The tool name unchanged.

    Raises:
      ValueError: When a non-empty name is not `server.tool`.
    '''
    if value and not _TOOL_NAME_RE.match(value):
      raise ValueError(f'tool name must be server.tool: {value!r}')
    return value


class CognitivePlan(BaseModel):
  '''A plan the Decision layer produced, in any of its three forms.

  Attributes:
    goal: Restated task goal.
    source: `dag` (JSON steps), `code` (agent-written solve()), or
      `fallback` (single safe step).
    strategy: Strategy that produced the plan.
    steps: DAG steps (used by `dag`/`fallback` sources).
    code: Python source for `code` plans.
    entrypoint: Function name to call in `code` plans (default `solve`).
    rationale: Why this plan shape was chosen.
    assumptions: Planner-stated assumptions.
  '''

  model_config = ConfigDict(extra='ignore')

  goal: str = ''
  source: PlanSource = 'dag'
  strategy: StrategyName = 'conservative'
  steps: List[PlanStep] = Field(default_factory=list)
  code: str = ''
  entrypoint: str = 'solve'
  rationale: str = ''
  assumptions: List[str] = Field(default_factory=list)

  @field_validator('steps')
  @classmethod
  def _validate_graph_shape(cls, steps: List[PlanStep]) -> List[PlanStep]:
    '''Check id uniqueness, self-dependencies, and known dependencies.

    Args:
      steps: Proposed plan steps.

    Returns:
      The validated step list.

    Raises:
      ValueError: On duplicate ids, self-dependency, or unknown dependency.
    '''
    ids = [step.id for step in steps]
    duplicates = sorted({step_id for step_id in ids if ids.count(step_id) > 1})
    if duplicates:
      raise ValueError(f'duplicate step ids: {duplicates}')

    known = set(ids)
    for step in steps:
      if step.id in step.depends_on:
        raise ValueError(f'step {step.id!r} depends on itself')
      unknown = [dep for dep in step.depends_on if dep not in known]
      if unknown:
        raise ValueError(
          f'step {step.id!r} depends on unknown steps: {unknown}'
        )
    return steps

  def step_map(self) -> Dict[str, PlanStep]:
    '''Map step id to step.

    Returns:
      Dictionary keyed by step id.
    '''
    return {step.id: step for step in self.steps}


class PlanInsight(BaseModel):
  '''Static introspection of a plan, produced before execution.

  Attributes:
    step_count: Number of steps (1 for code plans).
    risk_level: low/medium/high based on writes and failure surfaces.
    has_writes: Whether any step mutates state.
    missing_capabilities: Required capabilities the plan does not cover.
    warnings: Actionable planner-introspection warnings.
    confidence: 0..1 heuristic confidence that the plan can succeed.
  '''

  model_config = ConfigDict(extra='ignore')

  step_count: int = 0
  risk_level: RiskLevel = 'low'
  has_writes: bool = False
  missing_capabilities: List[str] = Field(default_factory=list)
  warnings: List[str] = Field(default_factory=list)
  confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Decision(BaseModel):
  '''Output of the Decision layer (typed input to Action).

  Attributes:
    strategy: Selected strategy profile.
    strategy_reason: Why the selector chose it.
    plan: The plan to execute.
    insight: Static introspection of the plan.
  '''

  model_config = ConfigDict(extra='ignore')

  strategy: StrategyName = 'conservative'
  strategy_reason: str = ''
  plan: CognitivePlan = Field(default_factory=CognitivePlan)
  insight: PlanInsight = Field(default_factory=PlanInsight)


# ---------------------------------------------------------------------------
# Layer 4 - Action
# ---------------------------------------------------------------------------

class ToolExecutionAudit(BaseModel):
  '''Audit record for one toolbox call (gated or not).'''

  model_config = ConfigDict(extra='ignore')

  tool: str
  args: Dict[str, Any] = Field(default_factory=dict)
  ok: bool = True
  approved: bool = True
  blocked: bool = False
  error: Optional[str] = None
  latency_ms: float = 0.0
  result_preview: str = ''


class StepAttempt(BaseModel):
  '''One attempt at executing a plan step.'''

  model_config = ConfigDict(extra='ignore')

  attempt: int
  tool: str = ''
  ok: bool = False
  error_class: ErrorClass = 'none'
  error: Optional[str] = None
  observation_preview: str = ''
  latency_ms: float = 0.0


class StepResult(BaseModel):
  '''Final result of one plan step after adaptive handling.'''

  model_config = ConfigDict(extra='ignore')

  step_id: str
  status: StepStatus = 'pending'
  output: str = ''
  tool_used: str = ''
  attempts: List[StepAttempt] = Field(default_factory=list)
  errors: List[str] = Field(default_factory=list)
  skipped_reason: str = ''
  duration_ms: float = 0.0


class AdaptationRecord(BaseModel):
  '''One adaptive decision taken by the executor.'''

  model_config = ConfigDict(extra='ignore')

  kind: AdaptationKind
  step_id: str = ''
  attempt: int = 0
  reason: str = ''
  detail: str = ''


class CodeRunResult(BaseModel):
  '''Result of executing an agent-written Python plan.'''

  model_config = ConfigDict(extra='ignore')

  ok: bool = False
  runner: CodeRunnerName = 'restricted'
  output: Dict[str, Any] = Field(default_factory=dict)
  stdout: str = ''
  error: Optional[str] = None
  error_class: ErrorClass = 'none'
  timed_out: bool = False
  latency_ms: float = 0.0
  calls_used: int = 0


class ActionResult(BaseModel):
  '''Output of the Action layer (returned to the agent/reflection).

  Attributes:
    goal: Plan goal.
    steps: Per-step results in execution order.
    status: Aggregate status.
    answer: Answer produced by a code plan, when one ran.
    adaptations: Every adaptive decision taken.
    tool_audits: Every toolbox call audit.
    errors: Aggregate errors.
    usage: Token usage (synthesis/reason steps).
    duration_ms: Wall duration.
  '''

  model_config = ConfigDict(extra='ignore')

  goal: str = ''
  steps: List[StepResult] = Field(default_factory=list)
  status: RunStatus = 'completed'
  answer: str = ''
  code_runner: Optional[CodeRunnerName] = None
  adaptations: List[AdaptationRecord] = Field(default_factory=list)
  tool_audits: List[ToolExecutionAudit] = Field(default_factory=list)
  errors: List[str] = Field(default_factory=list)
  usage: Dict[str, Any] = Field(default_factory=dict)
  duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Agent task input/output
# ---------------------------------------------------------------------------

class CognitiveTaskInput(BaseModel):
  '''Validated input for one Chapter 3 agent run.

  Attributes:
    task: Natural-language task.
    session_id: Session identifier for persistence.
    strategy_hint: Optional caller override of strategy selection.
    allow_writes: Whether state-changing tools may execute.
    allow_code: Whether agent-written Python plans may run.
    code_runner: Optional code runner override.
    plan_mode: `auto` (strategy decides), `dag`, or `code`.
    max_steps: Hard cap on plan steps.
  '''

  model_config = ConfigDict(extra='ignore')

  task: str = Field(min_length=1, max_length=4000)
  session_id: str = Field(min_length=1, max_length=128)
  strategy_hint: Optional[StrategyName] = None
  allow_writes: bool = False
  allow_code: bool = True
  code_runner: Optional[CodeRunnerName] = None
  plan_mode: Literal['auto', 'dag', 'code'] = 'auto'
  max_steps: int = Field(default=8, ge=1, le=24)


class CognitiveTaskOutput(BaseModel):
  '''Validated output of one Chapter 3 agent run.

  Attributes:
    session_id: Session identifier.
    answer: Final answer (post-synthesis/validation).
    status: Run status.
    percept: Perception-layer output.
    memory: Memory-layer output used for planning.
    decision: Decision-layer output (strategy, plan, insight).
    action: Action-layer output (steps, adaptations, audits).
    usage: Aggregated token usage.
    errors: Non-fatal and fatal error strings.
    duration_ms: End-to-end wall duration.
  '''

  model_config = ConfigDict(extra='ignore')

  session_id: str
  answer: str = ''
  status: RunStatus = 'completed'
  percept: Optional[Percept] = None
  memory: Optional[MemoryContext] = None
  decision: Optional[Decision] = None
  action: Optional[ActionResult] = None
  usage: Dict[str, Any] = Field(default_factory=dict)
  errors: List[str] = Field(default_factory=list)
  duration_ms: float = 0.0
