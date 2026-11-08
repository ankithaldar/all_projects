# Tabular ML Platform — Program Plan (P0–P11)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. This program plan is a **skeleton**: it fixes phase order, dependencies, and entry/exit gates. Contract details live in `.opencode/context/decisions.md` and are owned by `solution-architect`.

**Goal:** Complete Tabular ML Platform program, gate APPROVED by `anti-team-arch` Cycle 2.

**MVP v1 cut (frozen interpretation, see `D-MVP-READINESS-1`):** P0 + P1 contracts + P2 platform ***production-readiness***. Readiness ≠ production promotion — gates control exit; promotion only via P11.

**Architecture:** P0 freezes contracts → P1 implements them → P2 delivers platform readiness → P3–P7 build ML capability → P8 spikes de-risk multimodality → P9 executes MLOps → P10 serves → P11 releases. No IaC/code before P0 contract freeze.

**Out of scope for this skeleton:** contract field details (see ADRs), infra code, model code, serving code.

---

## Phase table

| Phase | Name / scope | Depends on | Entry gate | Exit gate (acceptance) |
|-------|--------------|------------|------------|------------------------|
| **P0** | Foundations + Contracts Freeze | none | Program APPROVED Cycle 2; `.opencode/context/decisions.md` skeleton created (this task) | All BLOCKING ADRs `frozen`; P1 entry check passes; no src/ edits made |
| **P1** | Contracts Implementation | P0 | P0 exit gate; `D-BEST-MODEL-V2-1` frozen | Contracts implemented to frozen ADRs; contract tests pass; anti-team sign-off |
| **P2** | Platform / IaC | P1 | P1 exit; `D-SECRETS-PII-GDPR-1` frozen | Platform production-**readiness** (not promotion): deploy + observability + secrets/retention checks pass |
| **P3** | Time-Split | P1 | P1 exit; `D-TEMPORAL-SPLIT-1` frozen | Temporal split API frozen & enforced; leakage checks pass |
| **P4** | Feature Eng v2 | P3 | P3 exit; `D-FITTED-REG-V2-1` frozen | Feature store v2 live; dual-read verified |
| **P5** | Supervised | P4 | P4 exit | Supervised training baseline passes acceptance metrics (metrics TBD by `solution-architect`) |
| **P6** | Fairness (early) | P5 | P5 exit | Early fairness eval passes agreed thresholds (thresholds TBD) |
| **P7** | Specialized split ×6 | P6 | P6 exit | 6 specialized splits implemented & evaluated; no leakage vs P3 contract |
| **P8** | Multimodal spikes ×3 | P7 | P7 exit | 3 spike charters (LLM / geo / graph) each closed with GO / KILL verdict per kill criteria below |
| **P9** | MLOps (executed) | P8 | P8 exit | End-to-end executed pipeline in staging; audit trail complete |
| **P10** | Serving | P9 | P9 exit | Serving behind readiness gates only; latency / error-budget SLOs met in staging |
| **P11** | Release gate | P10 | P10 exit | Release verdict APPROVED; sole path to production promotion |

**Dependencies (chain):** P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10 → P11. Any gate failure blocks the next phase; max 3 retries per phase (see `.opencode/status.md`), then BLOCKED escalation to Lead.

**MVP v1 = P0 + P1 + P2 exit.** Exiting P2 proves readiness; production traffic/promotion remains prohibited until P11 exit.

---

## P8 spike charters (skeleton ×3)

Each charter below is a fill-in template. Killing a spike is a successful outcome if kill criteria fire.

| Field | P8-S1 LLM | P8-S2 Geo | P8-S3 Graph |
|-------|-----------|-----------|-------------|
| **Question** | TBD — can an LLM text signal beat tabular baseline on the frozen label set without leakage? | TBD — does geo context add signal beyond tabular features under the P3 temporal contract? | TBD — do graph/relation features beat baseline under the P3 temporal contract? |
| **Dataset** | TBD (owner: `solution-architect`) — sources, split by P3 API, PII tag per `D-SECRETS-PII-GDPR-1` | TBD (same) | TBD (same) |
| **Baseline** | Frozen P5 supervised baseline (metric + value TBD at charter fill-in) | Same frozen baseline | Same frozen baseline |
| **Latency** | TBD budget (P50/P95 at serving shape); measured, not asserted | TBD budget; measured | TBD budget; measured |
| **3.14 proof** | Numeric-stability check labelled "3.14 proof": deterministic re-run reproduces score to agreed ?? precision (threshold TBD; no invented value) | Same | Same |
| **Kill criteria** | KILL if: no lift over baseline at charter threshold, OR leakage found, OR P95 latency > budget, OR PII/GDPR blocker | Same | Same |
| **Verdict** | GO / KILL (TBD) | GO / KILL (TBD) | GO / KILL (TBD) |

---

## Risks / assumptions

- `Assumption:` Contract details (field semantics, thresholds, budgets) are deliberately TBD — owned by `solution-architect`; this skeleton invents none.
- `Assumption:` Phase acceptance metrics for P5–P10 (accuracy, fairness, latency numbers) are set at each phase's charter fill-in, not in this skeleton.
- Open question for Lead: P8 dataset sources + P2 platform provider (GCP/AWS/Azure) — flagged, not decided here.

*Skeleton created by `product-planner` under P0-1. Planning artifacts only — no product code.*
