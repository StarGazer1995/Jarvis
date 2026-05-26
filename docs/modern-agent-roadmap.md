# Jarvis Modern Agent Roadmap

## Why This Roadmap Exists

Jarvis has now completed the highest-priority runtime foundation work for modern agent systems:

- session and user identity are threaded through tool execution and security validation
- approval gating is no longer a dead-end prompt and supports `list -> approve/reject`
- approval can resume real tool execution and continue LangGraph reasoning
- pending approval runtime state survives process restart through durable local persistence

This changes the project baseline. The next roadmap should no longer optimize for "basic agent completeness"; it should optimize for **shared governance, evaluation, memory, planning, and system-level scale**.

## Current Baseline

### Completed Runtime Foundation

- `in-memory + durable approval resume` is implemented
- `diff coverage = 100%` is enforced for all newly modified lines
- Deep Research now uses a structured answer protocol with shared auditing and evidence traceability
- ARK graph execution has bounded iterations and a real approval recovery path

### Remaining Architectural Gap

The biggest remaining system-level gap is no longer approval handling. It is now **governance drift** between:

- general ARK runtime
- Deep Research specialized runtime
- observability and evaluation
- memory and planning semantics

That gap defines the next phases.

## Roadmap Principles

### Principle 1: Shared Contracts Before More Features

Do not add large new behaviors until ARK and Deep Research share the same execution envelopes, event schema, and governance semantics.

### Principle 2: Eval Before Optimization

Do not optimize prompt logic, planning logic, or UI based on intuition alone. Add replayable task sets and measurable pass/fail signals first.

### Principle 3: Durable Simplicity

Prefer small, explicit runtime stores and contracts over abstract orchestration layers that hide protocol details.

### Principle 4: Memory Must Be Governed

Long-term memory is only useful if writes, reads, retention, and auditability are controlled by policy and measurable outcomes.

## Phase Overview

### Phase 1: Runtime Foundation

Status: `completed`

Scope:

- session identity propagation
- approval lifecycle
- approval resume into graph reasoning
- durable pending-approval checkpoint

Completion signal:

- restart-safe `list/approve/reject/continue`
- approval-resumed answers are written back into conversation context
- current diff coverage remains at `100%`

### Phase 2: Shared Governance Contract

Status: `next`

Goal:

Unify the runtime contracts used by ARK and Deep Research so that security, observability, audit, and approval semantics stop drifting.

Why now:

- Deep Research already has stricter evidence governance than ARK
- ARK now has a stronger runtime lifecycle than Deep Research
- without a shared contract, the two paths will keep diverging

Primary deliverables:

- shared tool execution envelope
- shared event and trace schema
- shared approval and denial result schema
- shared observation record contract
- shared audit metadata fields across ARK and Deep Research

Suggested work items:

1. Define `tool_execution_event` schema used by both runtimes.
2. Define shared `observation_data` and source-trace schema.
3. Define shared terminal status model: `success`, `denied`, `requires_approval`, `retry`, `error`.
4. Refactor Deep Research and ARK adapters to emit the same event payloads.
5. Add contract tests to prevent schema drift.

Done means:

- both ARK and Deep Research emit the same core runtime event fields
- both runtimes can be replayed by the same downstream analysis code
- no duplicate auditor-like governance logic exists in separate places

### Phase 3: Evaluation-Driven Observability

Status: `planned`

Goal:

Move from "metrics exist" to "agent quality is measurable and regressions are catchable".

Why now:

- the runtime is stable enough to observe
- the next improvements will otherwise become subjective prompt tuning

Primary deliverables:

- replayable agent task suite
- golden tasks for approval, recovery, tool failure, Deep Research evidence validation
- step-level latency metrics
- retry and approval-wait metrics
- run outcome taxonomy dashboards

Suggested work items:

1. Build a small evaluation corpus under deterministic mocks.
2. Add replay runner for ARK and Deep Research.
3. Record per-step latency, tool outcome class, retry count, approval wait time.
4. Add CI gate for task pass rate on critical scenarios.
5. Document expected operational baselines.

Done means:

- a regression in approval recovery, tool execution, or evidence validation is caught automatically
- performance and quality signals can be compared across runs

### Phase 4: Long-Term Memory and Context Governance

Status: `planned`

Goal:

Introduce long-term memory without turning context into an ungoverned dump.

Why now:

- short-term context and compression already exist
- durable runtime state is now available
- memory becomes more valuable once replay and observability exist

Primary deliverables:

- vector or structured long-term memory store
- explicit write policy
- retrieval policy and ranking rules
- memory audit metadata
- memory quality evaluation tasks

Suggested work items:

1. Define memory classes: user preference, durable task state, knowledge artifact, research artifact.
2. Add policy-driven write filters instead of saving everything.
3. Add retrieval budget and ranking diagnostics.
4. Add memory-hit replay tasks to measure usefulness.
5. Route history compression fully through shared LLM and memory policy surfaces.

Done means:

- memory writes are intentional and inspectable
- retrieval improves outcomes on defined tasks
- memory does not silently distort protocol fidelity

### Phase 5: Stronger Planning and Execution Model

Status: `planned`

Goal:

Upgrade Jarvis from "tool-using agent" to "task-executing agent" with clearer plan objects, dependencies, and local replanning.

Why now:

- the runtime can now survive approval pauses
- evaluation and observability will provide the feedback loop needed for planning upgrades

Primary deliverables:

- explicit plan object model
- dependency-aware execution shared across more agent paths
- local replanning after partial failures
- task state inspection APIs

Suggested work items:

1. Standardize plan representation across ARK and Deep Research.
2. Support partial failure handling and selective re-execution.
3. Add plan validation before execution.
4. Add plan-vs-outcome observability.
5. Add tests for dependency cycles, partial completion, and replanning.

Done means:

- plans are inspectable and replayable
- partial failures do not force a full restart
- execution state is visible and bounded

### Phase 6: Multi-Agent and UI

Status: `later`

Goal:

Expand the system outward only after governance, evaluation, and planning foundations are stable.

Why later:

- multi-agent systems amplify contract drift and observability gaps
- UI work is most valuable after runtime semantics stabilize

Primary deliverables:

- multi-agent orchestration built on shared runtime contracts
- richer Chainlit or web UI integration
- reasoning and execution timeline visualization
- approval management UI
- evaluation dashboard UI

Suggested work items:

1. Build multi-agent spawning only on top of the shared event contract.
2. Surface approval queues and resume actions in UI.
3. Visualize reasoning path, tool path, and evidence path separately.
4. Add operator views for replay and audit.

Done means:

- multi-agent behavior is explainable and observable
- UI exposes runtime truth rather than inferred summaries

## Suggested Execution Order

1. Shared governance contract
2. Evaluation-driven observability
3. Long-term memory and context governance
4. Stronger planning and execution model
5. Multi-agent and UI

## Near-Term Milestones

### Milestone A: Shared Governance MVP

Target:

- a single runtime event schema used by ARK and Deep Research
- contract tests that fail on schema drift

### Milestone B: Critical Eval Pack

Target:

- approval recovery regression tests
- Deep Research evidence regression tests
- step latency and outcome metrics reported per run

### Milestone C: Memory Governance MVP

Target:

- one durable memory store
- one controlled write policy
- one replayable memory usefulness benchmark

## Risks To Watch

- building multi-agent features before governance and replay are in place
- introducing long-term memory without write controls
- duplicating audit logic across ARK and Deep Research
- adding dashboards before the underlying event schema is stable

## Definition Of Healthy Progress

Jarvis is on the right roadmap if each new phase:

- reduces runtime ambiguity instead of adding it
- improves replayability and auditability
- ships with concrete tests and metrics
- preserves protocol fidelity across agent paths
- keeps newly changed lines at `100% diff coverage`
