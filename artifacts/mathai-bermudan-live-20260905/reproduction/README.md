# OpenHyra

![CI](https://github.com/MrSteeeve/OpenHyra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**English** | [中文](README.zh-CN.md)

An open, partial reproduction of the public architecture of Tencent Hunyuan's
**Hyra** (Hunyuan Research Agent) harness [1], currently demonstrated on the
**Bermudan optimal stopping** task: an autonomous loop in which LLM agents
propose solvers, a sandbox runs them, a trusted evaluator scores them, and
every outcome, whether success or failure, is banked as experience for the next round.

## Tasks

### Bermudan optimal stopping

OpenHyra now exposes two additive Bermudan tracks. The historical
`bermudan_optimal_stopping` task searches a bounded, typed feature-expression
program for a fixed Ridge LSMC algorithm. The new `bermudan_python_search`
task searches complete Python programs. A candidate-owned `algorithm.py`
implements both `fit` and `predict`; it controls representation, objectives,
training/search logic, model state and either continuation values or direct
stopping decisions. Its manifest declares only that output interface and does
not select a registered model family.

In both tracks the evaluator owns the risk-neutral model, simulation,
contracts, discounting, causal exercise, primal/dual audit, budgets, and
statistics. Candidate code runs as separate fit/predict processes and is never
imported into the evaluator. Public search uses independent
fit/pricing paths and paired Common Random Numbers against a frozen baseline;
its score is a conservative lower bound on strike-normalized lower-bound
improvement.

The hidden Top-K ranking/audit is a separate one-shot action. The harness validates and
freezes the distinct Top-K normalized artifacts first, then draws a fresh
private seed and evaluates every frozen artifact on the same hidden suite. The
audit constructs a conditionally centered nested martingale and ranks the
primal--dual confidence gap. Results are written to `final_audit.json`, never
fed back into the Experience Bank or another search round. A completed audit
means the frozen candidates were all evaluated; it is not an automatic claim
that the winner beats a hidden baseline or is scientifically novel. The public
termination record carries the seed commitment; exporting the completed audit
record makes the private seed available for independent reproduction.

The feature task remains the compatibility baseline, while the Python task is
the whole-program search surface. Direct-decision candidates supply a policy
lower bound; the hidden upper-bound diagnostic uses an independent
evaluator-owned approximation. None of this turns a public score into a
theorem, a novel-algorithm claim, or a production price. The complete protocols, financial model, scoring rules, and
claim boundaries are in the [feature task specification](tasks/bermudan_optimal_stopping/TASK.md)
and [Python task specification](tasks/bermudan_python_search/TASK.md).

The feedback protocol can also be exercised without an LLM or a financial
simulation:

```bash
python3 experiments/feedback_ablation.py
```

This writes a four-arm, equal-budget synthetic ablation under
`artifacts/feedback-ablation-20260904/`. It is a wiring and reproducibility
check for scalar versus directional/adaptive feedback, not evidence of a
better pricing algorithm.

### Task-independent discovery protocol

[`algorithm_discovery.py`](algorithm_discovery.py) exposes a small reusable
protocol for complete finite candidates: `AlgorithmSpec`, `SearchSpace`,
`EvaluationResult`, `FeedbackOracle`, deterministic acquisition, a round
barrier, and an append-only discovery ledger. An adaptive task receives the
same evaluator feedback and `ProblemState` with or without V5 islands; V5 adds
population scheduling, behavior retrieval, and richer lineage rather than
being required for recursion.

[`program_search.py`](program_search.py) is the concrete open implementation:
whole-program generation callbacks, multi-file source candidates, executable
AST mutation, two-parent function-graph crossover, score-based parent choice,
and recursive propose--evaluate--observe rounds. `AgentWholeProgramGenerator`
connects that standalone loop to the configured LLM CLI, while the main Harness
uses the same program operators through Proposal Agents and Experience Bank
parents. The Bermudan task supplies a real fit/predict evaluator for these
programs. This is program-synthesis capability, not evidence that OpenHyra has
already discovered a new algorithm; that claim still needs matched controls,
seeds, held-out results and mechanism inspection.

## How it works

```
┌───────────────┐   inspirations   ┌────────────────┐   solution    ┌─────────┐
│ Context Agent │ ───────────────► │ Proposal Agent │ ────────────► │ Sandbox │
│  (LLM reads the bank,            │  ×N workers    │               │ + trusted │
│   writes an analysis)            │ (Claude/Codex) │               │ evaluator │
└──────▲────────┘                  └────────────────┘               └────┬────┘
       │                     ┌──────────────────┐                        │
       └──────────────────── │ Experience Bank  │ ◄──────────────────────┘
                             └──────────────────┘        results
```

**Experience Bank** — every candidate's code, artifacts, logs and metrics,
committed as independent records whether it succeeded, crashed, or scored low.

**Context Agent** — an LLM that reads a structured summary of all records,
recent logs, recent failures and the current-best implementation, writes a
short situation analysis (persisted as cross-round memory), proposes a small
portfolio of mechanisms with predictions and falsifiers, and names a primary
direction. It does not yet retrieve arbitrary historical source trees or
artifacts.

**Proposal Agents** — headless Claude Code or Codex CLI processes that edit the
solver inside dedicated draft directories using backend-specific permissions;
these directories organize and validate changes but are not a uniform OpenHyra
OS security boundary. Each Context briefing fans out to several independent
candidates with distinct mechanism slots; when enabled, guided/control arms
share a parent and seed. Python-program candidates may replace the full fit and
predict structure, and proposal generation overlaps evaluation.

**Algorithm-design loop** — on `bermudan_python_search`, Context hypotheses are
frozen into traceable candidate slots, V5 persists the hypothesis/analogy
lineage, and the trusted evaluator adds behavior descriptors from independent
pricing paths. Paired contrasts are written to
`research/matched_controls.jsonl`, making the Workshop story “propose several
structures, test a counterfactual, and verify independently” rather than a
single best-score claim.

**Sandbox + trusted evaluation** — candidates run under macOS Seatbelt with no
network and writes confined to the sandbox. Most host reads remain allowed, so
this is write-confinement rather than a confidentiality sandbox. Candidate
`solution.json` files are accepted only as bounded, single-link regular files,
copied into a candidate-inaccessible trusted directory, and scored there. An
integrity whitelist rejects changes beyond the declared editable files, an AST
preflight catches known crash patterns, and every failed/repaired attempt is
stored as an immutable EB record linked by `repair_of`. Proposal source is
sealed before the final whitelist/preflight and execution; EB commits are
assembled from that parent-controlled source plus trusted evaluator outputs,
never from the still-writable draft. Bundle export revalidates source,
solution, evidence, and editable-file hashes before writing the verified bytes.

**Research promotion** — finite candidates remain the executable probes and
the only source of leaderboard scores. Reusable mechanisms can be attached as
typed research data, carried into the Experience Bank, and shown to later
Context Agents beside trusted numerical outcomes. The Context role is asked to
move deliberately through construction, falsification, formalization and proof
repair. A proof sketch, LLM judgment, or absence of small counterexamples is
never labelled a proof.

Each formal `proofs[]` entry contains only a claim ID and an inline Lean proof
term. The
task-owned specification fixes imports and theorem types, and a parent-owned
wrapper binds each proof to its claim template and rational target. Promotion
to `formal_checked` requires an isolated runner to compile that wrapper, then
run the trusted declaration and axiom audit in a separate process and output
channel. Candidate compiler output is never parsed as audit evidence. If no
such runner is configured, the verdict is `unavailable`, never accepted:
formal verification fails closed.
The default harness deliberately has no in-process or unsandboxed fallback.
An operator may opt in with `--formal-runner /absolute/path/to/runner`; the
executable must implement the strict JSON transport documented by
`external_formal_runner.py`, materialize every request in fresh isolation,
deny network and workspace writes, and enforce the supplied limits. Its exact
path, bytes, and SHA-256 are frozen in run provenance and rechecked before each
proof call. It must attest complete, untruncated output plus the actual Lean
binary, toolchain, Mathlib commit/tree, and immutable execution environment;
the probe and proof phases must have identical attestations. The trusted Lean
4.26.0 toolchain and Mathlib revision are pinned with the task-owned
specification. The repository currently supplies the fail-closed client
protocol, not a prebuilt isolated runner or offline Mathlib image.

Proof completion requires four `formal_checked` claims in one record at the
same normalized rational target: `universal_upper_bound`,
`approximating_family`, `supremum_eq`, and `nonattainment`, with zero trusted
claim, obligation, or certificate refutations in that record. A theorem may
still be individually `formal_checked` in a mixed record, but the aggregate
state becomes `formal_checked_with_refutation` and cannot complete the stop
gate. This machinery does not establish that OpenHyra has independently
discovered or proved
\(\sup C(A)=2\); that would require a successfully verified run artifact.

Each run freezes code, task, evaluator, model, concurrency, limits, seed and
stopping policy in `run_manifest.json`. Resume is refused if result-affecting
provenance drifts, and a process lock prevents two harnesses from writing the
same `run-id`.

### Guarded Agent stopping

Active stopping is opt-in with `--agent-stop`; `--iterations` is the
per-invocation upper bound. When enabled, Context rounds are sequential so each
decision sees the prior round's complete EB, while the candidates inside a
round still run concurrently. A Context `stop` is only a request. The
deterministic controller accepts it only after, by default, at least 6 completed
Contexts, 4 Contexts
without a meaningful gain of `0.0001`, and at least 4 successful candidates in
the latest 4 Contexts. Invalid Context JSON or a failed Context
call therefore continues rather than producing an accepted convergence stop. Every pipeline
invocation with `--iterations` writes its termination reason and evidence to
`termination.json`; an accepted stop also includes the raw Agent decision and
the controller review. The file is included in exported bundles.

`expected_gain` and `confidence` are recorded as Agent telemetry only; they do
not participate in the deterministic stop review. Context input is bounded to
80 representative EB records and 96,000 characters, preserving recent records,
the historical best, representative failures, direction coverage and aggregate
counts. A run with an incomplete Context fails closed on resume, and a
`terminal=true` run must be continued under a new `--run-id`. On Ctrl+C the
Harness cancels active CLI/solver process groups and joins all pipeline threads
before writing terminal state and releasing the run lock.

```bash
python3 harness.py --run-id guarded --init --workers 2 --agent-stop
python3 harness.py --run-id guarded --iterations 20 --workers 2 --agent-stop
```

The guards can be configured with `--min-contexts-before-stop`,
`--stop-patience`, `--stop-min-delta`, `--stop-recent-window`, and
`--stop-min-successful-candidates`. Use identical values when initializing and
resuming a run.

## Quick start

```bash
# Requirements: macOS, Python >= 3.10, numpy, and the Claude Code or Codex CLI

# Bermudan feature search, followed by a one-shot frozen Top-K audit.
python3 harness.py --run-id bermudan-demo --init --workers 2 --trial-seed 1729
python3 harness.py --run-id bermudan-demo --iterations 5 --workers 2 --trial-seed 1729
python3 harness.py --run-id bermudan-demo --final-audit
python3 harness.py --run-id bermudan-demo --status
python3 harness.py --run-id bermudan-demo --export-bundle bundles/bermudan-demo

# Whole Python-program search.
python3 harness.py --task bermudan_python_search --run-id bermudan-python-demo \
  --v5 --init --workers 1 --trial-seed 1729
python3 harness.py --task bermudan_python_search --run-id bermudan-python-demo \
  --v5 --iterations 5 --workers 1 --trial-seed 1729
python3 harness.py --task bermudan_python_search --run-id bermudan-python-demo \
  --v5 --final-audit
```

Pass the same `--backend`, `--model`, `--workers`, candidate count and trial
seed and stopping options at initialization and resume. To change them, start a
new `--run-id`.

## References

1. Hyra Team. *Hyra: Hunyuan Research Agent* — technical report, Tencent, 2026.
   <https://hy.tencent.com/research/hyra>
2. Tencent-Hunyuan. *Hyra-results: research artifacts from Hyra.*
   <https://github.com/Tencent-Hunyuan/Hyra-results>
