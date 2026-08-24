# OpenHyra

![CI](https://github.com/MrSteeeve/OpenHyra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**English** | [中文](README.zh-CN.md)

An open, partial reproduction of the public architecture of Tencent Hunyuan's
**Hyra** (Hunyuan Research Agent) harness [1], currently demonstrated on the
**sums_diffs** and **Bermudan optimal stopping** tasks: an autonomous loop in which LLM agents
propose solvers, a sandbox runs them, a trusted evaluator scores them, and
every outcome, whether success or failure, is banked as experience for the next round.

## Tasks

### Sum-difference search

Construct a finite set of integers $A$ maximizing the sum-vs-difference exponent

$$C(A) = \frac{\log\left(|A+A| \/\ |A|\right)}{\log\left(|A-A| \/\ |A|\right)}$$

where $A+A = \{a+b : a,b \in A\}$ and $A-A = \{a-b : a,b \in A\}$.

For most sets $C(A) < 1$, since addition commutes and differences tend to
outnumber sums; sum-dominant ("MSTD") constructions push it above 1 [4].

The current task accepts any finite explicit set with $|A| \ge 2$ and elements
within $[-10^6, 10^6]$; it has no fixed upper bound on set cardinality.
Candidates have a hard 180-second timeout, and a trusted evaluator outside the
sandbox exactly enumerates $A+A$ and $A-A$. Artifact size, evaluator time, and
evaluator memory budgets provide the operational limits. Nothing a candidate
reports about itself is ever trusted.

`solution.json` may also carry the single current `openhyra-research` schema;
there are no `v1`/`v2` protocol variants. Its construction is a typed positional
digit product with bounded exact check levels and allowlisted obligations.
Claims link explicitly to those obligations and, for formal claims, to a
normalized rational target. These fields never change the numerical score.
The trusted evaluator separately writes `evidence.json`: checked finite
obligations are `bounded_checked`, failed obligations are `refuted` with a
trusted counterexample. An obligation link is recorded as bounded or refuted
evidence, but it does not promote the linked natural-language claim: without a
trusted implication rule that claim remains `unverified`. Passing bounded
checks never proves an asymptotic statement. The candidate input field
contract and an example are in
[the task specification](tasks/sums_diffs/TASK.md).

### Bermudan optimal stopping

The second task searches a bounded, typed feature-expression program for a
fixed Ridge LSMC algorithm under evaluator-owned risk-neutral Black--Scholes
models. Candidates submit neither prices nor executable Python. The evaluator
owns simulation, contracts, discounting, regression, causal exercise, budgets,
and statistics. Public search uses independent fit/pricing paths and paired
Common Random Numbers against a frozen baseline; its score is a conservative
lower bound on strike-normalized lower-bound improvement.

Final acceptance is a separate one-shot action. The harness validates and
freezes the distinct Top-K normalized artifacts first, then draws a fresh
private seed and evaluates every frozen artifact on the same hidden suite. The
audit constructs a conditionally centered nested martingale and ranks the
primal--dual confidence gap. Results are written to `final_audit.json`, never
fed back into the Experience Bank or another search round. The public
termination record carries the seed commitment; exporting the completed audit
record makes the private seed available for independent reproduction.

This is deliberately a Phase-1 feature-search task with a Phase-4 acceptance
audit, not yet unrestricted policy/Python/full-algorithm search. Generating the
hidden seed only after a data-only artifact is frozen closes the relevant
feedback channel here; it does not turn the existing write-confinement sandbox
into a confidentiality boundary for arbitrary candidate code. The complete IR,
financial protocol, scoring rule, and claim boundary are in the
[Bermudan task specification](tasks/bermudan_optimal_stopping/TASK.md).

## Results

| System | $C(A)$ |
|---|---:|
| Official seed (17-element initial construction) | 1.059793 |
| **OpenHyra legacy run** | **1.111815** ($n = 405$) |
| SimpleTES [3] | 1.144887 |

These are historical reference points, not a current same-protocol leaderboard:
the OpenHyra legacy run and SimpleTES result were produced under size-bounded
settings, while the current task has no fixed cardinality ceiling. Published
Hyra artifacts [1, 2] have not been rerun through the current trusted evaluator
and its resource envelope, so they are not added to the table.

The OpenHyra set was found by a Codex-backed historical run (20 Context rounds
× 4 candidates per round), scored by the trusted evaluator and independently
re-verified: $n=405$, $|A+A|=2395$, $|A-A|=2003$. That run predates the current
all-outcomes and immutable-repair EB semantics: it retained one winner artifact
per Context and summaries for the other candidates. The set and standalone
verifier are published as a clearly labelled
[legacy artifact](artifacts/sums_diffs/openhyra-1.111814562869239-legacy/);
the current harness has not yet been rerun for a replacement headline result.

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
short situation analysis (persisted as cross-round memory), and picks the next
experiment direction. It does not yet retrieve arbitrary historical source
trees or artifacts.

**Proposal Agents** — headless Claude Code or Codex CLI processes that edit the
solver inside dedicated draft directories using backend-specific permissions;
these directories organize and validate changes but are not a uniform OpenHyra
OS security boundary. Each Context briefing fans out to several independent
candidates, and proposal generation overlaps evaluation.

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
the latest 4 Contexts. For this task it also requires the four formal claims
above at one common rational target. Invalid Context JSON, a failed Context
call, a missing formal runner, or incomplete formal evidence therefore
continues rather than producing an accepted convergence stop. Every pipeline
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
python3 harness.py --run-id demo --init --workers 2
python3 harness.py --run-id demo --iterations 5 --workers 2
python3 harness.py --run-id demo --status
python3 harness.py --run-id demo --export-bundle bundles/demo

# Bermudan feature search, followed by a one-shot frozen Top-K audit.
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-demo \
  --init --workers 2 --trial-seed 1729
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-demo \
  --iterations 5 --workers 2 --trial-seed 1729
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-demo \
  --final-audit
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-demo \
  --export-bundle bundles/bermudan-demo

# Formal runs must use the same trusted runner at initialization and resume.
python3 harness.py --run-id formal --init \
  --formal-runner /absolute/path/to/openhyra-formal-runner
```

Pass the same `--backend`, `--model`, `--workers`, candidate count and trial
seed and stopping options at initialization and resume. To change them, start a
new `--run-id`.

## References

1. Hyra Team. *Hyra: Hunyuan Research Agent* — technical report, Tencent, 2026.
   <https://hy.tencent.com/research/hyra>
2. Tencent-Hunyuan. *Hyra-results: research artifacts from Hyra.*
   <https://github.com/Tencent-Hunyuan/Hyra-results>
3. *SimpleTES: Evaluation-driven Scaling for Scientific Discovery.*
   arXiv:2604.19341. <https://arxiv.org/abs/2604.19341>
4. G. Martin, K. O'Bryant. *Many sets have more sums than differences.*
   In Additive Combinatorics, CRM Proc. Lecture Notes 43, 2007.
   <https://arxiv.org/abs/math/0608131>
