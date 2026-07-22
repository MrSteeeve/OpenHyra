# OpenHyra

![CI](https://github.com/MrSteeeve/OpenHyra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**English** | [中文](README.zh-CN.md)

An open reproduction of Tencent Hunyuan's **Hyra** (Hunyuan Research Agent,
[tech report](https://hy.tencent.com/research/hyra) /
[official results repo](https://github.com/Tencent-Hunyuan/Hyra-results)).

**Phase 1 target: `sums_diffs`** — the sum-vs-difference exponent problem:
construct a finite set of integers $A$ maximizing

$$C(A) = \frac{\log\left(|A+A| / |A|\right)}{\log\left(|A-A| / |A|\right)}$$

where $A+A = \{a+b : a,b \in A\}$ and $A-A = \{a-b : a,b \in A\}$. The task is
CPU-only, objectively and deterministically scored, and comparable across
machines — an ideal first arena for validating a research-agent loop. This
repository follows the public **SimpleTES sums_diffs v1 protocol** only:
$2 \le |A| \le 512$, elements within $[-10^6, 10^6]$, a hard 180-second
candidate timeout, and a reference score of **1.144887**. Hyra's published
large-set artifact does not satisfy this protocol, so it is excluded from this
project's scoreboard.

## Harness architecture (aligned with the tech report)

```
┌───────────────┐   inspirations   ┌────────────────┐   solution    ┌─────────┐
│ Context Agent │ ───────────────► │ Proposal Agent │ ────────────► │ Sandbox │
│  (LLM: reads the bank,           │  ×N workers      │             │ + trusted │
│   writes an analysis)            │ (Claude/Codex)   │             │ evaluator │
└──────▲────────┘                  └────────────────┘               └────┬────┘
       │                     ┌──────────────────┐                        │
       └──────────────────── │ Experience Bank  │ ◄──────────────────────┘
                             └──────────────────┘        results
```

- **Experience Bank** (`eb.py`): stores every candidate's code, artifacts, logs
  and evaluator metrics; failures, crashes, violations and low scores are
  committed as independent records just like successes; thread-safe.
- **Context Agent** (`context_agent.py`): an LLM agent that reads the full bank
  each round, writes a ≤250-word situation analysis and picks the next
  experiment direction; analyses persist as cross-round memory; falls back to a
  deterministic direction rotation if the call fails.
- **Proposal Agent** (`proposal_agent.py`): headless Claude Code or Codex CLI
  edits the task's single editable file inside a draft and writes `PROPOSAL.md`;
  by default each Context briefing fans out to 4 independent candidates, and
  with `--workers N` proposal generation overlaps evaluation. No local
  winner-selection — every outcome is committed.
- **Sandbox** (`sandbox.py`): macOS Seatbelt (`sandbox-exec`) isolation — no
  network, writes confined to the sandbox directory; a non-zero exit is a crash.
- **Trusted evaluation**: candidates emit only `solution.json` (the set
  itself); the score is computed outside the sandbox by
  `tasks/<name>/evaluator.py`, which enumerates $A+A$ and $A-A$ exactly and
  checks the SimpleTES constraints — nothing a candidate reports about itself
  is ever trusted.
- **Integrity whitelist**: any file added, removed or modified other than the
  task's editable files and `PROPOSAL.md` → status `violation`, never sandboxed.
- **Engineering preflight + bounded repair**: before sandboxing, candidates
  pass an AST lint distilled from observed crash patterns (unclamped annealing
  progress, dynamically-empty `randrange`; failures get status `rejected`);
  crashed or timed-out candidates receive `candidate_repair_attempts` repair
  rounds (the failure log is fed back to the proposal backend as untrusted
  data for a minimal fix, re-checked against the frozen-file whitelist, then
  re-evaluated), with the full attempt trail archived in EB metadata.

## Task plugins

```
tasks/sums_diffs/
  task.json        direction (max), editable files, timeouts, eval concurrency,
                   fallback directions, engineering invariants
  TASK.md          task statement and protocol constraints shown to agents
  evaluator.py     trusted exact evaluator (independent SimpleTES v1 reimplementation)
  seed_solution/   the official SimpleTES 17-element initial set (C ≈ 1.059793)
runs/<task>/       experience bank, drafts, sandboxes (never committed)
```

## Running

```bash
# Requirements: Python >= 3.10 + numpy; proposals/analyses default to the
# Claude Code CLI (headless)
python3 harness.py --task sums_diffs --init          # seed via the trusted pipeline
python3 harness.py --task sums_diffs --iterations 5 --workers 2
# Switch to the Codex CLI (--model optional; defaults to Codex's current model)
python3 harness.py --task sums_diffs --iterations 5 --workers 2 \
  --backend codex --model gpt-5.6-sol
python3 harness.py --task sums_diffs --status
```

`--iterations` counts Context rounds, not new EB records. With the default
`candidates_per_context = 4` each round normally adds 4 candidate records;
override with `--candidates-per-context N`. Every candidate outcome is written
to the EB, so later Context rounds can draw on successes, low-scoring
counterexamples and failure logs alike.

**Scheduling note**: the window of Context rounds allowed to run ahead of
results is fixed at `workers + eval_concurrency`. Historical experiments (the
`trial_01`–`trial_04` series) ran on an earlier scheduler revision in strict
serial-feedback mode (`--max-inflight 1`); that flag has since been removed —
when comparing bundles, refer to the scheduling parameters recorded in each
bundle's manifest.

## Known gaps vs the real Hyra

- Proposal models are the Claude Code / Codex CLIs (standing in for the
  internal Hunyuan model);
- single-machine concurrency is far below Hyra's semaphore-controlled agent
  fleet;
- the evaluator co-evolution outer loop is not implemented (this task's
  evaluator is fixed, so it is not needed);
- Seatbelt is process-level isolation, weaker than containers/VMs.

## Roadmap

`sums_diffs` search reproduction → independent audit of the official
`smallest_adder` artifact → `qubit_routing` and further AI4Science task
plugins.
