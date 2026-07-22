# OpenHyra

![CI](https://github.com/MrSteeeve/OpenHyra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**English** | [中文](README.zh-CN.md)

An open, partial reproduction of the public architecture of Tencent Hunyuan's
**Hyra** (Hunyuan Research Agent) harness [1], currently demonstrated on the
**sums_diffs** task: an autonomous loop in which LLM agents
propose solvers, a sandbox runs them, a trusted evaluator scores them, and
every outcome, whether success or failure, is banked as experience for the next round.

## The task

Construct a finite set of integers $A$ maximizing the sum-vs-difference exponent

$$C(A) = \frac{\log\left(|A+A| \/\ |A|\right)}{\log\left(|A-A| \/\ |A|\right)}$$

where $A+A = \{a+b : a,b \in A\}$ and $A-A = \{a-b : a,b \in A\}$.

For most sets $C(A) < 1$, since addition commutes and differences tend to
outnumber sums; sum-dominant ("MSTD") constructions push it above 1 [4].

We follow the public **SimpleTES sums_diffs task requirements** [3]:
$2 \le |A| \le 512$, elements within $[-10^6, 10^6]$, a hard 180-second
candidate timeout, and exact enumeration of $A+A$ and $A-A$ by a trusted
evaluator outside the sandbox. Nothing a candidate reports about itself is
ever trusted.

## Results

| System | $C(A)$ |
|---|---:|
| Official seed (17-element initial construction) | 1.059793 |
| **OpenHyra legacy run** | **1.111815** ($n = 405$) |
| SimpleTES [3] | 1.144887 |

All rows above use the declared SimpleTES v1 protocol. Hyra's published
1.159715 artifact [1, 2] is a cross-protocol reference only: its 181,131
elements exceed the $|A|\le512$ constraint and it is therefore not included in
the comparison table.

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
stored as an immutable EB record linked by `repair_of`.

Each run freezes code, task, evaluator, model, concurrency, limits and seed in
`run_manifest.json`. Resume is refused if result-affecting provenance drifts,
and a process lock prevents two harnesses from writing the same `run-id`.

## Quick start

```bash
# Requirements: macOS, Python >= 3.10, numpy, and the Claude Code or Codex CLI
python3 harness.py --run-id demo --init --workers 2
python3 harness.py --run-id demo --iterations 5 --workers 2
python3 harness.py --run-id demo --status
python3 harness.py --run-id demo --export-bundle bundles/demo
```

Pass the same `--backend`, `--model`, `--workers`, candidate count and trial
seed at initialization and resume. To change them, start a new `--run-id`.

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
