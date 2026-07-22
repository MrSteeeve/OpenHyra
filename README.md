# OpenHyra

![CI](https://github.com/MrSteeeve/OpenHyra/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**English** | [中文](README.zh-CN.md)

An open reproduction of Tencent Hunyuan's **Hyra** (Hunyuan Research Agent) harness [1],
applied to the **sums_diffs** task: an autonomous loop in which LLM agents
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

| System | $C(A)$ | Within SimpleTES v1 protocol |
|---|---|---|
| Official seed (17-element initial construction) | 1.059793 | ✓ |
| **OpenHyra (this repo)** | **1.111815** | ✓ ($n = 405$) |
| SimpleTES [3] | 1.144887 | ✓ |
| Hyra [1, 2] | 1.159715 | ✗ (published artifact has 181,131 elements) |

Our best set was found by a Codex-backed run (20 Context rounds × 4 candidates
per round), starting from the official seed. It was scored by the trusted
evaluator and independently re-verified: $n = 405$, $|A+A| = 2395$,
$|A-A| = 2003$.


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

**Context Agent** — an LLM that reads the full bank each round, writes a short
situation analysis (persisted as cross-round memory), and picks the next
experiment direction.

**Proposal Agents** — headless Claude Code or Codex CLI processes that edit the
solver inside isolated drafts; each Context briefing fans out to several
independent candidates, and proposal generation overlaps evaluation.

**Sandbox + trusted evaluation** — candidates run under macOS Seatbelt (no
network, writes confined to the sandbox) and emit only `solution.json`; the
score is recomputed outside the sandbox. An integrity whitelist rejects any
change beyond the declared editable files, an AST preflight catches known
crash patterns before launch, and crashed candidates get a bounded LLM repair
loop with a full audit trail.

## Quick start

```bash
# Requirements: macOS, Python >= 3.10, numpy, and the Claude Code or Codex CLI
python3 harness.py --init                      # score the official seed and bank it
python3 harness.py --iterations 5 --workers 2  # run the autonomous loop
python3 harness.py --status                    # inspect the experience bank
```

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
