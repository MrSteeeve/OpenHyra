"""Context Agent: distills the Experience Bank into diverse "inspiration" contexts.

Per the Hyra tech report, an inspiration is a bundle of context (past solutions,
scores, logs) plus a direction hint, pushed onto the task queue for Proposal
Agents to consume. Directions are rotated to keep exploration diverse.
"""

from pathlib import Path

TASK_DESCRIPTION = """\
# Task: NanoChat AutoResearch (Apple-Silicon scaled-down port)

You must lower `val_bpb` (validation bits-per-byte, LOWER IS BETTER) of a small
GPT trained with a FIXED 5-minute wall-clock budget on a single Apple M4 GPU (MPS).

Rules (same spirit as karpathy/autoresearch):
- You may ONLY modify `train.py` (architecture, optimizer, hyperparameters, batch size, ...).
- `prepare.py` is frozen: it defines the tokenizer, dataloader, seq len 512,
  the 300s time budget, and the ground-truth metric `evaluate_bpb`. DO NOT touch it,
  do not re-implement or bypass the eval. Leaking future tokens (non-causal
  attention) is reward hacking and strictly forbidden.
- No new dependencies. MPS eager mode: torch.compile is disabled off-CUDA;
  code must run on MPS (no CUDA-only ops, no tensor-scalar dtype tricks that MPS rejects).
- The run must finish within ~10 minutes total or it is killed.
"""

DIRECTIONS = [
    "Tune learning rates and the LR schedule (warmdown ratio, per-group LRs).",
    "Trade off batch size vs number of optimizer steps (TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE).",
    "Tune model size: DEPTH / width / head dim for this tiny compute budget.",
    "Architecture tweaks: attention window pattern, MLP width or activation, value embeddings, logit softcap.",
    "Optimizer settings: Muon momentum schedule, weight decay, Adam betas.",
    "Throughput: reduce per-step host overhead on MPS so more tokens fit in the budget.",
    "Combine the best ideas seen so far in the experience bank.",
]


def build_inspiration(eb, iteration: int):
    """Return (parent_record, prompt) for a Proposal Agent."""
    best = eb.best()
    records = eb.records()

    history_lines = ["| id | score (val_bpb) | status | description |", "|---|---|---|---|"]
    for r in records:
        score = f"{r['score']:.6f}" if r["score"] is not None else "-"
        history_lines.append(f"| {r['id']} | {score} | {r['status']} | {r['description']} |")
    history = "\n".join(history_lines)

    direction = DIRECTIONS[iteration % len(DIRECTIONS)]

    best_code = Path(best["path"], "train.py").read_text() if best else ""

    prompt = f"""{TASK_DESCRIPTION}

## Experience bank (all past attempts)

{history}

## Current best solution: {best['id']} (val_bpb {best['score']:.6f})

Its `train.py` is the file in your working directory. Recent log tail of the best run:

```
{best['log_tail']}
```

## Your assignment

Suggested exploration direction (you may deviate if you have a clearly better idea):
**{direction}**

Modify `train.py` in the current directory to implement ONE focused experiment
in that direction. Keep the change minimal and surgical — this is one iteration
of an experiment loop, not a rewrite. Then write a single line describing the
change to a new file named `PROPOSAL.md` (one short sentence, no markdown headers).

Do not run the training yourself. Do not touch prepare.py or solve.sh.
"""
    return best, prompt, direction, best_code
