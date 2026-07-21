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
    "Refine near the current best: pick the most promising hyperparameter seen so far and adjust it by a small step (10-30%). Exploitation, not exploration.",
    "Tune learning rates and the LR schedule (warmdown ratio, per-group LRs).",
    "Trade off batch size vs number of optimizer steps (TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE) — check the tokens_M/steps diagnostics first: fewer total tokens means the change hurt throughput.",
    "Tune model size: DEPTH / width / head dim — check tokens_M in the diagnostics: this budget feeds only a few M tokens, bigger models are undertrained.",
    "Architecture tweaks: attention window pattern, MLP width or activation, value embeddings, logit softcap.",
    "Optimizer settings: Muon momentum schedule, weight decay, Adam betas.",
    "Throughput: raise tokens/sec on MPS (bigger micro-batch, less per-step host overhead) so more tokens fit in the fixed budget.",
    "Combine the best ideas seen so far in the experience bank.",
]

# Exploit every other iteration: even -> refine-near-best, odd -> rotate exploration
def pick_direction(iteration):
    if iteration % 2 == 0:
        return DIRECTIONS[0]
    return DIRECTIONS[1 + (iteration // 2) % (len(DIRECTIONS) - 1)]


def build_inspiration(eb, iteration: int):
    """Return (parent_record, prompt, direction) for a Proposal Agent."""
    best = eb.best()
    records = eb.records()

    # Parent diversity (report: inspirations should be as diverse as possible):
    # exploitation iterations build on the best; exploration iterations cycle
    # through the top-3 scored solutions so search is not a single greedy chain.
    scored = sorted([r for r in records if r["score"] is not None], key=lambda r: r["score"])
    if iteration % 2 == 0 or len(scored) < 2:
        parent = best
    else:
        parent = scored[(iteration // 2) % min(3, len(scored))]

    # Full history WITH run diagnostics — score alone hides *why* an attempt lost
    # (e.g. a bigger model that starved itself of tokens in the fixed budget).
    history_lines = [
        "| id | val_bpb | tokens_M | steps | tok/s | mfu% | mem_MB | params_M | status | description |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        m = r.get("metrics", {})
        score = f"{r['score']:.6f}" if r["score"] is not None else "-"
        toks = m.get("total_tokens_M")
        secs = m.get("training_seconds")
        tok_s = f"{toks * 1e6 / secs:,.0f}" if toks and secs else "-"
        fmt = lambda k, p=1: f"{m[k]:.{p}f}" if k in m else "-"
        history_lines.append(
            f"| {r['id']} | {score} | {fmt('total_tokens_M')} | {fmt('num_steps', 0)} | {tok_s} "
            f"| {fmt('mfu_percent')} | {fmt('peak_vram_mb', 0)} | {fmt('num_params_M')} "
            f"| {r['status']} | {r['description']} |"
        )
    history = "\n".join(history_lines)

    # Crashed/timed-out attempts: show their log tails so failures aren't repeated
    failure_notes = ""
    failures = [r for r in records if r["status"] != "ok" and r.get("log_tail")]
    if failures:
        blocks = [f"### {r['id']} ({r['status']}): {r['description']}\n```\n{r['log_tail']}\n```"
                  for r in failures[-3:]]
        failure_notes = "\n## Recent failures (do not repeat these mistakes)\n\n" + "\n".join(blocks) + "\n"

    direction = pick_direction(iteration)

    prompt = f"""{TASK_DESCRIPTION}

## Experience bank (all past attempts, with run diagnostics)

The fixed budget is 300s of wall-clock training: `tokens_M` (total tokens seen)
and `tok/s` tell you how a change affected throughput — a "better" model that
feeds itself fewer tokens usually loses. Compare these columns before betting.

{history}
{failure_notes}
## Your starting point: {parent['id']} (val_bpb {parent['score']:.6f}; current best is {best['id']} @ {best['score']:.6f})

The `train.py` in your working directory is {parent['id']}'s. Log tail of its run:

```
{parent['log_tail']}
```

## Your assignment

Suggested exploration direction (you may deviate if you have a clearly better idea):
**{direction}**

Modify `train.py` in the current directory to implement ONE focused experiment
in that direction. Keep the change minimal and surgical — this is one iteration
of an experiment loop, not a rewrite. Then write a single line describing the
change to a new file named `PROPOSAL.md` (one short sentence, no markdown headers).

Do not run the training yourself. Do not touch prepare.py or solve.sh —
modifying them is a protocol violation and the harness will reject the solution.
"""
    return parent, prompt, direction
