"""Context Agent: distills the Experience Bank into "inspiration" contexts.

Per the Hyra tech report the Context Agent is itself an LLM agent: each round it
reads the experience bank, writes a short situation analysis (why attempts
won/lost, cross-run patterns) and picks the most promising next direction and
parent solution. The written analysis is the loop's only cross-iteration memory
— Proposal Agents are stateless, so conclusions must be distilled here or they
get re-derived (or re-guessed wrongly) every round.

The LLM call is deliberately light: text-only, no tools, capped output, fed by
the compact diagnostics table plus the previous round's analysis. If the call
fails, we fall back to a deterministic direction rotation so the loop never
stalls on the Context Agent.
"""

import re
import subprocess
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

# Deterministic fallback directions, used only if the LLM Context Agent fails.
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


def pick_direction(iteration):
    if iteration % 2 == 0:
        return DIRECTIONS[0]
    return DIRECTIONS[1 + (iteration // 2) % (len(DIRECTIONS) - 1)]


def _history_table(records):
    lines = [
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
        lines.append(
            f"| {r['id']} | {score} | {fmt('total_tokens_M')} | {fmt('num_steps', 0)} | {tok_s} "
            f"| {fmt('mfu_percent')} | {fmt('peak_vram_mb', 0)} | {fmt('num_params_M')} "
            f"| {r['status']} | {r['description']} |"
        )
    return "\n".join(lines)


def _failure_notes(records):
    failures = [r for r in records if r["status"] != "ok" and r.get("log_tail")]
    if not failures:
        return ""
    blocks = [f"### {r['id']} ({r['status']}): {r['description']}\n```\n{r['log_tail']}\n```"
              for r in failures[-3:]]
    return "\n## Recent failures (do not repeat these mistakes)\n\n" + "\n".join(blocks) + "\n"


def _analyses_dir(eb):
    d = eb.root / "analyses"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _previous_analysis(eb):
    files = sorted(_analyses_dir(eb).glob("iter_*.md"))
    return files[-1].read_text() if files else ""


def _llm_context_analysis(eb, records, best, history, iteration, timeout_s=240):
    """One light LLM call: situation analysis + next direction + parent choice.

    Returns (analysis_text, direction_label, parent_record) or None on failure.
    """
    recent_tails = "\n".join(
        f"### {r['id']} (val_bpb={r['score']})\n```\n{r.get('log_tail', '')}\n```"
        for r in records[-4:]
    )
    prev = _previous_analysis(eb)
    prev_block = f"\n## Your previous analysis (build on it, don't restate it)\n\n{prev}\n" if prev else ""

    prompt = f"""You are the Context Agent of an autonomous research loop (Hyra-style).
You do NOT write code. Your job: distill the experience bank below into guidance
for the next (stateless) Proposal Agent.

{TASK_DESCRIPTION}

## Experience bank (all attempts, run diagnostics)

Fixed budget = 300s wall-clock training, so `tokens_M` and `tok/s` show how each
change affected throughput; capacity gains that starve token throughput lose.

{history}

## Log tails of the most recent runs

{recent_tails}
{prev_block}
## Output format (STRICT, total under 250 words)

## Analysis
<=120 words. WHY attempts won/lost — cross-run patterns, throughput-vs-capacity
tradeoffs, what is now known/refuted. New conclusions only.

## Next
ONE concrete experiment for the next proposal, 1-3 sentences, with concrete
parameter names and values. Must be implementable by editing train.py only.

## Parent
The single solution id to start from (usually the best, unless a different
lineage is more promising). Format: exactly `sol_XXXX`.
"""
    try:
        res = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    out = res.stdout.strip()
    if res.returncode != 0 or "## Next" not in out:
        return None

    next_section = out.split("## Next", 1)[1]
    direction = next_section.split("## Parent", 1)[0].strip()

    parent = best
    m = re.search(r"sol_\d{4}", next_section.split("## Parent", 1)[1]) if "## Parent" in next_section else None
    if m:
        chosen = next((r for r in records if r["id"] == m.group(0) and r["score"] is not None), None)
        if chosen:
            parent = chosen

    (_analyses_dir(eb) / f"iter_{iteration:04d}.md").write_text(out)
    return out, direction, parent


def build_inspiration(eb, iteration: int):
    """Return (parent_record, prompt, direction) for a Proposal Agent."""
    best = eb.best()
    records = eb.records()
    history = _history_table(records)
    failure_notes = _failure_notes(records)

    llm = _llm_context_analysis(eb, records, best, history, iteration)
    if llm is not None:
        analysis, direction, parent = llm
        guidance = f"""## Context Agent briefing (analysis of all past attempts)

{analysis}

Implement the experiment described under "## Next" above. You may deviate only
if you see a clear flaw in the reasoning — say so in PROPOSAL.md if you do."""
    else:
        # Fallback: deterministic rotation (keeps the loop alive without the LLM)
        scored = sorted([r for r in records if r["score"] is not None], key=lambda r: r["score"])
        parent = best if (iteration % 2 == 0 or len(scored) < 2) else scored[(iteration // 2) % min(3, len(scored))]
        direction = pick_direction(iteration)
        guidance = f"""Suggested exploration direction (you may deviate if you have a clearly better idea):
**{direction}**"""

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

{guidance}

Modify `train.py` in the current directory to implement ONE focused experiment.
Keep the change minimal and surgical — this is one iteration of an experiment
loop, not a rewrite. Then write a single line describing the change to a new
file named `PROPOSAL.md` (one short sentence, no markdown headers).

Do not run the training yourself. Do not touch prepare.py or solve.sh —
modifying them is a protocol violation and the harness will reject the solution.
"""
    return parent, prompt, direction
