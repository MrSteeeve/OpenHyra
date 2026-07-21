"""OpenHyra harness: a minimal reproduction of the Hyra loop from the tech report.

    Context Agent -> inspiration queue -> Proposal Agent -> sandbox run -> Experience Bank

On a single-GPU laptop the producer/consumer pipeline degenerates to a sequential
loop (sandbox semaphore = 1), but the components and data flow are the same.

Usage:
    python harness.py --seed <solution_dir> [--iterations N]
    python harness.py --status
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from eb import ExperienceBank
from context_agent import build_inspiration
from proposal_agent import propose
from sandbox import run_solution, parse_metrics

ROOT = Path(__file__).resolve().parent
# Task environment (karpathy/autoresearch port) lives outside this repo.
AUTORESEARCH_DIR = Path(os.environ.get("OPENHYRA_AUTORESEARCH", Path.home() / "GitHub" / "autoresearch"))
PYTHON_BIN = str(AUTORESEARCH_DIR / ".venv" / "bin" / "python")


def seed_bank(eb, seed_dir, score, description, log_tail, runlog=None):
    metrics = {}
    if runlog and Path(runlog).exists():
        log_text = Path(runlog).read_text(errors="replace")
        metrics = parse_metrics(log_text)
        if not log_tail:
            log_tail = "\n".join(log_text.replace("\r", "\n").splitlines()[-15:])
    sol_id = eb.next_id()
    rec = eb.commit(sol_id, seed_dir, score, "ok", description, parent=None,
                    log_tail=log_tail, metrics=metrics)
    print(f"[eb] seeded {sol_id}: val_bpb={score} ({description})")
    return rec


def iterate(eb, iteration):
    """One full Hyra loop iteration. Returns the committed record."""
    parent, prompt, direction, _ = build_inspiration(eb, iteration)
    print(f"[context] iteration {iteration}: direction = {direction!r}, parent = {parent['id']}")

    draft = ROOT / "drafts" / f"iter_{iteration:04d}"
    ok, description = propose(Path(parent["path"]), draft, prompt)
    print(f"[proposal] {description}" if ok else f"[proposal] FAILED: {description}")
    if not ok:
        return eb.commit(eb.next_id(), draft, None, "crash", description, parent["id"], "")

    sandbox = ROOT / "sandboxes" / f"iter_{iteration:04d}"
    print(f"[sandbox] running solve.sh (fixed 5-minute budget + eval) ...")
    score, status, log_tail, metrics = run_solution(draft, sandbox, PYTHON_BIN)
    # keep the sandbox run.log with the solution artifact
    if (sandbox / "run.log").exists():
        shutil.copy(sandbox / "run.log", draft / "run.log")

    rec = eb.commit(eb.next_id(), draft, score, status, description, parent["id"], log_tail, metrics)
    best = eb.best()
    verdict = "IMPROVED — new best" if score is not None and rec["id"] == best["id"] else "kept in bank, best unchanged"
    print(f"[eb] {rec['id']}: val_bpb={score} status={status} -> {verdict} (best: {best['id']} @ {best['score']:.6f})")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", help="seed solution dir (with solve.sh, train.py, prepare.py)")
    ap.add_argument("--seed-score", type=float, help="known score of the seed solution")
    ap.add_argument("--seed-desc", default="seed solution")
    ap.add_argument("--seed-log", default="")
    ap.add_argument("--seed-runlog", help="path to the seed's run.log (metrics are parsed from it)")
    ap.add_argument("--iterations", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    eb = ExperienceBank(ROOT / "eb")

    if args.status:
        for r in eb.records():
            score = f"{r['score']:.6f}" if r["score"] is not None else "   -    "
            print(f"{r['id']}  {score}  {r['status']:7s}  {r['description']}")
        best = eb.best()
        if best:
            print(f"\nbest: {best['id']} @ {best['score']:.6f}")
        return

    if args.seed:
        if args.seed_score is None:
            sys.exit("--seed requires --seed-score (run the seed once and pass its val_bpb)")
        seed_bank(eb, Path(args.seed), args.seed_score, args.seed_desc, args.seed_log, args.seed_runlog)

    if not eb.records():
        sys.exit("Experience bank is empty; seed it first with --seed.")

    start = len([r for r in eb.records() if r["parent"] is not None])
    for i in range(start, start + args.iterations):
        iterate(eb, i)


if __name__ == "__main__":
    main()
