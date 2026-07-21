"""Proposal Agent: consumes an inspiration context and produces a new solution.

The configured LLM CLI edits the task's editable files inside a draft solution
folder; the harness then runs the draft in a sandbox, scores it with the trusted
evaluator, and commits the result to the Experience Bank.
"""

import shutil
import subprocess
from pathlib import Path

from llm_backend import run_agent

RUN_ARTIFACTS = [".venv", "__pycache__", ".git", "run.log", "train.log",
                 "PROPOSAL.md", "solution.json"]


def propose(parent_dir: Path, draft_dir: Path, prompt: str, editable_files,
            timeout_s: int = 600, backend: str = "claude", model=None):
    """Copy parent solution to draft_dir, let the agent edit the editable files.

    Returns (ok, description).
    """
    draft_dir = Path(draft_dir)
    if draft_dir.exists():
        shutil.rmtree(draft_dir)
    shutil.copytree(parent_dir, draft_dir, ignore=shutil.ignore_patterns(*RUN_ARTIFACTS))

    before = {f: (draft_dir / f).read_text() for f in editable_files if (draft_dir / f).exists()}
    try:
        res = run_agent(
            prompt, cwd=draft_dir, writable=True, timeout_s=timeout_s,
            backend=backend, model=model,
        )
    except subprocess.TimeoutExpired:
        return False, "proposal agent timed out"
    except FileNotFoundError:
        return False, f"{backend} CLI not found on PATH"

    if res.returncode != 0:
        detail = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else ""
        suffix = f": {detail[:300]}" if detail else ""
        return False, f"proposal agent ({backend}) exited with code {res.returncode}{suffix}"

    after = {f: (draft_dir / f).read_text() for f in editable_files if (draft_dir / f).exists()}
    if after == before:
        return False, "proposal agent made no change"

    proposal_md = draft_dir / "PROPOSAL.md"
    if not proposal_md.exists() and res.stdout.strip():
        # Codex sometimes makes the requested edit but reports its summary only
        # in the final response. Preserve that response as the experiment label.
        summary = " ".join(res.stdout.split())[:500]
        proposal_md.write_text(summary + "\n")
    description = proposal_md.read_text().strip().splitlines()[0] if proposal_md.exists() else "(no description)"
    return True, description
