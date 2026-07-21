"""Proposal Agent: consumes an inspiration context and produces a new solution.

The LLM backend is the Claude Code CLI in headless mode (`claude -p`), standing
in for the model that powers the real Hyra. It edits the task's editable files
inside a draft solution folder; the harness then runs the draft in a sandbox,
scores it with the trusted evaluator, and commits the result to the Experience
Bank.
"""

import shutil
import subprocess
from pathlib import Path

RUN_ARTIFACTS = [".venv", "__pycache__", ".git", "run.log", "train.log",
                 "PROPOSAL.md", "solution.json"]


def propose(parent_dir: Path, draft_dir: Path, prompt: str, editable_files, timeout_s: int = 600):
    """Copy parent solution to draft_dir, let the agent edit the editable files.

    Returns (ok, description).
    """
    draft_dir = Path(draft_dir)
    if draft_dir.exists():
        shutil.rmtree(draft_dir)
    shutil.copytree(parent_dir, draft_dir, ignore=shutil.ignore_patterns(*RUN_ARTIFACTS))

    before = {f: (draft_dir / f).read_text() for f in editable_files if (draft_dir / f).exists()}
    try:
        res = subprocess.run(
            ["claude", "-p", prompt,
             "--permission-mode", "acceptEdits",
             "--allowedTools", "Read,Edit,Write"],
            cwd=draft_dir, capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "proposal agent timed out"
    except FileNotFoundError:
        return False, "claude CLI not found on PATH"

    if res.returncode != 0:
        return False, f"proposal agent exited with code {res.returncode}"

    after = {f: (draft_dir / f).read_text() for f in editable_files if (draft_dir / f).exists()}
    if after == before:
        return False, "proposal agent made no change"

    proposal_md = draft_dir / "PROPOSAL.md"
    description = proposal_md.read_text().strip().splitlines()[0] if proposal_md.exists() else "(no description)"
    return True, description
