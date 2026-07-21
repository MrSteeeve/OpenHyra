"""Proposal Agent: consumes an inspiration context and produces a new solution folder.

The LLM backend is the Claude Code CLI in headless mode (`claude -p`), standing in
for the Hunyuan model that powers the real Hyra. It edits train.py inside a draft
solution folder; the harness then runs the draft in a sandbox and commits the
result to the Experience Bank.
"""

import shutil
import subprocess
from pathlib import Path


def propose(parent_dir: Path, draft_dir: Path, prompt: str, timeout_s: int = 600):
    """Copy parent solution to draft_dir, let the agent edit train.py there.

    Returns (ok, description).
    """
    draft_dir = Path(draft_dir)
    if draft_dir.exists():
        shutil.rmtree(draft_dir)
    shutil.copytree(parent_dir, draft_dir, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".git", "run.log"))

    before = (draft_dir / "train.py").read_text()
    try:
        subprocess.run(
            ["claude", "-p", prompt,
             "--permission-mode", "acceptEdits",
             "--allowedTools", "Read,Edit,Write"],
            cwd=draft_dir, capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "proposal agent timed out"

    after = (draft_dir / "train.py").read_text()
    if after == before:
        return False, "proposal agent made no change"

    proposal_md = draft_dir / "PROPOSAL.md"
    description = proposal_md.read_text().strip().splitlines()[0] if proposal_md.exists() else "(no description)"
    return True, description
