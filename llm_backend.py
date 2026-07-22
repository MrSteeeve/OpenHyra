"""LLM CLI adapters used by the Context and Proposal agents."""

import os
import subprocess


SUPPORTED_BACKENDS = ("claude", "codex")


def default_backend():
    return os.environ.get("OPENHYRA_BACKEND", "claude")


def default_model():
    return os.environ.get("OPENHYRA_MODEL") or None


def run_agent(prompt, *, cwd=None, writable=False, timeout_s=600,
              backend=None, model=None):
    """Run one stateless agent call and return CompletedProcess.

    Codex receives prompts over stdin so large experience banks do not run into
    the operating system's command-line length limit. Drafts intentionally have
    no .git directory, hence --skip-git-repo-check.
    """
    backend = backend or default_backend()
    model = model or default_model()

    if backend == "claude":
        cmd = ["claude", "-p", prompt]
        if writable:
            cmd += ["--permission-mode", "acceptEdits",
                    "--allowedTools", "Read,Edit,Write"]
        if model:
            cmd += ["--model", model]
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )

    if backend == "codex":
        cmd = [
            "codex", "--ask-for-approval", "never", "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color", "never",
            "--sandbox", "workspace-write" if writable else "read-only",
        ]
        if model:
            cmd += ["--model", model]
        cmd.append("-")
        return subprocess.run(
            cmd, cwd=cwd, input=prompt, capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )

    raise ValueError(
        f"Unknown LLM backend {backend!r}; choose one of {SUPPORTED_BACKENDS}"
    )
