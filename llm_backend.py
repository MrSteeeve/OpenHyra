"""LLM CLI adapters used by the Context and Proposal agents."""

import os
import signal
import subprocess
import time


SUPPORTED_BACKENDS = ("claude", "codex")


def default_backend():
    return os.environ.get("OPENHYRA_BACKEND", "claude")


def default_model():
    return os.environ.get("OPENHYRA_MODEL") or None


def _kill_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_cli(cmd, *, cwd, prompt_stdin, timeout_s, cancel_event):
    """Run a CLI with cancellable timeout handling and no orphan descendants."""
    if cancel_event is not None and cancel_event.is_set():
        return subprocess.CompletedProcess(cmd, 130, "", "agent call cancelled")
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdin=subprocess.PIPE if prompt_stdin is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    started = time.monotonic()
    pending_input = prompt_stdin
    while True:
        if cancel_event is not None and cancel_event.is_set():
            _kill_process_group(proc)
            stdout, stderr = proc.communicate()
            return subprocess.CompletedProcess(
                cmd, 130, stdout, (stderr + "\nagent call cancelled").strip(),
            )
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            _kill_process_group(proc)
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(
                cmd, timeout_s, output=stdout, stderr=stderr,
            )
        try:
            stdout, stderr = proc.communicate(
                input=pending_input, timeout=min(0.2, remaining),
            )
            return subprocess.CompletedProcess(
                cmd, proc.returncode, stdout, stderr,
            )
        except subprocess.TimeoutExpired:
            # Popen caches the partially written input; retries must pass None.
            pending_input = None


def run_agent(prompt, *, cwd=None, writable=False, timeout_s=600,
              backend=None, model=None, cancel_event=None):
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
        return _run_cli(
            cmd, cwd=cwd, prompt_stdin=None, timeout_s=timeout_s,
            cancel_event=cancel_event,
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
        return _run_cli(
            cmd, cwd=cwd, prompt_stdin=prompt, timeout_s=timeout_s,
            cancel_event=cancel_event,
        )

    raise ValueError(
        f"Unknown LLM backend {backend!r}; choose one of {SUPPORTED_BACKENDS}"
    )
