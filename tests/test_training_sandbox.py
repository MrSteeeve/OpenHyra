"""Cross-platform research worker lifecycle, not OS permission isolation."""

from pathlib import Path
from threading import Event
import json
import os
import subprocess
import sys
import time

import pytest

from worker_subprocess import LOG_BYTES, run_worker_subprocess


def python(code: str, *args: str) -> list[str]:
    return [sys.executable, "-s", "-B", "-c", code, *args]


def test_worker_runs_with_pythonpath_and_writes_json(tmp_path: Path) -> None:
    (tmp_path / "helper.py").write_text("VALUE = 41\n")
    target = tmp_path / "out.json"
    result = run_worker_subprocess(
        python("import json,sys,helper; from pathlib import Path; "
               "Path(sys.argv[1]).write_text(json.dumps({'value': helper.VALUE + 1}))", str(target)),
        timeout_s=10, extra_env={"PYTHONPATH": str(tmp_path)},
    )
    assert result["returncode"] == 0 and not result["timed_out"]
    assert json.loads(target.read_text()) == {"value": 42}


def test_timeout_reclaims_worker_and_subprocess_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    script = ("import subprocess,sys,time; from pathlib import Path; "
              "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
              "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)")
    result = run_worker_subprocess(python(script, str(pid_file)), timeout_s=1.5)
    assert result["timed_out"] and result["returncode"] != 0
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3
    while True:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
        ).stdout.strip()
        if not state or state.startswith("Z"):
            break
        assert time.monotonic() < deadline, "child remained running"
        time.sleep(0.05)


def test_worker_environment_excludes_host_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic")
    result = run_worker_subprocess(
        python("import os,json; print(json.dumps([key in os.environ for key in "
               "('OPENAI_API_KEY','DEEPSEEK_API_KEY')]))"), timeout_s=5,
    )
    assert result["returncode"] == 0
    assert json.loads(result["stdout"]) == [False, False]


def test_worker_owns_and_cleans_automatic_cwd_and_tmp() -> None:
    result = run_worker_subprocess(
        python("import os,json; print(json.dumps([os.getcwd(),os.environ['TMPDIR'],"
               "os.environ['TMP'],os.environ['TEMP']]))"), timeout_s=5,
    )
    assert result["returncode"] == 0
    cwd, tmp, short_tmp, temp = map(Path, json.loads(result["stdout"]))
    assert tmp == cwd / "tmp" == short_tmp == temp
    assert not cwd.exists()
    assert not tmp.exists()


def test_worker_preserves_exit_code_and_truncates_log_tail() -> None:
    result = run_worker_subprocess(
        python("import sys; print('a' * 70000 + 'END'); print('ERROR', file=sys.stderr); "
               "raise SystemExit(65)"), timeout_s=5,
    )
    assert result["returncode"] == 65
    assert not result["timed_out"] and not result["cancelled"]
    assert len(result["stdout"].encode()) == LOG_BYTES
    assert result["stdout"].endswith("END\n")
    assert result["stderr"] == "ERROR\n"


def test_worker_cancellation() -> None:
    event = Event()
    event.set()
    result = run_worker_subprocess(python("import time; time.sleep(60)"), timeout_s=5, cancel_event=event)
    assert result["cancelled"] and not result["timed_out"]
    assert result["returncode"] != 0


def test_public_worker_api_and_scientific_imports() -> None:
    from sandbox import run_worker_subprocess as public_worker
    from tasks.lob_world_model.gate import CandidateGate
    from stochopia.neural.primitives import seed_lob_source

    assert public_worker is run_worker_subprocess
    imports = "import typing\nimport functools\nimport torch.fft\n"
    source = seed_lob_source().replace("from __future__ import annotations\n",
                                      "from __future__ import annotations\n" + imports, 1)
    assert CandidateGate()(source) is None
