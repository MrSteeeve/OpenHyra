"""Execute the P1 zero-data checks through the existing training sandbox."""

from __future__ import annotations

import shutil
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path

from sandbox import read_regular_file, run_training_sandbox


def _runtime_roots() -> tuple[Path, ...]:
    candidates = (Path(sys.base_prefix), Path(sys.prefix),
                  Path(sys.executable).resolve().parent, Path("/System/Library"),
                  Path("/usr/lib"))
    roots: list[Path] = []
    for path in candidates:
        root = path.resolve()
        if root.is_dir() and not any(root == item or item in root.parents for item in roots):
            roots.append(root)
    return tuple(roots)


def _stage_neural_runtime(source_dir: Path) -> None:
    """Make only the two approved companion modules visible to the candidate."""
    location = find_spec("stochopia.neural.lob_interface")
    if location is None or location.origin is None:
        raise RuntimeError("P1 gate requires the Stochopia neural runtime")
    implementation = Path(location.origin).parent
    package = source_dir / "stochopia" / "neural"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    for name in ("lob_interface.py", "primitives.py"):
        shutil.copyfile(implementation / name, package / name)


class CandidateGate:
    """Run a candidate from an empty data directory with network denied."""

    def __init__(self, *, timeout_s: float = 20.0) -> None:
        self.timeout_s = timeout_s

    def __call__(self, source: str) -> str | None:
        with tempfile.TemporaryDirectory(prefix="stochopia-p1-gate-") as directory:
            root = Path(directory).resolve()
            source_dir = root / "source"
            input_dir = root / "input"
            output_dir = root / "output"
            scratch = root / "scratch"
            for path in (source_dir, input_dir, output_dir, scratch):
                path.mkdir()
            _stage_neural_runtime(source_dir)
            worker = source_dir / "gate_worker.py"
            shutil.copyfile(Path(__file__).with_name("gate_worker.py"), worker)
            candidate = source_dir / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            completion = output_dir / "checked"
            result = run_training_sandbox(
                [sys.executable, "-I", str(worker), str(candidate), str(completion)],
                source_dir=source_dir, input_dir=input_dir, output_dir=output_dir,
                tmp_dir=scratch, runtime_roots=_runtime_roots(), timeout_s=self.timeout_s,
                memory_bytes=2 * 1024 ** 3,
            )
            if result["status"] == "ok":
                try:
                    value = read_regular_file(completion, 3, label="P1 gate completion")
                except ValueError:
                    return "Candidate exited before completing its tensor checks"
                return None if value == b"ok\n" else "Candidate did not complete its tensor checks"
            trace = result["log_tail"].strip()
            trace = trace.replace(str(root), "<sandbox>").replace(str(Path.home()), "~")
            return f"Sandbox {result['status']} (returncode {result['returncode']})\n{trace[-3500:]}"
