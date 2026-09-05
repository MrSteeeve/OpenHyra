from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProbeResult:
    """Result of one probe check."""
    probe_id: str
    passed: bool
    expected: float | None = None
    actual: float | None = None
    error: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ProbeReport:
    """Aggregated probe suite report."""
    suite_id: str
    timestamp: str  # ISO 8601
    results: list[ProbeResult] = field(default_factory=list)
    all_passed: bool = False

    def to_dict(self) -> dict:
        d = {**self.__dict__}
        d["results"] = [r.to_dict() for r in self.results]
        return d


class ProbeSuite:
    """Runs integrity probes against a task evaluator.
    
    Each probe is a (solution_dict, expected_score, tolerance) triple.
    The suite runs the evaluator on each probe and checks the score.
    """

    def __init__(self, task_dir: Path, evaluator_timeout_s: int = 60):
        self.task_dir = Path(task_dir)
        self.evaluator_timeout_s = evaluator_timeout_s
        self._probes: list[dict] = []
        self._load_probes()

    def _load_probes(self) -> None:
        """Load probes from task_dir/probes.json if it exists."""
        probe_path = self.task_dir / "probes.json"
        if probe_path.exists():
            data = json.loads(probe_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._probes = data

    def add_probe(self, probe_id: str, solution: dict, expected_score: float,
                  tolerance: float = 1e-6) -> None:
        """Register a known-answer probe."""
        self._probes.append({
            "probe_id": probe_id,
            "solution": solution,
            "expected_score": expected_score,
            "tolerance": tolerance,
        })

    def run(self, evaluate_fn) -> ProbeReport:
        """Run all probes against evaluate_fn(solution_dict) -> (score, status, log, metrics).
        
        evaluate_fn should accept a solution dict and return (score, status, log, metrics).
        This is the evaluator's evaluate_submission or equivalent.
        
        Returns a ProbeReport. If any probe fails, all_passed is False.
        """
        import datetime
        suite_id = f"probe_{int(time.time())}"
        results = []
        
        for probe in self._probes:
            probe_id = probe["probe_id"]
            expected = probe["expected_score"]
            tolerance = probe.get("tolerance", 1e-6)
            start = time.monotonic()
            try:
                score, status, _log, _metrics = evaluate_fn(probe["solution"])
                elapsed = time.monotonic() - start
                if status != "ok":
                    results.append(ProbeResult(
                        probe_id=probe_id, passed=False,
                        expected=expected, actual=score,
                        error=f"evaluator returned status={status}",
                        elapsed_s=elapsed,
                    ))
                elif score is None:
                    results.append(ProbeResult(
                        probe_id=probe_id, passed=False,
                        expected=expected, actual=None,
                        error="evaluator returned None score",
                        elapsed_s=elapsed,
                    ))
                elif abs(float(score) - expected) > tolerance:
                    results.append(ProbeResult(
                        probe_id=probe_id, passed=False,
                        expected=expected, actual=float(score),
                        error=f"score mismatch: |{score} - {expected}| > {tolerance}",
                        elapsed_s=elapsed,
                    ))
                else:
                    results.append(ProbeResult(
                        probe_id=probe_id, passed=True,
                        expected=expected, actual=float(score),
                        elapsed_s=elapsed,
                    ))
            except Exception as exc:
                elapsed = time.monotonic() - start
                results.append(ProbeResult(
                    probe_id=probe_id, passed=False,
                    expected=expected, error=repr(exc),
                    elapsed_s=elapsed,
                ))
        
        report = ProbeReport(
            suite_id=suite_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            results=results,
            all_passed=all(r.passed for r in results),
        )
        return report

    def save_report(self, report: ProbeReport, output_dir: Path) -> Path:
        """Save probe report as JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{report.suite_id}.json"
        path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path
