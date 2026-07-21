import hashlib
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from eb import ExperienceBank
from sandbox import run_solution

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "tasks" / "sums_diffs" / "evaluator.py"
SPEC = importlib.util.spec_from_file_location("sums_diffs_evaluator", EVALUATOR_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class EvaluatorTests(unittest.TestCase):
    def test_official_simpletes_seed(self):
        values = [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29]
        score, metrics, normalized = EVALUATOR.evaluate_values(values)
        self.assertAlmostEqual(score, 1.0597930945472454, places=14)
        self.assertEqual(metrics["n"], 17)
        self.assertEqual(metrics["sums"], 59)
        self.assertEqual(metrics["diffs"], 55)
        self.assertEqual(normalized, values)

    def test_canonical_hash_removes_affine_symmetries(self):
        values = [0, 1, 3, 7]
        translated_scaled = [19 + 5 * value for value in values]
        reflected = [max(values) - value for value in values]
        expected = EVALUATOR.canonical_hash(values)
        self.assertEqual(expected, EVALUATOR.canonical_hash(translated_scaled))
        self.assertEqual(expected, EVALUATOR.canonical_hash(reflected))

    def test_simpletes_normalizes_integer_duplicates(self):
        score, metrics, normalized = EVALUATOR.evaluate_values(
            [3.0, 0, 1.0, 3, 0.0],
        )
        self.assertEqual(normalized, [0, 1, 3])
        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["sums"], 6)
        self.assertEqual(metrics["diffs"], 7)
        self.assertGreater(score, 0)

    def test_rejects_more_than_512_elements(self):
        with self.assertRaisesRegex(ValueError, "512"):
            EVALUATOR.evaluate_values(list(range(513)))


class ExperienceBankTests(unittest.TestCase):
    def test_concurrent_commits_are_complete_and_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solver.py").write_text("pass\n")
            bank = ExperienceBank(root / "eb", direction="max")

            def commit(index):
                bank.commit(
                    source, float(index), "ok", f"candidate {index}", None, "",
                    metrics={"artifact_sha256": hashlib.sha256(str(index).encode()).hexdigest()},
                )

            threads = [threading.Thread(target=commit, args=(index,)) for index in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            version, records = bank.snapshot()
            self.assertEqual(version, 20)
            self.assertEqual(len(records), 20)
            self.assertEqual({record["id"] for record in records}, {
                f"sol_{index:04d}" for index in range(20)
            })
            self.assertEqual(bank.best()["score"], 19.0)


@unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
class SandboxTests(unittest.TestCase):
    def test_background_writer_cannot_change_scored_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir()
            (source / "solve.sh").write_text(
                '#!/bin/bash\nexec "$OPENHYRA_PYTHON" solver.py\n'
            )
            (source / "solver.py").write_text(
                "import json, os, time\n"
                "seed=[0,1,2,4,5,9,12,13,14,16,17,21,24,25,26,28,29]\n"
                "with open('solution.json','w') as f: json.dump({'A':seed},f)\n"
                "pid=os.fork()\n"
                "if pid == 0:\n"
                "    time.sleep(0.5)\n"
                "    with open('solution.json','w') as f: json.dump({'A':[0,1]},f)\n"
                "    os._exit(0)\n"
            )
            stale = source / "solution.json"
            stale.write_text('{"A":[0,1]}')
            stale.chmod(0o444)
            task = SimpleNamespace(
                evaluator=EVALUATOR_PATH,
                python_bin=sys.executable,
                timeout_s=10,
                max_memory_mb=512,
                max_output_mb=8,
            )
            score, status, _tail, metrics = run_solution(
                source, root / "sandbox", task,
            )
            self.assertEqual(status, "ok")
            self.assertAlmostEqual(score, 1.0597930945472454, places=14)
            self.assertEqual(metrics["n"], 17)
            snapshot = json.loads((root / "sandbox" / "solution.snapshot.json").read_text())
            self.assertEqual(snapshot["A"], [0,1,2,4,5,9,12,13,14,16,17,21,24,25,26,28,29])
            evaluated = root / "sandbox" / "evaluated_solution.json"
            self.assertEqual(
                hashlib.sha256(evaluated.read_bytes()).hexdigest(),
                metrics["artifact_sha256"],
            )
            self.assertEqual(
                hashlib.sha256((root / "sandbox" / "solution.snapshot.json").read_bytes()).hexdigest(),
                metrics["candidate_artifact_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
