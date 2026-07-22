import hashlib
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eb import ExperienceBank
from harness import (
    _known_solver_issues,
    _next_context_iteration,
    check_frozen,
    run_pipeline,
)
from proposal_agent import prepare_draft
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


class DraftIsolationTests(unittest.TestCase):
    def test_draft_copies_code_without_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            draft = root / "draft"
            parent.mkdir()
            (parent / "solver.py").write_text("print('parent')\n")
            (parent / "solution.json").write_text('{"A":[0,1,3,7]}')

            prepare_draft(parent, draft)

            self.assertEqual((draft / "solver.py").read_text(), "print('parent')\n")
            self.assertFalse((draft / "solution.json").exists())
            self.assertEqual(check_frozen(parent, draft, ["solver.py"]), [])


class CandidatePipelineTests(unittest.TestCase):
    def test_next_context_iteration_ignores_candidates_per_context(self):
        records = [
            {"metadata": {}},
            *({"metadata": {"iteration": 0}} for _ in range(4)),
            *({"metadata": {"iteration": 1}} for _ in range(4)),
        ]
        self.assertEqual(_next_context_iteration(records), 2)


    def test_preflight_detects_unclamped_fractional_progress_power(self):
        with tempfile.TemporaryDirectory() as temporary:
            draft = Path(temporary)
            solver = draft / "solver.py"
            solver.write_text(
                "progress = elapsed / budget\n"
                "temperature = max(0.01, (1.0 - progress) ** 1.5)\n"
            )
            self.assertIn("without clamping", _known_solver_issues(draft, ["solver.py"])[0])

            solver.write_text(
                "progress = min(1.0, max(0.0, elapsed / budget))\n"
                "temperature = max(0.01, (1.0 - progress) ** 1.5)\n"
            )
            self.assertEqual(_known_solver_issues(draft, ["solver.py"]), [])

    def test_preflight_requires_nonempty_dynamic_randrange_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            draft = Path(temporary)
            solver = draft / "solver.py"
            solver.write_text("value = rng.randrange(width, new_width)\n")
            self.assertIn("proving stop > start", _known_solver_issues(draft, ["solver.py"])[0])

            solver.write_text(
                "if new_width > width:\n"
                "    value = rng.randrange(width, new_width)\n"
            )
            self.assertEqual(_known_solver_issues(draft, ["solver.py"]), [])

    def test_pipeline_commits_all_four_outcomes_for_one_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solver.py").write_text("print('seed')\n")
            (parent_dir / "solution.json").write_text('{"A":[0,1,3,7]}')
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(
                parent_dir, 1.0, "ok", "seed", None, "", metrics={"n": 4},
            )
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=2,
                candidates_per_context=4,
                candidate_repair_attempts=0,
                editable_files=["solver.py"],
                direction="max",
                protocol="test-v1",
                run_id="test",
            )

            def fake_context(*_args, **_kwargs):
                return parent, "seed=__OPENHYRA_CANDIDATE_SEED__", "vary", {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 9,
                    "direction": "vary",
                }

            def fake_propose(parent_path, draft, prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                commented_prompt = "\n".join(f"# {line}" for line in prompt.splitlines())
                (draft / editable_files[0]).write_text(commented_prompt + "\n")
                return True, "candidate"

            scores = [1.01, None, 1.09, 1.03]

            def fake_run_solution(_draft, sandbox, _task):
                index = int(Path(sandbox).name.split("_")[1])
                Path(sandbox).mkdir(parents=True)
                if index == 1:
                    (Path(sandbox) / "run.log").write_text("candidate crashed\n")
                    return None, "crash", "candidate crashed", {"candidate": index}
                (Path(sandbox) / "run.log").write_text("ok\n")
                (Path(sandbox) / "evaluated_solution.json").write_text(
                    json.dumps({"A": [0, index + 2]}),
                )
                return scores[index], "ok", "ok", {"candidate": index}

            with (patch("harness.build_inspiration", side_effect=fake_context),
                  patch("harness.propose", side_effect=fake_propose),
                  patch("harness.run_solution", side_effect=fake_run_solution)):
                run_pipeline(
                    task, bank, iterations=1, workers=2, backend="codex",
                    model="test", trial_seed=9,
                )

            records = bank.records()
            self.assertEqual(len(records), 5)
            candidates = [record for record in records if record["parent"] is not None]
            self.assertEqual(
                {record["metadata"]["candidate_index"] for record in candidates},
                {0, 1, 2, 3},
            )
            failed = next(
                record for record in candidates
                if record["metadata"]["candidate_index"] == 1
            )
            self.assertEqual(failed["status"], "crash")
            self.assertIsNone(failed["score"])
            self.assertIn("candidate crashed", failed["log_tail"])
            self.assertEqual(bank.best()["score"], 1.09)
            self.assertTrue(all(
                record["metadata"]["candidate_count"] == 4
                for record in candidates
            ))

    def test_single_candidate_can_repair_one_runtime_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solver.py").write_text("print('seed')\n")
            (parent_dir / "solution.json").write_text('{"A":[0,1,3,7]}')
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(
                parent_dir, 1.0, "ok", "seed", None, "",
                metrics={"n": 4, "set_hash": "parent"},
            )
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=1,
                candidate_repair_attempts=1,
                editable_files=["solver.py"],
                direction="max",
                protocol="test-v1",
                run_id="test-repair",
            )

            def fake_context(*_args, **_kwargs):
                return parent, "seed=__OPENHYRA_CANDIDATE_SEED__", "repair", {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 3,
                    "direction": "repair",
                }

            def fake_propose(parent_path, draft, _prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                (draft / editable_files[0]).write_text("print('broken')\n")
                return True, "candidate with a repairable bug"

            def fake_repair(draft, feedback, editable_files, **_kwargs):
                self.assertIn("TypeError", feedback)
                (Path(draft) / editable_files[0]).write_text("print('fixed')\n")
                return True, "clamped progress"

            evaluations = []

            def fake_run_solution(_draft, sandbox, _task):
                attempt = len(evaluations)
                evaluations.append(attempt)
                sandbox = Path(sandbox)
                sandbox.mkdir(parents=True, exist_ok=True)
                if attempt == 0:
                    (sandbox / "run.log").write_text("TypeError: complex temperature\n")
                    return None, "crash", "TypeError: complex temperature", {
                        "solver_seconds": 1.0,
                    }
                (sandbox / "run.log").write_text("fixed\n")
                (sandbox / "evaluated_solution.json").write_text('{"A":[0,1,4]}')
                return 1.2, "ok", "fixed", {
                    "set_hash": "novel", "solver_seconds": 2.0,
                }

            with (patch("harness.build_inspiration", side_effect=fake_context),
                  patch("harness.propose", side_effect=fake_propose),
                  patch("harness.repair_candidate", side_effect=fake_repair),
                  patch("harness.run_solution", side_effect=fake_run_solution)):
                run_pipeline(
                    task, bank, iterations=1, workers=1, backend="codex",
                    model="test", trial_seed=3,
                    candidates_per_context=1,
                )

            candidate = bank.records()[-1]
            self.assertEqual(candidate["status"], "ok")
            self.assertEqual(candidate["metadata"]["repair_count"], 1)
            self.assertEqual(
                [attempt["status"] for attempt in candidate["metadata"]["attempts"]],
                ["crash", "ok"],
            )
            self.assertIn(
                "TypeError", candidate["metadata"]["attempts"][0]["log_tail"],
            )
            self.assertEqual(len(evaluations), 2)

    def test_preflight_repairs_known_progress_bug_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_dir = root / "parent"
            parent_dir.mkdir()
            (parent_dir / "solver.py").write_text("print('seed')\n")
            (parent_dir / "solution.json").write_text('{"A":[0,1,3,7]}')
            bank = ExperienceBank(root / "eb", direction="max")
            parent = bank.commit(parent_dir, 1.0, "ok", "seed", None, "")
            task = SimpleNamespace(
                run_dir=root / "run",
                eval_concurrency=1,
                candidates_per_context=1,
                candidate_repair_attempts=1,
                editable_files=["solver.py"],
                direction="max",
                protocol="test-v1",
                run_id="test-preflight",
            )

            def fake_context(*_args, **_kwargs):
                return parent, "seed=__OPENHYRA_CANDIDATE_SEED__", "preflight", {
                    "iteration": 0,
                    "eb_version": 1,
                    "visible_solution_ids": [parent["id"]],
                    "trial_seed": 4,
                    "direction": "preflight",
                }

            def fake_propose(parent_path, draft, _prompt, editable_files, **_kwargs):
                prepare_draft(parent_path, draft)
                (draft / editable_files[0]).write_text(
                    "progress = elapsed / budget\n"
                    "temperature = max(0.01, (1.0 - progress) ** 1.5)\n"
                )
                return True, "unsafe annealing schedule"

            def fake_repair(draft, feedback, editable_files, **_kwargs):
                self.assertIn("Engineering preflight rejected", feedback)
                (Path(draft) / editable_files[0]).write_text(
                    "progress = min(1.0, max(0.0, elapsed / budget))\n"
                    "temperature = max(0.01, (1.0 - progress) ** 1.5)\n"
                )
                return True, "clamped progress"

            def fake_run_solution(draft, sandbox, _task):
                self.assertEqual(_known_solver_issues(draft, ["solver.py"]), [])
                sandbox = Path(sandbox)
                sandbox.mkdir(parents=True, exist_ok=True)
                (sandbox / "run.log").write_text("ok\n")
                (sandbox / "evaluated_solution.json").write_text('{"A":[0,1,4]}')
                return 1.2, "ok", "ok", {"set_hash": "safe"}

            with (patch("harness.build_inspiration", side_effect=fake_context),
                  patch("harness.propose", side_effect=fake_propose),
                  patch("harness.repair_candidate", side_effect=fake_repair),
                  patch("harness.run_solution", side_effect=fake_run_solution)):
                run_pipeline(
                    task, bank, iterations=1, workers=1, backend="codex",
                    model="test", trial_seed=4,
                    candidates_per_context=1,
                )

            candidate = bank.records()[-1]
            self.assertEqual(candidate["status"], "ok")
            self.assertTrue(candidate["metadata"]["preflight_notes"])
            self.assertEqual(len(candidate["metadata"]["attempts"]), 1)

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
