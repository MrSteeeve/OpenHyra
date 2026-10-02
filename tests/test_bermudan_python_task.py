"""Functional contract tests for the Bermudan whole-program task."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness import Task
from harness_v5 import adapt_bermudan_metrics, get_metrics_adapter
from sandbox import run_solution, snapshot_algorithm_source
from tasks.bermudan_optimal_stopping import evaluator
from tasks.bermudan_optimal_stopping.policy_protocols import (
    PYTHON_PROGRAM_SCHEMA,
    PythonProgramManifest,
    load_candidate_manifest,
)
from tasks.bermudan_optimal_stopping.training_pipeline import (
    training_instance_payload,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks" / "bermudan_python_search"


class BermudanPythonTaskTests(unittest.TestCase):
    def test_program_source_snapshot_keeps_task_owned_source_tree(self):
        task = Task("bermudan_python_search", "source-filter-program")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "algorithm.py").write_text("def main():\n    pass\n")
            (source / "manifest.json").write_text("{}\n")
            (source / "solve.sh").write_text("cp manifest.json solution.json\n")
            (source / "unregistered_helper.py").write_text("VALUE = 1\n")

            filtered = snapshot_algorithm_source(
                source,
                root / "filtered",
                task,
                task.max_output_mb * 1024 * 1024,
            )

            self.assertEqual(
                {path.name for path in filtered.iterdir()},
                {"algorithm.py", "manifest.json", "unregistered_helper.py"},
            )
            self.assertFalse((filtered / "solve.sh").exists())
            self.assertEqual(
                (filtered / "algorithm.py").read_text(),
                "def main():\n    pass\n",
            )

    def test_seed_fits_and_predicts_through_the_open_program_contract(self):
        task = Task("bermudan_python_search", "program-contract")
        self.assertEqual(task.protocol, "bermudan-python-program-search.v1")
        self.assertEqual(task.candidate_mode, "python_program")
        self.assertEqual(task.editable_files, ["algorithm.py", "manifest.json"])
        self.assertEqual(task.candidate_entrypoint, "algorithm.py")
        self.assertEqual(task.artifact_protocol, PYTHON_PROGRAM_SCHEMA)
        self.assertEqual(task.artifact_protocols, (PYTHON_PROGRAM_SCHEMA,))
        self.assertIs(
            get_metrics_adapter("bermudan-python-program-search.v1"),
            adapt_bermudan_metrics,
        )

        instance = evaluator.public_suite()[0]
        paths = evaluator.simulate_paths(instance, 48, 20260902)
        payoffs = evaluator.discounted_rewards(paths, instance)
        discounts = np.exp(
            -instance.rate * np.asarray(instance.exercise_times, dtype=np.float64)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            shutil.copytree(TASK_DIR / "seed_solution", source)
            subprocess.run(["bash", "solve.sh"], cwd=source, check=True)
            manifest = load_candidate_manifest(source / "manifest.json")
            self.assertIsInstance(manifest, PythonProgramManifest)
            self.assertEqual(manifest.interface, "continuation")

            input_dir = root / "training"
            model_dir = root / "model"
            input_dir.mkdir()
            np.save(input_dir / "training_paths.npy", paths, allow_pickle=False)
            np.save(input_dir / "payoffs.npy", payoffs, allow_pickle=False)
            np.save(input_dir / "discount_factors.npy", discounts, allow_pickle=False)
            (input_dir / "instance.json").write_text(
                json.dumps(training_instance_payload(instance)),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(source / "algorithm.py"),
                    "fit",
                    "--input",
                    str(input_dir),
                    "--output",
                    str(model_dir),
                    "--seed",
                    "17",
                ],
                check=True,
            )
            self.assertTrue((model_dir / "model.npz").is_file())

            query_dir = root / "query"
            result_dir = root / "result"
            query_dir.mkdir()
            time_index = 1
            np.save(query_dir / "states.npy", paths[:7, time_index, :], allow_pickle=False)
            np.save(
                query_dir / "history.npy",
                paths[:7, : time_index + 1, :],
                allow_pickle=False,
            )
            np.save(
                query_dir / "immediate_payoffs.npy",
                payoffs[:7, time_index],
                allow_pickle=False,
            )
            (query_dir / "request.json").write_text(
                json.dumps({"time_index": time_index}),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(source / "algorithm.py"),
                    "predict",
                    "--model",
                    str(model_dir),
                    "--input",
                    str(query_dir),
                    "--output",
                    str(result_dir),
                ],
                check=True,
            )
            predictions = np.load(result_dir / "predictions.npy", allow_pickle=False)
            self.assertEqual(predictions.shape, (7,))
            self.assertTrue(np.all(np.isfinite(predictions)))

    def test_harness_passes_the_sealed_program_to_the_real_evaluator(self):
        task = Task("bermudan_python_search", "program-harness-path")
        task.search_evaluation_request = {
            "schema": evaluator.REQUEST_SCHEMA,
            "stage": "search",
            "task": "bermudan_python_search",
            "protocol": "bermudan-python-program-search.v1",
            "seed": 1729,
            "suite_id": "program-harness-functional",
            "config": {
                "instance_count": 1,
                "repeats": 1,
                "training_paths": 64,
                "pricing_paths": 64,
                "training_timeout_s": 10,
                "prediction_timeout_s": 10,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            score, status, log_tail, metrics = run_solution(
                task.seed_dir,
                Path(temporary) / "sandbox",
                task,
            )

        self.assertEqual(status, "ok", log_tail)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(metrics["candidate_kind"], "python_program")
        self.assertEqual(metrics["policy_interface"], "continuation")
        self.assertEqual(metrics["entrypoint"], "algorithm.py")

    def test_evaluator_cli_accepts_a_python_program_directory(self):
        request = {
            "schema": evaluator.REQUEST_SCHEMA,
            "stage": "search",
            "task": "bermudan_python_search",
            "protocol": "bermudan-python-program-search.v1",
            "seed": 2718,
            "suite_id": "program-directory-functional",
            "config": {
                "instance_count": 1,
                "repeats": 1,
                "training_paths": 64,
                "pricing_paths": 64,
                "training_timeout_s": 10,
                "prediction_timeout_s": 10,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            source = temporary_root / "source"
            source.mkdir()
            shutil.copy2(TASK_DIR / "seed_solution" / "algorithm.py", source)
            shutil.copy2(TASK_DIR / "seed_solution" / "manifest.json", source)
            request_path = temporary_root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TASK_DIR / "evaluator.py"),
                    str(source),
                    str(request_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertNotIn("error", payload)
        self.assertEqual(payload["metrics"]["candidate_kind"], "python_program")


if __name__ == "__main__":
    unittest.main()
