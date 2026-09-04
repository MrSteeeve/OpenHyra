"""Focused contract smoke test for the additive Bermudan Python task."""

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
from sandbox import snapshot_algorithm_source
from tasks.bermudan_optimal_stopping import evaluator
from tasks.bermudan_optimal_stopping.policy_artifact import (
    MLPContinuationRunner,
    load_policy_artifact,
)
from tasks.bermudan_optimal_stopping.policy_protocols import (
    load_continuation_manifest,
)
from tasks.bermudan_optimal_stopping.training_pipeline import (
    training_instance_payload,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks" / "bermudan_python_search"


class BermudanPythonTaskTests(unittest.TestCase):
    def test_algorithm_source_snapshot_filters_solver_plumbing_and_helpers(self):
        task = Task("bermudan_python_search", "source-filter-smoke")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "train.py").write_text("print('train')\n")
            (source / "manifest.json").write_text("{}\n")
            (source / "solve.sh").write_text("cp manifest.json solution.json\n")
            (source / "helper.py").write_text("VALUE = 1\n")

            filtered = snapshot_algorithm_source(
                source,
                root / "filtered",
                task,
                task.max_output_mb * 1024 * 1024,
            )

            self.assertEqual(
                {path.name for path in filtered.iterdir()},
                {"train.py", "manifest.json"},
            )
            self.assertEqual(
                (filtered / "train.py").read_text(),
                "print('train')\n",
            )
            self.assertEqual((filtered / "train.py").stat().st_mode & 0o777, 0o400)

    def test_task_contract_and_seed_emit_loadable_mlp_artifact(self):
        task = Task("bermudan_python_search", "contract-smoke")
        self.assertEqual(task.name, "bermudan_python_search")
        self.assertEqual(task.protocol, "bermudan-lsmc-algorithm-bundle.v1")
        self.assertEqual(task.candidate_mode, "algorithm_bundle")
        self.assertEqual(task.editable_files, ["train.py", "manifest.json"])
        self.assertEqual(task.candidate_source_files, ("train.py", "manifest.json"))
        self.assertEqual(task.candidate_entrypoint, "train.py")
        self.assertEqual(task.solve_entrypoint, "solve.sh")
        self.assertGreaterEqual(task.evaluator_timeout_s, 600)
        self.assertIs(
            get_metrics_adapter("bermudan-lsmc-algorithm-bundle.v1"),
            adapt_bermudan_metrics,
        )
        adapted = adapt_bermudan_metrics({
            "artifact_sha256": "a" * 64,
            "algorithm_bundle_sha256": "b" * 64,
        })
        self.assertEqual(adapted["artifact_sha256"], "b" * 64)

        instance = evaluator.public_suite()[0]
        paths = evaluator.simulate_paths(instance, 32, 20260902)
        payoffs = evaluator.discounted_rewards(paths, instance)
        discounts = np.exp(
            -instance.rate * np.asarray(instance.exercise_times, dtype=np.float64)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            shutil.copyfile(TASK_DIR / "seed_solution" / "train.py", source / "train.py")
            shutil.copyfile(TASK_DIR / "seed_solution" / "manifest.json", source / "manifest.json")
            shutil.copyfile(TASK_DIR / "seed_solution" / "solve.sh", source / "solve.sh")

            subprocess.run(["bash", "solve.sh"], cwd=source, check=True)
            self.assertEqual(
                (source / "solution.json").read_bytes(),
                (source / "manifest.json").read_bytes(),
            )

            input_dir, output_dir = root / "input", root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            np.save(input_dir / "training_paths.npy", paths, allow_pickle=False)
            np.save(input_dir / "payoffs.npy", payoffs, allow_pickle=False)
            np.save(input_dir / "discount_factors.npy", discounts, allow_pickle=False)
            (input_dir / "instance.json").write_text(
                json.dumps(
                    training_instance_payload(instance),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            subprocess.run(
                [
                    sys.executable,
                    str(source / "train.py"),
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output_dir),
                    "--seed",
                    "17",
                ],
                check=True,
            )

            expected = {
                "normalization.json",
                *(f"step_{i:03d}.npy" for i in range(len(instance.exercise_times) - 1)),
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
            manifest = load_continuation_manifest(source / "manifest.json")
            artifact = load_policy_artifact(
                manifest,
                output_dir,
                n_exercise_times=len(instance.exercise_times),
                input_dim=instance.dimension,
            )
            values = MLPContinuationRunner(artifact).continuation(
                0, np.asarray([[0.9], [1.1]], dtype=np.float64),
            )
            self.assertEqual(values.shape, (2,))
            self.assertTrue(np.all(np.isfinite(values)))


if __name__ == "__main__":
    unittest.main()
