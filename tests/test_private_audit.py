import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auditing import run_final_audit, select_top_k
from eb import ExperienceBank
from harness import Task, _assemble_commit_snapshot
from provenance import (
    build_run_manifest,
    build_evaluation_request,
    derive_search_seed,
    sha256_json,
)
from sandbox import run_solution, source_tree_hash, trusted_artifact_dir
from stopping import write_termination


class PrivateAuditTests(unittest.TestCase):
    def _fixture(self, temporary, *, termination_reason="iteration_limit"):
        root = Path(temporary)
        run_dir = root / "run"
        evaluator = root / "evaluator.py"
        evaluator.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "artifact=json.loads(Path(sys.argv[1]).read_text())\n"
            "request=json.loads(Path(sys.argv[2]).read_text())\n"
            "assert set(request) == "
            "{'schema','stage','task','protocol','seed','suite_id','config'}\n"
            "assert request['stage'] == 'audit'\n"
            "assert request['config'] == {'direction': 'min'}\n"
            "print(json.dumps({\n"
            " 'score': artifact['audit_score'],\n"
            " 'normalized_solution': artifact,\n"
            " 'evidence': {'kind': 'private-test-evidence'},\n"
            " 'metrics': {\n"
            "   'seed_seen': request['seed'],\n"
            "   'omp_threads': os.environ.get('OMP_NUM_THREADS'),\n"
            "   'suite_id': request['suite_id'],\n"
            " }\n"
            "}))\n"
        )
        task = SimpleNamespace(
            run_dir=run_dir,
            run_id="audit-test",
            name="test_task",
            protocol="test-protocol.v1",
            direction="max",
            evaluation={
                "search_stage": {"suite_id": "public-suite"},
                "audit_stage": {
                    "suite_id": "private-suite",
                    "top_k": 2,
                    "direction": "min",
                },
            },
            evaluator=evaluator,
            evaluator_timeout_s=10,
            evaluator_max_memory_mb=256,
            max_output_mb=8,
            max_artifact_bytes=65536,
        )
        run_manifest = {
            "manifest_sha256": "a" * 64,
            "task": {"name": task.name, "protocol": task.protocol},
            "source_sha256": {"harness.py": "b" * 64},
        }
        bank = ExperienceBank(run_dir / "eb", direction=task.direction)
        if termination_reason == "agent_converged":
            terminal = True
        else:
            terminal = False
        write_termination(run_dir / "termination.json", {
            "reason": termination_reason,
            "terminal": terminal,
            "evidence": {"completed_contexts": 3, "sentinel": "preserve-me"},
        })
        return task, bank, run_manifest

    def _commit(self, task, bank, run_manifest, score, audit_score):
        source = Path(tempfile.mkdtemp(dir=task.run_dir.parent))
        artifact_bytes = json.dumps(
            {"audit_score": audit_score},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        (source / "solution.json").write_bytes(artifact_bytes)
        source_hash = source_tree_hash(source, 1024 * 1024)[0]
        return bank.commit(
            source, score, "ok", "candidate", None, "",
            metrics={
                "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "source_snapshot_sha256": source_hash,
            },
            metadata={
                "run_manifest_sha256": run_manifest["manifest_sha256"],
                "task_provenance": run_manifest["task"],
                "source_sha256": run_manifest["source_sha256"],
            },
        )

    def test_search_seed_is_run_level_deterministic_and_request_is_exact(self):
        task = SimpleNamespace(
            name="task", protocol="protocol",
            evaluation={"search_stage": {"suite_id": "shared-crn"}},
        )
        first = derive_search_seed(task, 17)
        self.assertEqual(first, derive_search_seed(task, 17))
        self.assertNotEqual(first, derive_search_seed(task, 18))
        request = build_evaluation_request(task, "search", first)
        self.assertEqual(set(request), {
            "schema", "stage", "task", "protocol", "seed", "suite_id",
            "config",
        })
        self.assertEqual(request["seed"], first)
        self.assertEqual(request["suite_id"], "shared-crn")
        self.assertEqual(request["config"], {})

    def test_top_k_deduplicates_normalized_artifacts(self):
        records = [
            {"id": "a", "status": "ok", "score": 10.0,
             "metrics": {"artifact_sha256": "x"}},
            {"id": "a-copy", "status": "ok", "score": 9.0,
             "metrics": {"artifact_sha256": "x"}},
            {"id": "b", "status": "ok", "score": 8.0,
             "metrics": {"artifact_sha256": "y"}},
        ]
        self.assertEqual(
            [item["id"] for item in select_top_k(records, "max", 2)],
            ["a", "b"],
        )

    def test_search_request_reaches_evaluator_and_hash_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "solve.sh").write_text(
                '#!/bin/bash\nexec "$OPENHYRA_PYTHON" solver.py\n'
            )
            (source / "solver.py").write_text(
                "from pathlib import Path\n"
                "Path('solution.json').write_text('{\"x\":1}')\n"
            )
            evaluator = root / "evaluator.py"
            evaluator.write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "request=json.loads(Path(sys.argv[2]).read_text())\n"
                "print(json.dumps({'score': request['seed'], 'metrics': "
                "{'threads': os.environ['OMP_NUM_THREADS']}}))\n"
            )
            task = SimpleNamespace(
                evaluator=evaluator,
                python_bin=sys.executable,
                timeout_s=10,
                max_memory_mb=256,
                max_output_mb=8,
                max_artifact_bytes=65536,
                evaluator_timeout_s=10,
                evaluator_max_memory_mb=256,
                search_evaluation_request={
                    "schema": "openhyra-evaluation-request.v1",
                    "stage": "search",
                    "task": "test",
                    "protocol": "test.v1",
                    "seed": 99,
                    "suite_id": "public",
                    "config": {},
                },
            )
            with patch.dict(
                os.environ, {"OPENHYRA_ALLOW_UNSANDBOXED": "1"},
            ):
                score, status, _tail, metrics = run_solution(
                    source, root / "sandbox", task,
                )
            self.assertEqual((score, status), (99.0, "ok"))
            self.assertEqual(metrics["threads"], "1")
            self.assertEqual(
                metrics["evaluation_request_sha256"],
                sha256_json(task.search_evaluation_request),
            )
            self.assertFalse(
                (trusted_artifact_dir(root / "sandbox")
                 / "evaluation_request.json").exists()
            )

    def test_real_bermudan_search_and_private_audit_integrate(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"OPENHYRA_ALLOW_UNSANDBOXED": "1"},
        ):
            task = Task("bermudan_optimal_stopping", "integration-test")
            task.run_dir = Path(temporary) / "run"
            manifest = build_run_manifest(
                task, Path(__file__).resolve().parents[1],
                backend="codex", model="test", workers=1,
                candidates_per_context=1, trial_seed=123,
                stopping_policy={},
            )
            task.run_manifest = manifest
            task.search_evaluation_request = manifest["search"][
                "evaluation_request"
            ]
            sandbox = task.run_dir / "sandboxes" / "seed"
            score, status, _tail, metrics = run_solution(
                task.seed_dir, sandbox, task,
            )
            self.assertEqual(status, "ok")
            self.assertAlmostEqual(score, 0.0)
            committed = _assemble_commit_snapshot(
                task.seed_dir, sandbox, task, metrics,
            )
            bank = ExperienceBank(task.run_dir / "eb", direction=task.direction)
            bank.commit(
                committed, score, status, task.seed_description, None, "",
                metrics=metrics,
                metadata={
                    "run_manifest_sha256": manifest["manifest_sha256"],
                    "task_provenance": manifest["task"],
                    "source_sha256": manifest["source_sha256"],
                },
            )
            write_termination(task.run_dir / "termination.json", {
                "reason": "iteration_limit",
                "terminal": False,
                "evidence": {"completed_contexts": 0},
            })
            report = run_final_audit(
                task, bank, manifest, seed_factory=lambda: 987654321,
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["candidates"][0]["status"], "ok")
            self.assertEqual(
                report["candidates"][0]["metrics"]["stage"], "audit",
            )
            self.assertEqual(
                report["candidates"][0]["normalized_solution_sha256"],
                report["candidates"][0]["artifact_sha256"],
            )

    def test_final_audit_freezes_before_seed_and_never_mutates_eb(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, bank, manifest = self._fixture(temporary)
            first = self._commit(task, bank, manifest, 10.0, 5.0)
            duplicate = self._commit(task, bank, manifest, 9.0, 5.0)
            second = self._commit(task, bank, manifest, 8.0, 1.0)
            # Make the middle record a normalized duplicate of the first.
            records_path = bank.records_path
            records = bank.records()
            records[1]["metrics"]["artifact_sha256"] = records[0]["metrics"][
                "artifact_sha256"
            ]
            duplicate_path = Path(duplicate["path"]) / "solution.json"
            duplicate_path.write_bytes(
                (Path(first["path"]) / "solution.json").read_bytes()
            )
            records_path.write_text(
                "".join(json.dumps(item) + "\n" for item in records)
            )
            before = bank.records_path.read_bytes()

            def seed_factory():
                freeze = task.run_dir / "final_audit_artifacts" / "manifest.json"
                self.assertTrue(freeze.is_file())
                frozen = json.loads(freeze.read_text())
                self.assertEqual(
                    [item["id"] for item in frozen["candidates"]],
                    [first["id"], second["id"]],
                )
                return 42

            report = run_final_audit(
                task, bank, manifest, seed_factory=seed_factory,
                now="2026-08-13T00:00:00+0800",
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["winner"]["id"], second["id"])
            self.assertEqual(report["evaluation_request"]["seed"], 42)
            self.assertEqual(
                report["evaluation_request_sha256"],
                sha256_json(report["evaluation_request"]),
            )
            self.assertEqual(bank.records_path.read_bytes(), before)
            self.assertEqual({item["metrics"]["seed_seen"] for item in report[
                "candidates"
            ]}, {42})
            self.assertEqual({item["metrics"]["omp_threads"] for item in report[
                "candidates"
            ]}, {"1"})
            self.assertEqual(len({item["metrics"][
                "evaluation_request_sha256"
            ] for item in report["candidates"]}), 1)
            for item in report["freeze_manifest"]["candidates"]:
                self.assertFalse(Path(item["frozen_artifact"]).is_absolute())
            for item in report["candidates"]:
                self.assertEqual(
                    item["normalized_solution_sha256"],
                    item["artifact_sha256"],
                )
                self.assertEqual(
                    item["evidence_sha256"], sha256_json(item["evidence"]),
                )

            report_path = task.run_dir / "final_audit.json"
            self.assertEqual(report_path.stat().st_mode & 0o777, 0o600)
            saved = json.loads(report_path.read_text())
            self.assertEqual(saved["seed"], 42)
            termination = json.loads(
                (task.run_dir / "termination.json").read_text()
            )
            self.assertTrue(termination["terminal"])
            self.assertEqual(termination["reason"], "final_audit_complete")
            self.assertNotIn("seed", termination)
            self.assertEqual(termination["audit_winner_id"], second["id"])
            self.assertEqual(termination["audit_winner_score"], 1.0)
            self.assertEqual(
                termination["search_termination"]["evidence"]["sentinel"],
                "preserve-me",
            )
            self.assertEqual(
                termination["final_audit_file_sha256"],
                hashlib.sha256(report_path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                run_final_audit(task, bank, manifest, seed_factory=lambda: 43)

    def test_clean_agent_convergence_is_auditable(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, bank, manifest = self._fixture(
                temporary, termination_reason="agent_converged",
            )
            self._commit(task, bank, manifest, 1.0, 2.0)
            report = run_final_audit(
                task, bank, manifest, seed_factory=lambda: 7,
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(
                report["search_termination"]["reason"], "agent_converged",
            )

    def test_freeze_tampering_fails_before_seed_and_seals_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, bank, manifest = self._fixture(temporary)
            record = self._commit(task, bank, manifest, 1.0, 2.0)
            (Path(record["path"]) / "solution.json").write_text(
                '{"audit_score":999}'
            )
            called = []
            with self.assertRaisesRegex(RuntimeError, "artifact provenance"):
                run_final_audit(
                    task, bank, manifest,
                    seed_factory=lambda: called.append(True) or 7,
                )
            self.assertEqual(called, [])
            report = json.loads(
                (task.run_dir / "final_audit.json").read_text()
            )
            self.assertEqual(report["status"], "failed")
            self.assertIsNone(report["seed"])
            termination = json.loads(
                (task.run_dir / "termination.json").read_text()
            )
            self.assertTrue(termination["terminal"])
            self.assertEqual(termination["reason"], "final_audit_failed")


if __name__ == "__main__":
    unittest.main()
