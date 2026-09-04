import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from auditing import _algorithm_bundle_digest, run_final_audit
from eb import ExperienceBank
from provenance import sha256_json
from sandbox import source_tree_hash
from stopping import write_termination


class AlgorithmBundlePrivateAuditTests(unittest.TestCase):
    def test_final_audit_freezes_and_passes_source_bundle_to_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            evaluator = root / "evaluator.py"
            evaluator.write_text(
                "import hashlib, json, sys\n"
                "from pathlib import Path\n"
                "artifact = json.loads(Path(sys.argv[1]).read_text())\n"
                "assert sys.argv[3] == '--candidate-source'\n"
                "source = Path(sys.argv[4])\n"
                "files = []\n"
                "for name in sorted(('manifest.json', 'train.py')):\n"
                "    data = (source / name).read_bytes()\n"
                "    files.append({'path': name, 'size_bytes': len(data),\n"
                "                  'sha256': hashlib.sha256(data).hexdigest()})\n"
                "digest = hashlib.sha256(json.dumps({\n"
                "    'schema': 'openhyra-algorithm-bundle.v1',\n"
                "    'files': files}, sort_keys=True, separators=(',', ':'),\n"
                "    ensure_ascii=False).encode()).hexdigest()\n"
                "print(json.dumps({'score': artifact['audit_score'],\n"
                " 'normalized_solution': artifact,\n"
                " 'evidence': {'source_seen': True},\n"
                " 'metrics': {'algorithm_bundle_sha256': digest}}))\n"
            )
            task = SimpleNamespace(
                run_dir=run_dir,
                run_id="algorithm-audit",
                name="algorithm_task",
                protocol="bermudan-lsmc-python.v1",
                direction="max",
                candidate_mode="algorithm_bundle",
                candidate_source_files=("train.py", "manifest.json"),
                artifact_protocol="openhyra-policy-spec.v1",
                evaluation={
                    "audit_stage": {"top_k": 1, "direction": "max"},
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
            source = root / "candidate"
            source.mkdir(parents=True)
            (source / "train.py").write_text("print('candidate')\n")
            (source / "manifest.json").write_text(
                '{"schema":"continuation-expression.v1"}\n'
            )
            artifact = b'{"audit_score":7}'
            source_hash = source_tree_hash(source, 1024 * 1024)[0]
            bundle_hash = _algorithm_bundle_digest(source, task)
            bank = ExperienceBank(run_dir / "eb", direction="max")
            record = bank.commit(
                source, 1.0, "ok", "candidate", None, "",
                metrics={
                    "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                    "source_snapshot_sha256": source_hash,
                    "algorithm_bundle_sha256": bundle_hash,
                },
                metadata={
                    "run_manifest_sha256": run_manifest["manifest_sha256"],
                    "task_provenance": run_manifest["task"],
                    "source_sha256": run_manifest["source_sha256"],
                    "algorithm_bundle_sha256": bundle_hash,
                },
            )
            (Path(record["path"]) / "solution.json").write_bytes(artifact)
            write_termination(run_dir / "termination.json", {
                "reason": "iteration_limit",
                "terminal": False,
                "evidence": {"completed_contexts": 1},
            })

            report = run_final_audit(
                task, bank, run_manifest, seed_factory=lambda: 17,
            )

            frozen = report["freeze_manifest"]["candidates"][0]
            self.assertEqual(frozen["candidate_mode"], "algorithm_bundle")
            self.assertEqual(frozen["source_files"], ["train.py", "manifest.json"])
            self.assertEqual(frozen["algorithm_bundle_sha256"], bundle_hash)
            self.assertEqual(
                frozen["artifact_protocol"], "continuation-expression.v1",
            )
            frozen_source = run_dir / "final_audit_artifacts" / frozen["frozen_source"]
            self.assertEqual(
                _algorithm_bundle_digest(frozen_source, task), bundle_hash,
            )
            self.assertTrue(report["candidates"][0]["evidence"]["source_seen"])
            self.assertEqual(
                report["candidates"][0]["algorithm_bundle_sha256"], bundle_hash,
            )
            self.assertEqual(
                report["candidates"][0]["artifact_protocol"],
                "continuation-expression.v1",
            )


if __name__ == "__main__":
    unittest.main()
