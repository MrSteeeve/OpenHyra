import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from auditing import _algorithm_bundle_digest, run_final_audit, select_top_k
from eb import ExperienceBank
from sandbox import source_tree_hash
from stopping import write_termination


class PythonProgramPrivateAuditTests(unittest.TestCase):
    def test_final_audit_keeps_distinct_programs_and_passes_frozen_source(self):
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
                "for name in sorted(('algorithm.py', 'manifest.json')):\n"
                "    data = (source / name).read_bytes()\n"
                "    files.append({'path': name, 'size_bytes': len(data),\n"
                "                  'sha256': hashlib.sha256(data).hexdigest()})\n"
                "digest = hashlib.sha256(json.dumps({\n"
                "    'schema': 'openhyra-algorithm-bundle.v1',\n"
                "    'files': files}, sort_keys=True, separators=(',', ':'),\n"
                "    ensure_ascii=False).encode()).hexdigest()\n"
                "score = float((source / 'algorithm.py').read_text().split('=')[-1])\n"
                "print(json.dumps({'score': score,\n"
                " 'normalized_solution': artifact,\n"
                " 'evidence': {'source_seen': True},\n"
                " 'metrics': {'candidate_hash': digest,\n"
                "             'artifact_protocol': artifact['schema']}}))\n",
                encoding="utf-8",
            )
            task = SimpleNamespace(
                run_dir=run_dir,
                run_id="python-program-audit",
                name="python_program_task",
                protocol="python-program-test.v1",
                direction="max",
                candidate_mode="python_program",
                candidate_source_files=("algorithm.py", "manifest.json"),
                artifact_protocol="openhyra-python-program.v1",
                evaluation={
                    "audit_stage": {"top_k": 3, "direction": "max"},
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
            bank = ExperienceBank(run_dir / "eb", direction="max")
            manifest_bytes = json.dumps(
                {
                    "schema": "openhyra-python-program.v1",
                    "interface": "continuation",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            records = []
            for index in range(1, 4):
                source = root / f"candidate-{index}"
                source.mkdir()
                (source / "algorithm.py").write_text(
                    f"PROGRAM_SCORE = {index}\n", encoding="utf-8"
                )
                (source / "manifest.json").write_bytes(manifest_bytes)
                (source / "solution.json").write_bytes(manifest_bytes)
                source_hash = source_tree_hash(source, 1024 * 1024)[0]
                candidate_hash = _algorithm_bundle_digest(source, task)
                records.append(bank.commit(
                    source,
                    float(index),
                    "ok",
                    f"candidate {index}",
                    None,
                    "",
                    metrics={
                        "artifact_sha256": hashlib.sha256(
                            manifest_bytes
                        ).hexdigest(),
                        "source_snapshot_sha256": source_hash,
                        "candidate_hash": candidate_hash,
                    },
                    metadata={
                        "run_manifest_sha256": run_manifest["manifest_sha256"],
                        "task_provenance": run_manifest["task"],
                        "source_sha256": run_manifest["source_sha256"],
                    },
                ))
            write_termination(run_dir / "termination.json", {
                "reason": "iteration_limit",
                "terminal": False,
                "evidence": {"completed_contexts": 1},
            })

            selected = select_top_k(bank.records(), "max", 3)
            report = run_final_audit(
                task,
                bank,
                run_manifest,
                seed_factory=lambda: 17,
            )

            self.assertEqual(len(selected), 3)
            self.assertEqual(len(report["candidates"]), 3)
            self.assertEqual(
                {item["candidate_mode"] for item in report["candidates"]},
                {"python_program"},
            )
            self.assertEqual(
                len({
                    item["candidate_source_sha256"]
                    for item in report["freeze_manifest"]["candidates"]
                }),
                3,
            )
            self.assertTrue(all(
                item["evidence"]["source_seen"]
                for item in report["candidates"]
            ))


if __name__ == "__main__":
    unittest.main()
