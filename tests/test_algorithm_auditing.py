import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from algorithm_auditing import (
    ALGORITHM_BUNDLE_SCHEMA,
    ALGORITHM_FREEZE_MANIFEST_SCHEMA,
    PER_CELL_POLICY_PROVENANCE_SCHEMA,
    build_per_cell_provenance,
    compute_eb_snapshot_sha256,
    freeze_top_k_algorithm_bundles,
    read_candidate_algorithm_bundle,
    validate_freeze_manifest,
    validate_per_cell_provenance,
    verify_frozen_algorithm_bundles,
)


RUN_SHA = "a" * 64


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class AlgorithmAuditingTests(unittest.TestCase):
    def _source(
            self, root, name, *, train="print('train')\n", features=None,
            bundle_subdir="."):
        slot = root / name
        slot.mkdir()
        bundle = slot if bundle_subdir == "." else slot / bundle_subdir
        bundle.mkdir(exist_ok=True)
        # Deliberately create files out of lexical order.
        (bundle / "train.py").write_text(train)
        if features is not None:
            (bundle / "features.json").write_text(
                json.dumps(features, separators=(",", ":"))
            )
        (bundle / "manifest.json").write_text(json.dumps({
            "schema": "openhyra-policy-spec.v1",
            "runner_type": "ridge_lsmc",
        }))
        return slot

    def _record(
            self, record_id, score, slot, *, bundle_subdir=".",
            run_sha=RUN_SHA, declared_hash=None):
        record = {
            "id": record_id,
            "status": "ok",
            "score": score,
            "path": str(slot),
            "metrics": {},
            "metadata": {"run_manifest_sha256": run_sha},
        }
        if declared_hash is None:
            inspected = read_candidate_algorithm_bundle(
                record,
                bundle_subdir=bundle_subdir,
                require_declared_bundle_hash=False,
            )
            declared_hash = inspected.algorithm_bundle_sha256
        record["metrics"]["algorithm_bundle_sha256"] = declared_hash
        return record

    def _freeze(
            self, root, records, *, destination=None, direction="max", top_k=2,
            bundle_subdir=".", source_root=None, destination_root=None):
        destination = destination or root / "freeze"
        source_root = source_root or root
        destination_root = destination_root or root
        version = len(records)
        snapshot_hash = compute_eb_snapshot_sha256(records, version)
        manifest = freeze_top_k_algorithm_bundles(
            records,
            destination,
            direction=direction,
            top_k=top_k,
            run_manifest_sha256=RUN_SHA,
            eb_snapshot_version=version,
            eb_snapshot_sha256=snapshot_hash,
            expected_source_root=source_root,
            expected_destination_root=destination_root,
            bundle_subdir=bundle_subdir,
            frozen_at="2026-08-29T00:00:00+0800",
        )
        return destination, manifest

    def _verify(self, root, destination, manifest, records, **kwargs):
        return verify_frozen_algorithm_bundles(
            destination,
            records=records,
            expected_manifest_sha256=manifest.sha256,
            expected_run_manifest_sha256=RUN_SHA,
            expected_destination_root=root,
            **kwargs,
        )

    def test_bundle_hash_order_and_declared_hash_are_deterministic_and_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._source(root, "first", features={"x": 1})
            second = self._source(root, "second", features={"x": 1})
            first_record = self._record("a", 1.0, first)
            second_record = self._record("b", 2.0, second)
            one = read_candidate_algorithm_bundle(first_record)
            two = read_candidate_algorithm_bundle(second_record)
            self.assertEqual(one.schema, ALGORITHM_BUNDLE_SCHEMA)
            self.assertEqual(one.algorithm_bundle_sha256, two.algorithm_bundle_sha256)
            self.assertEqual(
                [item.path for item in one.source_provenance.files],
                ["features.json", "manifest.json", "train.py"],
            )
            missing = dict(first_record)
            missing["metrics"] = {}
            with self.assertRaisesRegex(ValueError, "lacks declared"):
                read_candidate_algorithm_bundle(missing)
            with self.assertRaises(FrozenInstanceError):
                one.algorithm_bundle_sha256 = "0" * 64

    def test_freeze_binds_run_snapshot_selection_and_deduplicates_algorithm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            best = self._source(root, "best", train="x = 1\n")
            duplicate = self._source(root, "duplicate", train="x = 1\n")
            other = self._source(root, "other", train="x = 2\n")
            records = [
                self._record("other", 8.0, other),
                self._record("best-copy", 9.0, duplicate),
                self._record("best", 10.0, best),
            ]
            destination, manifest = self._freeze(root, records)
            self.assertEqual(manifest.schema, ALGORITHM_FREEZE_MANIFEST_SCHEMA)
            self.assertEqual(manifest.run_manifest_sha256, RUN_SHA)
            self.assertEqual(manifest.direction, "max")
            self.assertEqual(manifest.requested_top_k, 2)
            self.assertEqual(manifest.eb_snapshot_version, len(records))
            self.assertEqual(
                manifest.eb_snapshot_sha256,
                compute_eb_snapshot_sha256(records, len(records)),
            )
            self.assertEqual(manifest.bundle_subdir, ".")
            self.assertEqual(
                [item.record_id for item in manifest.candidates],
                ["best", "other"],
            )
            self.assertEqual([item.rank for item in manifest.candidates], [1, 2])
            self.assertEqual(
                len({item.algorithm_bundle_sha256 for item in manifest.candidates}),
                2,
            )
            self.assertEqual(self._verify(root, destination, manifest, records), manifest)

    def test_freeze_rejects_run_and_eb_snapshot_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slot = self._source(root, "candidate")
            wrong_run = [self._record("candidate", 1.0, slot, run_sha="b" * 64)]
            snapshot = compute_eb_snapshot_sha256(wrong_run, 1)
            with self.assertRaisesRegex(ValueError, "run manifest provenance"):
                freeze_top_k_algorithm_bundles(
                    wrong_run, root / "bad-run", direction="max", top_k=1,
                    run_manifest_sha256=RUN_SHA, eb_snapshot_version=1,
                    eb_snapshot_sha256=snapshot, expected_source_root=root,
                    expected_destination_root=root, bundle_subdir=".",
                )

            records = [self._record("candidate", 1.0, slot)]
            missing_declaration = [dict(records[0])]
            missing_declaration[0]["metrics"] = {}
            missing_snapshot = compute_eb_snapshot_sha256(missing_declaration, 1)
            with self.assertRaisesRegex(ValueError, "lacks declared"):
                freeze_top_k_algorithm_bundles(
                    missing_declaration, root / "missing-declaration",
                    direction="max", top_k=1, run_manifest_sha256=RUN_SHA,
                    eb_snapshot_version=1, eb_snapshot_sha256=missing_snapshot,
                    expected_source_root=root, expected_destination_root=root,
                    bundle_subdir=".",
                )
            with self.assertRaisesRegex(ValueError, "snapshot digest"):
                freeze_top_k_algorithm_bundles(
                    records, root / "bad-snapshot", direction="max", top_k=1,
                    run_manifest_sha256=RUN_SHA, eb_snapshot_version=1,
                    eb_snapshot_sha256="f" * 64, expected_source_root=root,
                    expected_destination_root=root, bundle_subdir=".",
                )
            with self.assertRaisesRegex(ValueError, "version"):
                compute_eb_snapshot_sha256(records, 2)

    def test_verifier_rejects_changed_snapshot_and_self_consistent_non_top_k(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._source(root, "first", train="x=1\n")
            second = self._source(root, "second", train="x=2\n")
            records = [
                self._record("first", 2.0, first),
                self._record("second", 1.0, second),
            ]
            destination, manifest = self._freeze(root, records, top_k=2)
            changed_records = [dict(item) for item in records]
            changed_records[0] = dict(changed_records[0], score=-100.0)
            with self.assertRaisesRegex(ValueError, "snapshot digest"):
                self._verify(root, destination, manifest, changed_records)

            payload = manifest.to_dict()
            payload["candidates"][0]["search_score"] = -123.0
            manifest_path = destination / "manifest.json"
            destination.chmod(0o700)
            manifest_path.chmod(0o600)
            manifest_path.write_text(json.dumps(payload, sort_keys=True))
            manifest_path.chmod(0o400)
            destination.chmod(0o500)
            with self.assertRaisesRegex(ValueError, "Top-K selection"):
                verify_frozen_algorithm_bundles(
                    destination,
                    records=records,
                    expected_manifest_sha256=canonical_sha256(payload),
                    expected_run_manifest_sha256=RUN_SHA,
                    expected_destination_root=root,
                )

    def test_bundle_subdir_allows_full_eb_slot_but_hashes_only_strict_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slot = self._source(root, "slot", bundle_subdir="algorithm")
            (slot / "solution.json").write_text("legacy output")
            (slot / "run.log").write_text("telemetry")
            record = self._record(
                "slot", 1.0, slot, bundle_subdir="algorithm",
            )
            destination, manifest = self._freeze(
                root, [record], top_k=1, bundle_subdir="algorithm",
            )
            self.assertEqual(manifest.bundle_subdir, "algorithm")
            self.assertEqual(
                Path(manifest.candidates[0].source_provenance.source_path).name,
                "algorithm",
            )
            self.assertEqual(
                {item.path for item in manifest.candidates[0].source_provenance.files},
                {"train.py", "manifest.json"},
            )
            self._verify(root, destination, manifest, [record])

            (slot / "algorithm" / "secret.txt").write_text("undeclared")
            with self.assertRaisesRegex(ValueError, "undeclared"):
                read_candidate_algorithm_bundle(
                    record, bundle_subdir="algorithm",
                )

    def test_destination_must_be_beneath_real_trusted_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slot = self._source(root, "slot")
            records = [self._record("slot", 1.0, slot)]
            version = len(records)
            snapshot = compute_eb_snapshot_sha256(records, version)
            outside = root.parent / f"{root.name}-outside-freeze"
            with self.assertRaisesRegex(ValueError, "escapes"):
                freeze_top_k_algorithm_bundles(
                    records, outside, direction="max", top_k=1,
                    run_manifest_sha256=RUN_SHA, eb_snapshot_version=version,
                    eb_snapshot_sha256=snapshot, expected_source_root=root,
                    expected_destination_root=root, bundle_subdir=".",
                )
            real_parent = root / "real-parent"
            real_parent.mkdir()
            link_parent = root / "link-parent"
            link_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "ancestors"):
                freeze_top_k_algorithm_bundles(
                    records, link_parent / "freeze", direction="max", top_k=1,
                    run_manifest_sha256=RUN_SHA, eb_snapshot_version=version,
                    eb_snapshot_sha256=snapshot, expected_source_root=root,
                    expected_destination_root=root, bundle_subdir=".",
                )

    def test_frozen_and_source_tampering_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slot = self._source(root, "slot")
            records = [self._record("slot", 1.0, slot)]
            destination, manifest = self._freeze(root, records, top_k=1)
            frozen_train = destination / "candidates" / "slot" / "train.py"
            frozen_train.chmod(0o600)
            frozen_train.write_text("tampered=True\n")
            with self.assertRaisesRegex(ValueError, "changed after algorithm freeze"):
                self._verify(root, destination, manifest, records)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slot = self._source(root, "slot")
            records = [self._record("slot", 1.0, slot)]
            destination, manifest = self._freeze(root, records, top_k=1)
            (slot / "train.py").write_text("changed=True\n")
            with self.assertRaisesRegex(ValueError, "provenance mismatch|source changed"):
                self._verify(
                    root, destination, manifest, records,
                    expected_source_root=root, verify_source_unchanged=True,
                )

    def test_source_symlink_hardlink_escape_and_byte_limit_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = self._source(root, "outside")
            eb_root = root / "eb" / "solutions"
            eb_root.mkdir(parents=True)

            linked = eb_root / "linked"
            linked.mkdir()
            (linked / "manifest.json").write_text("{}")
            os.symlink(outside / "train.py", linked / "train.py")
            linked_record = {
                "id": "linked", "status": "ok", "score": 1.0,
                "path": str(linked),
                "metrics": {"algorithm_bundle_sha256": "0" * 64},
                "metadata": {"run_manifest_sha256": RUN_SHA},
            }
            with self.assertRaisesRegex(ValueError, "regular file"):
                read_candidate_algorithm_bundle(
                    linked_record, expected_source_root=eb_root,
                )

            hardlinked = eb_root / "hardlinked"
            hardlinked.mkdir()
            (hardlinked / "manifest.json").write_text("{}")
            os.link(outside / "train.py", hardlinked / "train.py")
            hardlink_record = dict(linked_record, id="hardlinked", path=str(hardlinked))
            with self.assertRaisesRegex(ValueError, "hard link"):
                read_candidate_algorithm_bundle(
                    hardlink_record, expected_source_root=eb_root,
                )
            (hardlinked / "train.py").unlink()

            outside_record = self._record("outside", 1.0, outside)
            with self.assertRaisesRegex(ValueError, "escapes"):
                read_candidate_algorithm_bundle(
                    outside_record, expected_source_root=eb_root,
                )
            with self.assertRaisesRegex(ValueError, "exceeds"):
                read_candidate_algorithm_bundle(
                    outside_record, max_bundle_bytes=8,
                )
            with self.assertRaisesRegex(ValueError, "safe relative POSIX"):
                read_candidate_algorithm_bundle(
                    outside_record, bundle_subdir="../outside",
                )
            linked_bundle_slot = eb_root / "linked-bundle-slot"
            linked_bundle_slot.mkdir()
            (linked_bundle_slot / "algorithm").symlink_to(
                outside, target_is_directory=True,
            )
            linked_bundle_record = {
                "id": "linked-bundle", "status": "ok", "score": 1.0,
                "path": str(linked_bundle_slot),
                "metrics": {"algorithm_bundle_sha256": "0" * 64},
                "metadata": {"run_manifest_sha256": RUN_SHA},
            }
            with self.assertRaisesRegex(ValueError, "real directory"):
                read_candidate_algorithm_bundle(
                    linked_bundle_record,
                    expected_source_root=eb_root,
                    bundle_subdir="algorithm",
                )

    def test_per_cell_provenance_requires_all_external_hash_anchors(self):
        hashes = {
            "freeze_manifest_sha256": "0" * 64,
            "run_manifest_sha256": "1" * 64,
            "evaluation_request_sha256": "2" * 64,
            "trusted_runner_sha256": "3" * 64,
            "algorithm_bundle_sha256": "4" * 64,
            "instance_sha256": "5" * 64,
            "training_input_sha256": "6" * 64,
            "runtime_sha256": "7" * 64,
            "policy_artifact_sha256": "8" * 64,
        }
        record = build_per_cell_provenance(
            **hashes,
            train_seed=42,
            stage="audit",
            suite="private-v1",
            instance="bermudan-003",
            repeat=1,
        )
        self.assertEqual(record.schema, PER_CELL_POLICY_PROVENANCE_SCHEMA)
        expected_kwargs = {f"expected_{key}": value for key, value in hashes.items()}
        expected_kwargs.update({
            "expected_stage": "audit",
            "expected_suite": "private-v1",
            "expected_instance": "bermudan-003",
            "expected_repeat": 1,
            "expected_train_seed": 42,
        })
        self.assertEqual(
            validate_per_cell_provenance(record.to_dict(), **expected_kwargs),
            record,
        )
        bad = dict(expected_kwargs)
        bad["expected_trusted_runner_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "trusted runner provenance"):
            validate_per_cell_provenance(record.to_dict(), **bad)
        replacements = {
            "stage": "search",
            "suite": "private-v2",
            "instance": "bermudan-004",
            "repeat": 2,
            "train_seed": 43,
        }
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                tampered = record.to_dict()
                tampered[field] = replacement
                with self.assertRaisesRegex(
                        ValueError, rf"{field} binding mismatch"):
                    validate_per_cell_provenance(tampered, **expected_kwargs)
        with self.assertRaises(TypeError):
            validate_per_cell_provenance(record.to_dict())
        with self.assertRaises(FrozenInstanceError):
            record.repeat = 2

    def test_freeze_manifest_schema_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slot = self._source(root, "slot")
            records = [self._record("slot", 1.0, slot)]
            _destination, manifest = self._freeze(root, records, top_k=1)
            payload = manifest.to_dict()
            payload["unexpected"] = True
            with self.assertRaisesRegex(ValueError, "fields must be exactly"):
                validate_freeze_manifest(payload)


if __name__ == "__main__":
    unittest.main()
