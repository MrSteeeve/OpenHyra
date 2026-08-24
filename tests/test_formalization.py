import unittest

from tasks.sums_diffs.formalization import (
    ArtifactRejected,
    REQUEST_SCHEMA,
    RESERVED_AUDIT_FILE,
    RESERVED_CANDIDATE_FILE,
    build_formalization_audit,
    build_formalization_wrapper,
    validate_formalization_request,
    verify_formalization_request,
)


CLAIMS = [{
    "id": "upper",
    "template": "universal_upper_bound",
    "target": {"numerator": 2, "denominator": 1},
}]
MATHLIB_REVISION = "3" * 40
RUNTIME_ATTESTATION = {
    "environment_sha256": "1" * 64,
    "lean_binary_sha256": "2" * 64,
    "toolchain": "leanprover/lean4:v4.26.0",
    "mathlib_revision": MATHLIB_REVISION,
    "mathlib_tree_sha256": "4" * 64,
}


def _request(*, claim_id="upper", target="lean4", term="by exact proof"):
    return {
        "schema": REQUEST_SCHEMA,
        "target": target,
        "proofs": [{
            "claim_id": claim_id,
            "term": term,
        }],
    }


class InlineFormalizationTests(unittest.TestCase):
    def test_validate_and_build_derive_trusted_theorem_type(self):
        normalized, theorem_types = validate_formalization_request(
            _request(),
            CLAIMS,
        )

        self.assertEqual(normalized["schema"], "openhyra-lean4-request")
        self.assertEqual(
            theorem_types["upper"],
            (
                "OpenHyraSumDiff.UniversalUpperBoundAt "
                "((2 : ℝ) / (1 : ℝ))"
            ),
        )
        wrapper, theorem_names = build_formalization_wrapper(
            normalized,
            theorem_types,
        )
        source = wrapper.decode("utf-8")
        self.assertIn("import OpenHyraSumDiff.Spec", source)
        self.assertIn(
            "theorem claim_00 : "
            "OpenHyraSumDiff.UniversalUpperBoundAt "
            "((2 : ℝ) / (1 : ℝ)) :=",
            source,
        )
        self.assertNotIn("#print axioms", source)
        audit = build_formalization_audit(theorem_names).decode("utf-8")
        self.assertIn("import OpenHyraCandidate", audit)
        self.assertIn("#check OpenHyraCandidate.claim_00", audit)
        self.assertIn("#print axioms OpenHyraCandidate.claim_00", audit)
        self.assertEqual(
            theorem_names,
            {"upper": "OpenHyraCandidate.claim_00"},
        )

    def test_wrapper_can_inline_the_trusted_spec_for_offline_compilation(self):
        normalized, theorem_types = validate_formalization_request(
            _request(),
            CLAIMS,
        )
        wrapper, _theorem_names = build_formalization_wrapper(
            normalized,
            theorem_types,
            trusted_spec_source=(
                b"import Mathlib\n"
                b"namespace OpenHyraSumDiff\n"
                b"def UniversalUpperBoundAt (_q : Real) : Prop := True\n"
                b"end OpenHyraSumDiff\n"
            ),
        )
        source = wrapper.decode("utf-8")

        self.assertTrue(source.startswith("import Mathlib\n"))
        self.assertNotIn("import OpenHyraSumDiff.Spec", source)
        self.assertIn("namespace OpenHyraCandidate", source)

    def test_verify_succeeds_with_isolated_fake_runner(self):
        observed = []

        def runner(request):
            observed.append(request)
            if request.phase == "probe_lean":
                return {
                    "returncode": 0,
                    "stdout": "Lean (version 4.19.0, x86_64-apple-darwin)",
                    "stderr": "",
                    "timed_out": False,
                }
            return {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "audit_returncode": 0,
                "audit_stdout": (
                    "OpenHyraCandidate.claim_00 : "
                    "OpenHyraSumDiff.UniversalUpperBoundAt "
                    "((2 : ℝ) / (1 : ℝ))\n"
                    "'OpenHyraCandidate.claim_00' "
                    "does not depend on any axioms"
                ),
                "audit_stderr": "",
                "timed_out": False,
            }

        result = verify_formalization_request(
            _request(),
            CLAIMS,
            runner=runner,
            trusted_files={
                "OpenHyraSumDiff/Spec.lean": b"-- trusted task spec\n",
            },
            command_prefix=("lean",),
            toolchain="leanprover/lean4:v4.19.0",
        )

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["verified_claim_ids"], ["upper"])
        self.assertEqual(result["axioms"], {"upper": []})
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0].phase, "probe_lean")
        self.assertEqual(
            observed[0].argv,
            ("lean", "--version"),
        )
        run_request = observed[1]
        self.assertEqual(run_request.phase, "compile_then_audit")
        self.assertEqual(
            run_request.argv,
            (
                "lean",
                "-o",
                "OpenHyraCandidate.olean",
                RESERVED_CANDIDATE_FILE,
            ),
        )
        self.assertEqual(
            run_request.audit_argv,
            ("lean", RESERVED_AUDIT_FILE),
        )
        self.assertFalse(run_request.network_allowed)
        self.assertFalse(run_request.workspace_writable)
        self.assertIn(RESERVED_AUDIT_FILE, run_request.files)
        self.assertIn(RESERVED_CANDIDATE_FILE, run_request.files)

    def test_validate_rejects_unknown_claim(self):
        with self.assertRaises(ArtifactRejected) as caught:
            validate_formalization_request(
                _request(claim_id="missing"),
                CLAIMS,
            )

        self.assertEqual(caught.exception.code, "unknown_claim")

    def test_validate_rejects_target_mismatch(self):
        with self.assertRaises(ArtifactRejected) as caught:
            validate_formalization_request(
                _request(target="coq"),
                CLAIMS,
            )

        self.assertEqual(caught.exception.code, "unsupported_target")

    def test_validate_rejects_proof_hole(self):
        with self.assertRaises(ArtifactRejected) as caught:
            validate_formalization_request(
                _request(term="by sorry"),
                CLAIMS,
            )

        self.assertEqual(caught.exception.code, "forbidden_proof_hole")

    def test_validate_rejects_wrapper_escape_and_inline_meta_commands(self):
        cases = (
            "by exact proof) --",
            "by run_tac do pure ()",
            "by exact (proof",
            "by exact proof\naxiom injected : False",
        )
        for term in cases:
            with self.subTest(term=term):
                with self.assertRaises(ArtifactRejected) as caught:
                    validate_formalization_request(
                        _request(term=term),
                        CLAIMS,
                    )
                self.assertEqual(
                    caught.exception.code,
                    "forbidden_inline_syntax",
                )

    def test_candidate_cannot_submit_verdict_binding_hashes(self):
        request = _request()
        request["request_sha256"] = "0" * 64
        request["proofs"][0]["proof_sha256"] = "0" * 64
        with self.assertRaises(ArtifactRejected) as caught:
            validate_formalization_request(request, CLAIMS)

        self.assertEqual(caught.exception.code, "invalid_schema")

    def test_verify_reports_runner_unavailable(self):
        result = verify_formalization_request(
            _request(),
            CLAIMS,
            runner=None,
            trusted_files={},
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(
            result["reason"],
            "isolated_runner_not_configured",
        )

    def test_sealed_evaluator_request_round_trips_but_hash_tampering_fails(self):
        normalized, _types = validate_formalization_request(
            _request(),
            CLAIMS,
        )
        result = verify_formalization_request(
            dict(normalized),
            CLAIMS,
            runner=None,
            trusted_files={},
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(
            result["request_sha256"],
            normalized["request_sha256"],
        )

        tampered = {
            **dict(normalized),
            "proofs": [
                {
                    **dict(normalized["proofs"][0]),
                    "term": "by exact changed",
                }
            ],
        }
        rejected = verify_formalization_request(
            tampered,
            CLAIMS,
            runner=None,
            trusted_files={},
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["reason"], "sealed_hash_mismatch")

    def test_verify_rejects_unexpected_axiom(self):
        def runner(request):
            if request.phase == "probe_lean":
                return {
                    "returncode": 0,
                    "stdout": "Lean (version 4.26.0, x86_64-apple-darwin)",
                    "stderr": "",
                    "timed_out": False,
                }
            return {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "audit_returncode": 0,
                "audit_stdout": (
                    "'OpenHyraCandidate.claim_00' "
                    "depends on axioms: [OpenHyra.Untrusted]"
                ),
                "audit_stderr": "",
                "timed_out": False,
            }

        result = verify_formalization_request(
            _request(),
            CLAIMS,
            runner=runner,
            trusted_files={},
            command_prefix=("lean",),
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "unexpected_axioms")
        self.assertEqual(
            result["unexpected_axioms"],
            {"upper": ["OpenHyra.Untrusted"]},
        )

    def test_candidate_axiom_log_cannot_hide_separate_audit_output(self):
        def runner(request):
            if request.phase == "probe_lean":
                return {
                    "returncode": 0,
                    "stdout": "Lean (version 4.26.0, x86_64-apple-darwin)",
                    "stderr": "",
                    "timed_out": False,
                }
            return {
                "returncode": 0,
                "stdout": (
                    "'OpenHyraCandidate.claim_00' "
                    "depends on axioms: [propext]"
                ),
                "stderr": "",
                "audit_returncode": 0,
                "audit_stdout": (
                    "'OpenHyraCandidate.claim_00' "
                    "depends on axioms: [OpenHyra.Untrusted]"
                ),
                "audit_stderr": "",
                "timed_out": False,
            }

        result = verify_formalization_request(
            _request(),
            CLAIMS,
            runner=runner,
            trusted_files={},
            command_prefix=("lean",),
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "unexpected_axioms")

    def test_verify_fails_closed_on_toolchain_version_mismatch(self):
        def runner(_request):
            return {
                "returncode": 0,
                "stdout": "Lean (version 4.25.0, x86_64-apple-darwin)",
                "stderr": "",
                "timed_out": False,
            }

        result = verify_formalization_request(
            _request(),
            CLAIMS,
            runner=runner,
            trusted_files={},
            command_prefix=("lean",),
            toolchain="leanprover/lean4:v4.26.0",
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "lean_version_mismatch")

    def test_runtime_attestation_is_required_and_stable(self):
        calls = []

        def runner(request):
            calls.append(request)
            return {
                "returncode": 0,
                "stdout": (
                    "Lean (version 4.26.0, x86_64-apple-darwin)"
                    if request.phase == "probe_lean" else ""
                ),
                "stderr": "",
                "audit_returncode": (
                    None if request.phase == "probe_lean" else 0
                ),
                "audit_stdout": (
                    ""
                    if request.phase == "probe_lean" else
                    "'OpenHyraCandidate.claim_00' "
                    "does not depend on any axioms"
                ),
                "audit_stderr": "",
                "timed_out": False,
                "output_complete": True,
                "attestation": RUNTIME_ATTESTATION,
            }

        result = verify_formalization_request(
            _request(),
            CLAIMS,
            runner=runner,
            trusted_files={},
            command_prefix=("lean",),
            toolchain="leanprover/lean4:v4.26.0",
            mathlib_revision=MATHLIB_REVISION,
        )

        self.assertEqual(result["status"], "verified")
        self.assertEqual(
            result["runtime_attestation"],
            RUNTIME_ATTESTATION,
        )
        self.assertEqual(len(calls), 2)
        for request in calls:
            self.assertEqual(
                request.expected_toolchain,
                "leanprover/lean4:v4.26.0",
            )
            self.assertEqual(
                request.expected_mathlib_revision,
                MATHLIB_REVISION,
            )

    def test_runtime_attestation_change_fails_closed(self):
        def runner(request):
            attestation = dict(RUNTIME_ATTESTATION)
            if request.phase == "compile_then_audit":
                attestation["mathlib_tree_sha256"] = "5" * 64
            return {
                "returncode": 0,
                "stdout": (
                    "Lean (version 4.26.0, x86_64-apple-darwin)"
                    if request.phase == "probe_lean" else ""
                ),
                "stderr": "",
                "audit_returncode": (
                    None if request.phase == "probe_lean" else 0
                ),
                "audit_stdout": (
                    ""
                    if request.phase == "probe_lean" else
                    "'OpenHyraCandidate.claim_00' "
                    "does not depend on any axioms"
                ),
                "audit_stderr": "",
                "timed_out": False,
                "output_complete": True,
                "attestation": attestation,
            }

        result = verify_formalization_request(
            _request(),
            CLAIMS,
            runner=runner,
            trusted_files={},
            command_prefix=("lean",),
            toolchain="leanprover/lean4:v4.26.0",
            mathlib_revision=MATHLIB_REVISION,
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "runtime_attestation_changed")


if __name__ == "__main__":
    unittest.main()
