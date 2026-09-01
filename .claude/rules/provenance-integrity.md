run_manifest.json freezes code, task, evaluator, model, concurrency, limits, seed, and stopping policy.
Resume is refused if result-affecting provenance drifts — never bypass validate_run_manifest.
Hash chains (source_snapshot_sha256, artifact_sha256, evidence_sha256) must remain closed; any new artifact path must participate in the chain.
EB records are immutable — never implement deletion or mutation of committed records.
