"""Synthetic frozen receipts exercise the report's denominators and missing values."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.isomorphic_report import (
    ARMS, ALIGNED_IDS, ENVIRONMENTS, METRICS, OMITTED, Z_KEYS, _z,
    _table_control, _track_holdout, render,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _journal(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in values) + "\n", encoding="utf-8")


def _pair(estimate: float | None, interval: list[float] | None, days: int) -> dict:
    return {"estimate": estimate, "interval": interval, "paired_days": days,
            "registered_days": 12, "bootstrap_valid": 2000, "bootstrap_total": 2000}


def _fixture(root: Path) -> Path:
    artifact = root / "artifacts" / "p2-m3-isomorphic-control"
    source = "fixed seed\n"
    seed_digest = sha256(source.encode()).hexdigest()
    baseline = {}
    for i, env in enumerate(ENVIRONMENTS):
        evaluation = {"status": "VALID", "params": 24244, "state": 60, "aggregate": {},
                      "training_checkpoint_hashes": {"103": str(i)}}
        z = {key: {"median": float(i + 1), "intraday_iqr": .1} for key in Z_KEYS}
        baseline[env] = {"z": z, "evaluation": evaluation}
    approved = {env: [seed_digest] for env in ENVIRONMENTS}
    trajectories = {}
    holdout = {env: {} for env in ENVIRONMENTS}
    regret = {}
    calls = []
    alignment = {arm: 0 for arm in ARMS}
    for env in ENVIRONMENTS:
        for arm in ARMS:
            key = f"{env}/{arm}"
            active = sorted(ALIGNED_IDS[env])
            seed = {"outcome": "SEED", "cand_id": "B0", "source": source, "source_sha256": seed_digest,
                    "evaluation": baseline[env]["evaluation"]}
            rows = [seed]
            if arm == "d_only":
                z = OMITTED
            else:
                src_env = env if arm == "true_z_d" else ENVIRONMENTS[1 - ENVIRONMENTS.index(env)]
                z = _z(baseline[src_env]["z"])
            messages = [{"role": "system", "content": "objective"}, {"role": "user", "content":
                f"Anonymous environment: {env}\n[Current Parent]\nseed\n"
                f"[Macro Data Regime Z]\n{z}\n\n[Model Residuals D]\nidentical"}]
            for number in range(1, 7):
                cid = f"c{number:02d}"
                card = active[0] if arm == "true_z_d" else "C01"
                verdict = {1: "SUPPORTED", 2: "REFUTED", 3: "INCONCLUSIVE"}.get(number)
                row = {"round": number, "cand_id": cid, "parent_id": "B0", "outcome": "REJECTED",
                       "source": "", "prompt": messages, "prompt_tokens": 1500, "active_card_ids": active,
                       "selected_card_ids": [card], "prediction": {"target_metric": "dt_nll", "expected_direction": "decrease",
                       "selected_card_ids": [card], "regime_hypothesis": "causal state"}}
                if verdict:
                    row["prediction_verdict"] = verdict
                if number >= 4:
                    row["outcome"] = "INVALID_PROPOSAL"
                    row["prediction"] = {}
                    row["selected_card_ids"] = []
                if number == 1:
                    row["outcome"] = "PROMOTED"
                    row["source"] = f"variant {key}\n"
                    digest = sha256(row["source"].encode()).hexdigest()
                    row["source_sha256"] = digest
                    row["evaluation"] = {"status": "VALID"}
                    approved[env].append(digest)
                    trajectories[key] = {"final_id": cid, "final_sha256": digest}
                    holdout[env][digest] = {"evaluation": {"status": "VALID", "aggregate": {
                        "available_quote_fraction": .8, "exhausted_replicate_rate": .1},
                        "daily": [{"counterfactual_impact_mae": None} for _ in range(12)]},
                        "against_B0": {metric: _pair(None, None, 5) for metric in METRICS}}
                if arm == "true_z_d" and number <= 3:
                    alignment[arm] += 1
                rows.append(row)
                call_id = f"{key}/{cid}"
                calls.extend([{"event": "reserved", "call_id": call_id, "messages": messages},
                              {"event": "completed", "call_id": call_id,
                               "input_tokens": 1490, "output_tokens": 50}])
            _journal(artifact / "trajectories" / env / arm / "ledger.jsonl", rows)
            regret[key] = {"estimate": None, "paired_days": 5, "oracle_models": 2,
                           "reason": "oracle common-day support insufficient"}
        holdout[env][seed_digest] = {"evaluation": {"status": "VALID"}, "against_B0": {}}
    summary = {"wall_seconds": 1.23, "status": "PARTIAL", "alignment_supported": True, "holdout_supported": False,
               "holdout_refuted": False, "alignment_counts": alignment, "regret": regret,
               "formal_calls": 36, "formal_input_tokens": 1490 * 36, "formal_output_tokens": 50 * 36,
               "unknown_usage_calls": 0,
               "cross_arm_effects": {f"{env}/true_z_d-vs-{arm}": {metric: _pair(None, None, 5) for metric in METRICS}
                                     for env in ENVIRONMENTS for arm in ("d_only", "shuffled_z_d")}}
    _write(artifact / "registration.json", {"model": "offline-fixture", "promotion": {}, "data_protocol": {},
          "controlled_comparison_rule": "fixture"})
    _write(artifact / "baseline.json", baseline)
    _write(artifact / "freeze.json", {"approved": approved, "trajectories": trajectories})
    _write(artifact / "holdout.json", holdout)
    _write(artifact / "study_summary.json", summary)
    _write(artifact / "handshake.json", {"status": "PASS", "input_tokens": 20, "output_tokens": 4})
    _journal(artifact / "api_calls.jsonl", calls)
    return artifact


def test_five_table_report_preserves_finite_support_and_declared_denominators(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in artifact.rglob("*") if path.is_file()}
    report = render(artifact)
    assert report.count("## 表 ") == 5
    assert "**36/36**" in report
    assert "true_z_d=6/12" in report
    assert "NA 无区间 (5/12 日; 2000/2000 抽样)" in report
    assert "NA; 5/12 共同日" in report
    assert "1/1 (100.0%)" in report
    assert "**PARTIAL**" in report
    assert "总体合规审计：**通过**" in report
    assert "M2 参考文件未提供" in report
    after = {path: path.read_bytes() for path in artifact.rglob("*") if path.is_file()}
    assert after == before


def test_report_rejects_unfrozen_candidate_or_inconsistent_usage_receipt(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    path = artifact / "study_summary.json"
    summary = json.loads(path.read_text())
    _write(path, {**summary, "formal_output_tokens": 99})
    with pytest.raises(ValueError, match="accounting"):
        render(artifact)
    _write(path, summary)
    path = artifact / "freeze.json"
    freeze = json.loads(path.read_text())
    freeze["approved"]["env_alpha"].pop()
    _write(path, freeze)
    with pytest.raises(ValueError, match="roster"):
        render(artifact)


def test_nonz_prompt_tampering_is_visible_not_silently_repaired(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    path = artifact / "trajectories" / "env_alpha" / "shuffled_z_d" / "ledger.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["prompt"][1]["content"] += " altered D"
    _journal(path, rows)
    report = render(artifact)
    assert "总体合规审计：**未通过**" in report


@pytest.mark.parametrize("reshape", ["10", "2, 5", "10, 1"])
def test_posthoc_shape_failure_classification(reshape: str) -> None:
    from experiments.isomorphic_report import _failure_type

    row = {"outcome": "GATE_REJECTED", "source": f"q = obs.book_shares.reshape(*raw.shape[:-1], {reshape})",
           "error": "RuntimeError: shape '[1, 9, 10]' is invalid for input of size 180"}
    assert _failure_type(row) == "BOOK_SIDE_LEVEL_SHAPE"
    assert _failure_type({**row, "source": "h.reshape(*raw.shape[:-1], 10)"}) == "OTHER_TENSOR_SHAPE"
    assert _failure_type({**row, "error": "unimplemented method"}) == "OTHER_GATE"
    assert _failure_type({**row, "evaluation": {"status": "VALID"}}) is None
    assert _failure_type({"outcome": "INVALID_PROPOSAL", "error": "proposal uses a reserved environment label"}) == "RESERVED_LABEL"
