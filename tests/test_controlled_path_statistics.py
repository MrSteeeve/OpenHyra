"""Statistical fixtures are software tests, not research-study observations."""
import json
from pathlib import Path
import subprocess
import tarfile

import pytest
from scipy.stats import binom, binomtest

from tasks.stochopia_structuring.controlled_statistics import (
    choose_n, frontier_event, paired_summary, paired_summaries, power_grid,
)
from tasks.stochopia_structuring.research_provenance import (
    capture_provenance, export_blind_review, verify_provenance,
)


def rows(outcomes, model="a", phase="confirmatory"):
    return [{"pair_id": k, "model_group": model, "phase": phase,
             "feedback_event": f, "independent_event": i}
            for k, (f, i) in enumerate(outcomes)]


def test_exact_significance_no_effect_and_only_five_discordants():
    significant = paired_summary(rows([(1, 0)] * 10 + [(0, 1)]), bootstrap_samples=1000)
    assert significant["exact_mcnemar_two_sided_pvalue"] == pytest.approx(binomtest(10, 11, .5).pvalue)
    assert significant["reject_null"]
    assert significant["feedback_advantage_supported"]
    assert significant["rate_difference"] == pytest.approx(9 / 11)
    assert significant["paired_bootstrap_ci"][0] > 0
    assert not significant["significance_claim_allowed"]  # Preregistration not proven by rows.
    balanced = paired_summary(rows([(1, 0)] * 6 + [(0, 1)] * 6))
    assert balanced["rate_difference"] == 0
    assert balanced["exact_mcnemar_two_sided_pvalue"] == 1
    five = paired_summary(rows([(1, 0)] * 5 + [(1, 1)] * 35))
    assert five["exact_mcnemar_two_sided_pvalue"] == pytest.approx(.0625)
    assert not five["reject_null"]


def test_negative_effect_rejects_two_sided_null_without_feedback_advantage():
    negative = paired_summary(rows([(0, 1)] * 10))
    assert negative["reject_null"]
    assert negative["rate_difference"] == -1
    assert not negative["feedback_advantage_supported"]
    assert negative["paired_bootstrap_ci_degenerate"]
    assert "Approximate" in negative["paired_bootstrap_ci_method"]
    assert "zero population uncertainty" in negative["paired_bootstrap_ci_caveat"]


def test_missing_pairs_are_attrition_not_failures():
    values = rows([(1, 0), (None, 1), (1, None), (None, None)])
    del values[2]["independent_event"]
    result = paired_summary(values)
    assert result["n_complete"] == 1 and result["n_missing"] == 3
    assert result["feedback_rate"] == 1 and result["independent_rate"] == 0
    assert result["attrition_rate"] == .75
    assert result["missing_pairs"][1]["missing_arms"] == ["independent"]


@pytest.mark.parametrize("extra", [rows([(1, 0)], "b"), rows([(1, 0)], phase="pilot")])
def test_mixing_models_or_phases_rejected(extra):
    with pytest.raises(ValueError, match="mix"):
        paired_summary(rows([(1, 0)]) + extra)


def test_no_paths_candidates_duplicate_pairs_or_adaptive_stopping():
    for bad in ({"replicate_unit": "market_path"}, {"replicate_unit": "candidate"}, {"adaptive_stopping": True}):
        with pytest.raises(ValueError):
            paired_summary([rows([(1, 0)])[0] | bad])
    with pytest.raises(ValueError, match="Unique"):
        paired_summary(rows([(1, 0)]) * 2)
    with pytest.raises(ValueError, match="booleans"):
        paired_summary(rows([("unreviewed", 0)]))


def test_holm_family_excludes_pilot_and_preserves_groups():
    dataset = rows([(1, 0)] * 6, "a") + rows([(1, 0)] * 10, "b") + rows([(1, 0)] * 12, "a", "pilot")
    summary = paired_summaries(dataset)
    assert len(summary["groups"]) == 3
    assert summary["primary_model_groups"] == ["a", "b"]
    confirm = {r["model_group"]: r for r in summary["groups"] if r["phase"] == "confirmatory"}
    assert confirm["b"]["holm_adjusted_pvalue"] == pytest.approx(2 * 2 / 2 ** 10)
    assert confirm["a"]["holm_adjusted_pvalue"] == pytest.approx(2 / 2 ** 6)
    assert "holm_adjusted_pvalue" not in summary["groups"][2]


def enumerated_power(probs, n):
    discordance = probs[1] + probs[2]
    total = 0.
    for d in range(n + 1):
        for k in range(d + 1):
            if d and binomtest(k, d, .5).pvalue <= .05:
                total += binom.pmf(d, n, discordance) * binom.pmf(k, d, probs[2] / discordance)
    return total


def test_exact_power_matches_direct_conditional_enumeration():
    probs = (.2, .1, .4, .3)
    result = power_grid(probs, (12,))
    assert result["grid"][0]["power"] == pytest.approx(enumerated_power(probs, 12))
    assert power_grid((.5, 0, 0, .5), (40,))["grid"][0]["power"] == 0


def test_joint_correlation_sensitivity_even_with_identical_marginals():
    # Both cases: feedback=.6 and independent=.4, but discordance .2 vs 1.
    correlated = power_grid((.4, 0, .2, .4), (40,))["grid"][0]["power"]
    anticorrelated = power_grid((0, .4, .6, 0), (40,))["grid"][0]["power"]
    assert correlated > .8 and anticorrelated < .3
    result = choose_n((.4, 0, .2, .4), limit=60)
    assert result["target_reached"] and result["n_pairs"] < 40
    assert choose_n((0, .4, .6, 0), limit=40)["n_pairs"] is None
    with pytest.raises(ValueError):
        power_grid((.6, .4))


def test_frontier_endpoint_is_target_constrained_and_same_market():
    values = dict(candidate_cost=91., candidate_fulfillment=.9, catalog_cost=95.,
                  catalog_fulfillment=.9, frontier_cost=90., fulfillment_target=.9,
                  tolerance=2., candidate_market_id="tree", catalog_market_id="tree",
                  frontier_market_id="tree")
    assert frontier_event(**values)
    assert not frontier_event(**(values | {"candidate_fulfillment": .89}))
    assert not frontier_event(**(values | {"candidate_cost": 94.}))
    with pytest.raises(ValueError, match="same finite-tree"):
        frontier_event(**(values | {"frontier_market_id": "different"}))
    with pytest.raises(ValueError, match="certified"):
        frontier_event(**(values | {"frontier_status": "time_limit"}))


def _repo(root):
    root.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    git("init", "-q")
    (root / "base.py").write_text("a = 1\n")
    git("add", "base.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
    (root / "base.py").write_text("a = 2\n")
    return root


def test_provenance_copies_untracked_adapters_without_secret_or_data_contents(tmp_path):
    roots = {label: _repo(tmp_path / label) for label in ("OpenHyra", "Stochopia", "StructurerDSL")}
    stochopia = roots["Stochopia"]
    adapter = stochopia / "stochopia" / "research" / "untracked_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("ADAPTER = True\n")
    for path in (".env", "private_key.txt", "secret_config.json", "data/sensitive.py", "artifacts/private.py"):
        destination = stochopia / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("SECRET_SENTINEL\n")
    for public_file in ("api_config.py", "api_models.json"):
        (stochopia / public_file).write_text('ENV_VARIABLE_NAME = "OPENAI_API_KEY"\n')
    # Also ensure a tracked secret modification cannot leak through the diff.
    subprocess.check_call(["git", "-C", str(stochopia), "add", ".env"])
    output = tmp_path / "provenance"
    manifest = capture_provenance(output, roots)
    assert len(manifest["repositories"]) == 3
    assert manifest["scipy_version"]
    assert set(manifest["dependency_versions"]) == {"numpy", "scipy", "torch", "structurerdsl", "httpx"}
    assert manifest["dependency_versions"]["scipy"] == manifest["scipy_version"]
    assert all(r["dirty"] and len(r["commit_sha"]) == 40 for r in manifest["repositories"].values())
    assert "stochopia/research/untracked_adapter.py" in manifest["repositories"]["Stochopia"]["untracked_source_paths"]
    with tarfile.open(manifest["source_archive"]) as archive:
        members = archive.getnames()
        assert "Stochopia/stochopia/research/untracked_adapter.py" in members
        assert "Stochopia/api_config.py" in members
        assert "Stochopia/api_models.json" in members
        for member in archive.getmembers():
            assert b"SECRET_SENTINEL" not in archive.extractfile(member).read()
    for diff in output.glob("*.diff"):
        text = diff.read_text()
        assert "SECRET_SENTINEL" not in text
        assert "+a = 2" in text


def test_blind_cards_do_not_leak_origin_and_remain_unreviewed(tmp_path):
    candidates = [{"candidate_id": f"origin-{i}", "model": "MODEL_SENTINEL", "arm": "feedback",
                   "parent": "parent-origin", "round": 8, "rationale": "RATIONALE_SENTINEL",
                   "demand": {"target": .9}, "market": {"sigma": .2}, "cashflows": [[0., float(i)]]}
                  for i in range(5)]
    result = export_blind_review(candidates, tmp_path, seed=7)
    public = Path(result["review_cards"]).read_text()
    for marker in ("MODEL_SENTINEL", "feedback", "parent-origin", "RATIONALE_SENTINEL", "origin-", '"round"'):
        assert marker not in public
    cards = json.loads(public)["cards"]
    assert [card["economic_payload"]["cashflows"][0][1] for card in cards] != list(range(5))
    assert all(card["review_status"] == "awaiting_human" for card in cards)
    assert all(all(value is None for value in card["assessment"].values()) for card in cards)
    assert "MODEL_SENTINEL" in Path(result["private_mapping"]).read_text()
    assert not result["expert_review_complete"]
    nested = export_blind_review([{"market": {"model_group": "leak", "sigma": .2}}], tmp_path / "nested")
    assert "leak" not in Path(nested["review_cards"]).read_text()
    with pytest.raises(ValueError, match="numerical"):
        export_blind_review([{"market": {"description": "model rationale leak"}}], tmp_path / "bad")


def test_blind_cards_accept_actual_public_demand_and_fixed_condition_enums(tmp_path):
    from stochopia.research.path_demand import PathDemand
    from stochopia.research.tree_market import TreeMarket
    demand = PathDemand(exit_scenarios=({'month': 6, 'probability': 1.,
                        'condition': {'month': 3, 'direction': 'below', 'level': 1.}},))
    result = export_blind_review([{"demand": demand.to_dict(),
                                  "market": TreeMarket({'months': 2}).describe(),
                                  "cashflows": [[0., 1., 2.]], "model": "hidden-origin"},
                                 {"demand": PathDemand().to_dict(),
                                  "market": TreeMarket({'months': 2}).config,
                                  "cashflows": [[0., 0., 1.]]}], tmp_path)
    public = Path(result["review_cards"]).read_text()
    assert 'stochopia-path-demand.v2' in public
    assert 'stochopia-shared-vol-tree.v1' in public
    assert 'below' in public and 'hidden-origin' not in public
    assert 'minimum_service_capital' in public
    legacy = export_blind_review([{"demand": PathDemand(funding_mode="fixed_budget_leg_control").to_dict(),
                                  "market": TreeMarket({'months': 2}).config,
                                  "cashflows": [[0., 0., 1.]]}], tmp_path / "legacy")
    assert 'fixed_budget_leg_control' in Path(legacy["review_cards"]).read_text()
    with pytest.raises(ValueError, match="fixed factual"):
        export_blind_review([{"demand": {"schema": "hidden-model-origin"}}], tmp_path / "bad-schema")


def test_verify_provenance_detects_modified_added_deleted_sources_and_commit_drift(tmp_path):
    roots = {label: _repo(tmp_path / label) for label in ("OpenHyra", "Stochopia", "StructurerDSL")}
    source = roots["Stochopia"] / "stochopia/research/scientific.py"
    source.parent.mkdir(parents=True)
    source.write_text("SCIENTIFIC = 1\n")
    manifest = capture_provenance(tmp_path / "preflight", roots)
    path = manifest["manifest_path"]
    assert verify_provenance(path)["verified"]
    # Unrelated documentation does not change the selected scientific source.
    (roots["OpenHyra"] / "notes.md").write_text("unrelated change\n")
    assert verify_provenance(path)["verified"]
    source.write_text("SCIENTIFIC = 2\n")
    with pytest.raises(ValueError, match="scientific source modified"):
        verify_provenance(path)
    source.write_text("SCIENTIFIC = 1\n")
    added = source.parent / "added.py"
    added.write_text("ADDED = True\n")
    with pytest.raises(ValueError, match="scientific source added"):
        verify_provenance(path)
    added.unlink()
    source.unlink()
    with pytest.raises(ValueError, match="scientific source removed"):
        verify_provenance(path)
    source.write_text("SCIENTIFIC = 1\n")
    subprocess.check_call(["git", "-C", str(roots["OpenHyra"]), "-c", "user.name=Test",
                           "-c", "user.email=test@example.com", "commit", "--allow-empty", "-qm", "drift"])
    with pytest.raises(ValueError, match="commit differs"):
        verify_provenance(path)


def test_verify_provenance_custom_selection_never_reads_secret_paths(tmp_path):
    roots = {label: _repo(tmp_path / label) for label in ("OpenHyra", "Stochopia", "StructurerDSL")}
    private = roots["Stochopia"] / "secrets/private.py"
    private.parent.mkdir()
    private.write_text("SECRET_SENTINEL\n")
    manifest = capture_provenance(tmp_path / "preflight", roots)
    private.write_text("SECRET_CHANGED\n")
    result = verify_provenance(manifest["manifest_path"], {"Stochopia": ("secrets/",)})
    assert result["verified"] and result["checked_source_paths"]["Stochopia"] == []


def test_experiment_entry_and_public_model_config_are_versioned_scientific_sources(tmp_path):
    from tasks.stochopia_structuring.research_provenance import verify_provenance
    roots={label:_repo(tmp_path/label) for label in ('OpenHyra','Stochopia','StructurerDSL')}
    root=roots['OpenHyra'];entry=root/'experiments/run_stochopia_path_research.py'
    entry.parent.mkdir();entry.write_text('RESEARCH_ENTRY = 1\n')
    settings=root/'tasks/stochopia_structuring/api_models.json';settings.parent.mkdir(parents=True)
    settings.write_text('{"provider":"official"}')
    manifest=capture_provenance(tmp_path/'provenance',roots)
    checked=verify_provenance(manifest['manifest_path'])['checked_source_paths']['OpenHyra']
    assert 'experiments/run_stochopia_path_research.py' in checked
    assert 'tasks/stochopia_structuring/api_models.json' in checked
    settings.write_text('{"provider":"changed"}')
    with pytest.raises(ValueError,match='api_models.json'):
        verify_provenance(manifest['manifest_path'])
