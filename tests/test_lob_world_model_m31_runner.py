"""Offline roster, no-replay and conditional replicate aggregation regressions."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

import pytest

from experiments import run_isomorphic_control as runner
from tasks.lob_world_model.call_budget import BudgetExceededError, CallBudgetGuard
from tasks.lob_world_model.contracts import Messages, ProviderReply

Json = dict[str, Any]


def _count(messages: Messages) -> int:
    return 96 + sum((len(message['content']) + 3) // 4 for message in messages)


def test_exact_108_slots_and_separate_replicates_with_no_new_calls_after_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise all 18 trajectories with deterministic invalid mock replies, never a trained candidate."""
    prior = runner.ROOT / 'artifacts' / 'p2-m3-isomorphic-control' / 'baseline.json'
    baseline = json.loads(prior.read_text(encoding='utf-8'))
    monkeypatch.setattr(runner, 'openai_token_counter', lambda: _count)
    guard = CallBudgetGuard(tmp_path / 'api_calls.jsonl', limit=108)
    calls: list[str] = []
    lock = Lock()
    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        with lock:
            reply = f'invalid offline unique response {len(calls):03d}'
            calls.append(reply)
        return ProviderReply(reply, 100, 20)
    monkeypatch.setattr(runner, 'registry_transport', lambda *args: transport)
    monkeypatch.setattr(runner, '_check_model_cache', lambda evaluator, result: result)
    runner._audit_first_round(tmp_path, baseline)
    with pytest.raises(RuntimeError, match='incomplete'):
        runner._make_freeze(tmp_path, guard, {env: None for env in runner.ENVS})
    runner._run_tracks(tmp_path, {env: None for env in runner.ENVS}, baseline, guard)
    assert len(calls) == guard.count == runner.MAX_CALLS == 108
    freeze = runner._make_freeze(tmp_path, guard, {env: None for env in runner.ENVS})
    assert len(freeze['trajectories']) == 18
    assert set(freeze['trajectories']) == {f'{rep}/{env}/{arm}' for rep in runner.REPLICATES
                                           for env in runner.ENVS for arm in runner.ARMS}
    assert {row['final_id'] for row in freeze['trajectories'].values()} == {'B0'}
    audit = json.loads((tmp_path / 'first_round_sampling_audit.json').read_text())
    for env in runner.ENVS:
        for arm in runner.ARMS:
            cell = audit[env][arm]
            assert cell['distinct_raw_replies'] == 3
            assert len({row['call_id'] for row in cell['replicates'].values()}) == 3
    # Arms share each replicate's card order; replicates use their own shuffle.
    initial = json.loads((tmp_path / 'initial_control_audit.json').read_text())
    for env in runner.ENVS:
        for rep in runner.REPLICATES:
            cell = initial[rep][env]
            assert len({cell[arm]['non_z_sha256'] for arm in runner.ARMS}) == 1
            assert len({tuple(cell[arm]['card_order']) for arm in runner.ARMS}) == 1
            assert sorted(cell['d_only']['card_order']) == list(runner.CARDS)
        assert len({tuple(initial[rep][env]['d_only']['card_order']) for rep in runner.REPLICATES}) > 1
    # Reopening all tracks never invokes the provider for completed slots.
    runner._run_tracks(tmp_path, {env: None for env in runner.ENVS}, baseline, guard)
    assert len(calls) == 108
    with pytest.raises(BudgetExceededError):
        guard.call(transport, [{'role': 'user', 'content': 'beyond cap'}], call_id='extra')
    assert len(calls) == 108


def test_aggregate_resamples_common_days_not_search_day_pseudoreplicates() -> None:
    metric = 'counterfactual_impact_mae'
    dates = list(range(12))
    effects = {
        'rep_0': {metric: {'estimate': -1.0, 'daily': [-1.0] * 12}},
        'rep_1': {metric: {'estimate': -2.0, 'daily': [None] * 2 + [-2.0] * 10}},
        'rep_2': {metric: {'estimate': -3.0, 'daily': [-3.0] * 8 + [None] * 4}},
    }
    metric_result = runner._aggregate_metrics(effects, dates)[metric]
    assert metric_result['common_dates'] == list(range(2, 8))
    assert metric_result['paired_days'] == 6 and metric_result['registered_days'] == 12
    assert metric_result['estimate'] == -2.0
    assert metric_result['interval'] == [-2.0, -2.0]
    assert metric_result['replicate_mean'] == -2.0 and metric_result['replicate_sd'] == 1.0
    assert metric_result['daily'][0:2] == [None, None]
    effects['rep_2'][metric]['daily'][7] = None
    assert runner._aggregate_metrics(effects, dates)[metric]['interval'] is None
    effects['rep_2'] = {}
    unknown = runner._aggregate_metrics(effects, dates)[metric]
    assert unknown['estimate'] is None and unknown['replicate_mean'] is None
    assert unknown['replicate_estimate_support'] == 2


def test_posterior_counts_only_finite_valid_candidates_and_shared_cards() -> None:
    seed = {'outcome': 'SEED'}
    valid = {'evaluation': {'status': 'VALID', 'aggregate': {'val_nll_total': 4.0}},
             'selected_card_ids': ['C03', 'C08', 'C03'],
             'prediction': {'target_metric': 'dt_nll', 'expected_direction': 'increase'}}
    improved = {**valid, 'deltas': {'dt_nll': {'interval': [-.3, -.1]}}}
    deteriorated = {**valid, 'deltas': {'dt_nll': {'interval': [.1, .3]}}}
    missing = {**valid, 'deltas': {'dt_nll': {'interval': None}}}
    failed = {**improved, 'evaluation': {'status': 'GATE_REJECTED'}}
    posterior = runner._prior_posterior({'rep_0/env_beta/d_only': [seed, improved, deteriorated, missing, failed]})
    for card in ('C03', 'C08'):
        assert posterior['env_beta'][card] == {'n_valid': 3, 'n_plus': 1, 'n_minus': 1,
                                              'n_inconclusive': 1, 'posterior_mean': .5}
    assert posterior['env_beta']['C06']['n_valid'] == 0
    assert posterior['env_beta']['C06']['posterior_mean'] == .5


def test_old_artifact_root_cannot_be_reopened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('p2-m3-isomorphic-control', 'p2-m31-calibrated-control'):
        root = tmp_path / name
        monkeypatch.setattr('sys.argv', ['run', '--artifact-root', str(root), '--work-root', str(tmp_path / 'work'),
                                        '--stochopia-root', str(tmp_path / 'stochopia'), '--num-replicates', '3'])
        with pytest.raises(ValueError, match='isolated'):
            runner.main()
        assert not root.exists()


def test_r2_72_slot_pilot_roster_and_zero_api_empty_transplant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prior = runner.ROOT / 'artifacts' / 'p2-m3-isomorphic-control' / 'baseline.json'
    baseline = json.loads(prior.read_text(encoding='utf-8'))
    monkeypatch.setattr(runner, 'openai_token_counter', lambda: _count)
    replicates = ('rep_0', 'rep_1')
    guard = CallBudgetGuard(tmp_path / 'api_calls.jsonl', limit=72)
    calls: list[str] = []
    lock = Lock()

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        with lock:
            reply = f'invalid offline unique response {len(calls):03d}'
            calls.append(reply)
        return ProviderReply(reply, 100, 20)

    monkeypatch.setattr(runner, 'registry_transport', lambda *args: transport)
    monkeypatch.setattr(runner, '_check_model_cache', lambda evaluator, result: result)
    runner._audit_first_round(tmp_path, baseline, replicates)
    runner._run_tracks(tmp_path, {env: None for env in runner.ENVS}, baseline, guard, replicates)
    assert len(calls) == guard.count == 72
    freeze = runner._make_freeze(tmp_path, guard, {env: None for env in runner.ENVS}, replicates)
    assert len(freeze['trajectories']) == 12 and freeze['num_replicates'] == 2
    loop2 = runner.run_loop2_transplants(tmp_path, tmp_path / 'work', freeze, replicates)
    assert loop2['api_calls'] == 0 and loop2['total_claims'] == 0
