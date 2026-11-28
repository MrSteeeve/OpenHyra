"""Offline information-view, one-call, and end-to-end loop regressions."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pytest

from tasks.lob_world_model.call_budget import CallBudgetGuard
from tasks.lob_world_model.contracts import Messages, ProviderReply
from tasks.lob_world_model.discovery_loop import (
    D_KEYS, OMITTED, PROMPT_CAP, SEED_SOURCE, SYSTEM, Z_KEYS,
    DiscoveryLoop, View, _z_text, build_messages, finite_window_nulls, prediction_verdict,
)
from tasks.lob_world_model.prior_cards import shuffled_cards

Json = dict[str, Any]
SECTIONS = ('Anonymous environment: ', '[1. Market Data Profile Z]', '[2. Prior Mechanism Catalog (C01-C10, Open Reference)]',
            '[3. Core Parent Residuals D & Trial Ledger]', '[4. Current Parent Source Code]', '[5. Objective & Output Schema]')


def _count(messages: Messages) -> int:
    return 96 + sum((len(message['content']) + 2) // 3 for message in messages)


def _result(source: str = 'seed') -> Json:
    levels = {key: 0.15 for key in D_KEYS}
    levels.update(val_nll_total=5.0, dt_nll=2.0, event_nll=1.8, level_nll=.9, size_nll=.3,
                  counterfactual_impact_mae=1.0, exhausted_replicate_rate=.5)
    return {'status': 'VALID', 'aggregate': levels, 'params': 24000, 'state': 60,
            'support': {'counterfactual_impact_mae': {'paired_days': 7, 'registered_days': 12}},
            'training_checkpoint_hashes': {'103': 'a', '137': 'b', '211': 'c'}, 'source': source}


def _z(value: float) -> Json:
    return {key: {'median': value, 'intraday_iqr': .12} for key in Z_KEYS}


def _pair(candidate: Json, parent: Json) -> Json:
    return {key: {'estimate': -.02, 'interval': [-.025, -.015], 'paired_days': 12}
            for key in D_KEYS}


SOURCE = '''from torch import nn
from stochopia.neural.probe_features import ProbeBase

class CandidateLOBWorldModel(ProbeBase):
    """Thin-book Hawkes clock with long memory (ordinary wording is allowed)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(75, 60)
        self.mix = nn.Linear(60, 60)
    def step_context(self, obs, state):
        x = self.encoder(self.raw_features(obs)).tanh()
        h = x if state is None else (x + self.mix(state)).tanh()
        return h, h
    def sequence_context(self, obs):
        x = self.encoder(self.raw_features(obs)).tanh()
        h = x[:, 0]
        values = [h]
        for t in range(1, x.shape[1]):
            h = (x[:, t] + self.mix(h)).tanh()
            values.append(h)
        return __import__('torch').stack(values, 1)
'''


def _raw(cards: list[str], target_metric: str = 'dt_nll', z_metric: str = 'interarrival_cv2',
         z_dir: str = 'lower') -> str:
    obj = {'data_regime_diagnosis': 'Bursty, thick-tailed waits (Hawkes-like rather than Poisson) in a thin book.',
           'selected_card_ids': cards,
           'regime_hypothesis': 'A recurrent gain sharpens the clock; it should not help under long, short or thick '
                                'Poisson-like flow with constant rate.',
           'target_metric': target_metric, 'expected_direction': 'decrease',
           'weaker_when_z_metric': z_metric, 'weaker_when_z_direction': z_dir, 'unfold_ids': []}
    return f'```json\n{json.dumps(obj)}\n```\n```python\n{SOURCE}\n```'


def _reply(messages: Messages, *, response_json: bool) -> ProviderReply:
    catalog = messages[1]['content'].split(SECTIONS[2])[1].split(SECTIONS[3])[0]
    return ProviderReply(_raw(re.findall(r'^(C\d\d) ', catalog, flags=re.MULTILINE)[:1]), 100, 200)


def _without_z(messages: Messages) -> Messages:
    user = messages[1]['content']
    start, rest = user.split(SECTIONS[1] + '\n', 1)
    _, suffix = rest.split('\n\n' + SECTIONS[2], 1)
    return [messages[0], {'role': 'user', 'content': start + SECTIONS[1] + suffix}]


def _prompts(tmp_path: Path, parent: Json, z_true: Json, z_other: Json, env: str = 'env_alpha') -> list[Messages]:
    views = (View('d_only', env, None), View('true_z_d', env, z_true), View('shuffled_z_d', env, z_other))
    prompts = []
    for view in views:
        loop = DiscoveryLoop(view=view, root=tmp_path / view.arm, seed_result=parent, evaluator=_result,
                             pair=_pair, count=_count, replicate_id='rep_1')
        prompt, count = build_messages(view, 'B0', SEED_SOURCE, parent, loop.cards(1), loop.ledger, (), _count)
        assert count <= PROMPT_CAP
        prompts.append(prompt)
    return prompts


def test_data_first_layout_isomorphic_except_z(tmp_path: Path) -> None:
    prompts = _prompts(tmp_path, _result(), _z(1.0), _z(2.0))
    assert _without_z(prompts[0]) == _without_z(prompts[1]) == _without_z(prompts[2])
    user = prompts[1][1]['content']
    assert [user.index(section) for section in SECTIONS] == sorted(user.index(section) for section in SECTIONS)
    assert user.startswith('Anonymous environment: env_alpha\n[1. Market Data Profile Z]\n')
    assert OMITTED in prompts[0][1]['content']
    assert 'metric | median | IQR | reference_and_interpretation' in user
    assert 'interarrival_cv2 | 1 | 0.12 | var/mean^2 of waits; 1.0 for stationary Poisson arrivals' in user
    assert 'sign_acf_sum_1_32 | 2 |' in prompts[2][1]['content']
    # Six grouped residual lines with response support; the full catalog in shuffled order.
    d = user.split(SECTIONS[3] + '\n')[1].split('\n\nTrial ledger:')[0].splitlines()
    assert [line.split(':')[0] for line in d] == ['nll_overview', 'timing_clustering', 'rollout_viability',
                                                 'flow_memory', 'impact_response', 'drift_stability']
    assert 'counterfactual_impact_mae=1 (7/12 valid days)' in d[4]
    catalog = user.split(SECTIONS[2])[1].split(SECTIONS[3])[0]
    order = re.findall(r'^(C\d\d) ', catalog, flags=re.MULTILINE)
    assert sorted(order) == [f'C{i:02d}' for i in range(1, 11)]
    assert order == [card.card_id for card in shuffled_cards('env_alpha', 'rep_1', 1)]


def test_card_order_is_deterministic_and_shared_only_within_a_round() -> None:
    def ids(env: str, rep: str, round_idx: int) -> list[str]:
        return [card.card_id for card in shuffled_cards(env, rep, round_idx)]
    assert ids('env_alpha', 'rep_0', 1) == ids('env_alpha', 'rep_0', 1)
    orders = {tuple(ids(env, rep, r)) for env in ('env_alpha', 'env_beta') for rep in ('rep_0', 'rep_1') for r in (1, 2)}
    assert len(orders) > 1 and all(sorted(order) == [f'C{i:02d}' for i in range(1, 11)] for order in orders)


def test_calibrated_first_round_on_frozen_actual_residuals(tmp_path: Path) -> None:
    """Use M3's unmodified B0 receipts rather than an invented residual panel."""
    baseline_path = (Path(__file__).resolve().parents[1] / 'artifacts' /
                     'p2-m3-isomorphic-control' / 'baseline.json')
    baseline = json.loads(baseline_path.read_text())
    for env, other in (('env_alpha', 'env_beta'), ('env_beta', 'env_alpha')):
        parent = baseline[env]['evaluation']
        prompts = _prompts(tmp_path / env, parent, baseline[env]['z'], baseline[other]['z'], env)
        assert _without_z(prompts[0]) == _without_z(prompts[1]) == _without_z(prompts[2])
        assert all(f'C{i:02d} ' in prompts[0][1]['content'] for i in range(1, 11))
    for shape in ('[B, 2, 10]', '[B, T, 2, 10]', '[B, 4]', '[B, T, 4]', '[B]', '[B, T]', '[B, 60]', '[B, T, 60]'):
        assert shape in SYSTEM
    assert '20 elements per sample' in SYSTEM and 'NEVER reshape/view to just 10' in SYSTEM


def test_next_prompt_receives_one_line_gate_feedback(tmp_path: Path) -> None:
    guard = CallBudgetGuard(tmp_path / 'calls.jsonl', limit=2)
    sent = []

    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        sent.append(messages)
        return _reply(messages, response_json=response_json)

    error = "Worker crash (returncode 1)\n  File '/tmp/private/candidate.py', line 4\nRuntimeError: shape '[4, 10]' is invalid for input of size 80"
    loop = DiscoveryLoop(view=View('d_only', 'env_alpha', None), root=tmp_path / 'track', seed_result=_result(),
                         evaluator=lambda source: {'status': 'GATE_REJECTED', 'error': error}, pair=_pair,
                         count=_count, transport=transport, guard=guard, rounds=2)
    records = loop.run()['records']
    reason = "RuntimeError: shape '[4, 10]' is invalid for input of size 80"
    assert records[1]['outcome'] == 'GATE_REJECTED' and records[1]['gate_error'] == reason
    assert records[1]['error'] == error
    assert 'reason=' + reason in sent[1][1]['content'].split('[Branch taboos]')[1]
    assert '/tmp/private' not in sent[1][1]['content']
    assert guard.count == 2


def test_professional_terms_and_open_structures_are_not_rejected(tmp_path: Path) -> None:
    """Thin/thick/Hawkes/Poisson/long are ordinary terms; empty card list + expanded target metrics work."""
    assert PROMPT_CAP == 4500
    guard = CallBudgetGuard(tmp_path / 'calls.jsonl', limit=1)
    loop = DiscoveryLoop(view=View('true_z_d', 'env_beta', _z(1.0)), root=tmp_path / 'track', seed_result=_result(),
                         evaluator=_result, pair=_pair, count=_count,
                         transport=lambda messages, response_json: ProviderReply(
                             _raw([], target_metric='level_nll', z_metric='midprice_shift_rate', z_dir='lower'), 100, 200),
                         guard=guard, rounds=1)
    row = loop.run()['records'][1]
    assert row['outcome'] == 'PROMOTED' and row['promotion_reason'] == 'objective_improvement'
    assert row['selected_card_ids'] == []
    assert row['prediction']['target_metric'] == 'level_nll'
    assert row['prediction']['weaker_when_z_metric'] == 'midprice_shift_rate'
    assert row['prediction_verdict'] == 'SUPPORTED'
    assert 'Hawkes' in row['source'] and 'Poisson' in row['prediction']['regime_hypothesis']
    assert row['prediction']['data_regime_diagnosis'].startswith('Bursty')


@pytest.mark.parametrize('arm', ('no_llm', 'greedy_score_llm', 'residual_d_llm', 'full_zd_llm',
                                 'd_only', 'true_z_d', 'shuffled_z_d'))
def test_offline_modes_run_with_identical_promotion_policy(tmp_path: Path, arm: str) -> None:
    z = _z(1.0) if arm in ('full_zd_llm', 'true_z_d', 'shuffled_z_d') else None
    guard = CallBudgetGuard(tmp_path / 'calls.jsonl', limit=2)
    loop = DiscoveryLoop(view=View(arm, 'env_beta', z), root=tmp_path / 'track', seed_result=_result(),
                         evaluator=_result, pair=_pair, count=_count, transport=_reply, guard=guard, rounds=2)
    result = loop.run()
    assert len(result['records']) == 3
    assert result['records'][1]['outcome'] == 'PROMOTED'
    assert guard.count == (0 if arm == 'no_llm' else 2)
    if arm == 'greedy_score_llm':
        assert result['records'][1]['card_order'] == []
        d = result['records'][1]['prompt'][1]['content'].split(SECTIONS[3] + '\n')[1].split('\n\nTrial ledger:')[0]
        assert len(d.splitlines()) == 2
    for row in result['records'][1:]:
        assert row['prompt_tokens'] <= PROMPT_CAP
    # Completed track re-opens the ledger without another request or evaluation.
    assert loop.run()['final_id'] == result['final_id']
    assert guard.count == (0 if arm == 'no_llm' else 2)


def test_format_failure_counts_once_without_training(tmp_path: Path) -> None:
    guard = CallBudgetGuard(tmp_path / 'calls.jsonl', limit=1)
    def forbidden(source: str) -> Json:
        pytest.fail('invalid proposal should not train')
    loop = DiscoveryLoop(view=View('d_only', 'env_alpha', None), root=tmp_path / 'track', seed_result=_result(),
                         evaluator=forbidden, pair=_pair, count=_count,
                         transport=lambda messages, response_json: ProviderReply('not valid output', 40, 8),
                         guard=guard, rounds=1)
    result = loop.run()
    assert guard.count == 1
    assert result['records'][1]['outcome'] == 'INVALID_PROPOSAL'
    assert result['final_id'] == 'B0'


def test_direction_is_not_point_estimate_only() -> None:
    pred = {'target_metric': 'dt_nll', 'expected_direction': 'decrease'}
    assert prediction_verdict(pred, {'dt_nll': {'estimate': -.2, 'interval': [-.5, .1]}}) == 'INCONCLUSIVE'
    assert prediction_verdict(pred, {'dt_nll': {'estimate': .2, 'interval': [.1, .3]}}) == 'REFUTED'
    assert prediction_verdict(pred, {'dt_nll': {'estimate': -.2, 'interval': [-.3, -.1]}}) == 'SUPPORTED'


def test_dynamic_finite_window_nulls_calibrate_reference_without_altering_default() -> None:
    z_default = _z(0.5)
    default_rendered = _z_text(z_default)
    assert 'n ~ 100 aggressive trades' in default_rendered
    assert 'finite-window null about -32/n;' in default_rendered

    nulls_33 = finite_window_nulls(33.0)
    assert nulls_33['sign_acf_sum_1_32'] == pytest.approx(-0.5)

    z_szse = {**z_default, 'mean_window_trades': 60.0}
    calibrated = _z_text(z_szse)
    nulls_60 = finite_window_nulls(60.0)
    assert 'n ~ 60.0 aggressive trades' in calibrated
    assert f'{nulls_60["sign_acf_sum_1_32"]:.3f}' in calibrated
    assert f'{nulls_60["sign_acf_sum_approx"]:.3f}' in calibrated
    assert f'{nulls_60["flow_vr_32_4"]:.3f}' in calibrated
    assert f'{nulls_60["flow_sign_bias"]:.3f}' in calibrated
