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
    D_KEYS, LABELS, OMITTED, PROMPT_CAP, SEED_SOURCE, Z_KEYS,
    DiscoveryLoop, View, build_messages, prediction_verdict,
)
from tasks.lob_world_model.prior_cards import select_active_cards

Json = dict[str, Any]


def _count(messages: Messages) -> int:
    return 96 + sum((len(message['content']) + 2) // 3 for message in messages)


def _result(source: str = 'seed') -> Json:
    levels = {key: 0.15 for key in D_KEYS}
    levels.update(val_nll_total=5.0, dt_nll=2.0, event_nll=1.8, level_nll=.9, size_nll=.3,
                  counterfactual_impact_mae=1.0, exhausted_replicate_rate=.5)
    return {'status': 'VALID', 'aggregate': levels, 'params': 24000, 'state': 60,
            'training_checkpoint_hashes': {'103': 'a', '137': 'b', '211': 'c'}, 'source': source}


def _z(value: float) -> Json:
    return {key: {'median': value, 'intraday_iqr': .12} for key in Z_KEYS}


def _pair(candidate: Json, parent: Json) -> Json:
    return {key: {'estimate': -.02, 'interval': [-.025, -.015], 'paired_days': 12}
            for key in D_KEYS}


def _reply(messages: Messages, *, response_json: bool) -> ProviderReply:
    text = messages[1]['content']
    cards = text.split('[Active Prior Cards]\n')[1].split('\n\n[Objective]')[0]
    active = re.findall(r'\bC\d\d\b', cards)
    obj = {'selected_card_ids': active[:1], 'target_metric': 'dt_nll', 'expected_direction': 'decrease',
           'regime_hypothesis': 'Explicit recurrent gain can improve the clock residual.', 'unfold_ids': []}
    source = 'from torch import nn\nfrom stochopia.neural.probe_features import ProbeBase\n' + '''
class CandidateLOBWorldModel(ProbeBase):
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
    raw = f'```json\n{json.dumps(obj)}\n```\n```python\n{source}\n```'
    return ProviderReply(raw, 100, 200)


def _without_z(messages: Messages) -> Messages:
    user = messages[1]['content']
    start, rest = user.split('[Macro Data Regime Z]\n', 1)
    _, suffix = rest.split('\n\n[Model Residuals D]', 1)
    return [messages[0], {'role': 'user', 'content': start + '[Macro Data Regime Z]\n' + suffix}]


def test_first_round_isomorphic_except_z(tmp_path: Path) -> None:
    parent = _result()
    views = (View('d_only', 'env_alpha', None), View('true_z_d', 'env_alpha', _z(1.0)),
             View('shuffled_z_d', 'env_alpha', _z(2.0)))
    prompts = []
    for view in views:
        loop = DiscoveryLoop(view=view, root=tmp_path / view.arm, seed_result=parent,
                             evaluator=_result, pair=_pair, count=_count)
        cards = select_active_cards(parent['aggregate'], {})
        prompt, count = build_messages(view, 'B0', SEED_SOURCE, parent, cards, loop.ledger, (), _count)
        assert count <= PROMPT_CAP
        assert not any(LABELS.search(item['content']) for item in prompt)
        prompts.append(prompt)
    assert _without_z(prompts[0]) == _without_z(prompts[1]) == _without_z(prompts[2])
    assert OMITTED in prompts[0][1]['content']
    assert 'sign_acf_sum_1_32 | 1 |' in prompts[1][1]['content']
    assert 'sign_acf_sum_1_32 | 2 |' in prompts[2][1]['content']


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
        assert result['records'][1]['active_card_ids'] == []
        d = result['records'][1]['prompt'][1]['content'].split('[Model Residuals D]\n')[1].split('\n\n[Active')[0]
        assert len(d.splitlines()) == 2
    for row in result['records'][1:]:
        assert row['prompt_tokens'] <= PROMPT_CAP
        assert not any(LABELS.search(item['content']) for item in row['prompt'])
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
