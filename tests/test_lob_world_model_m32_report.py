"""Offline audit and five-table rendering checks for P2-M3.2 open-catalog & transplant report."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from experiments.open_catalog_report import _diagnosis_accurate, _z_axis_aligned, render
from tasks.lob_world_model.discovery_loop import SEED_SOURCE, View, build_messages, source_hash
from tasks.lob_world_model.folding_ledger import FoldingLedger
from tasks.lob_world_model.prior_cards import shuffled_cards


def _count(messages: list[dict[str, str]]) -> int:
    return 96 + sum((len(m['content']) + 3) // 4 for m in messages)


def _pair_box(est: float, lo: float, hi: float, days: int = 12) -> dict:
    return {'estimate': est, 'interval': [lo, hi], 'paired_days': days, 'registered_days': 12,
            'bootstrap_valid': 2000, 'bootstrap_total': 2000, 'daily': [est] * 12}


def test_diagnosis_and_z_axis_rules_distinguish_swapped_regimes() -> None:
    alpha_ok = {'outcome': 'PROMOTED', 'prediction': {
        'data_regime_diagnosis': 'interarrival_cv2=5.028 and midprice_shift_rate=0.639 show bursty thin book.',
        'weaker_when_z_metric': 'interarrival_cv2', 'weaker_when_z_direction': 'lower',
    }}
    alpha_swapped = {'outcome': 'PROMOTED', 'prediction': {
        'data_regime_diagnosis': 'sign_acf_sum=2.115 and flow_vr=1.724 indicate long-memory order splitting, midprice_shift_rate=0.',
        'weaker_when_z_metric': 'sign_acf_sum_1_32', 'weaker_when_z_direction': 'lower',
    }}
    assert _diagnosis_accurate(alpha_ok, 'env_alpha') and _z_axis_aligned(alpha_ok, 'env_alpha')
    assert not _diagnosis_accurate(alpha_ok, 'env_beta') and not _z_axis_aligned(alpha_ok, 'env_beta')
    assert not _diagnosis_accurate(alpha_swapped, 'env_alpha') and not _z_axis_aligned(alpha_swapped, 'env_alpha')
    assert _diagnosis_accurate(alpha_swapped, 'env_beta') and _z_axis_aligned(alpha_swapped, 'env_beta')


def test_render_m32_five_tables_with_open_innovation_and_transplant(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / 'artifacts' / 'p2-m3-isomorphic-control' / 'baseline.json').read_text())
    artifact = tmp_path / 'p2-m32-open-catalog-control'
    artifact.mkdir()
    replicates = ('rep_0', 'rep_1')
    arms = ('d_only', 'true_z_d', 'shuffled_z_d')
    envs = ('env_alpha', 'env_beta')

    cand_src = SEED_SOURCE.replace('input_projection(32)', 'input_projection(32)  # variant')
    cand_sha = source_hash(cand_src)
    b0_sha = source_hash(SEED_SOURCE)
    calls = []
    trajectories = {}
    approved = {'env_alpha': {b0_sha}, 'env_beta': {b0_sha}}

    for rep in replicates:
        for env in envs:
            parent_eval = baseline[env]['evaluation']
            other = 'env_beta' if env == 'env_alpha' else 'env_alpha'
            for arm in arms:
                z = None if arm == 'd_only' else baseline[env if arm == 'true_z_d' else other]['z']
                view = View(arm, env, z)
                t_root = artifact / 'trajectories' / rep / env / arm
                ledger = FoldingLedger(t_root)
                ledger.append({'cand_id': 'B0', 'parent_id': '', 'outcome': 'SEED', 'source': SEED_SOURCE,
                               'levels': parent_eval['aggregate'], 'parameters': parent_eval['params'],
                               'state_scalars': parent_eval['state'], 'source_sha256': b0_sha,
                               'evaluation': parent_eval})
                cur_id, cur_src, cur_eval, cur_sha = 'B0', SEED_SOURCE, parent_eval, b0_sha
                for r in range(1, 7):
                    cid = f'c{r:02d}'
                    cards = shuffled_cards(env, rep, r)
                    prompt, tokens = build_messages(view, cur_id, cur_src, cur_eval, cards, ledger, (), _count)
                    call_id = f'{rep}/{env}/{arm}/{cid}'
                    calls.append({'event': 'reserved', 'call_id': call_id, 'messages': prompt})
                    calls.append({'event': 'completed', 'call_id': call_id, 'raw_reply': f'reply-{call_id}',
                                  'input_tokens': tokens - 40, 'output_tokens': 180})
                    promote = (r == 1 and rep == 'rep_0' and arm == 'true_z_d')
                    open_cards = [] if (env == 'env_alpha' and promote) else ['C03']
                    reason = 'viability_recovery' if (env == 'env_alpha' and promote) else ('objective_improvement' if promote else None)
                    pred = {
                        'data_regime_diagnosis': ('interarrival_cv2=5.028 bursty thin book' if env == 'env_alpha'
                                                  else 'sign_acf_sum=2.115 persistent flow'),
                        'selected_card_ids': open_cards,
                        'regime_hypothesis': 'Mechanism works under active regime, weakens otherwise.',
                        'target_metric': 'dt_nll' if env == 'env_alpha' else 'sign_acf_abs_error',
                        'expected_direction': 'decrease',
                        'weaker_when_z_metric': 'interarrival_cv2' if env == 'env_alpha' else 'sign_acf_sum_1_32',
                        'weaker_when_z_direction': 'lower',
                        'unfold_ids': [],
                    }
                    eval_rec = {**parent_eval, 'source_sha256': cand_sha}
                    ledger.append({
                        'round': r, 'cand_id': cid, 'parent_id': cur_id, 'parent_source': cur_src,
                        'prompt': prompt, 'prompt_tokens': tokens, 'card_order': [c.card_id for c in cards],
                        'outcome': 'PROMOTED' if promote else 'REJECTED', 'promotion_reason': reason,
                        'source': cand_src, 'source_sha256': cand_sha, 'raw_reply': f'reply-{call_id}',
                        'prediction': pred, 'selected_card_ids': open_cards, 'unfold_ids': [],
                        'parameters': parent_eval['params'], 'state_scalars': parent_eval['state'],
                        'evaluation': eval_rec, 'levels': parent_eval['aggregate'],
                        'deltas': {'dt_nll': _pair_box(-.03, -.04, -.02), 'sign_acf_abs_error': _pair_box(-.02, -.03, -.01)},
                        'prediction_verdict': 'SUPPORTED' if promote else 'INCONCLUSIVE',
                    })
                    approved[env].add(cand_sha)
                    if promote:
                        cur_id, cur_src, cur_eval, cur_sha = cid, cand_src, eval_rec, cand_sha
                trajectories[f'{rep}/{env}/{arm}'] = {'final_id': cur_id, 'final_sha256': cur_sha}

    (artifact / 'registration.json').write_text(json.dumps({
        'stochopia_commit': 'deadbeef', 'openhyra_commit': 'cafebabe', 'model': 'gpt-6.2-sol',
        'replicates': list(replicates), 'formal_call_cap': 72,
    }))
    (artifact / 'baseline.json').write_text(json.dumps(baseline))
    (artifact / 'handshake.json').write_text(json.dumps({'status': 'PASS', 'input_tokens': 23, 'output_tokens': 11}))
    (artifact / 'api_calls.jsonl').write_text('\n'.join(json.dumps(c) for c in calls) + '\n')
    freeze_obj = {'approved': {k: sorted(v) for k, v in approved.items()}, 'trajectories': trajectories,
                  'formal_calls': 72, 'num_replicates': 2}
    (artifact / 'freeze.json').write_text(json.dumps(freeze_obj))
    freeze_sha = sha256((artifact / 'freeze.json').read_bytes()).hexdigest()
    holdout_dates = list(range(28, 40))
    holdout = {}
    for env in envs:
        (artifact / f'{env}-holdout-receipt.json').write_text(json.dumps({
            'env_id': env, 'dates': holdout_dates, 'freeze_sha256': freeze_sha, 'data_sha256': '00' * 32,
        }))
        base_eval = baseline[env]['evaluation']
        holdout[env] = {
            sha: {'evaluation': {**base_eval, 'source_sha256': sha},
                  'against_B0': {m: _pair_box(-.01 if sha == cand_sha else 0., -.02, -.005)
                                 for m in ('val_nll_total', 'dt_nll', 'exhausted_replicate_rate', 'counterfactual_impact_mae')}}
            for sha in approved[env]
        }
    (artifact / 'holdout.json').write_text(json.dumps(holdout))
    grp_metric = {**_pair_box(-.005, -.01, -.001), 'replicate_mean': -.005, 'replicate_sd': .007}
    grp_bundle = {m: grp_metric for m in ('val_nll_total', 'dt_nll', 'exhausted_replicate_rate', 'counterfactual_impact_mae')}
    in_tok = sum(c['input_tokens'] for c in calls if c['event'] == 'completed')
    out_tok = sum(c['output_tokens'] for c in calls if c['event'] == 'completed')
    (artifact / 'study_summary.json').write_text(json.dumps({
        'status': 'INCONCLUSIVE', 'alignment_supported': False, 'holdout_supported': False, 'holdout_refuted': False,
        'formal_calls': 72, 'formal_input_tokens': in_tok, 'formal_output_tokens': out_tok, 'unknown_usage_calls': 0,
        'wall_seconds': 123.4,
        'regret': {k: {'estimate': 0., 'paired_days': 12, 'oracle_models': 2} for k in trajectories},
        'track_effects': {k: holdout[k.split('/')[1]][v['final_sha256']]['against_B0'] for k, v in trajectories.items()},
        'cell_effects': {f'{e}/{a}': grp_bundle for e in envs for a in arms},
        'group_cross_arm_effects': {f'{e}/true_z_d-vs-{a}': grp_bundle for e in envs for a in ('d_only', 'shuffled_z_d')},
        'pooled_cross_arm_effects': {f'true_z_d-vs-{a}': grp_bundle for a in ('d_only', 'shuffled_z_d')},
    }))

    t_dir = artifact / 'transplants'
    t_dir.mkdir()
    world_z = {'w3_thin_short': {'interarrival_cv2': {'median': .94, 'intraday_iqr': .12}},
               'w6_thin_hawkes': {'interarrival_cv2': {'median': 5.03, 'intraday_iqr': .91}}}
    cand_entry = {
        'source_sha256': cand_sha, 'status': 'VALID',
        'tau': {'w3_thin_short': {'dt_nll': _pair_box(.004, .002, .006), 'val_nll_total': _pair_box(.0, -.01, .01),
                                  'exhausted_replicate_rate': _pair_box(.0, .0, .0)},
                'w6_thin_hawkes': {'dt_nll': _pair_box(-.02, -.03, -.01), 'val_nll_total': _pair_box(-.05, -.06, -.04),
                                   'exhausted_replicate_rate': _pair_box(-.12, -.15, -.09)}},
        'tau_difference': {'dt_nll': _pair_box(-.024, -.034, -.014), 'val_nll_total': _pair_box(-.05, -.07, -.03),
                           'exhausted_replicate_rate': _pair_box(-.12, -.15, -.09)},
    }
    (t_dir / 'clustering.json').write_text(json.dumps({
        'environments': ['w3_thin_short', 'w6_thin_hawkes'], 'world_z': world_z,
        'candidates': {cand_sha: cand_entry},
    }))
    claim = {
        'claim_id': 'rep_0-env_alpha-true_z_d-c01', 'origins': ['rep_0/env_alpha/true_z_d/c01'],
        'role': 'final_winner+open_innovation', 'promotion_reasons': ['viability_recovery'],
        'source_sha256': cand_sha, 'selected_card_ids': [], 'target_metric': 'dt_nll',
        'weaker_when_z_metric': 'interarrival_cv2', 'weaker_when_z_direction': 'lower',
        'pair_name': 'clustering', 'environments': ['w3_thin_short', 'w6_thin_hawkes'],
        'boundary_evaluation': {'verdict': 'BOUNDARY_CONFIRMED'},
    }
    (artifact / 'transplant-freeze.json').write_text(json.dumps({
        'pairs': {'clustering': {'environments': ['w3_thin_short', 'w6_thin_hawkes'], 'candidate_digests': [cand_sha]}},
        'claims': [claim],
    }))
    tf_sha = sha256((artifact / 'transplant-freeze.json').read_bytes()).hexdigest()
    (t_dir / 'transplant_summary.json').write_text(json.dumps({
        'api_calls': 0, 'transplant_freeze_sha256': tf_sha, 'unique_transplanted_models': 1,
        'total_claims': 1, 'pairs_evaluated': {'clustering': {}}, 'wall_seconds': 42.0,
        'verdict_counts': {'BOUNDARY_CONFIRMED': 1, 'BOUNDARY_REFUTED': 0, 'INCONCLUSIVE': 0},
        'claims': [claim],
    }))

    md = render(artifact, m31_artifact=root / 'artifacts' / 'p2-m31-calibrated-control')
    for heading in ('## 表 1', '## 表 2', '## 表 3', '## 表 4', '## 表 5'):
        assert heading in md
    assert 'viability_recovery' in md and 'objective_improvement' in md
    assert '[] (开放创新)' in md and 'BOUNDARY_CONFIRMED' in md
