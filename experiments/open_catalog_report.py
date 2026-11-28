"""Read sealed P2-M3.2 open-catalog & Loop 2 transplant receipts and render five tables (0 API, 0 fits)."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from math import exp, isfinite, log
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from experiments.isomorphic_report import (
    ARMS, ENVIRONMENTS, _journal, _load, _number, _pair, _primary_match, _table, _valid_evaluation,
)
from tasks.lob_world_model.discovery_loop import OMITTED, PROMPT_CAP, _z_text
from tasks.lob_world_model.synthetic_eval import evaluate_boundary_claim

Json = dict[str, Any]
CARDS = tuple(f'C{i:02d}' for i in range(1, 11))
ROUNDS = 6
Z_HEADER = '[1. Market Data Profile Z]\n'
CARD_HEADER = '\n\n[2. Prior Mechanism Catalog'
HOLDOUT_METRICS = ('val_nll_total', 'dt_nll', 'exhausted_replicate_rate', 'counterfactual_impact_mae')
HOLDOUT_HEADS = ('Δval_nll_total', 'Δdt_nll', 'Δexhausted_replicate_rate', 'Δcounterfactual_impact_mae (tick)')

# Active physical Z axes of the two anonymous worlds:
# env_alpha = W6_thin_hawkes (thin shifting book + Hawkes clustering; short memory)
# env_beta  = W2_thick_long  (thick zero-shift book + Poisson clock + long directional memory)
TRUE_Z_AXES: dict[str, set[str]] = {
    'env_alpha': {'trade_to_top1_depth', 'midprice_shift_rate', 'cancel_to_limit_ratio',
                  'zero_wait_prob', 'interarrival_cv2'},
    'env_beta': {'sign_acf_sum_1_32', 'flow_vr_32_4', 'flow_sign_bias'},
}


def _finite(value: object) -> bool:
    return type(value) in (int, float) and isfinite(value)


def _rate(count: int, total: int) -> str:
    return f'{count}/{total} ({100 * count / total:.1f}%)' if total else 'NA'


def _mean_sd(values: Sequence[float | None]) -> str:
    known = [float(x) for x in values if _finite(x)]
    if not known:
        return f'NA (0/{len(values)})'
    spread = _number(stdev(known)) if len(known) > 1 else '0'
    return f'{_number(fmean(known))} ± {spread} ({len(known)}/{len(values)})'


def _range(values: Sequence[float | None]) -> str:
    known = [float(x) for x in values if _finite(x)]
    if not known:
        return f'NA (0/{len(values)})'
    return f'{_number(fmean(known))} [{_number(min(known))}, {_number(max(known))}] ({len(known)}/{len(values)})'


def _parsed(row: Json) -> bool:
    pred = row.get('prediction')
    return (isinstance(pred, Mapping) and bool(pred.get('data_regime_diagnosis'))
            and row.get('outcome') not in ('INVALID_PROPOSAL', 'PROVIDER_ERROR', 'INTERRUPTED_CALL'))


def _z_axis_aligned(row: Json, env: str) -> bool:
    if not _parsed(row):
        return False
    metric = row['prediction'].get('weaker_when_z_metric')
    direction = row['prediction'].get('weaker_when_z_direction')
    if metric in TRUE_Z_AXES[env] and direction == 'lower':
        return True
    # In env_beta (stationary best quotes, Poisson waits), exploiting stationarity/Poisson
    # validly predicts weaker gains when shift rate or interarrival CV² is higher.
    if env == 'env_beta' and metric in {'midprice_shift_rate', 'interarrival_cv2', 'zero_wait_prob'} and direction == 'higher':
        return True
    return False


def _z_section(messages: list[Json]) -> tuple[str, list[Json]]:
    content = messages[1]['content']
    before, rest = content.split(Z_HEADER, 1)
    z_body, after = rest.split(CARD_HEADER, 1)
    stripped = [messages[0], {'role': 'user', 'content': before + Z_HEADER + CARD_HEADER + after}]
    return z_body, stripped


def _expected_z_section(baseline: Json, env: str, arm: str) -> str:
    if arm == 'd_only':
        return OMITTED
    other = ENVIRONMENTS[1 - ENVIRONMENTS.index(env)]
    return _z_text(baseline[env if arm == 'true_z_d' else other]['z'])


def _entropy_uniformity(counts: Mapping[str, int]) -> tuple[float | None, float | None]:
    total = sum(counts.get(card, 0) for card in CARDS)
    if total == 0:
        return None, None
    probs = [counts.get(card, 0) / total for card in CARDS if counts.get(card, 0) > 0]
    h = -sum(p * log(p) for p in probs)
    return h / log(len(CARDS)), exp(h)


def _load_tracks(artifact: Path, replicates: Sequence[str]) -> dict[str, list[Json]]:
    rows = {}
    for rep in replicates:
        for env in ENVIRONMENTS:
            for arm in ARMS:
                track = f'{rep}/{env}/{arm}'
                ledger = _journal(artifact / 'trajectories' / track / 'ledger.jsonl')
                if len(ledger) != ROUNDS + 1 or ledger[0]['outcome'] != 'SEED':
                    raise ValueError(f'incomplete ledger {track}')
                if [row['cand_id'] for row in ledger[1:]] != [f'c{i:02d}' for i in range(1, ROUNDS + 1)]:
                    raise ValueError(f'invalid round roster {track}')
                rows[track] = ledger
    return rows


def _audit_study(artifact: Path, registration: Json, tracks: Mapping[str, list[Json]],
                 baseline: Json, freeze: Json, holdout: Json, calls: list[Json]) -> Json:
    replicates = tuple(registration['replicates'])
    cap = registration['formal_call_cap']
    reservations = {x['call_id']: x for x in calls if x['event'] == 'reserved'}
    completed = {x['call_id']: x for x in calls if x['event'] == 'completed'}
    expected = {f'{track}/c{i:02d}' for track in tracks for i in range(1, ROUNDS + 1)}
    if set(reservations) != expected or sum(x['event'] == 'reserved' for x in calls) != cap:
        raise ValueError(f'formal call roster is not {cap} distinct reservations')
    if set(completed) - expected or sum(x['event'] == 'completed' for x in calls) != len(completed):
        raise ValueError('formal terminal receipt roster is inconsistent')
    summary: Json = {}
    for env in ENVIRONMENTS:
        summary[env] = {}
        for rep in replicates:
            initial = [tracks[f'{rep}/{env}/{arm}'][1] for arm in ARMS]
            stripped = [_z_section(row['prompt'])[1] for row in initial]
            if any(x != stripped[0] for x in stripped[1:]):
                raise ValueError(f'non-Z first prompt differs in {rep}/{env}')
            if any(row['card_order'] != initial[0]['card_order'] for row in initial[1:]):
                raise ValueError(f'round-1 card order differs across arms in {rep}/{env}')
            if sorted(initial[0]['card_order']) != list(CARDS):
                raise ValueError(f'catalog in {rep}/{env} is not all 10 cards')
        for arm in ARMS:
            first_sources = [tracks[f'{rep}/{env}/{arm}'][1].get('source', '') for rep in replicates]
            first_replies = [completed.get(f'{rep}/{env}/{arm}/c01', {}).get('raw_reply') for rep in replicates]
            summary[env][arm] = {
                'distinct_first_source_hashes': len({sha256(x.encode()).hexdigest() for x in first_sources if x}),
                'distinct_first_reply_hashes': len({sha256(x.encode()).hexdigest() for x in first_replies if x}),
                'valid_first_replies': sum(bool(x) for x in first_replies),
            }
    for track, rows in tracks.items():
        _, env, arm = track.split('/')
        seed = rows[0]
        if seed['evaluation'] != baseline[env]['evaluation']:
            raise ValueError(f'initial checkpoint or residual differs: {track}')
        active = seed
        for row in rows[1:]:
            call_id = f'{track}/{row["cand_id"]}'
            if reservations[call_id]['messages'] != row['prompt']:
                raise ValueError(f'prompt receipt differs: {call_id}')
            if _z_section(row['prompt'])[0] != _expected_z_section(baseline, env, arm):
                raise ValueError(f'Z treatment differs: {call_id}')
            actual = completed.get(call_id, {}).get('input_tokens')
            if row['prompt_tokens'] > PROMPT_CAP or (_finite(actual) and actual > PROMPT_CAP):
                raise ValueError(f'prompt budget overrun: {call_id}')
            if row['outcome'] == 'PROMOTED':
                active = row
        if freeze['trajectories'][track] != {'final_id': active['cand_id'], 'final_sha256': active['source_sha256']}:
            raise ValueError(f'final selection not frozen from development: {track}')
        for row in rows:
            if _valid_evaluation(row):
                digest = row['source_sha256']
                if digest not in freeze['approved'][env] or digest not in holdout[env]:
                    raise ValueError(f'valid candidate missing holdout authorization: {track}')
                test = holdout[env][digest]['evaluation']
                if test.get('training_checkpoint_hashes') != row['evaluation']['training_checkpoint_hashes']:
                    raise ValueError('holdout refitted or replaced checkpoint')
    freeze_digest = sha256((artifact / 'freeze.json').read_bytes()).hexdigest()
    for env in ENVIRONMENTS:
        if set(freeze['approved'][env]) != set(holdout[env]):
            raise ValueError('holdout roster differs from frozen approval')
        receipt = _load(artifact / f'{env}-holdout-receipt.json')
        development = baseline[env]['development_receipt']
        dates = development['train_dates'] + development['dev_dates'] + receipt['dates']
        if len(set(dates)) != 40 or receipt['freeze_sha256'] != freeze_digest:
            raise ValueError('holdout day split or freeze receipt differs')
    return summary


def _stage_stats(rows: Sequence[Json], completed_by_id: Mapping[str, Json], prefix_map: Mapping[int, str]) -> list[object]:
    n = len(rows)
    valid = sum(_valid_evaluation(r) for r in rows)
    est_max = max((r['prompt_tokens'] for r in rows), default=0)
    act_tokens = [completed_by_id[prefix_map[id(r)]]['input_tokens']
                  for r in rows if prefix_map.get(id(r)) in completed_by_id
                  and _finite(completed_by_id[prefix_map[id(r)]].get('input_tokens'))]
    act_max = max(act_tokens) if act_tokens else None
    parsed_rows = [r for r in rows if r.get('prediction') and r.get('outcome') not in ('INVALID_PROPOSAL', 'PROVIDER_ERROR', 'INTERRUPTED_CALL')]
    open_cnt = sum(r.get('selected_card_ids') == [] for r in parsed_rows)
    primary_counts = Counter(r['selected_card_ids'][0] for r in parsed_rows if r.get('selected_card_ids'))
    cited_counts = Counter(c for r in parsed_rows for c in set(r.get('selected_card_ids', [])))
    unif, eff = _entropy_uniformity(primary_counts)
    unif_str = f'{_number(unif)} ({_number(eff)} 卡)' if unif is not None else 'NA'
    denom = len(parsed_rows)
    return [n, _rate(valid, n), est_max, _number(act_max), _rate(open_cnt, denom), unif_str,
            _rate(primary_counts['C09'], denom),
            f"{_rate(primary_counts['C06'], denom)} (含副卡 {_rate(cited_counts['C06'], denom)})",
            f"{_rate(primary_counts['C08'], denom)} (含副卡 {_rate(cited_counts['C08'], denom)})"]


def _table1_comparison(m31_artifact: Path | None, m32_tracks: Mapping[str, list[Json]],
                       m32_calls: list[Json], audit: Json, replicates: Sequence[str]) -> str:
    m32_completed = {x['call_id']: x for x in m32_calls if x['event'] == 'completed'}
    m32_id_map = {id(r): f'{track}/{r["cand_id"]}' for track, rows in m32_tracks.items() for r in rows[1:]}
    rows_out = []
    dist_rows = []
    stages: list[tuple[str, Mapping[str, list[Json]], Mapping[str, Json], Mapping[int, str]]] = []
    if m31_artifact is not None and (m31_artifact / 'study_summary.json').exists():
        m31_reg = _load(m31_artifact / 'registration.json')
        m31_tracks = _load_tracks(m31_artifact, m31_reg['replicates'])
        m31_calls = _journal(m31_artifact / 'api_calls.jsonl')
        m31_comp = {x['call_id']: x for x in m31_calls if x['event'] == 'completed'}
        m31_map = {id(r): f'{track}/{r["cand_id"]}' for track, ledger in m31_tracks.items() for r in ledger[1:]}
        stages.append(('M3.1 (3卡筛卡)', m31_tracks, m31_comp, m31_map))
    stages.append(('M3.2 (10卡全展乱序)', m32_tracks, m32_completed, m32_id_map))

    for env in (*ENVIRONMENTS, '双环境合计'):
        for label, trks, comp, id_map in stages:
            subset = [r for key, ledger in trks.items()
                      if env == '双环境合计' or key.split('/')[1] == env
                      for r in ledger[1:]]
            rows_out.append([env, label, *_stage_stats(subset, comp, id_map)])
            parsed = [r for r in subset if r.get('prediction') and r.get('outcome') not in ('INVALID_PROPOSAL', 'PROVIDER_ERROR', 'INTERRUPTED_CALL')]
            prim = Counter(r['selected_card_ids'][0] if r.get('selected_card_ids') else '[]' for r in parsed)
            dist_rows.append([env, label, len(parsed), prim['[]'], *[prim[c] for c in CARDS]])

    t1a = _table(('环境', '版本', '尝试', '门禁及评估有效', '预估最大 token', '实际最大 token',
                  '开放创新 ([])', '主卡均匀度 H/ln10', '主卡 C09', 'C06 主卡(含副卡)', 'C08 主卡(含副卡)'), rows_out)
    t1b = _table(('环境', '版本', '解析有效', '[]', *CARDS), dist_rows)
    n_rep = len(replicates)
    div_rows = [[env, arm, audit[env][arm]['distinct_first_reply_hashes'],
                 audit[env][arm]['distinct_first_source_hashes'], audit[env][arm]['valid_first_replies']]
                for env in ENVIRONMENTS for arm in ARMS]
    t1c = _table(('环境', '臂', f'首轮不同回复哈希 / {n_rep}', f'不同源码哈希 / {n_rep}', f'有回复 / {n_rep}'), div_rows)
    return '\n\n'.join((t1a, '**主卡与开放创新 `[]` 分布明细**', t1b, '**首轮同构与跨副本采样独立性核验**', t1c))


def _feedback_audit_line(tracks: Mapping[str, list[Json]]) -> str:
    failure_types = {'GATE_REJECTED', 'INVALID_PROPOSAL', 'DUPLICATE_REJECTED', 'NONFINITE_LOSS', 'EVALUATION_FAILED'}
    failures = opportunities = visible = 0
    for ledger in tracks.values():
        for index, row in enumerate(ledger[1:], start=1):
            if row['outcome'] not in failure_types:
                continue
            reason = row.get('gate_error')
            if not isinstance(reason, str) or not reason or len(reason) > 120 or '\n' in reason:
                raise ValueError('missing or malformed gate-error receipt')
            failures += 1
            if index + 1 < len(ledger):
                opportunities += 1
                if any(reason in message['content'] for message in ledger[index + 1]['prompt']):
                    visible += 1
    return (f'失败回执单行脱敏错误共 {failures} 条（均 ≤120 字符）；有下一轮的 {opportunities} 次中，'
            f'实际下一轮提示保留该错误 {visible}/{opportunities} 次。')


def _table2_alignment(tracks: Mapping[str, list[Json]], replicates: Sequence[str]) -> str:
    rows_out = []
    n_rep = len(replicates)
    cell_n = n_rep * ROUNDS
    for env in ENVIRONMENTS:
        for arm in ARMS:
            per_rep = [tracks[f'{rep}/{env}/{arm}'][1:] for rep in replicates]
            flat = [r for trk in per_rep for r in trk]
            z_axis_ok = sum(_z_axis_aligned(r, env) for r in flat)
            aligned_per_rep = [sum(_primary_match(r, env) for r in trk) / ROUNDS for trk in per_rep]
            aligned_tot = sum(_primary_match(r, env) for r in flat)
            open_tot = sum(_parsed(r) and r.get('selected_card_ids') == [] for r in flat)
            metrics_used = sorted({r['prediction']['target_metric'] for r in flat if _parsed(r)})
            verdicts = Counter(r.get('prediction_verdict', 'NOT_EVALUABLE') for r in flat)
            s, ref, inc, ne = (verdicts[k] for k in ('SUPPORTED', 'REFUTED', 'INCONCLUSIVE', 'NOT_EVALUABLE'))
            rows_out.append([
                env, arm,
                _rate(z_axis_ok, cell_n),
                _rate(aligned_tot, cell_n),
                _mean_sd(aligned_per_rep),
                _rate(open_tot, cell_n),
                f'{len(metrics_used)} 种 ({", ".join(metrics_used)})' if metrics_used else '0',
                f'{s}/{ref}/{inc}/{ne}',
                _rate(s, s + ref),
            ])
    for arm in ARMS:
        flat_env = [(env, r) for env in ENVIRONMENTS for rep in replicates for r in tracks[f'{rep}/{env}/{arm}'][1:]]
        tot_n = len(flat_env)
        z_axis_ok = sum(_z_axis_aligned(r, env) for env, r in flat_env)
        aligned_tot = sum(_primary_match(r, env) for env, r in flat_env)
        open_tot = sum(_parsed(r) and r.get('selected_card_ids') == [] for _, r in flat_env)
        metrics_used = sorted({r['prediction']['target_metric'] for _, r in flat_env if _parsed(r)})
        verdicts = Counter(r.get('prediction_verdict', 'NOT_EVALUABLE') for _, r in flat_env)
        s, ref, inc, ne = (verdicts[k] for k in ('SUPPORTED', 'REFUTED', 'INCONCLUSIVE', 'NOT_EVALUABLE'))
        rows_out.append([
            '两环境合计', arm,
            _rate(z_axis_ok, tot_n),
            _rate(aligned_tot, tot_n),
            '—',
            _rate(open_tot, tot_n),
            f'{len(metrics_used)} 种',
            f'{s}/{ref}/{inc}/{ne}',
            _rate(s, s + ref),
        ])
    return _table((
        '环境', '臂', 'Z 边界轴对齐',
        '主卡对齐', '轨迹主卡比例均值 ± SD', '开放创新 ([])',
        '绑定 target_metric 覆盖', '方向 SUPPORTED/REFUTED/INCONCLUSIVE/未测', '命中 / 可判方向',
    ), rows_out)


def _table3_promotion_regret(tracks: Mapping[str, list[Json]], study: Json, replicates: Sequence[str]) -> str:
    rows_out = []
    for env in ENVIRONMENTS:
        for arm in ARMS:
            per_rep = [tracks[f'{rep}/{env}/{arm}'][1:] for rep in replicates]
            valid = [sum(_valid_evaluation(r) for r in trk) for trk in per_rep]
            prom = [sum(r['outcome'] == 'PROMOTED' for r in trk) for trk in per_rep]
            obj = [sum(r['outcome'] == 'PROMOTED' and r.get('promotion_reason') == 'objective_improvement' for r in trk)
                   for trk in per_rep]
            viab = [sum(r['outcome'] == 'PROMOTED' and r.get('promotion_reason') == 'viability_recovery' for r in trk)
                    for trk in per_rep]
            stone = [sum(r['outcome'] == 'STEPPING_STONE' for r in trk) for trk in per_rep]
            dedup = [sum(r['outcome'] == 'DUPLICATE_REJECTED' for r in trk) for trk in per_rep]
            regrets = [study['regret'][f'{rep}/{env}/{arm}'] for rep in replicates]
            rows_out.append([
                env, arm, _range(valid),
                f'{sum(prom)} ({_range(prom)})',
                f'{sum(obj)} ({_range(obj)})',
                f'{sum(viab)} ({_range(viab)})',
                f'{sum(stone)} ({_range(stone)})',
                f'{sum(dedup)} ({_range(dedup)})',
                _range([x['estimate'] for x in regrets]),
                ', '.join(str(x['paired_days']) for x in regrets),
                ', '.join(str(x['oracle_models']) for x in regrets),
            ])
    return _table((
        '环境', '臂', '有效候选/轨迹', '总晋升', 'objective_improvement',
        'viability_recovery', '踏脚石', '去重', '留出 Regret (tick)',
        f'共同有效日 (R={len(replicates)})', 'Oracle 模型数',
    ), rows_out)


def _group_pair(row: Json) -> str:
    return f'{_pair(row)}；R均值 {_number(row.get("replicate_mean"))}；SD {_number(row.get("replicate_sd"))}'


def _table4_holdout(tracks: Mapping[str, list[Json]], study: Json, freeze: Json, replicates: Sequence[str]) -> str:
    individual = []
    for rep in replicates:
        for env in ENVIRONMENTS:
            for arm in ARMS:
                key = f'{rep}/{env}/{arm}'
                final_id = freeze['trajectories'][key]['final_id']
                final_row = next(r for r in tracks[key] if r['cand_id'] == final_id)
                reason = final_row.get('promotion_reason')
                label = f'{final_id} ({reason})' if reason else final_id
                effects = study['track_effects'][key]
                individual.append([rep, env, arm, label, *[_pair(effects.get(m)) for m in HOLDOUT_METRICS]])
    grouped = []
    for env in ENVIRONMENTS:
        for arm in ARMS:
            key = f'{env}/{arm}'
            grouped.append([key, *[_group_pair(study['cell_effects'][key][m]) for m in HOLDOUT_METRICS]])
        for arm in ('d_only', 'shuffled_z_d'):
            key = f'{env}/true_z_d-vs-{arm}'
            grouped.append([key, *[_group_pair(study['group_cross_arm_effects'][key][m]) for m in HOLDOUT_METRICS]])
    for arm in ('d_only', 'shuffled_z_d'):
        key = f'true_z_d-vs-{arm}'
        grouped.append([f'双环境等权/{key}', *[_group_pair(study['pooled_cross_arm_effects'][key][m]) for m in HOLDOUT_METRICS]])
    return (_table(('轨迹', '环境', '臂', '最终父模型', *HOLDOUT_HEADS), individual) + '\n\n' +
            _table((f'R={len(replicates)} 汇总与组间对照', *HOLDOUT_HEADS), grouped))


def _audit_and_render_table5(artifact: Path) -> tuple[str, Json]:
    freeze_roster = _load(artifact / 'transplant-freeze.json')
    summary = _load(artifact / 'transplants' / 'transplant_summary.json')
    expected_digest = sha256((artifact / 'transplant-freeze.json').read_bytes()).hexdigest()
    if summary.get('transplant_freeze_sha256') != expected_digest or summary.get('api_calls') != 0:
        raise ValueError('Loop 2 transplant summary does not match sealed transplant-freeze.json')
    pair_files = {pair: _load(artifact / 'transplants' / f'{pair}.json') for pair in freeze_roster['pairs']}
    if not summary['claims']:
        return '本轮无晋升或显著开放创新候选进入单因子跨环境移植名单（0 条候选）。', summary
    rows_out = []
    for claim in summary['claims']:
        pair_name = claim['pair_name']
        pair_data = pair_files[pair_name]
        low, high = pair_data['environments']
        entry = pair_data['candidates'][claim['source_sha256']]
        rechecked = evaluate_boundary_claim(
            entry, pair_data['world_z'], claim['weaker_when_z_metric'],
            claim['weaker_when_z_direction'], claim['target_metric'],
        )
        if rechecked['verdict'] != claim['boundary_evaluation']['verdict']:
            raise ValueError(f"boundary verdict mismatch for {claim['claim_id']}")
        cards = ', '.join(claim['selected_card_ids']) if claim['selected_card_ids'] else '[] (开放创新)'
        reasons = '/'.join(claim.get('promotion_reasons') or []) or 'dev显著开放候选'
        role = f"{claim['role']} ({reasons})"
        target = claim['target_metric']
        if entry.get('status') == 'VALID':
            tau_low = _pair(entry['tau'][low][target])
            tau_high = _pair(entry['tau'][high][target])
            tau_diff_hl = _pair(entry['tau_difference'][target])
            delta_sw = _pair(rechecked['delta_strong_minus_weak'])
            nll_diff = _pair(entry['tau_difference']['val_nll_total'])
            exh_diff = _pair(entry['tau_difference']['exhausted_replicate_rate'])
            aux = f'Δτ_NLL={nll_diff}；Δτ_Exh={exh_diff}'
        else:
            tau_low = tau_high = tau_diff_hl = delta_sw = aux = f"UNAVAILABLE ({entry.get('dev_status')})"
        rows_out.append([
            ', '.join(claim['origins']) + f" (`{claim['source_sha256'][:10]}`)",
            role,
            cards,
            f"{claim['weaker_when_z_metric']} ({claim['weaker_when_z_direction']})",
            target,
            f'{pair_name}: {low} ↔ {high}',
            tau_low,
            tau_high,
            tau_diff_hl,
            delta_sw,
            aux,
            rechecked['verdict'],
        ])
    table = _table((
        '候选来源 (源码SHA)', '角色与晋升类别', '引用卡', '声明弱化边界 (Z / 方向)',
        '目标指标', '单因子配对世界 (z0 ↔ z1)', 'τ(z0) 目标效应', 'τ(z1) 目标效应',
        'τ(z1)−τ(z0) [95% CI]', 'Δτ(strong−weak) [95% CI]', '全样本无空值护栏差 (z1−z0)', '边界证伪判定',
    ), rows_out)
    return table, summary


def render(artifact: Path, *, m31_artifact: Path | None = None) -> str:
    registration = _load(artifact / 'registration.json')
    baseline = _load(artifact / 'baseline.json')
    freeze = _load(artifact / 'freeze.json')
    holdout = _load(artifact / 'holdout.json')
    study = _load(artifact / 'study_summary.json')
    handshake = _load(artifact / 'handshake.json')
    calls = _journal(artifact / 'api_calls.jsonl')
    replicates = tuple(registration['replicates'])
    cap = registration['formal_call_cap']
    tracks = _load_tracks(artifact, replicates)
    audit = _audit_study(artifact, registration, tracks, baseline, freeze, holdout, calls)
    if m31_artifact is None:
        candidate_m31 = artifact.parent / 'p2-m31-calibrated-control'
        if candidate_m31.exists():
            m31_artifact = candidate_m31
    completed = [row for row in calls if row['event'] == 'completed']
    inputs = [row['input_tokens'] for row in completed if _finite(row.get('input_tokens'))]
    outputs = [row['output_tokens'] for row in completed if _finite(row.get('output_tokens'))]
    unknown = sum(not _finite(row.get('input_tokens')) or not _finite(row.get('output_tokens')) for row in completed) + cap - len(completed)
    if (cap, sum(inputs), sum(outputs), unknown) != (study['formal_calls'], study['formal_input_tokens'], study['formal_output_tokens'], study['unknown_usage_calls']):
        raise ValueError('formal usage differs from summary')
    table5, transplant_summary = _audit_and_render_table5(artifact)
    vc = transplant_summary['verdict_counts']
    return '\n\n'.join([
        '# P2-M3.2 · 开放先验目录小规模同构对照与单因子跨环境移植研究',
        f'**执行代码锁**：Stochopia `{registration["stochopia_commit"]}`；OpenHyra `{registration["openhyra_commit"]}`。'
        f'\n\n受测模型 `{registration["model"]}`，保持 `models.yaml` 注册参数与原生随机采样，不设贪心温度，不跨槽缓存回复。'
        f'Loop 1 正式预约 **{len(calls) - len(completed) if False else study["formal_calls"]}/{cap}** 次（`R={len(replicates)}` 独立重复 × 3 同构臂 × 2 环境 × 6 轮 = {len(tracks)} 条轨迹），'
        f'终止回执 {len(completed)}/{cap}；正式输入 {sum(inputs)} tokens、输出 {sum(outputs)} tokens、总计 {sum(inputs) + sum(outputs)} tokens；未知用量 {unknown} 次；'
        f'预估最大提示词 {max(r["prompt_tokens"] for ledger in tracks.values() for r in ledger[1:])} tokens，供应商实测最大输入 {_number(max(inputs) if inputs else None)} tokens（预算上限 {PROMPT_CAP}）。'
        f'启动前连通性自检 `{handshake.get("status")}`（输入 {_number(handshake.get("input_tokens"))} / 输出 {_number(handshake.get("output_tokens"))} tokens，独立于正式 {cap} 次配额）；'
        f'Loop 1 墙钟 {_number(study["wall_seconds"])} 秒，独立留出集唯一登记评估模型 {sum(len(x) for x in holdout.values())} 个。'
        f'Loop 2 单因子跨环境移植调用 API **0** 次，唯一移植算子 {transplant_summary["unique_transplanted_models"]} 个（共 {transplant_summary["total_claims"]} 条边界声明，'
        f'覆盖单因子配对组 {len(transplant_summary["pairs_evaluated"])} 组），墙钟 {_number(transplant_summary["wall_seconds"])} 秒。',
        '数据与复验隔离协议：训练 16 日、开发 12 日、留出 12 日；三固定训练初值 `(103, 137, 211)`，模型回放 `K=4`、合成真值 `K=16`。'
        'Loop 1 留出集在全部 12 条轨迹冻结后实体化，与历史 M3/M3.1 共用同一组合成日历（对本轮在线提案严格未见，但不称为跨里程碑新日期）。'
        'Loop 2 在隔离工作目录对 `P2-M2` 单因子控制世界（`w1_thick_short`、`w2_thick_long`、`w3_thin_short`、`w6_thin_hawkes`）重新训练冻结源码并在独立留出日评估，先写入 `transplant-freeze.json` 再开启留出。',
        '## 表 1 · `M3.1` 筛卡旧版 vs. `M3.2` 开放目录新版对照审计表',
        _table1_comparison(m31_artifact, tracks, calls, audit, replicates),
        _feedback_audit_line(tracks),
        '审计说明：`M3.1` 每轮仅按残差紧急度展示 3 张竞争卡（上限 2,800 tokens，不允许 `[]`）；`M3.2` 每轮全量展示 `C01–C10` 共 10 张卡并按 `random.Random(f"{env}:{rep}:{round}")` 做确定性洗牌（同一轮次三臂卡片顺序逐字符一致，不同重复与轮次位置打散，上限 4,500 tokens，允许 `selected_card_ids = []` 提出目录外开放算子）。'
        '主卡均匀度定义为 `C01–C10` 主卡经验分布的归一化香农熵 $H/\\ln 10 \\in [0,1]$，括号内 $\\exp(H)$ 为等效均匀卡数。',
        f'## 表 2 · 账本一：`Z → X` 诊断与机制选择对齐表（`R = {len(replicates)}`）',
        _table2_alignment(tracks, replicates),
        '口径说明：本表仅统计 100% 事前锁定的结构化 JSON 硬指标，不对自由文本 `data_regime_diagnosis` 做事后正则或主观评分。'
        '`Z 边界轴对齐`检验结构化声明 `(weaker_when_z_metric, weaker_when_z_direction)` 是否落在真实环境的主导物理轴上。'
        '主卡对齐沿用预注册规则（`env_alpha`: `C04/C05/C06`；`env_beta`: `C03/C08` 且不含 `C06`），开放创新 `[]` 单独列计不混入主卡匹配。'
        '`target_metric` 已放开至六行残差面板全部 11 项越低越好的误差指标；单步方向预测仅当候选减父模型 95% 按日配对区间完全位于预测侧记 `SUPPORTED`、完全位于反侧记 `REFUTED`，跨零或共同有效日不足 6 日记 `INCONCLUSIVE`。',
        '## 表 3 · 账本二：两阶段晋升构成与留出集 Regret 表',
        _table3_promotion_regret(tracks, study, replicates),
        '晋升分流规则：两条基础护栏恒为 `Δval_nll_total <= +0.005` 且 `Δexhausted_replicate_rate <= +0.05`。'
        '当反事实冲击共同有效日 `>= 6` 时，按冲击改善判据晋升并记为 `objective_improvement`；当共同有效日 `< 6` 时，不再允许仅凭单步 NLL 下降晋升，必须满足 `Δexhausted_replicate_rate <= -0.10`（且若幸存日存在非空冲击误差点估计须 `<= +0.50` tick）方可记为 `viability_recovery`。'
        '留出 Regret 仅在同一轨迹全部有效候选与 `B0` 的共同有效留出日（至少 6 日）上计算，缺失报价不作零填充。',
        f'## 表 4 · 账本三：`{len(tracks)}` 条轨迹独立留出集配对表现表',
        _table4_holdout(tracks, study, freeze, replicates),
        f'单轨迹效应为冻结终局模型减同环境固定 `B0`；`R={len(replicates)}` 汇总先在每个物理留出日对所有轨迹取等权均值（要求该日在全部副本上有效），再对独立交易日做 2,000 次共同随机数 Bootstrap。区间条件于本次实现的 {len(replicates)} 条搜索轨迹，不外推为无限次搜索总体显著性。',
        '## 表 5 · Loop 2：单因子跨环境反事实移植 $\\Delta\\hat{\\tau}_X$ 与边界证伪核验表',
        table5,
        f'移植汇总：`BOUNDARY_CONFIRMED = {vc["BOUNDARY_CONFIRMED"]}`，`BOUNDARY_REFUTED = {vc["BOUNDARY_REFUTED"]}`，`INCONCLUSIVE = {vc["INCONCLUSIVE"]}`（共 {transplant_summary["total_claims"]} 条冻结声明，0 次 API 调用）。'
        '判定规则：每个晋升胜出算子（及开发期目标指标 95% CI 完全为负的 `[]` 开放算子）按事前声明的 `weaker_when_z_metric` 自动映射至唯一单因子配对世界（记忆维 `w1_thick_short ↔ w2_thick_long`；薄度/换档维 `w1_thick_short ↔ w3_thin_short`；时间簇聚维 `w3_thin_short ↔ w6_thin_hawkes`），'
        '在两世界各自独立重训 3 个固定种子并在未见留出集计算相对各自世界 `B0` 的配对效应 $\\hat{\\tau}_X(z)$。记实测该 $Z$ 指标中位数满足 `weaker_when_z_direction` 的世界为 $z_{\\text{weak}}$、另一世界为 $z_{\\text{strong}}$：'
        '因全部 11 项 `target_metric` 均为越低越好的误差读数，当 $\\Delta\\hat{\\tau}_X = \\hat{\\tau}_X(z_{\\text{strong}}) - \\hat{\\tau}_X(z_{\\text{weak}})$ 的 95% 按日 Bootstrap CI 完全低于 0 且 $\\hat{\\tau}_X(z_{\\text{strong}}) < 0$ 时判 `BOUNDARY_CONFIRMED`；'
        '当 $\\Delta\\hat{\\tau}_X$ 的 95% CI 完全高于 0（在声明应弱化的工况反而显著更优）时判 `BOUNDARY_REFUTED`；其余（含共同有效日不足 6 日或区间跨零）判 `INCONCLUSIVE`。',
        f'**Loop 1 预注册综合判定：{study["status"]}**（主卡对齐规则支持：`{study["alignment_supported"]}`；留出整体规则支持：`{study["holdout_supported"]}`；留出整体证伪：`{study["holdout_refuted"]}`）。'
        '全部封存回执、跨环境移植原始日面板与 `SHA256SUMS` 位于 `OpenHyra/artifacts/p2-m32-open-catalog-control/`。严格停在 `P2-M3.2`，未启动 `P2-M4` 实盘实验。',
    ]) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--m31-artifact', type=Path, default=None)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(args.artifact, m31_artifact=args.m31_artifact), encoding='utf-8')


if __name__ == '__main__':
    main()
