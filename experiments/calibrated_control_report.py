"""Read sealed M3.1 results and render six research tables without API or fitting."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from statistics import fmean, stdev
from typing import Any

from experiments.isomorphic_report import (
    ARMS, ENVIRONMENTS, LABELS, _expect_z, _journal, _load, _number, _pair, _primary_match,
    _table, _valid_evaluation, _z_block,
)

Json = dict[str, Any]
REPS = ('rep_0', 'rep_1', 'rep_2')
METRICS = ('val_nll_total', 'dt_nll', 'counterfactual_impact_mae', 'exhausted_replicate_rate')
HEADS = ('Δ 总 NLL', 'Δ dt NLL', 'Δ CF MAE (tick)', 'Δ 耗尽率')
CARDS = tuple(f'C{i:02d}' for i in range(1, 11))
ROUNDS = 6
N_CALLS = 108


def _finite(value: object) -> bool:
    return type(value) in (int, float) and isfinite(value)


def _rate(count: int, total: int) -> str:
    return f'{count}/{total} ({100 * count / total:.1f}%)' if total else 'NA'


def _mean_sd(values: Sequence[float | None]) -> str:
    """Unconditional R mean requires all replicates; otherwise report explicit observed coverage."""
    known = [float(x) for x in values if _finite(x)]
    if not known:
        return f'NA (0/{len(values)})'
    spread = _number(stdev(known)) if len(known) > 1 else 'NA'
    return f'{_number(fmean(known))} ± {spread} ({len(known)}/{len(values)})'


def _range(values: Sequence[float | None]) -> str:
    known = [float(x) for x in values if _finite(x)]
    if not known:
        return f'NA (0/{len(values)})'
    return f'{_number(fmean(known))} [{_number(min(known))}, {_number(max(known))}] ({len(known)}/{len(values)})'


def _shape_error(row: Json) -> bool:
    """Same physical side/level reshape failure as M3; generic shape errors not conflated."""
    text = str(row.get('error') or '') + ' ' + str(row.get('gate_error') or '')
    shape = re.search(r'shape\s+[\'\"\[][^\n]*invalid for input of size', text, re.I)
    return row.get('outcome') == 'GATE_REJECTED' and bool(shape and re.search(r'book_(shares|ticks)', text + ' ' + row.get('source', '')))


def _tracks(artifact: Path) -> dict[str, list[Json]]:
    rows = {}
    for rep in REPS:
        for env in ENVIRONMENTS:
            for arm in ARMS:
                track = f'{rep}/{env}/{arm}'
                ledger = _journal(artifact / 'trajectories' / track / 'ledger.jsonl')
                if len(ledger) != ROUNDS + 1 or ledger[0]['outcome'] != 'SEED':
                    raise ValueError(f'incomplete ledger {track}')
                if [row['cand_id'] for row in ledger[1:]] != [f'c{i:02d}' for i in range(1, 7)]:
                    raise ValueError(f'invalid round roster {track}')
                rows[track] = ledger
    return rows


def _old_tracks(artifact: Path) -> dict[str, list[Json]]:
    return {f'{env}/{arm}': _journal(artifact / 'trajectories' / env / arm / 'ledger.jsonl')
            for env in ENVIRONMENTS for arm in ARMS}


def _group(tracks: Mapping[str, list[Json]], env: str, arm: str) -> list[list[Json]]:
    return [tracks[f'{rep}/{env}/{arm}'][1:] for rep in REPS]


def _env_tracks(tracks: Mapping[str, list[Json]], env: str) -> list[list[Json]]:
    return [rows[1:] for key, rows in tracks.items() if env in key.split('/')]


def _audit(artifact: Path, tracks: Mapping[str, list[Json]], baseline: Json, freeze: Json,
           holdout: Json, calls: list[Json]) -> Json:
    """Recompute controls from durable receipts, never re-run proposal or evaluation."""
    reservations = {x['call_id']: x for x in calls if x['event'] == 'reserved'}
    completed = {x['call_id']: x for x in calls if x['event'] == 'completed'}
    expected = {f'{track}/c{i:02d}' for track in tracks for i in range(1, 7)}
    if set(reservations) != expected or sum(x['event'] == 'reserved' for x in calls) != N_CALLS:
        raise ValueError('formal call roster is not exactly 108 distinct reservations')
    if set(completed) - expected or sum(x['event'] == 'completed' for x in calls) != len(completed):
        raise ValueError('formal terminal receipt roster is inconsistent')
    summary = {}
    for env in ENVIRONMENTS:
        initial = []
        for rep in REPS:
            for arm in ARMS:
                initial.append(tracks[f'{rep}/{env}/{arm}'][1])
        stripped = [_z_block(row['prompt'])[1] for row in initial]
        if any(x != stripped[0] for x in stripped[1:]):
            raise ValueError(f'non-Z first prompt differs in {env}')
        if any(row['active_card_ids'] != initial[0]['active_card_ids'] for row in initial[1:]):
            raise ValueError('initial cards differ')
        summary[env] = {'initial_cards': initial[0]['active_card_ids'], 'non_z_equal': True}
        for arm in ARMS:
            group = _group(tracks, env, arm)
            first_sources = [rows[0].get('source', '') for rows in group]
            first_replies = [completed.get(f'{rep}/{env}/{arm}/c01', {}).get('raw_reply') for rep in REPS]
            summary[env][arm] = {'distinct_first_source_hashes': len({sha256(x.encode()).hexdigest() for x in first_sources if x}),
                                 'distinct_first_reply_hashes': len({sha256(x.encode()).hexdigest() for x in first_replies if x}),
                                 'valid_first_replies': sum(bool(x) for x in first_replies)}
    for track, rows in tracks.items():
        rep, env, arm = track.split('/')
        seed = rows[0]
        if seed['evaluation'] != baseline[env]['evaluation']:
            raise ValueError(f'initial checkpoint or residual differs: {track}')
        active = seed
        for row in rows[1:]:
            call_id = f'{track}/{row["cand_id"]}'
            if reservations[call_id]['messages'] != row['prompt']:
                raise ValueError(f'prompt receipt differs: {call_id}')
            if _z_block(row['prompt'])[0] != _expect_z(baseline, env, arm):
                raise ValueError(f'Z treatment differs: {call_id}')
            if any(LABELS.search(message['content']) for message in row['prompt']):
                raise ValueError(f'nonanonymous prompt: {call_id}')
            actual = completed.get(call_id, {}).get('input_tokens')
            if row['prompt_tokens'] > 2800 or (_finite(actual) and actual > 2800):
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


def _calibration_table(old: Mapping[str, list[Json]], current: Mapping[str, list[Json]], audit: Json) -> str:
    result = []
    for env in ENVIRONMENTS:
        for stage, records in (('M3', old), ('M3.1', current)):
            groups = _env_tracks(records, env)
            rows = [row for track in groups for row in track]
            errors = sum(_shape_error(row) for row in rows)
            repeats = sum(max(0, sum(_shape_error(row) for row in track) - 1) for track in groups)
            result.append([env, stage, len(rows), _rate(sum(_valid_evaluation(row) for row in rows), len(rows)),
                           _rate(errors, len(rows)), _rate(repeats, len(rows)),
                           _rate(sum('C06' in row['active_card_ids'] for row in rows), len(rows)),
                           ', '.join(rows[0]['active_card_ids'])])
    control = _table(('环境', '版本', '尝试', '门禁及评估有效', '同类双侧形状错误', '同轨迹重复错误', 'C06 曝光', '首轮卡'), result)
    random_rows = []
    for env in ENVIRONMENTS:
        for arm in ARMS:
            info = audit[env][arm]
            random_rows.append([env, arm, info['distinct_first_reply_hashes'], info['distinct_first_source_hashes'], info['valid_first_replies']])
    return control + '\n\n' + _table(('环境', '臂', '首轮不同回复哈希 / 3', '不同源码哈希 / 3', '有回复 / 3'), random_rows)


def _alignment_table(tracks: Mapping[str, list[Json]]) -> str:
    result = []
    for env in ENVIRONMENTS:
        for arm in ARMS:
            groups = _group(tracks, env, arm)
            rows = [row for track in groups for row in track]
            counts = [sum(_primary_match(row, env) for row in track) for track in groups]
            choices = Counter(row.get('prediction_verdict', 'NOT_EVALUABLE') for row in rows)
            valid_aligned = sum(_primary_match(row, env) and _valid_evaluation(row) for row in rows)
            selected6 = sum('C06' in row.get('selected_card_ids', []) for row in rows)
            avoid6 = sum(bool(row.get('prediction')) and 'C06' not in row.get('selected_card_ids', []) for row in rows)
            valid_cards = {card: sum(_valid_evaluation(row) and card in row.get('selected_card_ids', []) for row in rows) for card in ('C06', 'C04', 'C05')}
            result.append([env, arm, _rate(sum(_primary_match(track[0], env) for track in groups), 3),
                           _rate(sum(counts), 18), _mean_sd([n / 6 for n in counts]), _rate(valid_aligned, 18),
                           '/'.join(str(valid_cards[card]) for card in ('C06', 'C04', 'C05')),
                           _rate(avoid6, 18) if env == 'env_beta' else _rate(selected6, 18),
                           '/'.join(str(choices[v]) for v in ('SUPPORTED', 'REFUTED', 'INCONCLUSIVE', 'NOT_EVALUABLE'))])
    return _table(('环境', '臂', '首轮匹配', '全轮匹配', '轨迹匹配比例均值 ± SD', '有效且主卡匹配',
                   '有效引用 C06/C04/C05', 'β规避 / α选择 C06', '方向支持/反向/不定/未测'), result)


def _selection_table(tracks: Mapping[str, list[Json]], study: Json) -> str:
    result = []
    for env in ENVIRONMENTS:
        for arm in ARMS:
            groups = _group(tracks, env, arm)
            valid = [sum(_valid_evaluation(row) for row in track) for track in groups]
            counts = [Counter(row['outcome'] for row in track) for track in groups]
            regrets = [study['regret'][f'{rep}/{env}/{arm}'] for rep in REPS]
            result.append([env, arm, _range(valid), _range([c['PROMOTED'] for c in counts]),
                           _range([c['STEPPING_STONE'] for c in counts]), _range([c['DUPLICATE_REJECTED'] for c in counts]),
                           _range([x['estimate'] for x in regrets]), ', '.join(str(x['paired_days']) for x in regrets),
                           ', '.join(str(x['oracle_models']) for x in regrets)])
    return _table(('环境', '臂', '有效候选/轨迹', '晋升', '踏脚石', '去重', 'Regret tick', '共同日(3轨迹)', 'Oracle 模型数'), result)


def _group_pair(row: Json) -> str:
    return f'{_pair(row)}；R均值 {_number(row.get("replicate_mean"))}；SD {_number(row.get("replicate_sd"))}'


def _holdout_tables(study: Json, freeze: Json) -> str:
    individual = []
    for rep in REPS:
        for env in ENVIRONMENTS:
            for arm in ARMS:
                key = f'{rep}/{env}/{arm}'
                effects = study['track_effects'][key]
                individual.append([rep, env, arm, freeze['trajectories'][key]['final_id'], *[_pair(effects.get(metric)) for metric in METRICS]])
    grouped = []
    for env in ENVIRONMENTS:
        for arm in ARMS:
            key = f'{env}/{arm}'
            grouped.append([key, *[_group_pair(study['cell_effects'][key][metric]) for metric in METRICS]])
        for arm in ('d_only', 'shuffled_z_d'):
            key = f'{env}/true_z_d-vs-{arm}'
            grouped.append([key, *[_group_pair(study['group_cross_arm_effects'][key][metric]) for metric in METRICS]])
    for arm in ('d_only', 'shuffled_z_d'):
        key = f'true_z_d-vs-{arm}'
        grouped.append([f'双环境等权/{key}', *[_group_pair(study['pooled_cross_arm_effects'][key][metric]) for metric in METRICS]])
    return (_table(('轨迹', '环境', '臂', '最终父模型', *HEADS), individual) + '\n\n' +
            _table(('R=3 汇总/组间对照', *HEADS), grouped))


def posterior(tracks: Mapping[str, list[Json]]) -> Json:
    """Count valid cited candidates once/card; only decisive improvement/harm updates Beta(1,1)."""
    result = {env: {card: {'n_valid': 0, 'n_plus': 0, 'n_minus': 0} for card in CARDS} for env in ENVIRONMENTS}
    for key, ledger in tracks.items():
        env = key.split('/')[-2]
        for row in ledger[1:]:
            if not _valid_evaluation(row):
                continue
            interval = row.get('deltas', {}).get(row.get('prediction', {}).get('target_metric'), {}).get('interval')
            for card in set(row.get('selected_card_ids', [])):
                count = result[env][card]
                count['n_valid'] += 1
                if interval is not None and interval[1] < 0:
                    count['n_plus'] += 1
                elif interval is not None and interval[0] > 0:
                    count['n_minus'] += 1
    for group in result.values():
        for count in group.values():
            count['n_inconclusive'] = count['n_valid'] - count['n_plus'] - count['n_minus']
            count['posterior_mean'] = (1 + count['n_plus']) / (2 + count['n_plus'] + count['n_minus'])
    return result


def _posterior_table(values: Json) -> str:
    result = []
    for env, cards in values.items():
        for card, row in cards.items():
            result.append([env, card, row['n_valid'], row['n_plus'], row['n_minus'], row['n_inconclusive'], _number(row['posterior_mean'])])
    return _table(('环境', '卡', '有效验证', '显著改善', '显著恶化', '不确定', 'Beta 后验均值'), result)


def _ablation_table(abl: Json) -> str:
    """Render exact frozen ablation schema (levels and independent daily contrasts)."""
    if abl['formal_fits'] != 12:
        raise ValueError('ablation must have twelve fitted instances')
    metrics = ('val_nll_total', 'counterfactual_impact_mae', 'post_stop_relaxation_delta',
               'sign_acf_abs_error', 'memory_gap_1', 'memory_gap_4', 'memory_gap_16', 'memory_gap_64')
    rows = []
    for name in ('A00', 'A10', 'A01', 'A11'):
        model = abl['models'][name]
        if model['state'] != 60 or len(model['fits']) != 3:
            raise ValueError('ablation state or fit roster differs')
        result = [name, model['params'], model.get('used_params', model['params']), model['state']]
        for metric in metrics:
            result.append(_number(model['aggregate'][metric]))
        rows.append(result)
    levels = _table(('模型', '登记参数', '使用参数', '状态', *metrics), rows)
    diffs = []
    for contrast, values in abl['contrasts'].items():
        diffs.append([contrast, *[_pair(values.get(metric)) for metric in metrics]])
    diffs.append(['(A11−A10)−(A01−A00)', *[_pair(abl['synergy'].get(metric)) for metric in metrics]])
    return levels + '\n\n' + _table(('配对差', *metrics), diffs)


def _ablation_interpretation(abl: Json) -> str:
    def verdict(metric: str) -> str:
        band = abl['synergy'][metric]['interval']
        if band is None or band[0] <= 0 <= band[1]:
            return '不可判定'
        return '支持超可加改善' if band[1] < 0 else '反对超可加改善（组合收益低于加总）'

    reproduction = abl['m3_reference_reproduction']
    matches = sum(sum(row['checkpoint_hash_match'].values()) for row in reproduction.values())
    return (f'消融正式拟合 {abl["formal_fits"]} 次，API {abl["api_calls"]} 次，墙钟 {_number(abl["wall_seconds"])} 秒；'
            f'A00/A11 对照旧 M3 权重 SHA256 共 {matches}/6 一致。'
            f'总 NLL 交互：{verdict("val_nll_total")}；冲击 MAE 交互：{verdict("counterfactual_impact_mae")}。'
            '交互判定依据冻结的共同日区间，不以事后最佳模型排名代替。')


def render(artifact: Path) -> str:
    registration = _load(artifact / 'registration.json')
    baseline = _load(artifact / 'baseline.json')
    freeze = _load(artifact / 'freeze.json')
    holdout = _load(artifact / 'holdout.json')
    study = _load(artifact / 'study_summary.json')
    handshake = _load(artifact / 'handshake.json')
    calls = _journal(artifact / 'api_calls.jsonl')
    tracks = _tracks(artifact)
    old = _old_tracks(artifact.parent / 'p2-m3-isomorphic-control')
    audit = _audit(artifact, tracks, baseline, freeze, holdout, calls)
    completed = [row for row in calls if row['event'] == 'completed']
    inputs = [row['input_tokens'] for row in completed if _finite(row.get('input_tokens'))]
    outputs = [row['output_tokens'] for row in completed if _finite(row.get('output_tokens'))]
    unknown = sum(not _finite(row.get('input_tokens')) or not _finite(row.get('output_tokens')) for row in completed) + N_CALLS - len(completed)
    if (N_CALLS, sum(inputs), sum(outputs), unknown) != (study['formal_calls'], study['formal_input_tokens'], study['formal_output_tokens'], study['unknown_usage_calls']):
        raise ValueError('formal usage differs from summary')
    values = posterior(tracks)
    if 'prior_posterior' in study and study['prior_posterior'] != values:
        raise ValueError('offline prior posterior differs from independent ledger audit')
    abl = _load(artifact / 'ablation' / 'ablation.json')
    return '\n\n'.join([
        '# P2-M3.1 · Harness 校准、2×2 消融与 R=3 同构复测',
        f'**执行代码锁**：Stochopia `{registration["stochopia_commit"]}`；OpenHyra `{registration["openhyra_commit"]}`。'
        f'\n\nGPT-6.2 Sol 同一注册参数，原生随机采样，不设贪心温度，不缓存跨轨迹回复。正式预约108/108次，终止回执 {len(completed)}/108；输入 {sum(inputs)}，输出 {sum(outputs)} tokens；未知用量 {unknown} 次；'
        f'估计最大输入 {max(row["prompt_tokens"] for track in tracks.values() for row in track[1:])}、实际最大 {_number(max(inputs) if inputs else None)}。自检 {handshake.get("status")}，'
        f'输入 {_number(handshake.get("input_tokens"))} 输出 {_number(handshake.get("output_tokens"))}；正式闭环墙钟 {_number(study["wall_seconds"])} 秒。'
        f'独立留出登记模型 {sum(len(x) for x in holdout.values())} 个（同环境同源码只评估一次）。',
        '训练16日、开发12日、留出12日；三训练初值、模型K=4、生成真值K=16。M3.1 留出日对本轮搜索隔离，在18条轨迹冻结后才实体化；但它与历史M3使用相同日期，不能称为跨里程碑全新数据集。消融复用相同日期，独立于在线闭环。',
        '## 表 1 · M3→M3.1 校准前后与同构、独立采样审计',
        _calibration_table(old, tracks, audit),
        '所有108个实际请求和账本提示逐字符匹配；每环境9条轨迹首轮去除Z块后逐字符一致，初始D和权重一致；逐轮Z正确置换，匿名标签与预算检查通过。有效率为完成契约及开发测量/全部尝试；同轨迹重复错误只计同类形状错误首次之后的次数。三个修复联合上线，前后率差是整体Harness校准对照，不能分别归因；M3只有R=1。不同回复哈希是独立请求输出的实测多样性，而不是声称供应商独立同分布。',
        '## 表 2 · c06 的 2×2 机制消融（独立、零 API）',
        _ablation_table(abl),
        _ablation_interpretation(abl),
        '配对差均为左模型减右模型。交互=(A11−A10)−(A01−A00)，四模型共同有效日、三训练种子先聚合、2,000次按日重抽；负值对于误差/NLL表示超可加改善，正值表示劣化。A00是原60维GRU、其余52+8，A01同时移除动态时钟；A10保留原预测层参数但不使用预测相减，因此按指定设计得到的是架构组合交互，不是严格只改两个独立算子的纯机制因果量。记忆谱不是越大越好，弛豫值也不能直接以符号判定优劣。',
        '## 表 3 · 账本一：R=3 主卡匹配、边界规避与事前方向',
        _alignment_table(tracks),
        '主卡匹配按事前规则：α C04/C05/C06，β C03/C08且不含C06。每条轨迹六轮、相互依赖，不把18轮当作独立研究样本；SD为三轨迹比例的样本标准差。有效引用仅指声明该卡且候选通过门禁及测量，不是机制忠实实现的证明。β规避分母包括全部尝试，缺少预测不计规避。方向支持/证伪依据目标指标配对95%区间严格位于预测侧/反侧，不以排名代替预测。',
        '## 表 4 · 账本二：开发演化与留出 Regret',
        _selection_table(tracks, study),
        '单元为均值[最小,最大]与可用轨迹数。Regret只在同一轨迹所有有效Oracle模型的共同有效日上测量（至少6日），不足则NA；仅B0的零后悔值不证明选择能力。三条轨迹中有缺失时不把可用均值解释为完整R=3均值。',
        '## 表 5 · 账本三：冻结胜出模型的独立留出配对效应',
        _holdout_tables(study, freeze),
        '单轨迹为冻结模型减B0；R汇总为每个物理日对三个轨迹取等权差，再对日重抽2,000次，要求全部轨迹共同有效支持。R均值与SD另外描述三个独立搜索的估计量差异；区间条件于本次三个研究轨迹，不能推广成跨研究轨迹总体显著性。在所有主读数上，缺失报价不可记作零。',
        '## 表 6 · 探索性审计：先验卡条件 Beta 后验（离线）',
        _posterior_table(values),
        '仅对有效训练及测量候选计一次每张引用卡；目标读数95%配对区间完全负计改善，完全正计恶化，其余不更新Beta(1,1)。多卡共享同一结果并非独立验证，后验均值是本任务的探索性条件统计，不是关于单一理论正确性的因果概率。无决定性数据保持0.5，不意味着理论获得支持。所有在线卡库未动态删改。',
        f'**预注册综合判定：{study["status"]}。** 匹配规则支持：{study.get("alignment_supported")}；留出整体规则支持：{study.get("holdout_supported")}。注册规则：{registration.get("controlled_comparison_rule")}。'
        '单一环境或单一臂涨分不能证明正确Z普遍优于两个控制臂；无显著差异也不能证明等效。所有原始账本、冻结回执、消融数据与校验值见 [OpenHyra M3.1 证据](https://github.com/MrSteeeve/OpenHyra/tree/world-model/artifacts/p2-m31-calibrated-control)。未启动M4。',
    ]) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(args.artifact), encoding='utf-8')


if __name__ == '__main__':
    main()
