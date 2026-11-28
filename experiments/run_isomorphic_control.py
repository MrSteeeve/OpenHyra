"""Run exactly six anonymous research trajectories; no holdout before freeze."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import os
import subprocess
from threading import Event, Lock, Semaphore
from time import perf_counter
from typing import Any

from stochopia.common import atomic_write_json
from tasks.lob_world_model.call_budget import CallBudgetGuard
from tasks.lob_world_model.contracts import Messages, ProviderReply
from tasks.lob_world_model.discovery_loop import (
    DiscoveryLoop, View, PROMPT_CAP, SOURCE_TOKEN_CAP, SEED_SOURCE, build_messages, source_hash,
)
from tasks.lob_world_model.folding_ledger import FoldingLedger
from tasks.lob_world_model.provider import DEFAULT_MODELS_PATH, provider_handshake, registry_transport
from tasks.lob_world_model.provider_client import load_model_registry
from tasks.lob_world_model.prior_cards import select_active_cards
from tasks.lob_world_model.synthetic_eval import DATA_PROTOCOL, SyntheticEvaluator, paired_summary
from tasks.lob_world_model.token_count import openai_token_counter

Json = dict[str, Any]
ARMS = ('d_only', 'true_z_d', 'shuffled_z_d')
ENVS = ('env_alpha', 'env_beta')
ROUNDS = 6
MAX_CALLS = len(ARMS) * len(ENVS) * ROUNDS
MODEL = 'gpt-6.2-sol'
MIN_DAYS = 6
ALIGNMENT = {'env_alpha': ['C04', 'C05', 'C06'], 'env_beta': ['C03', 'C08']}
ROOT = Path(__file__).resolve().parents[1]
METRICS = ('val_nll_total', 'dt_nll', 'counterfactual_impact_mae',
           'exhausted_replicate_rate', 'memory_gap_64')


def _read(path: Path) -> Json:
    return json.loads(path.read_text(encoding='utf-8'))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *parts: str) -> str:
    return subprocess.check_output(['git', *parts], cwd=root, text=True).strip()


def _freeze(path: Path, value: Json) -> None:
    if path.exists():
        if _read(path) != value:
            raise ValueError(f'frozen registration differs: {path.name}')
    else:
        atomic_write_json(path, value)


def registration(stochopia: Path) -> Json:
    """Record all decision rules before seeing any candidate or API response."""
    model = load_model_registry(DEFAULT_MODELS_PATH)[MODEL]
    return {
        'schema': 'isomorphic-control-v1', 'stochopia_commit': _git(stochopia, 'rev-parse', 'HEAD'),
        'openhyra_commit': _git(ROOT, 'rev-parse', 'HEAD'), 'model': MODEL,
        'provider': {'model': model.model, 'max_output_tokens': model.max_tokens,
                     'temperature': model.temperature, 'extra_body': model.extra_body,
                     'strict_model_match': model.strict_model_match, 'registry_sha256': _digest(DEFAULT_MODELS_PATH)},
        'environments': list(ENVS), 'arms': list(ARMS), 'rounds': ROUNDS, 'formal_call_cap': MAX_CALLS,
        'handshake_cap': 1, 'prompt_token_cap': PROMPT_CAP, 'candidate_source_token_cap': SOURCE_TOKEN_CAP,
        'seed_model_sha256': source_hash(SEED_SOURCE), 'data_protocol': DATA_PROTOCOL,
        'paired_days_minimum': MIN_DAYS, 'bootstrap_draws': 2000,
        'promotion': {'nll_max_increase': .005, 'exhaustion_max_increase': .05,
                      'cf_min_improvement_ticks': .01, 'cf_equivalence_ticks': .01,
                      'exhaustion_min_improvement': .10, 'nll_min_improvement': .008},
        'stepping_stones': {'nll_max_increase': .025, 'dt_improvement': .010,
                            'exhaustion_improvement': .10, 'acf_relative_improvement': .25,
                            'memory_gap64_absolute_change': .05},
        'dedup_similarity_strict_gt': .94, 'cooldown_rounds': 2, 'active_cards_max': 3,
        'alignment_primary_card': ALIGNMENT, 'alignment_beta_excluded': 'C06',
        'alignment_denominator': 'all attempted rounds; primary selected card only; malformed/absent counted as nonaligned, report coverage separately',
        'prediction_rule': 'candidate-parent 95% paired day CI wholly predicted side supports, opposite side refutes; crossing zero or absent CI inconclusive; no evaluated candidate non-evaluable',
        'regret_rule': 'holdout CF MAE of final minus minimum over B0 and all valid evaluated candidates, on same intersection of finite paired days across oracle set; require >=6 days; no imputation',
        'controlled_comparison_rule': ('True-Z directional alignment must strictly exceed each comparison arm over all 12 attempts; '
            'holdout support requires both envs vs both arms CF CI upper < 0 and point NLL/exhaust deltas <= .005/.05. '
            'Report descriptive alignment separately; if both support then SUPPORTED_WITH_BOUNDARY, '
            'if one supports PARTIAL, if both envs vs both CF CI lower > 0 REFUTED, otherwise INCONCLUSIVE. '
            'One trajectory per cell, rounds not independent replicates; no population-level inference.'),
        'holdout_rule': 'materialize 12 independent days only after all 36 slots completed and approved source hashes frozen; reuse trained models, oracle does not feed search',
    }


def _handshake(artifact: Path) -> None:
    path = artifact / 'handshake.json'
    if path.exists():
        if _read(path).get('status') != 'PASS':
            raise RuntimeError('preflight was already consumed and not successful; stop')
        return
    config = load_model_registry(DEFAULT_MODELS_PATH)[MODEL]
    if not os.environ.get(config.api_key_env, '').strip():
        raise RuntimeError(f'configured API credential is absent ({config.api_key_env})')
    # Durable reservation: a crash or failure does not grant a second request.
    atomic_write_json(path, {'status': 'RESERVED', 'calls': 1})
    try:
        reply = provider_handshake(DEFAULT_MODELS_PATH, MODEL, completion_limit=128)
    except Exception as exc:
        atomic_write_json(path, {'status': 'FAILED', 'calls': 1, 'error': type(exc).__name__,
                                  'input_tokens': getattr(exc, 'input_tokens', None),
                                  'output_tokens': getattr(exc, 'output_tokens', None)})
        raise RuntimeError('single preflight did not pass; no formal calls authorized') from exc
    atomic_write_json(path, {'status': 'PASS', 'calls': 1, 'input_tokens': reply.input_tokens,
                             'output_tokens': reply.output_tokens})
    print('API preflight passed; starting registered proposal calls', flush=True)


def _baseline(artifact: Path, evaluators: Mapping[str, SyntheticEvaluator]) -> Json:
    path = artifact / 'baseline.json'
    if path.exists():
        return _read(path)
    baseline = {}
    for env, evaluator in evaluators.items():
        z = evaluator.prepare_dev()
        result = evaluator.evaluate(SEED_SOURCE)
        if result['status'] != 'VALID' or result['state'] != 60:
            raise RuntimeError(f'baseline did not meet the fixed contract: {env}')
        baseline[env] = {'z': z, 'evaluation': result, 'development_receipt': evaluator.development}
        print(f'BASELINE {env} READY', flush=True)
    _freeze(path, baseline)
    return baseline


def _records(artifact: Path, env: str, arm: str) -> list[Json]:
    return FoldingLedger(artifact / 'trajectories' / env / arm).records


def _non_z(prompt: list[dict[str, str]]) -> str:
    start, rest = prompt[1]['content'].split('[Macro Data Regime Z]\n', 1)
    _, suffix = rest.split('\n\n[Model Residuals D]', 1)
    return prompt[0]['content'] + '\n' + start + '[Macro Data Regime Z]\n' + suffix


def _audit_first_round(artifact: Path, baseline: Json) -> None:
    """Verify the actual first-round packets before the first network request."""
    count = openai_token_counter()
    audit = {}
    for env in ENVS:
        parent = baseline[env]['evaluation']
        cards = select_active_cards(parent['aggregate'], {})
        entries = {}
        for arm in ARMS:
            other = ENVS[1 - ENVS.index(env)]
            z = None if arm == 'd_only' else baseline[env if arm == 'true_z_d' else other]['z']
            view = View(arm, env, z)
            loop = DiscoveryLoop(view=view, root=artifact / 'trajectories' / env / arm,
                                 seed_result=parent, evaluator=lambda source: parent,
                                 pair=paired_summary, count=count)
            prompt, tokens = build_messages(view, 'B0', SEED_SOURCE, parent, cards, loop.ledger, (), count)
            entries[arm] = {'non_z_sha256': sha256(_non_z(prompt).encode()).hexdigest(),
                            'tokens': tokens, 'active_card_ids': [c.card_id for c in cards]}
        if len({row['non_z_sha256'] for row in entries.values()}) != 1:
            raise RuntimeError('first-round information control failed before API')
        audit[env] = entries
    _freeze(artifact / 'initial_control_audit.json', audit)


def _make_freeze(artifact: Path, guard: CallBudgetGuard) -> Json:
    approved = {env: set() for env in ENVS}
    trajectories = {}
    for env in ENVS:
        for arm in ARMS:
            rows = _records(artifact, env, arm)
            if len(rows) != ROUNDS + 1 or rows[0]['outcome'] != 'SEED':
                raise RuntimeError('all six tracks must finish before the independent holdout opens')
            active = rows[0]
            for row in rows:
                if row.get('evaluation', {}).get('status') == 'VALID':
                    approved[env].add(row['source_sha256'])
                if row['outcome'] == 'PROMOTED':
                    active = row
            trajectories[f'{env}/{arm}'] = {'final_id': active['cand_id'], 'final_sha256': active['source_sha256']}
    reservations = [r for r in guard.entries() if r['event'] == 'reserved']
    expected = {f'{env}/{arm}/c{r:02d}' for env in ENVS for arm in ARMS for r in range(1, ROUNDS + 1)}
    if {r['call_id'] for r in reservations} != expected or len(reservations) != MAX_CALLS:
        raise RuntimeError('formal call roster is incomplete; do not open holdout')
    value = {'approved': {env: sorted(values) for env, values in approved.items()}, 'trajectories': trajectories,
             'formal_calls': guard.count}
    _freeze(artifact / 'freeze.json', value)
    print('All six tracks frozen; holdout may now be opened', flush=True)
    return value


def _run_tracks(artifact: Path, evaluators: Mapping[str, SyntheticEvaluator], baseline: Json,
                guard: CallBudgetGuard) -> None:
    count = openai_token_counter()
    raw_transport = registry_transport(DEFAULT_MODELS_PATH, MODEL)
    network = Semaphore(3)
    stop = Event()
    def transport(messages: Messages, *, response_json: bool) -> ProviderReply:
        with network:
            if stop.is_set():
                raise RuntimeError('study interrupted before provider dispatch')
            return raw_transport(messages, response_json=response_json)
    resource = Semaphore(2)
    locks = {env: Lock() for env in ENVS}
    def evaluate(env: str, source: str) -> Json:
        with locks[env], resource:
            return evaluators[env].evaluate(source)

    def track(env: str, arm: str) -> Json:
        other = ENVS[1 - ENVS.index(env)]
        z = None if arm == 'd_only' else baseline[env if arm == 'true_z_d' else other]['z']
        loop = DiscoveryLoop(view=View(arm, env, z), root=artifact / 'trajectories' / env / arm,
                             seed_result=baseline[env]['evaluation'], evaluator=lambda source: evaluate(env, source),
                             pair=paired_summary, count=count, transport=transport, guard=guard,
                             rounds=ROUNDS, should_stop=stop.is_set)
        result = loop.run()
        print(f'TRACK {env}/{arm} COMPLETE', flush=True)
        return result

    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = {pool.submit(track, env, arm): (env, arm) for env in ENVS for arm in ARMS}
        for future in as_completed(pending):
            try:
                future.result()
            except BaseException:
                stop.set()
                raise

    # Audit exact non-Z first-round equality and model receipt equality before holdout.
    for env in ENVS:
        initial = [_records(artifact, env, arm)[1] for arm in ARMS]
        if any(row['active_card_ids'] != initial[0]['active_card_ids'] or row['prompt'][0] != initial[0]['prompt'][0]
               or _non_z(row['prompt']) != _non_z(initial[0]['prompt']) for row in initial[1:]):
            raise RuntimeError('initial non-Z prompt sections are not isomorphic')
        seeds = [_records(artifact, env, arm)[0]['evaluation']['training_checkpoint_hashes'] for arm in ARMS]
        if any(item != seeds[0] for item in seeds[1:]):
            raise RuntimeError('initial model checkpoints differ across arms')


def _sources(artifact: Path, env: str) -> dict[str, str]:
    sources = {}
    for arm in ARMS:
        for row in _records(artifact, env, arm):
            if row.get('evaluation', {}).get('status') == 'VALID':
                if source_hash(row['source']) != row['source_sha256']:
                    raise RuntimeError('frozen candidate source hash mismatch')
                sources[row['source_sha256']] = row['source']
    return sources


def _verify(artifact: Path, evaluators: Mapping[str, SyntheticEvaluator], freeze: Json) -> Json:
    path = artifact / 'holdout.json'
    if path.exists():
        return _read(path)
    holdout = {}
    for env, evaluator in evaluators.items():
        receipt = evaluator.open_holdout(artifact / 'freeze.json')
        atomic_write_json(artifact / f'{env}-holdout-receipt.json', receipt)
        sources = _sources(artifact, env)
        if set(sources) != set(freeze['approved'][env]):
            raise RuntimeError('holdout request does not match authorized source set')
        base = evaluator.evaluate(SEED_SOURCE, split='holdout_test')
        holdout[env] = {}
        for digest, source in sources.items():
            candidate = base if digest == source_hash(SEED_SOURCE) else evaluator.evaluate(source, split='holdout_test')
            against = paired_summary(candidate, base) if candidate['status'] == base['status'] == 'VALID' else {}
            holdout[env][digest] = {'evaluation': candidate, 'against_B0': against}
        print(f'HOLDOUT {env} COMPLETE ({len(sources)} authorized models)', flush=True)
    _freeze(path, holdout)
    return holdout


def _regret(rows: list[Json], final: str, holdout: Json) -> Json:
    """Same-day intersection across every oracle candidate; missing is not zero."""
    ids = {r['source_sha256'] for r in rows if r.get('evaluation', {}).get('status') == 'VALID'}
    by_model = {}
    for digest in ids:
        evaluation = holdout[digest]['evaluation']
        if evaluation['status'] != 'VALID':
            return {'estimate': None, 'reason': 'oracle evaluation unavailable', 'paired_days': 0, 'oracle_models': len(ids)}
        values = {}
        for day in evaluation['daily']:
            value = day.get('counterfactual_impact_mae')
            if value is not None and isfinite(value):
                values[day['trading_date']] = float(value)
        by_model[digest] = values
    common = set.intersection(*(set(values) for values in by_model.values())) if by_model else set()
    if len(common) < MIN_DAYS:
        return {'estimate': None, 'reason': 'oracle common-day support insufficient',
                'paired_days': len(common), 'oracle_models': len(ids)}
    means = {digest: sum(values[day] for day in sorted(common)) / len(common) for digest, values in by_model.items()}
    best = min(means, key=lambda digest: (means[digest], digest))
    return {'estimate': max(0., means[final] - means[best]), 'reason': None, 'paired_days': len(common),
            'oracle_models': len(ids), 'best_sha256': best, 'final_mae': means[final], 'oracle_mae': means[best]}


def _aligned(row: Json, env: str) -> bool:
    chosen = row.get('selected_card_ids', [])
    return bool(chosen and row.get('prediction') and chosen[0] in ALIGNMENT[env]
                and (env != 'env_beta' or 'C06' not in chosen)
                and row.get('outcome') not in ('INVALID_PROPOSAL', 'PROVIDER_ERROR', 'INTERRUPTED_CALL'))


def _summarize(artifact: Path, freeze: Json, holdout: Json, guard: CallBudgetGuard, seconds: float) -> Json:
    regrets = {}
    alignment = {arm: 0 for arm in ARMS}
    comparisons = {}
    for env in ENVS:
        for arm in ARMS:
            track = f'{env}/{arm}'
            rows = _records(artifact, env, arm)
            final = freeze['trajectories'][track]['final_sha256']
            regrets[track] = _regret(rows, final, holdout[env])
            alignment[arm] += sum(_aligned(row, env) for row in rows[1:])
        true_final = freeze['trajectories'][f'{env}/true_z_d']['final_sha256']
        for arm in ('d_only', 'shuffled_z_d'):
            other_final = freeze['trajectories'][f'{env}/{arm}']['final_sha256']
            candidate = holdout[env][true_final]['evaluation']
            parent = holdout[env][other_final]['evaluation']
            comparisons[f'{env}/true_z_d-vs-{arm}'] = paired_summary(candidate, parent) if candidate['status'] == parent['status'] == 'VALID' else {}
    align_support = all(alignment['true_z_d'] > alignment[arm] for arm in ('d_only', 'shuffled_z_d'))
    def supported(row: Json) -> bool:
        cf = row.get('counterfactual_impact_mae', {}).get('interval')
        nll = row.get('val_nll_total', {}).get('estimate')
        exhaust = row.get('exhausted_replicate_rate', {}).get('estimate')
        return cf is not None and cf[1] < 0 and nll is not None and exhaust is not None and nll <= .005 and exhaust <= .05
    def refuted(row: Json) -> bool:
        cf = row.get('counterfactual_impact_mae', {}).get('interval')
        return cf is not None and cf[0] > 0
    hold_support = all(supported(row) for row in comparisons.values())
    hold_refuted = all(refuted(row) for row in comparisons.values())
    status = 'SUPPORTED_WITH_BOUNDARY' if align_support and hold_support else 'PARTIAL' if align_support or hold_support else 'REFUTED' if hold_refuted else 'INCONCLUSIVE'
    completed = [row for row in guard.entries() if row['event'] == 'completed']
    return {'status': status, 'alignment_supported': align_support, 'holdout_supported': hold_support,
            'holdout_refuted': hold_refuted, 'alignment_counts': alignment, 'regret': regrets,
            'cross_arm_effects': comparisons, 'formal_calls': guard.count,
            'formal_input_tokens': sum(r['input_tokens'] for r in completed if r['input_tokens'] is not None),
            'formal_output_tokens': sum(r['output_tokens'] for r in completed if r['output_tokens'] is not None),
            'unknown_usage_calls': sum(r['input_tokens'] is None or r['output_tokens'] is None for r in completed) + guard.count - len(completed),
            'wall_seconds': seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--stochopia-root', type=Path, required=True)
    args = parser.parse_args()
    artifact, work = args.artifact_root.resolve(), args.work_root.resolve()
    if artifact.name != 'p2-m3-isomorphic-control' or work.is_relative_to(artifact):
        raise ValueError('the study requires an isolated M3-only artifact root and private work root')
    artifact.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    _freeze(artifact / 'registration.json', registration(args.stochopia_root.resolve()))
    evaluators = {env: SyntheticEvaluator(work, env) for env in ENVS}
    baseline = _baseline(artifact, evaluators)
    guard = CallBudgetGuard(artifact / 'api_calls.jsonl', limit=MAX_CALLS)
    if not (artifact / 'freeze.json').exists():
        if guard.count == 0:
            _audit_first_round(artifact, baseline)
        _handshake(artifact)
        _run_tracks(artifact, evaluators, baseline, guard)
        freeze = _make_freeze(artifact, guard)
    else:
        freeze = _read(artifact / 'freeze.json')
        if guard.count != MAX_CALLS:
            raise RuntimeError('cannot replay an incomplete frozen call roster')
    holdout = _verify(artifact, evaluators, freeze)
    if not (artifact / 'study_summary.json').exists():
        atomic_write_json(artifact / 'study_summary.json', _summarize(artifact, freeze, holdout, guard, perf_counter() - started))
    print('M3 complete; evidence ready for report and checksums', flush=True)


if __name__ == '__main__':
    main()
