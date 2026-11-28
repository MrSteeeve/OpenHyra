"""Run the calibrated, replicated isomorphic control; no holdout before global freeze."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
import os
from statistics import mean, stdev
import subprocess
from threading import Event, Lock, Semaphore
from time import perf_counter
from typing import Any

from experiments.synthetic_tau_z.design import EVAL_DAYS, SEEDS, summarize_days
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
REPLICATES = ('rep_0', 'rep_1', 'rep_2')
MAX_CALLS = len(REPLICATES) * len(ARMS) * len(ENVS) * ROUNDS
ARTIFACT_NAME = 'p2-m31-calibrated-control'
MODEL = 'gpt-6.2-sol'
MIN_DAYS = 6
ALIGNMENT = {'env_alpha': ['C04', 'C05', 'C06'], 'env_beta': ['C03', 'C08']}
ROOT = Path(__file__).resolve().parents[1]
METRICS = ('val_nll_total', 'dt_nll', 'counterfactual_impact_mae',
           'exhausted_replicate_rate', 'memory_gap_64')
CARDS = tuple(f'C{number:02d}' for number in range(1, 11))
COMPARATORS = ('d_only', 'shuffled_z_d')


def _tracks(replicates: Sequence[str] = REPLICATES) -> list[tuple[str, str, str]]:
    return [(rep, env, arm) for rep in replicates for env in ENVS for arm in ARMS]


def _track_id(rep: str, env: str, arm: str) -> str:
    return f'{rep}/{env}/{arm}'


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


def registration(stochopia: Path, replicates: Sequence[str] = REPLICATES) -> Json:
    """Record all decision rules before seeing any candidate or API response."""
    model = load_model_registry(DEFAULT_MODELS_PATH)[MODEL]
    if tuple(replicates) != REPLICATES:
        raise ValueError('the formal M3.1 study requires three registered replicates')
    if model.temperature is not None and model.temperature <= 0:
        raise ValueError('the sampled search cannot use zero temperature')
    return {
        'schema': 'isomorphic-control-v2-calibrated', 'stochopia_commit': _git(stochopia, 'rev-parse', 'HEAD'),
        'openhyra_commit': _git(ROOT, 'rev-parse', 'HEAD'), 'model': MODEL,
        'provider': {'model': model.model, 'max_output_tokens': model.max_tokens,
                     'temperature': model.temperature, 'extra_body': model.extra_body,
                     'strict_model_match': model.strict_model_match, 'registry_sha256': _digest(DEFAULT_MODELS_PATH),
                     'sampling': 'provider-native stochastic reasoning; omit unsupported explicit temperature and seed',
                     'response_cache': 'none between logical call slots; durable receipt replay only within the same slot after interruption'},
        'environments': list(ENVS), 'arms': list(ARMS), 'rounds': ROUNDS, 'formal_call_cap': MAX_CALLS,
        'num_replicates': len(replicates), 'replicates': list(replicates),
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
        'controlled_comparison_rule': ('True-Z directional alignment must strictly exceed each comparison arm over all 36 attempts; '
            'holdout support requires both envs vs both arms R-mean CF common-day CI upper < 0 and point NLL/exhaust deltas <= .005/.05. '
            'Report descriptive alignment separately; if both support then SUPPORTED_WITH_BOUNDARY, '
            'if one supports PARTIAL, if both envs vs both CF CI lower > 0 REFUTED, otherwise INCONCLUSIVE. '
            'Three independent search trajectories per cell; day-bootstrap CIs condition on these realized trajectories, not a search-population inference.'),
        'replicate_summary_rule': ('Report per-track effects and descriptive mean/sample SD over all three separate estimates; '
            'R-mean day bootstrap uses only days with finite effects in all three trajectories, shared day draws; '
            'pooled world contrast requires both worlds per day/replicate and weights worlds equally; no R×day pseudo-replication.'),
        'prior_posterior_rule': ('Offline only, per environment/card, Beta(1,1). Count a VALID numerically converged candidate once for each explicitly selected card; '
            'multi-card candidates contribute shared evidence, not isolated causal attribution. n_valid includes inconclusive or unavailable CI. '
            'n_plus if target-metric candidate-minus-parent paired-day 95% CI wholly negative; n_minus if wholly positive; otherwise neither; '
            'all target metrics lower is better, irrespective of declared direction verdict. Posterior mean=(1+n_plus)/(2+n_plus+n_minus).'),
        'holdout_rule': ('materialize 12 independent days only after all 108 slots consumed and all 18 trajectories finished and approved source hashes frozen; '
            'reuse trained models, oracle does not feed search. Same registered dates as historical M3; unseen by M3.1 search, not novel relative to M3. '
            'Ablation and search evaluations use independent artifact and work roots.'),
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
        baseline = _read(path)
        for env, evaluator in evaluators.items():
            z = evaluator.prepare_dev()  # rechecks train/dev file hashes on resume
            saved = baseline[env]
            if saved['z'] != z or saved['development_receipt'] != evaluator.development:
                raise ValueError(f'resumed baseline data differs: {env}')
            cached = _check_model_cache(evaluator, saved['evaluation'])
            if cached != saved['evaluation']:
                raise ValueError(f'resumed baseline evaluation differs: {env}')
        return baseline
    baseline = {}
    for env, evaluator in evaluators.items():
        z = evaluator.prepare_dev()
        result = evaluator.evaluate(SEED_SOURCE)
        if result['status'] != 'VALID' or result['state'] != 60:
            raise RuntimeError(f'baseline did not meet the fixed contract: {env}')
        _check_model_cache(evaluator, result)
        baseline[env] = {'z': z, 'evaluation': result, 'development_receipt': evaluator.development}
        print(f'BASELINE {env} READY', flush=True)
    _freeze(path, baseline)
    return baseline


def _check_model_cache(evaluator: SyntheticEvaluator, result: Json) -> Json:
    """Verify a registered development receipt and all three training weight files."""
    if result['status'] != 'VALID' or result['env_id'] != evaluator.env_id:
        raise ValueError('invalid frozen development receipt')
    cache = evaluator.root / 'models' / result['source_sha256']
    cached = _read(cache / 'dev.json')
    fit = _read(cache / 'fit.json')
    hashes = result['training_checkpoint_hashes']
    if cached != result or fit['training_checkpoint_hashes'] != hashes or set(hashes) != set(map(str, SEEDS)):
        raise ValueError('frozen development fit receipt mismatch')
    for seed in SEEDS:
        if _digest(cache / f'{seed}.pt') != hashes[str(seed)]:
            raise ValueError('frozen development checkpoint hash mismatch')
    return cached


def _audit_historical_baseline(artifact: Path, baseline: Json) -> None:
    """Isolate the calibration intervention from a changed B0 or data panel."""
    historical = ROOT / 'artifacts' / 'p2-m3-isomorphic-control' / 'baseline.json'
    prior = _read(historical)
    audit = {}
    for env in ENVS:
        now, old = baseline[env], prior[env]
        current_eval, old_eval = now['evaluation'], old['evaluation']
        current_receipt, old_receipt = now['development_receipt'], old['development_receipt']
        def same_metric(key: str, old_value: Any) -> bool:
            current = current_eval['aggregate'].get(key)
            if old_value is None or current is None:
                return old_value is current
            return isclose(float(current), float(old_value), rel_tol=1e-7, abs_tol=1e-7)
        checks = {
            'source': current_eval['source_sha256'] == old_eval['source_sha256'] == source_hash(SEED_SOURCE),
            'capacity': (current_eval['params'], current_eval['state']) == (old_eval['params'], old_eval['state']),
            'data': all(current_receipt[key] == old_receipt[key] for key in ('protocol', 'file_hashes', 'train_dates', 'dev_dates', 'reference')),
            'z': now['z'] == old['z'],
            'checkpoints': current_eval['training_checkpoint_hashes'] == old_eval['training_checkpoint_hashes'],
            'prior_d_metrics': all(same_metric(key, value) for key, value in old_eval['aggregate'].items()),
        }
        audit[env] = {'checks': checks, 'm3_baseline_sha256': _digest(historical),
                      'training_checkpoint_hashes': current_eval['training_checkpoint_hashes']}
    _freeze(artifact / 'baseline_calibration_audit.json', audit)
    if not all(all(item['checks'].values()) for item in audit.values()):
        raise RuntimeError('baseline differs from historical M3; do not dispatch formal API calls')


def _track_path(artifact: Path, rep: str, env: str, arm: str) -> Path:
    return artifact / 'trajectories' / rep / env / arm


def _records(artifact: Path, rep: str, env: str, arm: str) -> list[Json]:
    return FoldingLedger(_track_path(artifact, rep, env, arm)).records


def _non_z(prompt: list[dict[str, str]]) -> str:
    start, rest = prompt[1]['content'].split('[Macro Data Regime Z]\n', 1)
    _, suffix = rest.split('\n\n[Model Residuals D]', 1)
    return prompt[0]['content'] + '\n' + start + '[Macro Data Regime Z]\n' + suffix


def _audit_first_round(artifact: Path, baseline: Json) -> None:
    """Verify the actual first-round packets before the first network request."""
    count = openai_token_counter()
    audit: Json = {rep: {env: {} for env in ENVS} for rep in REPLICATES}
    for rep, env, arm in _tracks():
        parent = baseline[env]['evaluation']
        cards = select_active_cards(parent['aggregate'], {})
        other = ENVS[1 - ENVS.index(env)]
        z = None if arm == 'd_only' else baseline[env if arm == 'true_z_d' else other]['z']
        view = View(arm, env, z)
        loop = DiscoveryLoop(view=view, root=_track_path(artifact, rep, env, arm),
                             seed_result=parent, evaluator=lambda source: parent,
                             pair=paired_summary, count=count, call_prefix=_track_id(rep, env, arm))
        prompt, tokens = build_messages(view, 'B0', SEED_SOURCE, parent, cards, loop.ledger, (), count)
        prompt_hash = sha256(json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        audit[rep][env][arm] = {'non_z_sha256': sha256(_non_z(prompt).encode()).hexdigest(),
                                'prompt_sha256': prompt_hash, 'tokens': tokens,
                                'active_card_ids': [c.card_id for c in cards],
                                'training_checkpoint_hashes': parent['training_checkpoint_hashes']}
    for env in ENVS:
        if len({audit[rep][env][arm]['non_z_sha256'] for rep in REPLICATES for arm in ARMS}) != 1:
            raise RuntimeError('first-round information control failed before API')
        for arm in ARMS:
            if len({audit[rep][env][arm]['prompt_sha256'] for rep in REPLICATES}) != 1:
                raise RuntimeError('replicates did not start from the same prompt')
    _freeze(artifact / 'initial_control_audit.json', audit)


def _validate_roster(guard: CallBudgetGuard) -> dict[str, Json]:
    reservations = [r for r in guard.entries() if r['event'] == 'reserved']
    expected = {f'{_track_id(rep, env, arm)}/c{r:02d}'
                for rep, env, arm in _tracks() for r in range(1, ROUNDS + 1)}
    if {r['call_id'] for r in reservations} != expected or len(reservations) != MAX_CALLS:
        raise RuntimeError('formal call roster is incomplete; do not open holdout')
    return {r['call_id']: r for r in reservations}


def _audit_sampling(artifact: Path, guard: CallBudgetGuard, reservations: Mapping[str, Json]) -> None:
    """Verify separate first-round HTTP slots and report observed reply variability.

    Equal replies, if any, are retained and disclosed rather than retried. Distinct
    replies provide observed variability; provider RNG internals are not observable.
    """
    prereg = _read(artifact / 'initial_control_audit.json')
    completed = [row for row in guard.entries() if row['event'] == 'completed']
    receipts = {row['call_id']: row for row in completed}
    if len(receipts) != len(completed) or not set(receipts).issubset(reservations):
        raise RuntimeError('duplicate or unregistered API completion receipt')
    audit: Json = {env: {} for env in ENVS}
    for env in ENVS:
        for arm in ARMS:
            entries = {}
            for rep in REPLICATES:
                cid = f'{_track_id(rep, env, arm)}/c01'
                row = _records(artifact, rep, env, arm)[1]
                recorded = prereg[rep][env][arm]
                messages = reservations[cid]['messages']
                prompt_hash = sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if (row['prompt'] != messages or prompt_hash != recorded['prompt_sha256']
                    or row['active_card_ids'] != recorded['active_card_ids']
                    or sha256(_non_z(messages).encode()).hexdigest() != recorded['non_z_sha256']):
                    raise RuntimeError('first-round request differs from frozen audit')
                raw = receipts.get(cid, {}).get('raw_reply')
                if raw is not None and raw != row.get('raw_reply'):
                    raise RuntimeError('raw API response differs from trajectory receipt')
                entries[rep] = {'call_id': cid, 'prompt_sha256': prompt_hash,
                                'completed_receipt': cid in receipts,
                                'raw_reply_sha256': source_hash(raw) if raw else None,
                                'source_sha256': source_hash(row['source']) if row.get('source') else None}
            if len({item['prompt_sha256'] for item in entries.values()}) != 1:
                raise RuntimeError('replicate first-round treatment prompts diverged')
            audit[env][arm] = {'replicates': entries,
                               'distinct_raw_replies': len({item['raw_reply_sha256'] for item in entries.values() if item['raw_reply_sha256']}),
                               'distinct_sources': len({item['source_sha256'] for item in entries.values() if item['source_sha256']})}
    _freeze(artifact / 'first_round_sampling_audit.json', audit)


def _make_freeze(artifact: Path, guard: CallBudgetGuard, evaluators: Mapping[str, SyntheticEvaluator]) -> Json:
    approved = {env: set() for env in ENVS}
    trajectories = {}
    reservations = _validate_roster(guard)
    for rep, env, arm in _tracks():
        rows = _records(artifact, rep, env, arm)
        if (len(rows) != ROUNDS + 1 or rows[0]['outcome'] != 'SEED'
            or [row['cand_id'] for row in rows] != ['B0', *(f'c{r:02d}' for r in range(1, ROUNDS + 1))]):
            raise RuntimeError('all 18 tracks must finish before the independent holdout opens')
        active = rows[0]
        for row in rows:
            if row['cand_id'] != 'B0':
                call_id = f'{_track_id(rep, env, arm)}/{row["cand_id"]}'
                if row['prompt'] != reservations[call_id]['messages'] or row['prompt_tokens'] > PROMPT_CAP:
                    raise RuntimeError('round ledger request differs from its reserved slot or exceeds prompt cap')
            if row.get('evaluation', {}).get('status') == 'VALID':
                digest = row['source_sha256']
                if source_hash(row['source']) != digest:
                    raise RuntimeError('candidate source differs from evaluation receipt')
                if digest not in approved[env]:
                    _check_model_cache(evaluators[env], row['evaluation'])
                approved[env].add(digest)
            if row['outcome'] == 'PROMOTED':
                active = row
        if active['source_sha256'] not in approved[env]:
            raise RuntimeError('promoted final has no valid development receipt')
        trajectories[_track_id(rep, env, arm)] = {'final_id': active['cand_id'], 'final_sha256': active['source_sha256']}
    _audit_sampling(artifact, guard, reservations)
    value = {'approved': {env: sorted(values) for env, values in approved.items()}, 'trajectories': trajectories,
             'formal_calls': guard.count, 'num_replicates': len(REPLICATES)}
    _freeze(artifact / 'freeze.json', value)
    print('All 18 tracks frozen; holdout may now be opened', flush=True)
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

    def track(rep: str, env: str, arm: str) -> Json:
        other = ENVS[1 - ENVS.index(env)]
        z = None if arm == 'd_only' else baseline[env if arm == 'true_z_d' else other]['z']
        loop = DiscoveryLoop(view=View(arm, env, z), root=_track_path(artifact, rep, env, arm),
                             seed_result=baseline[env]['evaluation'], evaluator=lambda source: evaluate(env, source),
                             pair=paired_summary, count=count, transport=transport, guard=guard,
                             rounds=ROUNDS, should_stop=stop.is_set, call_prefix=_track_id(rep, env, arm))
        result = loop.run()
        print(f'TRACK {_track_id(rep, env, arm)} COMPLETE', flush=True)
        return result

    with ThreadPoolExecutor(max_workers=len(_tracks())) as pool:
        pending = {pool.submit(track, rep, env, arm): (rep, env, arm) for rep, env, arm in _tracks()}
        for future in as_completed(pending):
            try:
                future.result()
            except BaseException:
                stop.set()
                raise

    # Audit exact non-Z first-round equality and model receipt equality before holdout.
    for env in ENVS:
        initial = [_records(artifact, rep, env, arm)[1] for rep in REPLICATES for arm in ARMS]
        if any(row['active_card_ids'] != initial[0]['active_card_ids'] or row['prompt'][0] != initial[0]['prompt'][0]
               or _non_z(row['prompt']) != _non_z(initial[0]['prompt']) for row in initial[1:]):
            raise RuntimeError('initial non-Z prompt sections are not isomorphic')
        seeds = [_records(artifact, rep, env, arm)[0]['evaluation']['training_checkpoint_hashes']
                 for rep in REPLICATES for arm in ARMS]
        if any(item != seeds[0] for item in seeds[1:]):
            raise RuntimeError('initial model checkpoints differ across arms')


def _sources(artifact: Path, env: str) -> dict[str, str]:
    sources = {}
    for rep in REPLICATES:
        for arm in ARMS:
            for row in _records(artifact, rep, env, arm):
                if row.get('evaluation', {}).get('status') == 'VALID':
                    if source_hash(row['source']) != row['source_sha256']:
                        raise RuntimeError('frozen candidate source hash mismatch')
                    sources[row['source_sha256']] = row['source']
    return sources


def _verify(artifact: Path, evaluators: Mapping[str, SyntheticEvaluator], freeze: Json) -> Json:
    path = artifact / 'holdout.json'
    if path.exists():
        return _read(path)
    def verify_env(env: str, evaluator: SyntheticEvaluator) -> tuple[str, Json]:
        receipt = evaluator.open_holdout(artifact / 'freeze.json')
        _freeze(artifact / f'{env}-holdout-receipt.json', receipt)
        sources = _sources(artifact, env)
        if set(sources) != set(freeze['approved'][env]):
            raise RuntimeError('holdout request does not match authorized source set')
        base = evaluator.evaluate(SEED_SOURCE, split='holdout_test')
        evaluated = {}
        for digest, source in sources.items():
            candidate = base if digest == source_hash(SEED_SOURCE) else evaluator.evaluate(source, split='holdout_test')
            against = paired_summary(candidate, base) if candidate['status'] == base['status'] == 'VALID' else {}
            evaluated[digest] = {'evaluation': candidate, 'against_B0': against}
        print(f'HOLDOUT {env} COMPLETE ({len(sources)} authorized models)', flush=True)
        return env, evaluated

    with ThreadPoolExecutor(max_workers=len(ENVS)) as pool:
        holdout = dict(future.result() for future in [pool.submit(verify_env, env, evaluator)
                                                     for env, evaluator in evaluators.items()])
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


def _finite(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and isfinite(value) else None


def _complete_stats(values: Mapping[str, float | None]) -> tuple[float | None, float | None]:
    """A descriptive replicate summary requires all three estimates, not selected successes."""
    if set(values) != set(REPLICATES) or any(value is None for value in values.values()):
        return None, None
    available = [float(value) for value in values.values() if value is not None]
    return mean(available), stdev(available)


def _metric_daily(effect: Json, metric: str, days: int) -> list[float | None]:
    values = effect.get(metric, {}).get('daily')
    if values is None:
        return [None] * days
    if len(values) != days:
        raise ValueError('effect panel does not have the registered holdout calendar')
    return [_finite(value) for value in values]


def _aggregate_metrics(effects: Mapping[str, Json], dates: Sequence[int]) -> Json:
    """R-mean effects use common complete days; the bootstrap samples days, not R×day cells."""
    if set(effects) != set(REPLICATES) or len(dates) != EVAL_DAYS or len(set(dates)) != len(dates):
        raise ValueError('replicated effects require the entire registered panel')
    result = {}
    for metric in METRICS:
        estimates = {rep: _finite(effect.get(metric, {}).get('estimate')) for rep, effect in effects.items()}
        replicate_mean, replicate_sd = _complete_stats(estimates)
        panel = {rep: _metric_daily(effects[rep], metric, len(dates)) for rep in REPLICATES}
        common = [index for index in range(len(dates)) if all(panel[rep][index] is not None for rep in REPLICATES)]
        daily = [mean(float(panel[rep][i]) for rep in REPLICATES) if i in common else None for i in range(len(dates))]
        common_estimates = {rep: mean(float(panel[rep][index]) for index in common) if common else None for rep in REPLICATES}
        common_mean, common_sd = _complete_stats(common_estimates)
        result[metric] = {**summarize_days(daily), 'daily': daily,
                          'replicate_estimates': estimates, 'replicate_mean': replicate_mean, 'replicate_sd': replicate_sd,
                          'replicates': list(REPLICATES), 'replicate_count': len(REPLICATES),
                          'replicate_estimate_support': sum(value is not None for value in estimates.values()),
                          'common_dates': [dates[i] for i in common], 'common_panel_replicate_estimates': common_estimates,
                          'common_panel_replicate_mean': common_mean, 'common_panel_replicate_sd': common_sd}
    return result


def _pool_worlds(effects: Mapping[str, Json], dates: Sequence[int]) -> Json:
    """Equal-weight paired worlds per day, then aggregate trajectories; never pool 24 day rows."""
    result = {}
    for metric in METRICS:
        panels = [_metric_daily(effects[env], metric, len(dates)) for env in ENVS]
        values = [mean(float(panel[i]) for panel in panels) if all(panel[i] is not None for panel in panels) else None
                  for i in range(len(dates))]
        result[metric] = {**summarize_days(values), 'daily': values}
    return result


def _prior_posterior(rows: Mapping[str, Sequence[Json]]) -> Json:
    """Exploratory conditional evidence, not an isolated effect per co-selected mechanism."""
    result = {env: {card: {'n_valid': 0, 'n_plus': 0, 'n_minus': 0} for card in CARDS} for env in ENVS}
    for track, records in rows.items():
        env = track.split('/')[1]
        for row in records[1:]:
            evaluation = row.get('evaluation', {})
            if evaluation.get('status') != 'VALID' or _finite(evaluation.get('aggregate', {}).get('val_nll_total')) is None:
                continue
            metric = row.get('prediction', {}).get('target_metric')
            interval = row.get('deltas', {}).get(metric, {}).get('interval')
            for card in set(row.get('selected_card_ids', [])):
                counts = result[env][card]
                counts['n_valid'] += 1
                if interval is not None and interval[1] < 0:
                    counts['n_plus'] += 1
                elif interval is not None and interval[0] > 0:
                    counts['n_minus'] += 1
    for env in ENVS:
        for counts in result[env].values():
            counts['n_inconclusive'] = counts['n_valid'] - counts['n_plus'] - counts['n_minus']
            counts['posterior_mean'] = (1 + counts['n_plus']) / (2 + counts['n_plus'] + counts['n_minus'])
    return result


def _alignment_cell(rows: Mapping[str, Sequence[Json]], env: str) -> Json:
    result = {}
    for scope in ('first', 'all'):
        attempts = 1 if scope == 'first' else ROUNDS
        counts = {}
        for rep in REPLICATES:
            selected = rows[rep][1:2] if scope == 'first' else rows[rep][1:]
            matched = sum(_aligned(row, env) for row in selected)
            valid = [row for row in selected if row.get('evaluation', {}).get('status') == 'VALID']
            declared = [row for row in selected if row.get('selected_card_ids') and row.get('prediction')
                        and row.get('outcome') not in ('INVALID_PROPOSAL', 'PROVIDER_ERROR', 'INTERRUPTED_CALL')]
            counts[rep] = {'attempts': attempts, 'matched': matched, 'valid': len(valid),
                           'valid_matched': sum(_aligned(row, env) for row in valid), 'declared': len(declared),
                           'avoided_C06': sum('C06' not in row['selected_card_ids'] for row in declared) if env == 'env_beta' else None,
                           'rate': matched / attempts}
        rates = [counts[rep]['rate'] for rep in REPLICATES]
        result[scope] = {'replicates': counts, 'attempts': attempts * len(REPLICATES),
                         'matched': sum(count['matched'] for count in counts.values()),
                         'mean_rate': mean(rates), 'sample_sd_rate': stdev(rates)}
    return result


def _summarize(artifact: Path, freeze: Json, holdout: Json, guard: CallBudgetGuard, seconds: float) -> Json:
    regrets = {}
    alignment = {arm: 0 for arm in ARMS}
    tracks, track_effects, comparisons = {}, {}, {}
    for rep, env, arm in _tracks():
        track = _track_id(rep, env, arm)
        rows = _records(artifact, rep, env, arm)
        tracks[track] = rows
        final = freeze['trajectories'][track]['final_sha256']
        regrets[track] = _regret(rows, final, holdout[env])
        alignment[arm] += sum(_aligned(row, env) for row in rows[1:])
        track_effects[track] = holdout[env][final]['against_B0']
    for rep in REPLICATES:
        for env in ENVS:
            true_final = freeze['trajectories'][_track_id(rep, env, 'true_z_d')]['final_sha256']
            for arm in COMPARATORS:
                other_final = freeze['trajectories'][_track_id(rep, env, arm)]['final_sha256']
                candidate = holdout[env][true_final]['evaluation']
                parent = holdout[env][other_final]['evaluation']
                comparisons[f'{rep}/{env}/true_z_d-vs-{arm}'] = paired_summary(candidate, parent) if candidate['status'] == parent['status'] == 'VALID' else {}
    dates = {env: _read(artifact / f'{env}-holdout-receipt.json')['dates'] for env in ENVS}
    if dates[ENVS[0]] != dates[ENVS[1]]:
        raise ValueError('paired worlds have different registered holdout date indexes')
    cells, cross, pooled, alignment_by_cell = {}, {}, {}, {}
    for env in ENVS:
        for arm in ARMS:
            cells[f'{env}/{arm}'] = _aggregate_metrics(
                {rep: track_effects[_track_id(rep, env, arm)] for rep in REPLICATES}, dates[env])
            alignment_by_cell[f'{env}/{arm}'] = _alignment_cell(
                {rep: tracks[_track_id(rep, env, arm)] for rep in REPLICATES}, env)
        for arm in COMPARATORS:
            cross[f'{env}/true_z_d-vs-{arm}'] = _aggregate_metrics(
                {rep: comparisons[f'{rep}/{env}/true_z_d-vs-{arm}'] for rep in REPLICATES}, dates[env])
    for arm in COMPARATORS:
        pooled[f'true_z_d-vs-{arm}'] = _aggregate_metrics(
            {rep: _pool_worlds({env: comparisons[f'{rep}/{env}/true_z_d-vs-{arm}'] for env in ENVS}, dates[ENVS[0]])
             for rep in REPLICATES}, dates[ENVS[0]])
    align_support = all(alignment['true_z_d'] > alignment[arm] for arm in COMPARATORS)
    def supported(row: Json) -> bool:
        cf = row.get('counterfactual_impact_mae', {}).get('interval')
        nll = row.get('val_nll_total', {}).get('estimate')
        exhaust = row.get('exhausted_replicate_rate', {}).get('estimate')
        return cf is not None and cf[1] < 0 and nll is not None and exhaust is not None and nll <= .005 and exhaust <= .05
    def refuted(row: Json) -> bool:
        cf = row.get('counterfactual_impact_mae', {}).get('interval')
        return cf is not None and cf[0] > 0
    hold_support = all(supported(row) for row in cross.values())
    hold_refuted = all(refuted(row) for row in cross.values())
    status = 'SUPPORTED_WITH_BOUNDARY' if align_support and hold_support else 'PARTIAL' if align_support or hold_support else 'REFUTED' if hold_refuted else 'INCONCLUSIVE'
    completed = [row for row in guard.entries() if row['event'] == 'completed']
    return {'status': status, 'alignment_supported': align_support, 'holdout_supported': hold_support,
            'holdout_refuted': hold_refuted, 'alignment_counts': alignment, 'alignment_by_cell': alignment_by_cell,
            'regret': regrets, 'track_effects': track_effects,
            'cross_arm_effects': comparisons, 'cell_effects': cells, 'group_cross_arm_effects': cross,
            'pooled_cross_arm_effects': pooled, 'prior_posterior': _prior_posterior(tracks),
            'num_replicates': len(REPLICATES), 'formal_calls': guard.count,
            'formal_completed_receipts': len(completed),
            'formal_input_tokens': sum(r['input_tokens'] for r in completed if r['input_tokens'] is not None),
            'formal_output_tokens': sum(r['output_tokens'] for r in completed if r['output_tokens'] is not None),
            'unknown_usage_calls': sum(r['input_tokens'] is None or r['output_tokens'] is None for r in completed) + guard.count - len(completed),
            'wall_seconds': seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--stochopia-root', type=Path, required=True)
    parser.add_argument('--num-replicates', type=int, choices=(3,), default=3)
    args = parser.parse_args()
    artifact, work = args.artifact_root.resolve(), args.work_root.resolve()
    historical_names = {'p2-m1-validation', 'p2-m2-synthetic', 'p2-m3-isomorphic-control'}
    if (artifact.name != ARTIFACT_NAME or work.is_relative_to(artifact) or artifact.is_relative_to(work)
        or historical_names.intersection(artifact.parts) or historical_names.intersection(work.parts)):
        raise ValueError('the study requires an isolated M3.1-only artifact root and private work root')
    artifact.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    reg = registration(args.stochopia_root.resolve(), tuple(f'rep_{i}' for i in range(args.num_replicates)))
    _freeze(artifact / 'registration.json', reg)
    work_registration = work / 'm31-registration.json'
    if not work_registration.exists() and any((work / env).exists() for env in ENVS):
        raise ValueError('private work root already contains an unrelated evaluation cache')
    _freeze(work_registration, {'registration_sha256': _digest(artifact / 'registration.json'),
                                'artifact_root': str(artifact)})
    evaluators = {env: SyntheticEvaluator(work, env) for env in ENVS}
    baseline = _baseline(artifact, evaluators)
    _audit_historical_baseline(artifact, baseline)
    guard = CallBudgetGuard(artifact / 'api_calls.jsonl', limit=MAX_CALLS)
    if not (artifact / 'freeze.json').exists():
        if guard.count == 0:
            _audit_first_round(artifact, baseline)
        _handshake(artifact)
        _run_tracks(artifact, evaluators, baseline, guard)
        freeze = _make_freeze(artifact, guard, evaluators)
    else:
        freeze = _make_freeze(artifact, guard, evaluators)
    holdout = _verify(artifact, evaluators, freeze)
    if not (artifact / 'study_summary.json').exists():
        atomic_write_json(artifact / 'study_summary.json', _summarize(artifact, freeze, holdout, guard, perf_counter() - started))
    print('M3.1 complete; evidence ready for report and checksums', flush=True)


if __name__ == '__main__':
    main()
