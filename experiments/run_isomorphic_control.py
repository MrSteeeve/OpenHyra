"""Run the replicated open-catalog isomorphic control and Loop 2 transplants; no holdout before freeze."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
import os
import shutil
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
from tasks.lob_world_model.proposal import TARGET_METRICS, Z_METRICS
from tasks.lob_world_model.provider import DEFAULT_MODELS_PATH, provider_handshake, registry_transport
from tasks.lob_world_model.provider_client import load_model_registry
from tasks.lob_world_model.synthetic_data import SINGLE_FACTOR_PAIRS
from tasks.lob_world_model.synthetic_eval import (
    DATA_PROTOCOL, SyntheticEvaluator, evaluate_boundary_claim, paired_summary,
    single_factor_pair_for_z, transplant_contrast,
)
from tasks.lob_world_model.token_count import openai_token_counter

Json = dict[str, Any]
ARMS = ('d_only', 'true_z_d', 'shuffled_z_d')
ENVS = ('env_alpha', 'env_beta')
ROUNDS = 6
REPLICATES = ('rep_0', 'rep_1', 'rep_2')
MAX_CALLS = len(REPLICATES) * len(ARMS) * len(ENVS) * ROUNDS
Z_HEADER = '[1. Market Data Profile Z]\n'
CARD_HEADER = '\n\n[2. Prior Mechanism Catalog'
ARTIFACT_NAME = 'p2-m32-open-catalog-control'
MODEL = 'gpt-6.2-sol'
MIN_DAYS = 6
ALIGNMENT = {'env_alpha': ['C04', 'C05', 'C06'], 'env_beta': ['C03', 'C08']}
ROOT = Path(__file__).resolve().parents[1]
METRICS = ('val_nll_total', 'dt_nll', 'counterfactual_impact_mae',
           'exhausted_replicate_rate', 'memory_gap_64')
CARDS = tuple(f'C{number:02d}' for number in range(1, 11))
COMPARATORS = ('d_only', 'shuffled_z_d')


def _call_cap(replicates: Sequence[str]) -> int:
    return len(replicates) * len(ARMS) * len(ENVS) * ROUNDS


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
    expected_reps = tuple(f'rep_{i}' for i in range(len(replicates)))
    if len(replicates) not in (2, 3) or tuple(replicates) != expected_reps:
        raise ValueError('the formal study requires two or three canonical registered replicates')
    if model.temperature is not None and model.temperature <= 0:
        raise ValueError('the sampled search cannot use zero temperature')
    n_rep = len(replicates)
    cap = _call_cap(replicates)
    attempts_per_arm = n_rep * len(ENVS) * ROUNDS
    n_tracks = n_rep * len(ENVS) * len(ARMS)
    return {
        'schema': 'isomorphic-control-v3.2-open-catalog-transplant',
        'stochopia_commit': _git(stochopia, 'rev-parse', 'HEAD'),
        'openhyra_commit': _git(ROOT, 'rev-parse', 'HEAD'), 'model': MODEL,
        'provider': {'model': model.model, 'max_output_tokens': model.max_tokens,
                     'temperature': model.temperature, 'extra_body': model.extra_body,
                     'strict_model_match': model.strict_model_match, 'registry_sha256': _digest(DEFAULT_MODELS_PATH),
                     'sampling': 'provider-native stochastic reasoning; omit unsupported explicit temperature and seed',
                     'response_cache': 'none between logical call slots; durable receipt replay only within the same slot after interruption'},
        'environments': list(ENVS), 'arms': list(ARMS), 'rounds': ROUNDS, 'formal_call_cap': cap,
        'num_replicates': n_rep, 'replicates': list(replicates),
        'handshake_cap': 1, 'prompt_token_cap': PROMPT_CAP, 'candidate_source_token_cap': SOURCE_TOKEN_CAP,
        'seed_model_sha256': source_hash(SEED_SOURCE), 'data_protocol': DATA_PROTOCOL,
        'paired_days_minimum': MIN_DAYS, 'bootstrap_draws': 2000,
        'promotion': {'nll_max_increase': .005, 'exhaustion_max_increase': .05,
                      'cf_min_improvement_ticks': .01, 'cf_equivalence_ticks': .01,
                      'exhaustion_min_improvement': .10, 'nll_min_improvement': .008,
                      'viability_max_surviving_cf_point_increase_ticks': .50,
                      'objective_improvement': 'CF paired estimate with >=6 common valid days decides',
                      'viability_recovery': 'CF support <6 common days: guardrails, dExhaust<=-0.10, and surviving point dCF<=+0.50 when present'},
        'stepping_stones': {'nll_max_increase': .025, 'dt_improvement': .010,
                            'exhaustion_improvement': .10, 'acf_relative_improvement': .25,
                            'memory_gap64_absolute_change': .05},
        'dedup_similarity_strict_gt': .94,
        'card_catalog': 'all C01-C10 every round, order random.Random(f"{env}:{rep}:{round}"), shared by the three arms',
        'target_metrics_allowed': list(TARGET_METRICS),
        'z_boundary_metrics_allowed': list(Z_METRICS),
        'prediction_fields': ['data_regime_diagnosis', 'selected_card_ids (may be empty)', 'regime_hypothesis',
                              'target_metric', 'expected_direction', 'weaker_when_z_metric', 'weaker_when_z_direction'],
        'single_factor_transplant_pairs': {k: list(v) for k, v in SINGLE_FACTOR_PAIRS.items()},
        'alignment_primary_card': ALIGNMENT, 'alignment_beta_excluded': 'C06',
        'alignment_denominator': ('all attempted rounds; primary selected card only; malformed/absent counted as nonaligned, '
                                  'empty selection counted as nonaligned open structure; report coverage separately'),
        'prediction_rule': 'candidate-parent 95% paired day CI wholly predicted side supports, opposite side refutes; crossing zero or absent CI inconclusive; no evaluated candidate non-evaluable',
        'regret_rule': 'holdout CF MAE of final minus minimum over B0 and all valid evaluated candidates, on same intersection of finite paired days across oracle set; require >=6 days; no imputation',
        'controlled_comparison_rule': (
            f'True-Z directional alignment must strictly exceed each comparison arm over all {attempts_per_arm} attempts; '
            'holdout support requires both envs vs both arms R-mean CF common-day CI upper < 0 and point NLL/exhaust deltas <= .005/.05. '
            'Report descriptive alignment separately; if both support then SUPPORTED_WITH_BOUNDARY, '
            'if one supports PARTIAL, if both envs vs both CF CI lower > 0 REFUTED, otherwise INCONCLUSIVE. '
            f'{n_rep} independent search trajectories per cell; day-bootstrap CIs condition on these realized trajectories, not a search-population inference.'),
        'replicate_summary_rule': (
            f'Report per-track effects and descriptive mean/sample SD over all {n_rep} separate estimates; '
            f'R-mean day bootstrap uses only days with finite effects in all {n_rep} trajectories, shared day draws; '
            'pooled world contrast requires both worlds per day/replicate and weights worlds equally; no R×day pseudo-replication.'),
        'prior_posterior_rule': ('Offline only, per environment/card, Beta(1,1). Count a VALID numerically converged candidate once for each explicitly selected card; '
            'multi-card candidates contribute shared evidence, not isolated causal attribution. n_valid includes inconclusive or unavailable CI. '
            'n_plus if target-metric candidate-minus-parent paired-day 95% CI wholly negative; n_minus if wholly positive; otherwise neither; '
            'all target metrics lower is better, irrespective of declared direction verdict. Posterior mean=(1+n_plus)/(2+n_plus+n_minus).'),
        'holdout_rule': (
            f'materialize 12 independent days only after all {cap} slots consumed and all {n_tracks} trajectories finished and approved source hashes frozen; '
            'reuse trained models, oracle does not feed search. Same registered dates as historical M3/M3.1; unseen by this search, not novel relative to them. '
            'Transplant and search evaluations use independent work roots.'),
        'loop2_transplant_rule': (
            'Zero API calls. Extract all unique non-B0 final promoted winners across trajectories plus any valid open-innovation '
            '(selected_card_ids == []) candidate whose target_metric paired dev 95% CI is wholly negative; seal transplant-freeze.json '
            'before opening any single-factor holdout; refit unchanged in mapped P2-M2 single-factor pair and test delta_tau = tau(z_strong) - tau(z_weak) '
            'on target_metric (BOUNDARY_CONFIRMED if 95% CI wholly < 0 and tau(z_strong) < 0; BOUNDARY_REFUTED if 95% CI wholly > 0; else INCONCLUSIVE).'),
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
    """Isolate the intervention from a changed B0 or data panel."""
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
    start, rest = prompt[1]['content'].split(Z_HEADER, 1)
    _, suffix = rest.split(CARD_HEADER, 1)
    return prompt[0]['content'] + '\n' + start + Z_HEADER + CARD_HEADER + suffix


def _audit_first_round(artifact: Path, baseline: Json, replicates: Sequence[str] = REPLICATES) -> None:
    """Verify the actual first-round packets before the first network request.

    Within one replicate the three arms differ only in the Z block; card order
    differs between replicates by design (per-replicate shuffle).
    """
    count = openai_token_counter()
    audit: Json = {rep: {env: {} for env in ENVS} for rep in replicates}
    for rep, env, arm in _tracks(replicates):
        parent = baseline[env]['evaluation']
        other = ENVS[1 - ENVS.index(env)]
        z = None if arm == 'd_only' else baseline[env if arm == 'true_z_d' else other]['z']
        view = View(arm, env, z)
        loop = DiscoveryLoop(view=view, root=_track_path(artifact, rep, env, arm),
                             seed_result=parent, evaluator=lambda source: parent,
                             pair=paired_summary, count=count, call_prefix=_track_id(rep, env, arm),
                             replicate_id=rep)
        cards = loop.cards(1)
        prompt, tokens = build_messages(view, 'B0', SEED_SOURCE, parent, cards, loop.ledger, (), count)
        prompt_hash = sha256(json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        audit[rep][env][arm] = {'non_z_sha256': sha256(_non_z(prompt).encode()).hexdigest(),
                                'prompt_sha256': prompt_hash, 'tokens': tokens,
                                'card_order': [c.card_id for c in cards],
                                'training_checkpoint_hashes': parent['training_checkpoint_hashes']}
    for rep in replicates:
        for env in ENVS:
            if len({audit[rep][env][arm]['non_z_sha256'] for arm in ARMS}) != 1:
                raise RuntimeError('first-round information control failed before API')
    _freeze(artifact / 'initial_control_audit.json', audit)


def _validate_roster(guard: CallBudgetGuard, replicates: Sequence[str] = REPLICATES) -> dict[str, Json]:
    reservations = [r for r in guard.entries() if r['event'] == 'reserved']
    expected = {f'{_track_id(rep, env, arm)}/c{r:02d}'
                for rep, env, arm in _tracks(replicates) for r in range(1, ROUNDS + 1)}
    if {r['call_id'] for r in reservations} != expected or len(reservations) != _call_cap(replicates):
        raise RuntimeError('formal call roster is incomplete; do not open holdout')
    return {r['call_id']: r for r in reservations}


def _audit_sampling(artifact: Path, guard: CallBudgetGuard, reservations: Mapping[str, Json],
                    replicates: Sequence[str] = REPLICATES) -> None:
    """Verify separate first-round HTTP slots and report observed reply variability."""
    prereg = _read(artifact / 'initial_control_audit.json')
    completed = [row for row in guard.entries() if row['event'] == 'completed']
    receipts = {row['call_id']: row for row in completed}
    if len(receipts) != len(completed) or not set(receipts).issubset(reservations):
        raise RuntimeError('duplicate or unregistered API completion receipt')
    audit: Json = {env: {} for env in ENVS}
    for env in ENVS:
        for arm in ARMS:
            entries = {}
            for rep in replicates:
                cid = f'{_track_id(rep, env, arm)}/c01'
                row = _records(artifact, rep, env, arm)[1]
                recorded = prereg[rep][env][arm]
                messages = reservations[cid]['messages']
                prompt_hash = sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if (row['prompt'] != messages or prompt_hash != recorded['prompt_sha256']
                    or row['card_order'] != recorded['card_order']
                    or sha256(_non_z(messages).encode()).hexdigest() != recorded['non_z_sha256']):
                    raise RuntimeError('first-round request differs from frozen audit')
                raw = receipts.get(cid, {}).get('raw_reply')
                if raw is not None and raw != row.get('raw_reply'):
                    raise RuntimeError('raw API response differs from trajectory receipt')
                entries[rep] = {'call_id': cid, 'prompt_sha256': prompt_hash,
                                'completed_receipt': cid in receipts,
                                'raw_reply_sha256': source_hash(raw) if raw else None,
                                'source_sha256': source_hash(row['source']) if row.get('source') else None}
            audit[env][arm] = {'replicates': entries,
                               'distinct_raw_replies': len({item['raw_reply_sha256'] for item in entries.values() if item['raw_reply_sha256']}),
                               'distinct_sources': len({item['source_sha256'] for item in entries.values() if item['source_sha256']})}
    _freeze(artifact / 'first_round_sampling_audit.json', audit)


def _make_freeze(artifact: Path, guard: CallBudgetGuard, evaluators: Mapping[str, SyntheticEvaluator],
                 replicates: Sequence[str] = REPLICATES) -> Json:
    approved = {env: set() for env in ENVS}
    trajectories = {}
    reservations = _validate_roster(guard, replicates)
    tracks = _tracks(replicates)
    for rep, env, arm in tracks:
        rows = _records(artifact, rep, env, arm)
        if (len(rows) != ROUNDS + 1 or rows[0]['outcome'] != 'SEED'
            or [row['cand_id'] for row in rows] != ['B0', *(f'c{r:02d}' for r in range(1, ROUNDS + 1))]):
            raise RuntimeError(f'all {len(tracks)} tracks must finish before the independent holdout opens')
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
    _audit_sampling(artifact, guard, reservations, replicates)
    value = {'approved': {env: sorted(values) for env, values in approved.items()}, 'trajectories': trajectories,
             'formal_calls': guard.count, 'num_replicates': len(replicates)}
    _freeze(artifact / 'freeze.json', value)
    print(f'All {len(tracks)} tracks frozen; holdout may now be opened', flush=True)
    return value


def _run_tracks(artifact: Path, evaluators: Mapping[str, SyntheticEvaluator], baseline: Json,
                guard: CallBudgetGuard, replicates: Sequence[str] = REPLICATES) -> None:
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
                             rounds=ROUNDS, should_stop=stop.is_set, call_prefix=_track_id(rep, env, arm),
                             replicate_id=rep)
        result = loop.run()
        print(f'TRACK {_track_id(rep, env, arm)} COMPLETE', flush=True)
        return result

    tracks = _tracks(replicates)
    with ThreadPoolExecutor(max_workers=len(tracks)) as pool:
        pending = {pool.submit(track, rep, env, arm): (rep, env, arm) for rep, env, arm in tracks}
        for future in as_completed(pending):
            try:
                future.result()
            except BaseException:
                stop.set()
                raise

    # Audit exact non-Z first-round equality and model receipt equality before holdout.
    for rep in replicates:
        for env in ENVS:
            initial = [_records(artifact, rep, env, arm)[1] for arm in ARMS]
            if any(row['card_order'] != initial[0]['card_order']
                   or _non_z(row['prompt']) != _non_z(initial[0]['prompt']) for row in initial[1:]):
                raise RuntimeError('initial non-Z prompt sections are not isomorphic')
    for env in ENVS:
        seeds = [_records(artifact, rep, env, arm)[0]['evaluation']['training_checkpoint_hashes']
                 for rep in replicates for arm in ARMS]
        if any(item != seeds[0] for item in seeds[1:]):
            raise RuntimeError('initial model checkpoints differ across arms')


def _sources(artifact: Path, env: str, replicates: Sequence[str] = REPLICATES) -> dict[str, str]:
    sources = {}
    for rep in replicates:
        for arm in ARMS:
            for row in _records(artifact, rep, env, arm):
                if row.get('evaluation', {}).get('status') == 'VALID':
                    if source_hash(row['source']) != row['source_sha256']:
                        raise RuntimeError('frozen candidate source hash mismatch')
                    sources[row['source_sha256']] = row['source']
    return sources


def _verify(artifact: Path, evaluators: Mapping[str, SyntheticEvaluator], freeze: Json,
            replicates: Sequence[str] = REPLICATES) -> Json:
    path = artifact / 'holdout.json'
    if path.exists():
        return _read(path)
    def verify_env(env: str, evaluator: SyntheticEvaluator) -> tuple[str, Json]:
        receipt = evaluator.open_holdout(artifact / 'freeze.json')
        _freeze(artifact / f'{env}-holdout-receipt.json', receipt)
        sources = _sources(artifact, env, replicates)
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


def _complete_stats(values: Mapping[str, float | None],
                    replicates: Sequence[str] = REPLICATES) -> tuple[float | None, float | None]:
    """A descriptive replicate summary requires all registered estimates, not selected successes."""
    if set(values) != set(replicates) or any(value is None for value in values.values()):
        return None, None
    available = [float(value) for value in values.values() if value is not None]
    return mean(available), stdev(available) if len(available) > 1 else 0.0


def _metric_daily(effect: Json, metric: str, days: int) -> list[float | None]:
    values = effect.get(metric, {}).get('daily')
    if values is None:
        return [None] * days
    if len(values) != days:
        raise ValueError('effect panel does not have the registered holdout calendar')
    return [_finite(value) for value in values]


def _aggregate_metrics(effects: Mapping[str, Json], dates: Sequence[int],
                       replicates: Sequence[str] = REPLICATES) -> Json:
    """R-mean effects use common complete days; the bootstrap samples days, not R×day cells."""
    if set(effects) != set(replicates) or len(dates) != EVAL_DAYS or len(set(dates)) != len(dates):
        raise ValueError('replicated effects require the entire registered panel')
    result = {}
    for metric in METRICS:
        estimates = {rep: _finite(effect.get(metric, {}).get('estimate')) for rep, effect in effects.items()}
        replicate_mean, replicate_sd = _complete_stats(estimates, replicates)
        panel = {rep: _metric_daily(effects[rep], metric, len(dates)) for rep in replicates}
        common = [index for index in range(len(dates)) if all(panel[rep][index] is not None for rep in replicates)]
        daily = [mean(float(panel[rep][i]) for rep in replicates) if i in common else None for i in range(len(dates))]
        common_estimates = {rep: mean(float(panel[rep][index]) for index in common) if common else None for rep in replicates}
        common_mean, common_sd = _complete_stats(common_estimates, replicates)
        result[metric] = {**summarize_days(daily), 'daily': daily,
                          'replicate_estimates': estimates, 'replicate_mean': replicate_mean, 'replicate_sd': replicate_sd,
                          'replicates': list(replicates), 'replicate_count': len(replicates),
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


def _alignment_cell(rows: Mapping[str, Sequence[Json]], env: str,
                    replicates: Sequence[str] = REPLICATES) -> Json:
    result = {}
    for scope in ('first', 'all'):
        attempts = 1 if scope == 'first' else ROUNDS
        counts = {}
        for rep in replicates:
            selected = rows[rep][1:2] if scope == 'first' else rows[rep][1:]
            matched = sum(_aligned(row, env) for row in selected)
            valid = [row for row in selected if row.get('evaluation', {}).get('status') == 'VALID']
            declared = [row for row in selected if row.get('prediction')
                        and row.get('outcome') not in ('INVALID_PROPOSAL', 'PROVIDER_ERROR', 'INTERRUPTED_CALL')]
            open_innov = [row for row in declared if row.get('selected_card_ids') == []]
            counts[rep] = {'attempts': attempts, 'matched': matched, 'valid': len(valid),
                           'valid_matched': sum(_aligned(row, env) for row in valid), 'declared': len(declared),
                           'open_innovations': len(open_innov),
                           'avoided_C06': sum('C06' not in row.get('selected_card_ids', []) for row in declared) if env == 'env_beta' else None,
                           'rate': matched / attempts}
        rates = [counts[rep]['rate'] for rep in replicates]
        result[scope] = {'replicates': counts, 'attempts': attempts * len(replicates),
                         'matched': sum(count['matched'] for count in counts.values()),
                         'mean_rate': mean(rates), 'sample_sd_rate': stdev(rates) if len(rates) > 1 else 0.0}
    return result


def _summarize(artifact: Path, freeze: Json, holdout: Json, guard: CallBudgetGuard, seconds: float,
               replicates: Sequence[str] = REPLICATES) -> Json:
    regrets = {}
    alignment = {arm: 0 for arm in ARMS}
    tracks, track_effects, comparisons = {}, {}, {}
    for rep, env, arm in _tracks(replicates):
        track = _track_id(rep, env, arm)
        rows = _records(artifact, rep, env, arm)
        tracks[track] = rows
        final = freeze['trajectories'][track]['final_sha256']
        regrets[track] = _regret(rows, final, holdout[env])
        alignment[arm] += sum(_aligned(row, env) for row in rows[1:])
        track_effects[track] = holdout[env][final]['against_B0']
    for rep in replicates:
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
                {rep: track_effects[_track_id(rep, env, arm)] for rep in replicates}, dates[env], replicates)
            alignment_by_cell[f'{env}/{arm}'] = _alignment_cell(
                {rep: tracks[_track_id(rep, env, arm)] for rep in replicates}, env, replicates)
        for arm in COMPARATORS:
            cross[f'{env}/true_z_d-vs-{arm}'] = _aggregate_metrics(
                {rep: comparisons[f'{rep}/{env}/true_z_d-vs-{arm}']} if len(replicates) == 1 else
                {rep: comparisons[f'{rep}/{env}/true_z_d-vs-{arm}'] for rep in replicates}, dates[env], replicates)
    for arm in COMPARATORS:
        pooled[f'true_z_d-vs-{arm}'] = _aggregate_metrics(
            {rep: _pool_worlds({env: comparisons[f'{rep}/{env}/true_z_d-vs-{arm}'] for env in ENVS}, dates[ENVS[0]])
             for rep in replicates}, dates[ENVS[0]], replicates)
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
            'num_replicates': len(replicates), 'formal_calls': guard.count,
            'formal_completed_receipts': len(completed),
            'formal_input_tokens': sum(r['input_tokens'] for r in completed if r['input_tokens'] is not None),
            'formal_output_tokens': sum(r['output_tokens'] for r in completed if r['output_tokens'] is not None),
            'unknown_usage_calls': sum(r['input_tokens'] is None or r['output_tokens'] is None for r in completed) + guard.count - len(completed),
            'wall_seconds': seconds}


def _select_transplant_roster(artifact: Path, freeze: Json,
                              replicates: Sequence[str] = REPLICATES) -> tuple[dict[str, str], list[Json]]:
    """Extract all non-B0 final winners plus significantly improving open-innovation candidates."""
    sources: dict[str, str] = {}
    claims_by_key: dict[tuple[str, str, str, str], Json] = {}
    for rep, env, arm in _tracks(replicates):
        track = _track_id(rep, env, arm)
        rows = _records(artifact, rep, env, arm)
        final_id = freeze['trajectories'][track]['final_id']
        for row in rows[1:]:
            if row.get('evaluation', {}).get('status') != 'VALID':
                continue
            pred = row.get('prediction') or {}
            if not pred:
                continue
            is_winner = row['cand_id'] == final_id and final_id != 'B0'
            target = pred['target_metric']
            interval = (row.get('deltas') or {}).get(target, {}).get('interval')
            is_open_sig = (row.get('selected_card_ids') == []
                           and interval is not None and interval[1] < 0)
            if not (is_winner or is_open_sig):
                continue
            digest = row['source_sha256']
            sources[digest] = row['source']
            z_metric = pred['weaker_when_z_metric']
            z_dir = pred['weaker_when_z_direction']
            pair_name, worlds = single_factor_pair_for_z(z_metric)
            role = ('final_winner+open_innovation' if is_winner and is_open_sig
                    else 'final_winner' if is_winner else 'open_innovation')
            key = (digest, z_metric, z_dir, target)
            origin = f'{track}/{row["cand_id"]}'
            if key not in claims_by_key:
                claims_by_key[key] = {
                    'claim_id': f'{origin.replace("/", "-")}',
                    'origins': [origin],
                    'origin_envs': [env],
                    'origin_arms': [arm],
                    'role': role,
                    'promotion_reasons': [row.get('promotion_reason')] if row.get('promotion_reason') else [],
                    'source_sha256': digest,
                    'selected_card_ids': list(row.get('selected_card_ids', [])),
                    'target_metric': target,
                    'expected_direction': pred['expected_direction'],
                    'weaker_when_z_metric': z_metric,
                    'weaker_when_z_direction': z_dir,
                    'data_regime_diagnosis': pred['data_regime_diagnosis'],
                    'regime_hypothesis': pred['regime_hypothesis'],
                    'pair_name': pair_name,
                    'environments': list(worlds),
                }
            else:
                existing = claims_by_key[key]
                existing['origins'].append(origin)
                if env not in existing['origin_envs']:
                    existing['origin_envs'].append(env)
                if arm not in existing['origin_arms']:
                    existing['origin_arms'].append(arm)
                if row.get('promotion_reason') and row['promotion_reason'] not in existing['promotion_reasons']:
                    existing['promotion_reasons'].append(row['promotion_reason'])
                if is_winner and 'final_winner' not in existing['role']:
                    existing['role'] = 'final_winner+open_innovation'
    claims = sorted(claims_by_key.values(), key=lambda c: (c['pair_name'], c['claim_id']))
    return sources, claims


def run_loop2_transplants(artifact: Path, work: Path, freeze: Json,
                          replicates: Sequence[str] = REPLICATES) -> Json:
    """Seal transplant-freeze.json, then run zero-API single-factor counterfactual transplants."""
    out_dir = artifact / 'transplants'
    summary_path = out_dir / 'transplant_summary.json'
    if summary_path.exists():
        return _read(summary_path)
    started = perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources, claims = _select_transplant_roster(artifact, freeze, replicates)
    by_pair: dict[str, list[str]] = {}
    for claim in claims:
        by_pair.setdefault(claim['pair_name'], [])
        if claim['source_sha256'] not in by_pair[claim['pair_name']]:
            by_pair[claim['pair_name']].append(claim['source_sha256'])
    for digests in by_pair.values():
        digests.sort()
    roster_doc = {
        'schema': 'loop2-single-factor-transplant-v1',
        'api_calls': 0,
        'reference_sha256': source_hash(SEED_SOURCE),
        'pairs': {pair: {'environments': list(SINGLE_FACTOR_PAIRS[pair]), 'candidate_digests': digests}
                  for pair, digests in sorted(by_pair.items())},
        'claims': claims,
    }
    _freeze(artifact / 'transplant-freeze.json', roster_doc)
    src_dir = out_dir / 'sources'
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / 'B0.py').write_text(SEED_SOURCE, encoding='utf-8')
    for digest, src in sorted(sources.items()):
        (src_dir / f'{digest[:12]}.py').write_text(src, encoding='utf-8')

    pair_results: dict[str, Json] = {}
    for pair_name, digests in sorted(by_pair.items()):
        pair_work = work / 'transplant' / pair_name
        pair_candidates = {digest: sources[digest] for digest in digests}
        envs = SINGLE_FACTOR_PAIRS[pair_name]
        print(f'TRANSPLANT {pair_name} ({envs[0]} <-> {envs[1]}): {len(digests)} frozen models', flush=True)
        res = transplant_contrast(pair_work, pair_candidates, SEED_SOURCE, environments=envs)
        shutil.copyfile(pair_work / 'transplant-freeze.json', out_dir / f'{pair_name}-freeze.json')
        atomic_write_json(out_dir / f'{pair_name}.json', res)
        pair_results[pair_name] = res

    evaluated_claims = []
    for claim in claims:
        pair_res = pair_results[claim['pair_name']]
        entry = pair_res['candidates'][claim['source_sha256']]
        boundary = evaluate_boundary_claim(
            entry, pair_res['world_z'], claim['weaker_when_z_metric'],
            claim['weaker_when_z_direction'], claim['target_metric'],
        )
        low, high = pair_res['environments']
        guardrails = {}
        if entry.get('status') == 'VALID':
            for m in (claim['target_metric'], 'val_nll_total', 'exhausted_replicate_rate', 'counterfactual_impact_mae'):
                guardrails[m] = {
                    'tau_low': entry['tau'][low][m],
                    'tau_high': entry['tau'][high][m],
                    'tau_high_minus_low': entry['tau_difference'][m],
                }
        evaluated_claims.append({**claim, 'candidate_status': entry.get('status'),
                                 'boundary_evaluation': boundary, 'metrics': guardrails})
    summary = {
        'schema': 'loop2-single-factor-transplant-summary-v1',
        'api_calls': 0,
        'transplant_freeze_sha256': _digest(artifact / 'transplant-freeze.json'),
        'unique_transplanted_models': len(sources),
        'total_claims': len(evaluated_claims),
        'pairs_evaluated': {k: {'environments': v['environments'], 'freeze_sha256': v['freeze_sha256'],
                                'world_z': v['world_z']} for k, v in pair_results.items()},
        'verdict_counts': {
            v: sum(c['boundary_evaluation']['verdict'] == v for c in evaluated_claims)
            for v in ('BOUNDARY_CONFIRMED', 'BOUNDARY_REFUTED', 'INCONCLUSIVE')
        },
        'claims': evaluated_claims,
        'wall_seconds': perf_counter() - started,
    }
    atomic_write_json(summary_path, summary)
    print(f'LOOP 2 TRANSPLANT COMPLETE ({len(evaluated_claims)} claims, 0 API calls)', flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--stochopia-root', type=Path, required=True)
    parser.add_argument('--num-replicates', type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    artifact, work = args.artifact_root.resolve(), args.work_root.resolve()
    historical_names = {'p2-m1-validation', 'p2-m2-synthetic', 'p2-m3-isomorphic-control', 'p2-m31-calibrated-control'}
    if (artifact.name != ARTIFACT_NAME or work.is_relative_to(artifact) or artifact.is_relative_to(work)
        or historical_names.intersection(artifact.parts) or historical_names.intersection(work.parts)):
        raise ValueError(f'the study requires an isolated {ARTIFACT_NAME} artifact root and private work root')
    artifact.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    replicates = tuple(f'rep_{i}' for i in range(args.num_replicates))
    reg = registration(args.stochopia_root.resolve(), replicates)
    _freeze(artifact / 'registration.json', reg)
    work_registration = work / 'study-registration.json'
    if not work_registration.exists() and any((work / env).exists() for env in ENVS):
        raise ValueError('private work root already contains an unrelated evaluation cache')
    _freeze(work_registration, {'registration_sha256': _digest(artifact / 'registration.json'),
                                'artifact_root': str(artifact)})
    evaluators = {env: SyntheticEvaluator(work, env) for env in ENVS}
    baseline = _baseline(artifact, evaluators)
    _audit_historical_baseline(artifact, baseline)
    guard = CallBudgetGuard(artifact / 'api_calls.jsonl', limit=_call_cap(replicates))
    if not (artifact / 'freeze.json').exists():
        if guard.count == 0:
            _audit_first_round(artifact, baseline, replicates)
        _handshake(artifact)
        _run_tracks(artifact, evaluators, baseline, guard, replicates)
        freeze = _make_freeze(artifact, guard, evaluators, replicates)
    else:
        freeze = _make_freeze(artifact, guard, evaluators, replicates)
    holdout = _verify(artifact, evaluators, freeze, replicates)
    if not (artifact / 'study_summary.json').exists():
        atomic_write_json(artifact / 'study_summary.json',
                          _summarize(artifact, freeze, holdout, guard, perf_counter() - started, replicates))
    run_loop2_transplants(artifact, work, freeze, replicates)
    print('Study complete; evidence ready for report and checksums', flush=True)


if __name__ == '__main__':
    main()
