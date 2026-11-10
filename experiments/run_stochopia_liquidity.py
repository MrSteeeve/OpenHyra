"""One-model paired diagnostic pilot; offline by default, eight paid calls explicitly."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent/'Stochopia'))

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import torch

from tasks.stochopia_structuring.liquidity_research import (
    pilot_config, LiquidityEvaluator, known_families, scaled_variant, diagnostic_rank, compare_catalog)
from tasks.stochopia_structuring.api_config import api_generator
from tasks.stochopia_structuring.research_provenance import capture_provenance


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def try_evaluation(evaluator, artifact, quote=None):
    try:
        return evaluator.evaluate(artifact, quote=quote, keep_behavior=True)
    except (ValueError, KeyError, TypeError, RuntimeError, IndexError) as exc:
        return {'status': 'invalid', 'error': f'{type(exc).__name__}: {exc}', 'feasible': False}, None


def optimize_proposal(evaluator, artifact):
    attempts = []
    for index in range(evaluator.config['evaluations_per_proposal']):
        try:
            candidate = scaled_variant(artifact, index)
            metrics, behavior = try_evaluation(evaluator, candidate)
        except (ValueError, KeyError, TypeError) as exc:
            candidate = deepcopy(artifact)
            metrics, behavior = {'status':'invalid','feasible':False,
                                 'error':f'{type(exc).__name__}: {exc}'}, None
        attempts.append({'variant': index, 'artifact': candidate, 'metrics': metrics, 'behavior': behavior})
    return max(attempts, key=lambda row: diagnostic_rank(row['metrics'])), attempts


def catalog(evaluator, track):
    steps = evaluator.panels['normal'].paths.n_steps
    entries = []
    for family in known_families(track, steps):
        for index in (1, 3):
            artifact = scaled_variant(family['artifact'], index)
            metrics, behavior = try_evaluation(evaluator, artifact)
            entries.append({'family': family['family'], 'variant': index, 'artifact': artifact,
                            'metrics': metrics, 'behavior': behavior})
    assert len(entries) == evaluator.config['evaluation_budget_per_arm']
    return entries


def portable(rows):
    return [{k:v for k,v in row.items() if k!='behavior'} for row in rows]


def run(output, config=None, generator=None):
    cfg = deepcopy(config or pilot_config())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    write_json(output/'protocol.json', cfg)
    provenance = capture_provenance(output/'provenance', {name: ROOT.parent/name
                         for name in ('OpenHyra', 'Stochopia', 'StructurerDSL')})
    write_json(output/'run_config.json', {'protocol': cfg, 'model_api': generator.metadata() if generator else None,
        'planned_api_calls': 8 if generator else 0, 'provenance': provenance['manifest_path'],
        'actual_system_prompt': 'Stored beside every API request; existing transport unchanged',
        'risk_decisions': cfg['decisions'], 'extra_calls_or_retries': 0})
    dev = LiquidityEvaluator(cfg, 'development')
    held = LiquidityEvaluator(cfg, 'holdout')
    write_json(output/'reference-development.json',dev.reference_report())
    write_json(output/'reference-holdout.json',held.reference_report())
    summary = {'schema': cfg['schema'], 'phase': cfg['phase'], 'model': cfg['model'],
        'api_calls': 0, 'completed_proposals': 0, 'tracks': {}, 'new_mechanisms_confirmed': 0,
        'sampling_limit': 'One pair per track; no significance or market-first conclusion',
        'theory_headroom': 'not certified; this run validates the evaluation/search flow'}
    for track in ('synthetic', 'contract'):
        refs = catalog(dev, track)
        write_json(output/track/'catalog-development.json', portable(refs))
        held_refs = []
        for row in refs:
            metrics, behavior = try_evaluation(held, row['artifact'], row['metrics'].get('quote'))
            held_refs.append({**row, 'metrics': metrics, 'behavior': behavior})
        write_json(output/track/'catalog-holdout.json', portable(held_refs))
        track_summary = {'catalog_evaluations': len(refs), 'catalog_holdout_evaluations': len(held_refs),
            'catalog_feasible_development': sum(r['metrics']['feasible'] for r in refs),
            'catalog_feasible_holdout': sum(r['metrics']['feasible'] for r in held_refs), 'arms': {}}
        for mode in ('feedback', 'independent'):
            history, reps = [], []
            for round_index in range(cfg['rounds']):
                destination = output/track/mode/f'round-{round_index+1}'
                request = dev.public_request(track, mode, round_index, history)
                write_json(destination/'generation-request.json', request)
                if generator:
                    write_json(destination/'api-request.json', generator.request_body(request))
                    print(json.dumps({'event': 'api-start', 'track': track, 'mode': mode,
                                      'round': round_index+1, 'call': generator.calls+1}), flush=True)
                    generation = generator.generate(request)
                    write_json(destination/'generation.json', asdict(generation))
                    summary['api_calls'] += generation.usage.get('api_calls', 0)
                    write_json(output/'progress.json', summary)
                    if generation.error:
                        history.append({'round': round_index+1, 'generation_error': generation.error})
                        print(json.dumps({'event': 'api-failure', 'track': track, 'mode': mode,
                                          'error': generation.error}), flush=True)
                        continue
                    artifact = generation.artifact
                    if artifact['track'] != track:
                        history.append({'round': round_index+1, 'generation_error': 'wrong track'})
                        continue
                else:
                    artifact = known_families(track, dev.panels['normal'].paths.n_steps)[round_index+1]['artifact']
                best, attempts = optimize_proposal(dev, artifact)
                write_json(destination/'development.json', portable(attempts))
                summary['completed_proposals'] += 1
                reps.append((round_index, best))
                history.append({'round': round_index+1, 'artifact': best['artifact'],
                    'metrics': {k:v for k,v in best['metrics'].items() if k not in ('quote',)},
                    'instruction': 'Use executed failure/response differences to propose another mechanism; magnitude alone is not novelty.'})
                print(json.dumps({'event': 'proposal-evaluated', 'track': track, 'mode': mode,
                    'round': round_index+1, 'coverage': best['metrics'].get('coverage'),
                    'feasible': best['metrics']['feasible']}), flush=True)
            # All valid proposal representatives are frozen before holdout; results
            # below never enter generation history or change the selected variant.
            reports = []
            for round_index, row in reps:
                metrics, behavior = try_evaluation(held, row['artifact'], row['metrics'].get('quote'))
                comparison = compare_catalog(metrics, behavior, held_refs) if behavior is not None else {
                    'novelty_status': 'invalid', 'new_mechanism_confirmed': False}
                sensitivities = {}
                if track == 'contract' and behavior is not None:
                    quote = row['metrics']['quote']
                    for coefficient in (0., .2):
                        # Reselect on the same independent quotation panel only;
                        # neither fit nor price uses holdout market paths.
                        alternative = held.quote(row['artifact'], risk_fee=coefficient)
                        sensitivity, _ = try_evaluation(held, row['artifact'], alternative)
                        sensitivities[str(coefficient)] = sensitivity
                reports.append({'round': round_index+1, 'artifact': row['artifact'], 'metrics': metrics,
                                'comparison': comparison, 'risk_fee_sensitivity': sensitivities})
            write_json(output/track/mode/'heldout.json', reports)
            write_json(output/track/mode/'history.json', history)
            track_summary['arms'][mode] = {'proposals': len(history), 'evaluated_proposals': len(reps),
                'development_evaluation_budget': cfg['evaluation_budget_per_arm'],
                'development_evaluations_used': len(reps)*cfg['evaluations_per_proposal'],
                'heldout_feasible': sum(r['metrics']['feasible'] for r in reports),
                'reports': str(output/track/mode/'heldout.json')}
        summary['tracks'][track] = track_summary
        write_json(output/'progress.json', summary)
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api', action='store_true')
    parser.add_argument('--direction', choices=('bearish','bullish'), default='bearish')
    parser.add_argument('--env-file', type=Path, default=Path('/Users/williamqian/.config/stochopia/model-api.env'))
    args = parser.parse_args()
    generator = None
    cfg = pilot_config(args.direction)
    if args.api:
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
        generator = api_generator(cfg['model'], max_calls=8, max_tokens=cfg['max_tokens'], timeout=120.)
    print(json.dumps(run(args.output, cfg, generator), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
