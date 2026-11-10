"""Offline preflight first; paid paired pilot/confirmatory runs are explicit.

Theoretical solutions and catalog optima stay in evaluation artifacts, never
in generation requests. This entry is separate from historical v1/v2 runs.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'Stochopia'))

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import random
import time
import numpy as np
import torch

from stochopia.research.tree_market import TreeMarket
from stochopia.research.path_demand import PathDemand,evaluate_cashflows
from stochopia.research.dsl_cashflows import table_from_artifact
from tasks.stochopia_structuring.path_research_protocol import research_config,protocol_identity,public_generation_request,pareto_outward
from tasks.stochopia_structuring.known_path_families import optimize_known_families,FAMILY_NAMES
from tasks.stochopia_structuring.theoretical_frontier import theoretical_frontier,preflight_gate
from tasks.stochopia_structuring.controlled_statistics import frontier_event,power_grid,paired_summary
from tasks.stochopia_structuring.research_provenance import capture_provenance,export_blind_review,verify_provenance
from tasks.stochopia_structuring.economic_behavior import complexity_stratum,match_known_behaviors


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def compact(metrics):
    return {k:v for k,v in metrics.items() if k!='case_results'}


def branch_config(config,branch):
    value=deepcopy(config)
    if 'p-equals-q' in branch:value['market']['sharpe']=0.
    if 'partial-sale' in branch:
        value['demand'].update(partial_sales=True,partial_sale_discount=config['liquidity_control']['partial_sale_discount'])
    return value


def run_preflight(output,config):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    if config['rounds']*config['per_round']!=config['evaluation_budget_per_arm']:
        raise ValueError('model proposal and evaluator attempt budgets must match')
    if config['evaluation_budget_per_arm']<2*len(FAMILY_NAMES):
        raise ValueError('budget must optimize every family with at least two trials')
    config=deepcopy(config)
    config['assumption_status']='public synthetic parameters; not market calibrated; final preregistration pending pilot'
    write_json(output/'protocol.json',config)
    provenance=capture_provenance(output/'provenance',{
       name:ROOT.parent/name for name in ('OpenHyra','Stochopia','StructurerDSL')})
    run_config={'protocol':config,'repositories':{name:{'commit_sha':r['commit_sha'],'dirty':r['dirty'],
        'tracked_diff':r['tracked_diff'],'untracked_source_paths':r['untracked_source_paths']} for name,r in provenance['repositories'].items()},
        'source_snapshot':provenance['source_archive'],'dependency_versions':provenance['dependency_versions'],
        'new_api_calls':0,'purpose':'offline preflight, not an innovation experiment'}
    write_json(output/'run_config.json',run_config)
    results={}
    for branch in ('primary','p-equals-q','partial-sale','p-equals-q-partial-sale'):
        spec=branch_config(config,branch);tree=TreeMarket(spec['market']);demand=PathDemand.from_dict(spec['demand'])
        start=time.monotonic()
        catalog=optimize_known_families(tree,demand,lambda table:compact(evaluate_cashflows(tree,demand,table)),
           budget=config['evaluation_budget_per_arm'],seed=config['catalog_seed'])
        write_json(output/branch/'catalog.json',catalog)
        if demand.partial_sales:
            theory={'points':[],'evaluation_only':True,'status':'unsupported',
                    'reason':'partial-sale theory not implemented; primary no-sale bound cannot certify this control'}
            gate={'allowed':False,'api_allowed':False,'reasons':[theory['reason']],
                  'role':'exploratory control; excluded from primary theory certificate'}
        else:
            theory=theoretical_frontier(tree,demand,targets=(demand.min_success_probability,),
                 time_limit=config['solver_time_limit'],max_constraints=config['solver_max_constraints'])
            gate=preflight_gate(theory,catalog,cost_tolerance=config['tolerances']['cost_cny'])
        write_json(output/branch/'theory-evaluation-only.json',theory)
        write_json(output/branch/'gate.json',gate)
        best=catalog['best_cost_meeting_target']
        results[branch]={'market':tree.describe(),'protocol_identity':protocol_identity(spec),
          'catalog_evaluations':catalog['evaluations'],'catalog_families':len(FAMILY_NAMES),
          'valid_catalog_candidates':sum(r['metrics']['status']=='ok' for r in catalog['attempts']),
          'feasible_catalog_candidates':sum(r['eligible'] for r in catalog['attempts']),
          'best_cost_meeting_target':None if best is None else {'family':best['family'],'metrics':best['metrics']},
          'theory_points':[{k:v for k,v in point.items() if k not in ('payment_table','node_payments')} for point in theory['points']],
          'gate':gate,'elapsed_seconds':time.monotonic()-start}
        write_json(output/'preflight.json',{'branches':results,'api_calls':0})
        print(json.dumps({'branch':branch,'gate':gate,'elapsed_seconds':results[branch]['elapsed_seconds']}),flush=True)
    # Marginal event rates do not define paired power. Sensitivity assumptions
    # remain explicit until an independent pilot estimates the full joint table.
    power=[]
    for label,joint in [('positively-paired',{'p00':.7,'p01':0.,'p10':.25,'p11':.05}),
                        ('independent-arms',{'p00':.665,'p01':.035,'p10':.285,'p11':.015})]:
        power.append({'label':label,'hypothetical_not_pilot_estimate':True,**power_grid(joint)})
    write_json(output/'power-sensitivity.json',power)
    headroom=all(results[branch]['gate']['allowed'] for branch in ('primary','p-equals-q'))
    report={'schema':'stochopia-path-preflight.v1','branches':results,'api_calls':0,
      'economic_headroom_certified':headroom,'paid_execution_allowed':False,
      'paid_execution_requirements':['Validated headroom in primary and P=Q','Explicit pilot budget, one model, one demand',
          'For confirmatory: separate pilot, full paired joint power, fixed N and preregistration'],
      'primary_endpoint':config['endpoint'],'expert_review_status':'not performed'}
    write_json(output/'preflight.json',report);return report


def _audit_endpoint(tree,demand,table,config,folder,metrics,program_complexity=None):
    catalog=json.loads((folder/'catalog.json').read_text());theory=json.loads((folder/'theory-evaluation-only.json').read_text())
    point=next(p for p in theory['points'] if p['target_success']==demand.min_success_probability)
    best=catalog['best_cost_meeting_target'];identity=protocol_identity(config)
    event=frontier_event(candidate_cost=metrics['cost'],candidate_fulfillment=metrics['satisfaction_probability'],
        catalog_cost=best['metrics']['cost'],catalog_fulfillment=best['metrics']['satisfaction_probability'],
        frontier_cost=point['lower_cost_bound'],fulfillment_target=demand.min_success_probability,
        tolerance=config['tolerances']['cost_cny'],candidate_market_id=identity,catalog_market_id=identity,
        frontier_market_id=identity,frontier_status=point['status'],
        numerical_tolerance=max(1e-8,demand.budget*1e-9))
    return {'fixed_target_cost_event':event,
       'cost_gap_to_finite_reference':metrics['cost']-point['primal_cost'],
       'pareto_outward':pareto_outward(metrics,catalog['pareto_front'],
            cost_tolerance=config['tolerances']['cost_cny'],success_tolerance=config['tolerances']['success_probability'],
            drawdown_tolerance=config['tolerances']['drawdown']),
       'behavior_match':match_known_behaviors(tree,table,catalog['attempts'],
            max_legs=config['matching']['max_legs'],max_fit_attempts=config['matching']['max_fit_attempts'],
            abs_tol=config['tolerances']['matching_abs_cny'],rel_tol=config['tolerances']['matching_relative'],
            program_complexity=program_complexity)}


def complexity_match_summary(candidates):
    strata={}
    for candidate in candidates:
        key=complexity_stratum(candidate.get('program_complexity'))['stratum']
        row=strata.setdefault(key,{'candidates':0,'matched':0,'provisional_unmatched':0,'inconclusive':0})
        row['candidates']+=1
        status=candidate['primary_audit'].get('behavior_match',{}).get('novelty_status')
        field='matched' if status=='sampled_behavior_matched' else 'provisional_unmatched' if status=='provisional_unfamiliarity_to_finite_catalog' else 'inconclusive'
        row[field]+=1
    for row in strata.values():
        row['provisional_unmatched_rate']=row['provisional_unmatched']/row['candidates']
    return {'strata':strata,'population':'frozen feasible candidates, auxiliary descriptive audit',
        'inconclusive_is_not_unmatched':True,'originality_claim':False}


def run_arm(output,config,mode,generator,preflight_root):
    """Fixed proposal attempts; freeze development candidates before audit."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    tree=TreeMarket(config['market']);demand=PathDemand.from_dict(config['demand'])
    records=[];tables={};groups=[];uses={};feedback=[]
    for round_index in range(config['rounds']):
        frozen_feedback=deepcopy(feedback);cohort=[]
        for slot in range(config['per_round']):
            parent=None
            if mode=='feedback' and groups:
                selected=min(range(len(groups)),key=lambda g:uses.get(g,0));uses[selected]=uses.get(selected,0)+1
                parent=groups[selected]['artifact']
            request=public_generation_request(config,round_index=round_index,slot=slot,feedback=frozen_feedback,
                parent=parent,search_mode=mode)
            generation=generator.generate(request);cid=f'r{round_index}-s{slot}'
            record={'candidate_id':cid,'request':request,'generation':asdict(generation)}
            records.append(record);cohort.append(record)
            write_json(output/'generations.json',records)
        for record in cohort:
            cid=record['candidate_id'];gen=record['generation']
            if gen['error']:
                metrics={'status':'rejected','feasible':False,'error':gen['error']}
            else:
                try:
                    table,complexity=table_from_artifact(tree,gen['artifact']);metrics=compact(evaluate_cashflows(tree,demand,table))
                    record['program_complexity']=complexity
                    if metrics['status']=='ok':
                        tables[cid]=table
                        # Archive grouping is numeric payment behavior only; not
                        # an originality score or part of the primary event.
                        centered=table-table.mean(axis=0)
                        size=np.max(np.abs(centered));shape=centered/max(size,1e-12)
                        months=tuple(np.flatnonzero(np.any(table!=0,axis=0)))
                        group=next((g for g in groups if g['months']==months and np.allclose(g['shape'],shape,atol=1e-8)),None)
                        if metrics['feasible'] and group is None:
                            groups.append({'months':months,'shape':shape,'artifact':gen['artifact'],'cost':metrics['cost']})
                        elif metrics['feasible'] and metrics['cost']<group['cost']:
                            group.update(artifact=gen['artifact'],cost=metrics['cost'])
                except (ValueError,KeyError,TypeError) as exc:
                    metrics={'status':'rejected','feasible':False,'error':str(exc)}
            record['development']=metrics
            feedback.append({'candidate_id':cid,'artifact':gen['artifact'],'metrics':metrics})
        write_json(output/'experience.json',records)
    # No later generation sees audit/reference/P=Q values.
    candidates=[]
    for record in records:
        cid=record['candidate_id'];metrics=record['development']
        if metrics.get('feasible'):
            table=tables[cid]
            audit=_audit_endpoint(tree,demand,table,config,Path(preflight_root)/'primary',metrics,record.get('program_complexity'))
            qconfig=branch_config(config,'p-equals-q');qtree=TreeMarket(qconfig['market']);qd=PathDemand.from_dict(qconfig['demand'])
            qm=compact(evaluate_cashflows(qtree,qd,table))
            qaudit=_audit_endpoint(qtree,qd,table,qconfig,Path(preflight_root)/'p-equals-q',qm,record.get('program_complexity')) if qm['status']=='ok' else {'fixed_target_cost_event':False}
            candidates.append({'candidate_id':cid,'development':metrics,'primary_audit':audit,
                'p_equals_q_evaluation':qm,'p_equals_q_audit':qaudit,
                'main_event':audit['fixed_target_cost_event'] and qaudit['fixed_target_cost_event'],
                'program_complexity':record.get('program_complexity'),
                'demand':config['demand'],'market':config['market'],'cashflows':table.tolist()})
    write_json(output/'frozen-candidate-audit.json',candidates)
    write_json(output/'complexity-match-summary.json',complexity_match_summary(candidates))
    usages=[r['generation']['usage'] for r in records]
    result={'model_group':config['model_group'],'mode':mode,'api_calls':getattr(generator,'calls',0),
       'attempts':len(records),'failed_generations':sum(bool(r['generation']['error']) for r in records),
       'feasible_candidates':len(candidates),'main_event':any(c['main_event'] for c in candidates),
       'reported_tokens':{key:(sum(u.get(key) or 0 for u in usages) if any(u.get(key) is not None for u in usages) else None) for key in ('prompt_tokens','completion_tokens','reasoning_tokens')},
       'token_reporting_calls':{key:sum(u.get(key) is not None for u in usages) for key in ('prompt_tokens','completion_tokens','reasoning_tokens')},
       'theory_and_catalog_feedback':False,'economic_identity':protocol_identity(config)}
    write_json(output/'summary.json',result)
    export_blind_review(candidates,output/'human-review');return result


def run_study(output,preflight_root,config,generator_factory):
    if config['phase'] not in ('pilot','confirmatory'):
        raise ValueError('phase must be pilot or confirmatory before model calls')
    if any(type(config[key]) is not int or config[key]<1 for key in ('pairs','rounds','per_round')):
        raise ValueError('pairs, rounds and per_round must be positive integer budgets')
    saved=json.loads((Path(preflight_root)/'protocol.json').read_text())
    if protocol_identity(saved)!=protocol_identity(config):
        raise ValueError('demand, market, fees or tolerance changed since preflight')
    if saved['evaluation_budget_per_arm']!=config['evaluation_budget_per_arm'] or saved['catalog_seed']!=config['catalog_seed']:
        raise ValueError('catalog and model budgets/settings differ from preflight')
    preflight=json.loads((Path(preflight_root)/'preflight.json').read_text())
    if not preflight['economic_headroom_certified']:
        raise ValueError('offline economic preflight failed; do not spend API budget')
    if config['rounds']*config['per_round']!=config['evaluation_budget_per_arm']:
        raise ValueError('model proposal and evaluator attempt budgets must match')
    for branch in ('primary','p-equals-q'):
        spec=branch_config(config,branch);folder=Path(preflight_root)/branch
        if preflight['branches'][branch]['protocol_identity']!=protocol_identity(spec):
            raise ValueError('branch market/demand identity does not match the preflight')
        theory=json.loads((folder/'theory-evaluation-only.json').read_text())
        catalog=json.loads((folder/'catalog.json').read_text())
        if catalog['evaluations']!=config['evaluation_budget_per_arm']:
            raise ValueError('catalog evaluation budget differs from model arm')
        if not preflight_gate(theory,catalog,cost_tolerance=config['tolerances']['cost_cny'])['allowed']:
            raise ValueError('branch reference/frontier no longer passes economic preflight')
    verify_provenance(Path(preflight_root)/'provenance'/'manifest.json')
    if config['phase']=='confirmatory' and (not config['formal_study']['preregistered'] or
       config['formal_study']['fixed_n']!=config['pairs']):
        raise ValueError('confirmatory runs require separate pilot, registered fixed N and power plan')
    if config['phase']=='confirmatory':
        power_plan=config['formal_study'].get('power_plan')
        if not power_plan or not power_plan.get('independent_pilot_report'):
            raise ValueError('confirmatory study needs a separate pilot joint-event power plan')
        pilot_path=Path(power_plan['independent_pilot_report']);pilot_root=pilot_path.parent
        pilot=json.loads(pilot_path.read_text())
        if pilot.get('phase')!='pilot' or pilot.get('model_group')!=config['model_group'] or pilot.get('n_complete',0)<2:
            raise ValueError('pilot must be separate and use the chosen primary model')
        pilot_plan=json.loads((pilot_root/'study-plan.json').read_text())
        if protocol_identity(pilot_plan)!=protocol_identity(config) or pilot_plan['evaluation_budget_per_arm']!=config['evaluation_budget_per_arm']:
            raise ValueError('pilot must use the same economic protocol and proposal budget')
        if pilot_root.resolve()==Path(output).resolve():
            raise ValueError('pilot cannot be the confirmatory data directory')
        pilot_rows=json.loads((pilot_root/'paired-events.json').read_text())
        recomputed=paired_summary(pilot_rows,alpha=config['formal_study']['alpha'])
        if recomputed['phase']!='pilot' or recomputed['model_group']!=config['model_group'] or recomputed['joint_counts']!=pilot['joint_counts'] or recomputed['n_complete']!=pilot['n_complete']:
            raise ValueError('pilot paired evidence and reported joint table disagree')
        for row in pilot_rows:
            if any(row.get('api_attempts',{}).get(mode)!=config['evaluation_budget_per_arm'] for mode in ('feedback','independent')):
                raise ValueError('pilot needs real equal-budget model run pairs, not offline fixtures')
        counts=recomputed['joint_counts'];n=recomputed['n_complete']
        estimated={f'p{cell[1:]}':count/n for cell,count in counts.items()}
        if power_plan.get('assumption_source')=='pilot-frequency':
            if any(abs(power_plan['joint_probs'][key]-estimated[key])>1e-9 for key in estimated):
                raise ValueError('registered pilot-frequency assumptions differ from pilot evidence')
        elif power_plan.get('assumption_source')!='registered-conservative-sensitivity' or not power_plan.get('assumption_rationale'):
            raise ValueError('power assumptions need pilot frequency or explicitly registered sensitivity justification')
        power=power_grid(power_plan['joint_probs'],sample_sizes=(config['pairs'],),alpha=config['formal_study']['alpha'])
        if power['grid'][0]['power']<config['formal_study']['target_power']:
            raise ValueError('registered N does not attain the declared power assumption')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json(output/'study-plan.json',config)
    capture_provenance(output/'provenance',{name:ROOT.parent/name for name in ('OpenHyra','Stochopia','StructurerDSL')})
    pairs=[]
    for pair in range(config['pairs']):
        modes=['feedback','independent'];random.Random(config['catalog_seed']+pair).shuffle(modes)
        result={}
        for mode in modes:
            result[mode]=run_arm(output/f'pair-{pair:03d}'/mode,config,mode,generator_factory(config),preflight_root)
        pairs.append({'pair_id':pair,'model_group':config['model_group'],'phase':config['phase'],
            'feedback_event':result['feedback']['main_event'],'independent_event':result['independent']['main_event'],
            'replicate_unit':'research_run_pair','arm_order':modes,
            'api_attempts':{mode:result[mode]['api_calls'] for mode in modes}})
        write_json(output/'paired-events.json',pairs)
    summary=paired_summary(pairs,alpha=config['formal_study']['alpha']);summary['interpretation']='optimization discovery at a finite-model bound, not proof of financial originality'
    summary['economic_identity']=protocol_identity(config)
    summary['evaluation_budget_per_arm']=config['evaluation_budget_per_arm']
    write_json(output/'statistical-summary.json',summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--preflight',type=Path)
    parser.add_argument('--env-file',type=Path,default=Path('/Users/williamqian/.config/stochopia/model-api.env'))
    args=parser.parse_args();config=research_config(json.loads(args.config.read_text()) if args.config else None)
    if args.execute:
        if args.preflight is None:parser.error('--execute requires frozen --preflight')
        from dotenv import load_dotenv
        from tasks.stochopia_structuring.api_config import api_generator
        load_dotenv(args.env_file)
        def factory(cfg):
            tokens=393216 if cfg['model_group']=='deepseek-v4.1-flash' else None
            return api_generator(cfg['model_group'],max_calls=cfg['evaluation_budget_per_arm'],max_tokens=tokens)
        result=run_study(args.output,args.preflight,config,factory)
    else:
        result=run_preflight(args.output,config)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
