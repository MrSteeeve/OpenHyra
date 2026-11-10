"""Frozen research assumptions; generation sees demand and execution only."""
from copy import deepcopy
import json


GOAL='path-demand-cost-discovery.v2'


def research_config(overrides=None):
    from stochopia.research.path_demand import PathDemand
    config={'schema':'stochopia-controlled-path-study.v2',
       'market':{'months':12,'sigma':.2,'rate':.02,'dividend':0.,'sharpe':.3},
       'demand':PathDemand(withdrawal_ratios={1:.01,2:.01,4:.02,6:.03,9:.02}).to_dict(),
       'research_goal':GOAL,'model_group':'deepseek-v4.1-flash',
       'phase':'pilot','pairs':3,'rounds':4,'per_round':13,
       'evaluation_budget_per_arm':52,'catalog_seed':20261012,
       'solver_time_limit':20.,'solver_max_constraints':400000,
       'tolerances':{'cost_cny':500.,'success_probability':1e-8,'drawdown':1e-4,
                     'matching_abs_cny':.01,'matching_relative':1e-6},
       'matching':{'max_legs':2,'max_fit_attempts':256},
       'formal_study':{'primary_models':1,'primary_demands':1,'alpha':.05,
          'target_power':.8,'fixed_n':None,'preregistered':False,
          'pilot_in_confirmatory_analysis':False},
       'endpoint':'Fixed-target funded service capital closer to the finite-tree reference than optimized catalog, also valid at P=Q',
       'cost_scope':'Q contract purchase plus minimum required operational cash reserve; unused budget is outside the service account',
       'measurement_note':'Absolute delivery obligations are fixed; no discretionary optimization of wealth outside this mandate',
       'liquidity_control':{'partial_sales':True,'partial_sale_discount':0.,'role':'exploratory preregistered control'},
       'sampling':'exact finite tree; monthly observation, not continuous maximum drawdown'}
    if overrides:
        # Nested market/demand knobs are intentionally explicit, not silently reset.
        for key,value in deepcopy(overrides).items():
            config[key]={**config[key],**value} if isinstance(value,dict) and isinstance(config.get(key),dict) else value
    return config


def protocol_identity(config):
    """Full literal economic specification; no new hash/security mechanism."""
    value={'schema':config['schema'],'market':config['market'],
                       'demand':config['demand'],'tolerances':config['tolerances'],
                       'cost_scope':config['cost_scope']}
    # JSON persists integer month keys as strings; normalize before sorting.
    return json.dumps(json.loads(json.dumps(value)),sort_keys=True,separators=(',',':'))


def public_generation_request(config,*,round_index,slot,feedback,parent,search_mode='feedback'):
    request={'schema':'stochopia-contract-generation-path.v1','research_goal':GOAL,
        'horizon_months':config['market']['months'],'market':deepcopy(config['market']),
        'demand':deepcopy(config['demand']),'round_index':round_index,'slot':slot,
        'search_mode':search_mode,'feedback':deepcopy(feedback) if search_mode=='feedback' else [],
        'objective':'Design useful feasible cashflow mechanisms; reduce total funded service capital (Q purchase plus required reserve) without degrading path demand.',
        'design_brief':{'track':('mechanism-explore','mechanism-refine','mechanism-recombine')[(round_index*config['per_round']+slot)%3]},
        'protocol_rules':{'leftover_cash':'outside this mandate; required reserve has zero interest and is included in cost','contract_observables':'stock history only',
             'theoretical_solutions_provided':False,'optimized_catalog_results_provided':False}}
    if parent is not None and search_mode=='feedback':
        request['parent']=deepcopy(parent)
    return request


def pareto_outward(metrics,catalog,*,cost_tolerance,success_tolerance=1e-8,drawdown_tolerance=1e-4):
    """Three-axis observed envelope, distinct from fixed-target cost endpoint.

    Not being dominated is insufficient: at least one coordinate must exceed
    the observed reference envelope by the registered tolerance. No convex
    combination of probability/CVaR is called an attainable portfolio.
    """
    rows=[r.get('metrics',r) for r in catalog]
    rows=[r for r in rows if all(r.get(k) is not None for k in ('cost','satisfaction_probability','drawdown_loss'))]
    if not rows or any(metrics.get(k) is None for k in ('cost','satisfaction_probability','drawdown_loss')):
        return {'outward':False,'reason':'valid reference/candidate metrics missing','dominates_every_reference':False,'outward_dimensions':[]}
    def weakly_dominates(a,b):
        return (a['cost']<=b['cost']+cost_tolerance and
                a['satisfaction_probability']>=b['satisfaction_probability']-success_tolerance and
                a['drawdown_loss']<=b['drawdown_loss']+drawdown_tolerance)
    dominated=any(weakly_dominates(row,metrics) for row in rows)
    dimensions=[]
    if metrics['cost']<min(r['cost'] for r in rows)-cost_tolerance:dimensions.append('cost')
    if metrics['satisfaction_probability']>max(r['satisfaction_probability'] for r in rows)+success_tolerance:dimensions.append('satisfaction_probability')
    if metrics['drawdown_loss']<min(r['drawdown_loss'] for r in rows)-drawdown_tolerance:dimensions.append('drawdown_loss')
    return {'outward':not dominated and bool(dimensions),'weakly_dominated':dominated,
            'outward_dimensions':dimensions,'dominates_every_reference':all(weakly_dominates(metrics,row) for row in rows),
            'scope':'observed three-axis envelope, conservative sufficient outward test; no attainable convex-hull claim'}
