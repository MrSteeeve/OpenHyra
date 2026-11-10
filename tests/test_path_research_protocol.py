import copy
import pytest

from tasks.stochopia_structuring.path_research_protocol import (
    research_config, protocol_identity, public_generation_request, pareto_outward,
)
from tasks.stochopia_structuring.generator import system_prompt_for


def test_theoretical_solution_and_optimized_catalog_never_enter_model_feedback():
    config=research_config();private={'payment_table':[[99]],'frontier_cost':42,'pareto_front':['private']}
    request=public_generation_request(config,round_index=0,slot=0,feedback=[],parent=None)
    assert 'theory' not in str(request) and 'frontier_cost' not in str(request)
    assert private['payment_table'] not in request.values()
    assert request['demand']['withdrawal_ratios'][1]>0
    assert request['horizon_months']==12
    prompt=system_prompt_for(request)
    assert 'Fixed horizon is 3 months' not in prompt
    assert '12 months' in prompt and 'existing DSL' in prompt


def test_independent_arm_has_neither_parent_nor_previous_feedback():
    config=research_config()
    r=public_generation_request(config,round_index=1,slot=0,
      feedback=[{'private_feedback':42}],parent={'parent':42},search_mode='independent')
    assert r['feedback']==[] and 'parent' not in r


def test_full_protocol_identity_distinguishes_exit_permissions_and_p_equals_q():
    a=research_config();b=copy.deepcopy(a)
    b['demand']['partial_sales']=True
    assert protocol_identity(a)!=protocol_identity(b)
    b=copy.deepcopy(a);b['market']['sharpe']=0.
    assert protocol_identity(a)!=protocol_identity(b)


def test_cost_frontier_improvement_does_not_claim_dominating_other_axes():
    candidate={'cost':90,'satisfaction_probability':.95,'drawdown_loss':.14}
    catalog=[{'metrics':{'cost':100,'satisfaction_probability':1.,'drawdown_loss':.01}}]
    result=pareto_outward(candidate,catalog,cost_tolerance=5,success_tolerance=1e-8,drawdown_tolerance=1e-4)
    assert result['outward'] and not result['dominates_every_reference']
    assert result['outward_dimensions']==['cost']
    identical=pareto_outward(catalog[0]['metrics'],catalog,cost_tolerance=5)
    assert not identical['outward']


def test_paired_runner_blocks_before_generator_when_preflight_fails(tmp_path):
    import json
    from experiments.run_stochopia_path_research import run_study
    config=research_config();f=tmp_path/'preflight';f.mkdir()
    (f/'protocol.json').write_text(json.dumps(config))
    (f/'preflight.json').write_text(json.dumps({'economic_headroom_certified':False}))
    called=[]
    with pytest.raises(ValueError,match='preflight failed'):
        run_study(tmp_path/'study',f,config,lambda c:called.append(c))
    assert not called and not (tmp_path/'study').exists()


def test_offline_arm_freezes_before_reference_audit_and_never_feeds_theory(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tasks.stochopia_structuring.generator import Generation,parse_generation
    from experiments import run_stochopia_path_research as runner
    import json
    import numpy as np
    config=research_config({'market':{'months':2,'rate':.12,'sharpe':0.},
      'demand':{'budget':100.,'withdrawal_ratios':{1:.1},'terminal_total_ratio':1.001,
        'max_drawdown':1.,'min_success_probability':1.,'exit_scenarios':[{'month':None,'probability':1.}]},
      'rounds':2,'per_round':1,'evaluation_budget_per_arm':2})
    raw=json.dumps({'contract_sexpr':'(contract (meta (underlyings u) (maturity 2) (schedules (obs 1 2))) (premium (pay 0.051 maturity)))','notional':100.,'params':{},'rationale':'fixture, not a model discovery'})
    generator=SimpleNamespace(calls=0);requests=[];events=[]
    def generate(request):
        assert not events
        requests.append(request)
        return Generation(parse_generation(raw),usage={'api_calls':0})
    generator.generate=generate
    def audit(*args):
        events.append('post-generation-only')
        return {'fixed_target_cost_event':False,'private_frontier_cost':42.}
    monkeypatch.setattr(runner,'_audit_endpoint',audit)
    result=runner.run_arm(tmp_path/'arm',config,'feedback',generator,tmp_path/'references')
    assert result['attempts']==2 and result['feasible_candidates']==2
    assert len(events)==4
    assert len(requests[0]['feedback'])==0 and len(requests[1]['feedback'])==1
    assert all('private_frontier_cost' not in str(r) for r in requests)
    assert result['api_calls']==0


def test_json_roundtrip_preserves_month_key_protocol_identity():
    import json
    config=research_config({'demand':{'withdrawal_ratios':{1:.01,2:.01,12:.01}}})
    assert protocol_identity(config)==protocol_identity(json.loads(json.dumps(config)))


def _passing_references(tmp_path,config,runner,monkeypatch):
    import json
    folder=tmp_path/'preflight';folder.mkdir()
    (folder/'protocol.json').write_text(json.dumps(config))
    branches={}
    for branch in ('primary','p-equals-q'):
        branches[branch]={'protocol_identity':protocol_identity(runner.branch_config(config,branch))}
        b=folder/branch;b.mkdir()
        (b/'theory-evaluation-only.json').write_text('{}')
        (b/'catalog.json').write_text(json.dumps({'evaluations':config['evaluation_budget_per_arm']}))
    (folder/'preflight.json').write_text(json.dumps({'economic_headroom_certified':True,'branches':branches}))
    monkeypatch.setattr(runner,'preflight_gate',lambda *a,**k:{'allowed':True})
    monkeypatch.setattr(runner,'verify_provenance',lambda *a:None)
    return folder


def test_execution_rechecks_branch_certificate_before_any_api_factory(tmp_path,monkeypatch):
    from experiments import run_stochopia_path_research as runner
    config=research_config();folder=_passing_references(tmp_path,config,runner,monkeypatch)
    monkeypatch.setattr(runner,'preflight_gate',lambda *a,**k:{'allowed':False})
    called=[]
    with pytest.raises(ValueError,match='branch reference'):
        runner.run_study(tmp_path/'study',folder,config,lambda c:called.append(c))
    assert not called


def test_confirmatory_cannot_substitute_a_dummy_pilot_summary(tmp_path,monkeypatch):
    import json
    from experiments import run_stochopia_path_research as runner
    config=research_config({'phase':'confirmatory','formal_study':{
      'preregistered':True,'fixed_n':3,'power_plan':{'independent_pilot_report':str(tmp_path/'pilot.json')}}})
    (tmp_path/'pilot.json').write_text(json.dumps({'phase':'pilot','model_group':config['model_group']}))
    folder=_passing_references(tmp_path,config,runner,monkeypatch);called=[]
    with pytest.raises(ValueError,match='pilot must be separate'):
        runner.run_study(tmp_path/'study',folder,config,lambda c:called.append(c))
    assert not called


def test_registered_alpha_reaches_summary(tmp_path,monkeypatch):
    from experiments import run_stochopia_path_research as runner
    config=research_config({'pairs':2,'formal_study':{'alpha':.01}})
    folder=_passing_references(tmp_path,config,runner,monkeypatch)
    monkeypatch.setattr(runner,'capture_provenance',lambda *a:None)
    monkeypatch.setattr(runner,'run_arm',lambda *a:{'main_event':False,'api_calls':52})
    result=runner.run_study(tmp_path/'study',folder,config,lambda c:object())
    assert result['alpha']==.01 and result['n_complete']==2
    assert not result['significance_claim_allowed']


def test_invalid_study_phase_is_rejected_before_paid_generation(tmp_path):
    from experiments import run_stochopia_path_research as runner
    config=research_config({'phase':'confirmatroy'})
    called=[]
    with pytest.raises(ValueError,match='phase must be'):
        runner.run_study(tmp_path/'study',tmp_path/'absent',config,lambda c:called.append(c))
    assert not called


def test_complexity_strata_do_not_convert_failed_fits_into_novelty():
    from experiments.run_stochopia_path_research import complexity_match_summary
    rows=[{'program_complexity':{'node_count':n},'primary_audit':{'behavior_match':{'novelty_status':s}}}
      for n,s in ((10,'sampled_behavior_matched'),(120,'fit_budget_inconclusive'),
                  (120,'provisional_unfamiliarity_to_finite_catalog'))]
    result=complexity_match_summary(rows)
    assert result['strata']['large']['provisional_unmatched_rate']==.5
    assert result['strata']['large']['inconclusive']==1
    assert result['strata']['small']['matched']==1 and not result['originality_claim']
