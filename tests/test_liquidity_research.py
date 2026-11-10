from copy import deepcopy
import json
import httpx
import numpy as np
import pytest

from tasks.stochopia_structuring.liquidity_research import (
    GOAL, pilot_config, expected_shortfall, LiquidityEvaluator, known_families,
    compare_catalog, scaled_variant)
from tasks.stochopia_structuring.liquidity_research import parse_liquidity_generation
from tasks.stochopia_structuring.liquidity_research import liquidity_prompt
from tasks.stochopia_structuring.api_config import api_generator
from experiments.run_stochopia_liquidity import run
from experiments.run_stochopia_liquidity import optimize_proposal


def small_config():
    cfg = pilot_config()
    cfg['market'].update(days=1, steps_per_day=8)
    cfg['paths'] = dict(development=16, quote=32, holdout=32)
    return cfg


def test_fractional_tail_and_cash_cannot_pass_as_liquidity_trade():
    assert expected_shortfall([0,10,20], .5)==pytest.approx(50/3)
    evaluator = LiquidityEvaluator(small_config())
    result = evaluator.evaluate(known_families('synthetic',8)[0]['artifact'])
    assert result['coverage']==0 and result['cost']==0
    assert not result['gates']['net_view_gain'] and not result['feasible']
    assert all(v==0 for v in result['scenario_net_means'].values())


def test_direct_liquidity_contract_is_known_and_quote_frozen_on_holdout():
    cfg = small_config()
    dev, hold = LiquidityEvaluator(cfg), LiquidityEvaluator(cfg,'holdout')
    artifact = known_families('contract',8)[1]['artifact']
    dm, db = dev.evaluate(artifact, keep_behavior=True)
    hm, hb = hold.evaluate(artifact, quote=dm['quote'], keep_behavior=True)
    assert dm['quote']['price']==hm['quote']['price']
    assert set(dm['quote']['label'] for _ in [0]) <= {'none','static-basket','static-basket-put'}
    assert len(hb)>len(db)
    report = compare_catalog(hm,hb,[dict(family='liquidity-call',metrics=hm,behavior=hb)])
    assert report['matched_parameterized_controls']==['liquidity-call']
    assert not report['new_mechanism_confirmed']


def test_independent_requests_never_carry_history_and_recovery_units_correct():
    evaluator = LiquidityEvaluator(small_config())
    history = [{'secret_parent': 'development-only'}]
    independent = evaluator.public_request('contract','independent',1,history)
    feedback = evaluator.public_request('contract','feedback',1,history)
    assert independent['feedback']==[] and feedback['feedback']==history
    assert independent['root_observables']['recovery'] == pytest.approx(.2)
    assert independent['steps']==8 and independent['risk']['buyer_capital']==1e6


def test_api_transport_uses_new_prompt_and_parse_without_changing_default_reasoning(monkeypatch):
    artifact = {'track':'synthetic','policy':{'targets':{},'rebalance_every':78}, 'params':{}}
    monkeypatch.setenv('DEEPSEEK_API_KEY','unit-test-placeholder')
    observed=[]
    def post(url, **kwargs):
        observed.append(kwargs['json'])
        return httpx.Response(200,json={'model':'deepseek-flash','choices':[{
            'message':{'content':json.dumps(artifact)},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':10,'completion_tokens':20}},request=httpx.Request('POST',url))
    monkeypatch.setattr(httpx,'post',post)
    generator = api_generator('deepseek-v4.1-flash',max_calls=1,max_tokens=393216)
    request = LiquidityEvaluator(small_config()).public_request('synthetic','independent',0,[])
    result = generator.generate(request)
    assert result.error is None and result.artifact==artifact
    assert observed[0]['max_tokens']==393216
    assert 'reasoning_effort' not in observed[0] and 'thinking' not in observed[0]
    assert 'COMMON equity liquidity' in observed[0]['messages'][0]['content']
    assert result.usage['api_calls']==1


def test_offline_full_budget_and_all_frozen_representatives_saved(tmp_path, monkeypatch):
    # Provenance is separately tested in existing research tests; this test
    # exercises budgets and independent/feedback artifacts, not git snapshots.
    monkeypatch.setattr('experiments.run_stochopia_liquidity.capture_provenance',
                        lambda output,roots: {'manifest_path':'unit-test-provenance'})
    summary = run(tmp_path/'pilot',small_config())
    assert summary['api_calls']==0 and summary['completed_proposals']==8
    for track in ('synthetic','contract'):
        assert summary['tracks'][track]['catalog_evaluations']==16
        for mode in ('feedback','independent'):
            arm = summary['tracks'][track]['arms'][mode]
            assert arm['development_evaluations_used']==16
            reports = json.loads((tmp_path/'pilot'/track/mode/'heldout.json').read_text())
            assert len(reports)==2
    assert summary['new_mechanisms_confirmed']==0


def test_extra_leverage_or_syntax_is_not_confirmation():
    artifact = known_families('contract',8)[0]['artifact']
    original = deepcopy(artifact)
    assert scaled_variant(artifact,7)['notional']==2*artifact['notional']
    assert artifact==original
    metrics=dict(status='ok',feasible=True,coverage=.9,nuisance=.1,cost=20.)
    behavior=np.arange(10.)
    refs=[dict(family='linear',metrics={**metrics,'coverage':.95},behavior=behavior*2+7)]
    report = compare_catalog(metrics,behavior,refs)
    assert report['catalog_weakly_dominates']
    assert report['matched_parameterized_controls']==['linear']
    assert not report['economic_advantage_diagnostic'] and not report['new_mechanism_confirmed']


def test_malformed_candidate_is_archived_and_consumes_evaluation_slots():
    evaluator = LiquidityEvaluator(small_config())
    best, attempts = optimize_proposal(evaluator, {'track':'synthetic','params':{}})
    assert len(attempts)==8
    assert all(a['metrics']['status']=='invalid' for a in attempts)
    for bad_params in ([],None,{'explanation':'not a scalar'}):
        malformed={'track':'synthetic','policy':{'targets':{}},'params':bad_params}
        with pytest.raises(ValueError,match='params'):
            parse_liquidity_generation(json.dumps(malformed))


def test_quote_rejects_interim_obligation_beyond_seller_capital():
    evaluator = LiquidityEvaluator(small_config())
    artifact = dict(track='contract',params={},notional=1_200_000.,contract_sexpr=
        '(contract (meta (underlyings basket) (maturity 8) (schedules (obs 2))) '
        '(premium (both (pay 1 (at obs 0)) (pay -1 maturity))))')
    quote = evaluator.quote(artifact)
    assert quote['fair_mean']==pytest.approx(0.)
    assert not quote['quote_feasible']
    assert all(not q['feasible'] for q in quote['all_hedge_trials'])


def test_seller_account_and_market_capacity_have_distinct_capital_bases():
    evaluator = LiquidityEvaluator(small_config())
    evaluator.config['risk']['seller_capital']=100_000.
    cfg = evaluator.hedge_config()
    assert cfg['buyer_capital']==100_000. and cfg['market_capital']==1_000_000.


def test_bullish_view_distribution_reverses_without_changing_reference_market():
    bear,bull=pilot_config(),pilot_config('bullish')
    assert bear['market']==bull['market'] and bear['risk']==bull['risk']
    assert bear['view_weights']['common_bad']==bull['view_weights']['common_good']


def test_prompt_reference_contract_is_executable_without_schedule_alias_or_new_grammar():
    from stochopia.research.liquidity_execution import contract_cashflows
    evaluator=LiquidityEvaluator(small_config())
    request=evaluator.public_request('contract','independent',0,[])
    prompt=liquidity_prompt(request)
    program='(contract (meta (underlyings basket) (maturity 8) (schedules)) (premium (pay 0.001 maturity)))'
    artifact={'track':'contract','contract_sexpr':program,'params':{},'notional':100_000.}
    table=contract_cashflows(artifact,evaluator.panels['normal'])
    assert np.all(table[:,-1]==100.) and np.all(table[:,:-1]==0.)
    assert '(schedules)) (premium body)' in prompt
    assert 'HIGHER = WORSE' in prompt and 'IDENTICAL within pairs' in prompt
