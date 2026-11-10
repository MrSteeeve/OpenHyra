"""Exact finite-tree reference checks, never continuous-market optimum claims."""
import math

import numpy as np
import pytest

from stochopia.research.tree_market import TreeMarket
from stochopia.research.path_demand import PathDemand, evaluate_cashflows
from tasks.stochopia_structuring.theoretical_frontier import (
    preflight_gate, solve_frontier_point, theoretical_frontier,
)


def demand(**kwargs):
    fields = dict(budget=100., withdrawal_ratios={}, terminal_total_ratio=1.01,
                  max_drawdown=1., min_success_probability=1.,
                  exit_scenarios=({'month': None, 'probability': 1.},),
                  exit_discount=0., partial_sales=False,
                  funding_mode='fixed_budget_leg_control')
    fields.update(kwargs)
    return PathDemand(**fields)


@pytest.mark.parametrize('budget', [100., 1_000_000.])
def test_hard_lp_attains_analytic_cost_floor_with_fixed_reserve(budget):
    tree = TreeMarket({'months': 2, 'sigma': .2, 'rate': .12, 'sharpe': 0.})
    point = solve_frontier_point(tree, demand(budget=budget))
    expected = .01 * budget / math.expm1(.12 * 2 / 12)
    assert point['status'] == 'optimal'
    assert point['proven_optimal']
    assert point['primal_cost'] == pytest.approx(expected, abs=1e-5)
    assert point['lower_cost_bound'] == pytest.approx(expected, abs=1e-5)
    assert point['attained_success'] == pytest.approx(1.)
    table = np.asarray(point['payment_table'])
    tree.node_cashflows(table)  # Rejects any future-dependent payment.
    assert tree.price(table) == pytest.approx(expected, abs=1e-5)
    assert point['independent_evaluation']['joint_success_probability'] == pytest.approx(1.)


@pytest.mark.parametrize('budget', [100., 1_000_000.])
def test_primary_service_capital_matches_discounted_delivery_obligation(budget):
    tree = TreeMarket({'months': 2, 'rate': .12, 'sharpe': 0.})
    spec = demand(budget=budget, funding_mode='minimum_service_capital')
    point = solve_frontier_point(tree, spec)
    expected = 1.01 * budget * math.exp(-.12 * 2 / 12)
    assert point['status'] == 'optimal'
    assert point['primal_cost'] == pytest.approx(expected, abs=1e-5)
    assert point['lower_cost_bound'] == pytest.approx(expected, abs=1e-5)
    assert point['contract_purchase_cost'] + point['cash_reserve'] == pytest.approx(expected, abs=1e-5)
    assert point['initial_funded_wealth'] == pytest.approx(expected, abs=1e-5)
    assert point['cash_reserve'] == pytest.approx(0., abs=1e-5)
    assert point['funding_mode'] == 'minimum_service_capital'
    assert 'minimum initial service capital' in point['model_scope']


def test_primary_zero_cashflow_cost_is_the_required_service_reserve():
    tree = TreeMarket({'months': 2, 'rate': 0.})
    spec = demand(terminal_total_ratio=.8, funding_mode='minimum_service_capital')
    evaluation = evaluate_cashflows(tree, spec, np.zeros_like(tree.paths))
    assert evaluation['joint_success_probability'] == pytest.approx(1.)
    assert evaluation['cost'] == pytest.approx(80.)
    assert evaluation['contract_purchase_cost'] == pytest.approx(0.)
    assert evaluation['cash_reserve'] == pytest.approx(80.)
    assert evaluation['initial_funded_wealth'] == pytest.approx(80.)


def test_primary_cash_obligation_has_its_own_discounted_funding_cost():
    tree = TreeMarket({'months': 2, 'rate': .12, 'sharpe': 0.})
    spec = demand(funding_mode='minimum_service_capital', withdrawal_ratios={1: .3})
    point = solve_frontier_point(tree, spec)
    expected = 30. * math.exp(-.01) + 71. * math.exp(-.02)
    assert point['status'] == 'optimal'
    assert point['primal_cost'] == pytest.approx(expected, abs=1e-6)
    assert point['initial_funded_wealth'] < spec.budget
    assert point['attained_success'] == pytest.approx(1.)


def test_primary_soft_one_step_p_equals_q_cost_is_discounted_claim_cost():
    tree = TreeMarket({'months': 1, 'rate': 0., 'sharpe': 0.})
    probability = float(min(tree.q_leaf))
    point = solve_frontier_point(tree, demand(funding_mode='minimum_service_capital'),
                                 target_success=probability - 1e-7)
    assert point['status'] == 'optimal'
    assert point['primal_cost'] == pytest.approx(101. * probability, abs=1e-6)
    assert point['cash_reserve'] == pytest.approx(0., abs=1e-6)


@pytest.mark.parametrize('budget', [100., 1_000_000.])
def test_primary_soft_joint_cash_exit_frontier_uses_target_specific_reserve(budget):
    from dataclasses import replace
    tree = TreeMarket({'months': 3, 'rate': .12})
    spec = demand(budget=budget, funding_mode='minimum_service_capital', withdrawal_ratios={1: .3},
                  terminal_total_ratio=1., max_drawdown=.15,
                  min_success_probability=1.,
                  exit_scenarios=({'month': 1, 'probability': .3},
                                  {'month': None, 'probability': .7}), exit_discount=.01)
    point = solve_frontier_point(tree, spec, target_success=.5)
    assert point['status'] == 'optimal'
    assert point['attained_success'] >= .5
    assert point['variable_count'] == 2 * len(tree.node_times) + 2 + point['compressed_case_count']
    evaluation = evaluate_cashflows(tree, replace(spec, min_success_probability=.5), np.asarray(point['payment_table']))
    assert point['primal_cost'] == pytest.approx(evaluation['cost'], abs=1e-6)
    assert point['cash_reserve'] == pytest.approx(evaluation['cash_reserve'], abs=1e-6)
    assert point['primal_cost'] == pytest.approx(point['contract_purchase_cost'] + point['cash_reserve'], abs=1e-6)


def test_soft_frontier_is_scale_invariant_with_binding_cash_and_exit_requirements():
    tree = TreeMarket({'months': 2, 'rate': .12})
    outputs = []
    for budget in (100., 1_000_000.):
        spec = demand(budget=budget, withdrawal_ratios={1: .3},
                      terminal_total_ratio=1.01, max_drawdown=.05,
                      exit_scenarios=({'month': 1, 'probability': .3},
                                      {'month': None, 'probability': .7}),
                      exit_discount=.01)
        point = solve_frontier_point(tree, spec, target_success=.5)
        assert point['status'] == 'optimal'
        assert point['proven_optimal']
        evaluation = evaluate_cashflows(tree, spec, np.asarray(point['payment_table']))
        assert evaluation['joint_success_probability'] >= .5
        assert point['attained_success'] == pytest.approx(evaluation['joint_success_probability'])
        assert point['normalization']['optimization_budget'] == 1.
        assert point['normalization']['reported_budget'] == budget
        assert point['validation_tolerances']['monetary_original_currency'] == pytest.approx(budget * 1e-9)
        outputs.append(point)
    assert outputs[0]['primal_cost'] / 100. == pytest.approx(outputs[1]['primal_cost'] / 1_000_000., abs=1e-8)
    assert outputs[0]['attained_success'] == pytest.approx(outputs[1]['attained_success'])


@pytest.mark.parametrize('funding_mode', ['fixed_budget_leg_control', 'minimum_service_capital'])
def test_early_exit_case_compression_preserves_exact_frontier_and_original_evaluation(funding_mode):
    from stochopia.research.path_demand import demand_cases
    from tasks.stochopia_structuring.theoretical_frontier import _coalesce_cases
    tree = TreeMarket({'months': 3, 'rate': .12})
    spec = demand(funding_mode=funding_mode, withdrawal_ratios={1: .1, 3: .2}, terminal_total_ratio=1.005,
                  max_drawdown=.1,
                  exit_scenarios=({'month': 1, 'probability': .3},
                                  {'month': 2, 'probability': .2},
                                  {'month': None, 'probability': .5}))
    compressed = solve_frontier_point(tree, spec, target_success=.5)
    original = solve_frontier_point(tree, spec, target_success=.5, compress_cases=False)
    assert compressed['status'] == original['status'] == 'optimal'
    assert compressed['original_case_count'] == original['original_case_count'] == 24
    assert compressed['compressed_case_count'] == 14  # 2 + 4 + 8 histories.
    assert original['compressed_case_count'] == 24
    assert compressed['variable_count'] == 2 * len(tree.node_times) + 1 + 14 + (funding_mode == 'minimum_service_capital')
    assert compressed['primal_cost'] == pytest.approx(original['primal_cost'], abs=1e-6)
    for point in (compressed, original):
        from dataclasses import replace
        evaluated = evaluate_cashflows(tree, replace(spec, min_success_probability=.5), np.asarray(point['payment_table']))
        assert evaluated['case_count'] == 24
        assert evaluated['joint_success_probability'] == pytest.approx(point['attained_success'])
        assert evaluated['joint_success_probability'] >= .5
        outcomes = {(row['leaf_index'], row['event_index']): row['success']
                    for row in evaluated['case_results']}
        grouped = _coalesce_cases(tree, demand_cases(tree, spec))
        for measure, output_key in [('p', 'joint_success_probability'), ('q', 'q_satisfaction_probability')]:
            grouped_success = sum(case[f'weight_{measure}'] * outcomes[(case['leaf_index'], case['event_index'])]
                                  for case in grouped)
            assert grouped_success == pytest.approx(evaluated[output_key])


def test_case_compression_keeps_different_future_withdrawal_requirements():
    from stochopia.research.path_demand import demand_cases
    from tasks.stochopia_structuring.theoretical_frontier import _coalesce_cases
    tree = TreeMarket({'months': 3})
    spec = demand(exit_scenarios=({'month': 1, 'probability': .5, 'withdrawal_ratios': {3: .2}},
                                  {'month': 1, 'probability': .5, 'withdrawal_ratios': {3: .8}}))
    original = demand_cases(tree, spec)
    compressed = _coalesce_cases(tree, original)
    assert len(original) == 16
    assert len(compressed) == 4  # Two exit histories times two distinct needs.
    assert sum(case['weight_p'] for case in compressed) == pytest.approx(1.)
    assert sum(case['weight_q'] for case in compressed) == pytest.approx(1.)
    assert {tuple(case['withdrawals']) for case in compressed} == {(0., 0., 0., 20.), (0., 0., 0., 80.)}


def test_zero_rate_hard_growth_is_infeasible_and_gate_blocks():
    tree = TreeMarket({'months': 2, 'rate': 0., 'sharpe': 0.})
    point = solve_frontier_point(tree, demand())
    assert point['status'] == 'infeasible'
    assert point['primal_cost'] is None
    assert not preflight_gate({'points': [point]}, {'points': []})['allowed']


def test_soft_joint_success_milp_exact_one_step_p_equals_q():
    tree = TreeMarket({'months': 1, 'rate': 0., 'sharpe': 0.})
    np.testing.assert_allclose(tree.p_leaf, tree.q_leaf)
    probability = float(min(tree.q_leaf))
    point = solve_frontier_point(tree, demand(), target_success=probability - 1e-7)
    expected = probability / (1. - probability)
    assert point['solver_kind'] == 'milp'
    assert point['status'] == 'optimal'
    assert point['primal_cost'] == pytest.approx(expected, abs=1e-5)
    assert point['attained_success'] >= probability - 1e-7


def test_exit_liquidation_does_not_fund_same_month_withdrawal():
    tree = TreeMarket({'months': 2, 'rate': .12, 'sharpe': 0.})
    spec = demand(withdrawal_ratios={1: .8, 2: .2}, terminal_total_ratio=1.,
                  max_drawdown=.02, exit_discount=.1,
                  exit_scenarios=({'month': 1, 'probability': .5},
                                  {'month': None, 'probability': .5}))
    point = solve_frontier_point(tree, spec)
    assert point['status'] == 'optimal'
    assert point['primal_cost'] == pytest.approx(0., abs=1e-7)
    assert point['attained_success'] == pytest.approx(1.)


def test_positive_cost_exit_requires_payment_before_withdrawal():
    tree = TreeMarket({'months': 2, 'rate': .12, 'sharpe': 0.})
    spec = demand(withdrawal_ratios={1: 1.}, terminal_total_ratio=1.005,
                  exit_scenarios=({'month': 1, 'probability': 1.},))
    point = solve_frontier_point(tree, spec)
    assert point['status'] == 'optimal'
    table = np.asarray(point['payment_table'])
    assert np.all(table[:, 1] >= point['primal_cost'] - 1e-6)
    assert point['attained_success'] == pytest.approx(1.)


def test_exogenous_events_do_not_add_payment_information():
    tree = TreeMarket({'months': 2, 'rate': .12})
    spec = demand(withdrawal_ratios={}, terminal_total_ratio=1.005,
                  exit_scenarios=({'month': 1, 'probability': .5,
                                   'withdrawal_ratios': {1: 1.}},
                                  {'month': None, 'probability': .5}))
    point = solve_frontier_point(tree, spec)
    assert point['status'] == 'optimal'
    assert point['variable_count'] == 2 * len(tree.node_times) + 1
    table = np.asarray(point['payment_table'])
    tree.node_cashflows(table)
    assert np.all(table[:, 1] >= point['primal_cost'] - 1e-6)


def test_hard_drawdown_constraints_include_intermediate_wealth():
    tree = TreeMarket({'months': 3, 'rate': .12, 'sharpe': .3})
    spec = demand(max_drawdown=0., withdrawal_ratios={1: .1, 2: .1})
    point = solve_frontier_point(tree, spec)
    assert point['status'] == 'optimal'
    assert point['attained_success'] == pytest.approx(1.)
    tree.node_cashflows(np.asarray(point['payment_table']))


def test_partial_sales_has_explicit_unsupported_scope():
    point = solve_frontier_point(TreeMarket({'months': 1}), demand(partial_sales=True))
    assert point['status'] == 'unsupported'
    assert not point['proven_optimal']
    assert point['primal_cost'] is None


def test_frontier_gate_blocks_catalog_at_bound_and_unproven_results():
    report = {'points': [{'status': 'optimal', 'proven_optimal': True,
                         'target_success': .9, 'primal_cost': 10., 'lower_cost_bound': 10.}]}
    baseline = {'points': [{'success_probability': .95, 'cost': 10.}]}
    assert not preflight_gate(report, baseline)['allowed']
    assert preflight_gate(report, {'points': [{'success_probability': .95, 'cost': 20.}]})['allowed']
    report['points'][0]['proven_optimal'] = False
    assert not preflight_gate(report, baseline)['allowed']


def test_gate_uses_all_valid_comparators_below_catalog_primary_target():
    report = {'points': [{'status': 'optimal', 'proven_optimal': True,
                         'target_success': .8, 'primal_cost': 10., 'lower_cost_bound': 10.}]}
    cheap = {'metrics': {'status': 'ok', 'cost': 10., 'satisfaction_probability': .8,
                         'feasible': False}, 'eligible': False}
    expensive = {'metrics': {'status': 'ok', 'cost': 30., 'satisfaction_probability': .95}}
    catalog = {'pareto_front': [expensive], 'pareto_front_all_valid': [cheap, expensive]}
    gate = preflight_gate(report, catalog)
    assert not gate['allowed']
    assert any('already reaches' in reason for reason in gate['reasons'])
    # Older catalog exports still include all attempts, before target filtering.
    del catalog['pareto_front_all_valid']
    catalog['attempts'] = [cheap, expensive]
    assert not preflight_gate(report, catalog)['allowed']


def test_gate_blocks_missing_comparator_and_rejects_invalid_metrics():
    report = {'points': [{'status': 'optimal', 'proven_optimal': True,
                         'target_success': .9, 'primal_cost': 10., 'lower_cost_bound': 10.}]}
    records = [{'metrics': {'status': 'ok', 'cost': 5., 'satisfaction_probability': .8}},
               {'metrics': {'status': 'rejected', 'cost': 20., 'satisfaction_probability': 1.}},
               {'metrics': {'status': 'ok', 'cost': float('nan'), 'satisfaction_probability': 1.}}]
    gate = preflight_gate(report, {'attempts': records})
    assert not gate['allowed']
    assert any('lacks optimized feasible comparator at target' in reason for reason in gate['reasons'])


def test_gate_reports_model_mismatch_when_catalog_beats_proven_bound():
    report = {'points': [{'status': 'optimal', 'proven_optimal': True,
                         'target_success': .9, 'primal_cost': 10., 'lower_cost_bound': 10.}]}
    gate = preflight_gate(report, {'points': [{'cost': 9., 'success_probability': .95}]})
    assert not gate['allowed']
    assert any('model/evaluator mismatch' in reason for reason in gate['reasons'])
    assert gate['api_allowed'] is False


def test_gate_does_not_compare_service_capital_with_legacy_leg_cost():
    report = {'funding_mode': 'minimum_service_capital',
              'points': [{'status': 'optimal', 'proven_optimal': True,
                          'target_success': .9, 'primal_cost': 10., 'lower_cost_bound': 10.}]}
    catalog = {'funding_mode': 'fixed_budget_leg_control',
               'points': [{'cost': 20., 'success_probability': .95}]}
    gate = preflight_gate(report, catalog)
    assert not gate['allowed']
    assert any('funding objectives differ' in reason for reason in gate['reasons'])


def test_solver_limit_never_fabricates_optimum(monkeypatch):
    import scipy.optimize
    from scipy.optimize import OptimizeResult
    monkeypatch.setattr(scipy.optimize, 'milp', lambda **kw: OptimizeResult(
        status=1, success=False, message='time limit', x=None, fun=None,
        mip_dual_bound=.25, mip_gap=None))
    point = solve_frontier_point(TreeMarket({'months': 1}), demand(), target_success=.5)
    assert point['status'] == 'limit'
    assert point['lower_cost_bound'] == pytest.approx(.25 * 100.)
    assert point['primal_cost'] is None
    assert not point['proven_optimal']


def test_time_limited_valid_incumbent_is_separate_from_bound(monkeypatch):
    import scipy.optimize
    actual_milp = scipy.optimize.milp

    def limited_milp(**kwargs):
        result = actual_milp(**kwargs)
        result.status = 1
        result.success = False
        result.message = 'test incumbent at time limit'
        result.mip_dual_bound = result.fun * .5
        return result

    monkeypatch.setattr(scipy.optimize, 'milp', limited_milp)
    point = solve_frontier_point(TreeMarket({'months': 1, 'rate': 0., 'sharpe': 0.}),
                                 demand(), target_success=.4)
    assert point['status'] == 'limit'
    assert point['primal_cost'] > point['lower_cost_bound']
    assert point['gap'] == pytest.approx(.5)
    assert not point['proven_optimal']


def test_resource_limit_is_explicit_and_blocks_gate():
    point = solve_frontier_point(TreeMarket({'months': 2}), demand(),
                                 target_success=.9, max_binary_cases=1)
    assert point['status'] == 'resource_limit'
    assert point['primal_cost'] is None
    assert not preflight_gate({'points': [point]}, {'points': []})['allowed']


def test_theory_output_scope_is_evaluation_only():
    report = theoretical_frontier(TreeMarket({'months': 1}), demand(terminal_total_ratio=1.), targets=(1.,))
    assert report['evaluation_only'] is True
    assert 'finite' in report['model_scope']
    assert report['points'][0]['primal_cost'] == pytest.approx(0.)
