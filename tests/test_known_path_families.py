"""Economic checks on the new monthly catalog, independent of legacy DSL syntax."""
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.stochopia_structuring.known_path_families import (
    FAMILY_NAMES, family_candidate, optimize_known_families,
)
from tasks.stochopia_structuring.economic_behavior import (
    behavior_signature, match_known_behaviors,
)


class SmallTree:
    """Exact complete non-recombining tree for independent analytic checks."""
    def __init__(self, months=4, rate=.02):
        self.months, self.dt, self.rate = months, 1 / 12, rate
        self.up = np.exp(.2 * np.sqrt(self.dt))
        self.down = 1 / self.up
        self.q = (np.exp(rate * self.dt) - self.down) / (self.up - self.down)
        steps = np.asarray(list(itertools.product([0, 1], repeat=months)))
        self.paths = np.column_stack([np.ones(len(steps)), np.cumprod(np.where(steps, self.up, self.down), axis=1)])
        self.q_leaf = np.prod(np.where(steps, self.q, 1 - self.q), axis=1)
        self.p_leaf = self.q_leaf.copy()
        self.path_nodes = np.column_stack([np.zeros(len(steps), dtype=int)] + [
            2 ** t - 1 + sum(steps[:, j] * 2 ** (t - j - 1) for j in range(t))
            for t in range(1, months + 1)])

    def node_cashflows(self, table):
        table = np.asarray(table)
        result = np.zeros(2 ** (self.months + 1) - 1)
        for t in range(self.months + 1):
            for node in np.unique(self.path_nodes[:, t]):
                values = table[self.path_nodes[:, t] == node, t]
                if not np.allclose(values, values[0], atol=1e-9, rtol=0):
                    raise ValueError('not adapted')
                result[node] = values[0]
        return result

    def price(self, table):
        self.node_cashflows(table)
        return float(self.q_leaf @ (np.asarray(table) @ np.exp(-self.rate * self.dt * np.arange(self.months + 1))))

    def conditional_values(self, table):
        self.node_cashflows(table)
        values = np.zeros_like(table, dtype=float)
        for t in range(self.months):
            future = np.asarray(table)[:, t + 1:] @ np.exp(-self.rate * self.dt * np.arange(1, self.months - t + 1))
            for node in np.unique(self.path_nodes[:, t]):
                mask = self.path_nodes[:, t] == node
                values[mask, t] = np.average(future[mask], weights=self.q_leaf[mask])
        return values


@pytest.fixture
def tree():
    return SmallTree()


@pytest.mark.parametrize('family', FAMILY_NAMES)
def test_every_family_is_nonnegative_cny_adapted_and_initial_payment_zero(tree, family):
    row = family_candidate(tree, family, {}, notional=100_000.)
    table = row['table']
    assert table.shape == tree.paths.shape
    assert np.all(table >= 0) and np.all(table[:, 0] == 0)
    tree.node_cashflows(table)
    assert row['price'] == pytest.approx(tree.price(table))


@pytest.mark.parametrize('family', ['cppi', 'tipp', 'drawdown_cushion', 'periodic_payout'])
def test_strategy_self_financing_coupons_preserve_initial_q_cost(tree, family):
    row = family_candidate(tree, family, {'coupon': .012, 'multiplier': 3.}, notional=100.)
    assert tree.price(row['table']) == pytest.approx(100., abs=1e-9)
    audit = row['strategy_diagnostics']
    assert audit['minimum_cash_holding'] >= -1e-10
    assert audit['minimum_stock_value'] >= -1e-10
    assert audit['maximum_self_financing_error'] < 1e-9
    assert np.any(row['table'][:, 1:-1] > 0)
    if family == 'drawdown_cushion':
        assert 'discrete approximation' in row['definition_scope']


def test_budget_is_balanced_exact_and_all_infeasible_is_honest(tree):
    calls = []
    def evaluate(table):
        calls.append(table)
        return {'cost': tree.price(table), 'satisfaction_probability': .4,
                'drawdown_loss': .2, 'feasible': False}
    n = len(FAMILY_NAMES)
    result = optimize_known_families(tree, SimpleNamespace(budget=100., min_success_probability=.8), evaluate, 2 * n + 3, seed=7)
    assert result['evaluations'] == len(calls) == 2 * n + 3
    assert [r['family'] for r in result['attempts'][:n]] == list(FAMILY_NAMES)
    counts = [sum(r['family'] == f for r in result['attempts']) for f in FAMILY_NAMES]
    assert min(counts) == 2 and max(counts) == 3
    assert result['best'] is None and result['best_cost_meeting_target'] is None
    assert result['pareto_front'] == []
    assert set(result['parameter_boundaries']) == set(FAMILY_NAMES)
    assert 'not global optimum' in result['scope']
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match='two evaluations'):
        optimize_known_families(tree, SimpleNamespace(budget=100.), evaluate, n)


def test_terminal_rewrite_matches_behavior_without_ast_or_novelty_claim(tree):
    row = family_candidate(tree, 'vanilla_call', {'strike': 1.}, notional=100.)
    rewrite = np.zeros_like(row['table'])
    rewrite[:, -1] = 50. * (tree.paths[:, -1] - 1 + np.abs(tree.paths[:, -1] - 1))
    result = match_known_behaviors(tree, rewrite, [row], max_legs=1, abs_tol=1e-8)
    assert result['matched']
    assert result['match']['max_absolute_error'] < 1e-8
    assert result['novelty_status'] == 'sampled_behavior_matched'
    assert not result['economic_proof']
    assert behavior_signature(tree, rewrite)['same_terminal_different_history_witness'] is None


def test_average_trigger_has_same_terminal_different_history_witness(tree):
    row = family_candidate(tree, 'average_trigger', {'strike': 1.}, notional=100.)
    sig = behavior_signature(tree, row['table'])
    assert sig['history_dependent_on_tree']
    assert sig['same_terminal_different_history_witness'] is not None
    assert 'sampled' in sig['scope']


def test_fixed_leg_limit_rejects_match_that_needs_three_independent_payments(tree):
    legs = []
    for t in [1, 2, 3]:
        table = np.zeros_like(tree.paths)
        table[:, t] = 1.
        legs.append({'family': f'coupon{t}', 'params': {}, 'table': table})
    target = sum(r['table'] for r in legs)
    assert not match_known_behaviors(tree, target, legs, max_legs=2, abs_tol=1e-8)['matched']
    matched = match_known_behaviors(tree, target, legs, max_legs=3, abs_tol=1e-8)
    assert matched['matched'] and len(matched['match']['legs']) == 3
    assert matched['settings']['max_legs'] == 3


def test_real_evaluator_rejections_do_not_abort_balanced_search():
    from stochopia.research.tree_market import TreeMarket
    from stochopia.research.path_demand import PathDemand, evaluate_cashflows
    market = TreeMarket({'months': 3, 'sharpe': 0.})
    demand = PathDemand(budget=100., withdrawal_ratios={}, exit_scenarios=({'month': None, 'probability': 1.},))
    result = optimize_known_families(market, demand, lambda t: evaluate_cashflows(market, demand, t), 2 * len(FAMILY_NAMES))
    assert result['evaluations'] == 2 * len(FAMILY_NAMES)
    assert any(r['metrics']['status'] == 'rejected' for r in result['attempts'])
    assert all(not r['eligible'] for r in result['attempts'] if r['metrics']['status'] == 'rejected')
    json.dumps(result, allow_nan=False)


def test_dynamic_strategies_reinvest_declared_dividends_self_financing():
    from stochopia.research.tree_market import TreeMarket
    market = TreeMarket({'months': 4, 'dividend': .03, 'sharpe': 0.})
    row = family_candidate(market, 'periodic_payout', {'equity_fraction': 1., 'coupon': .01}, notional=100.)
    assert market.price(row['table']) == pytest.approx(100., abs=1e-9)


def test_zero_payments_match_zero_weight_catalog_and_bad_tolerances_reject(tree):
    zero = np.zeros_like(tree.paths)
    row = family_candidate(tree, 'vanilla_call', {}, notional=100.)
    result = match_known_behaviors(tree, zero, [row], abs_tol=1e-8)
    assert result['matched'] and result['match']['legs'] == []
    with pytest.raises(ValueError, match='tolerances'):
        match_known_behaviors(tree, zero, [row], abs_tol=-1.)


def test_behavior_risk_flags_follow_payments_and_q_hedge_not_family_name(tree):
    put = family_candidate(tree, 'vanilla_put', {}, notional=100.)
    flags = behavior_signature(tree, put['table'])
    assert flags['negative_stock_delta_on_tree']
    assert flags['can_underperform_same_cost_cash_on_tree']
    renamed = {**put, 'family': 'syntactically_complex_novel_name'}
    result = match_known_behaviors(tree, put['table'], [renamed], program_complexity={'node_count': 120})
    assert result['matched'] and not result['behavior']['program_complexity']['used_for_novelty']
    assert result['behavior']['program_complexity']['stratum'] == 'large'


def test_reference_frontier_keeps_valid_candidates_below_primary_target(tree):
    n = len(FAMILY_NAMES)
    def evaluate(table):
        return {'cost': tree.price(table), 'satisfaction_probability': .4,
                'drawdown_loss': .2, 'feasible': False, 'status': 'ok'}
    result = optimize_known_families(tree, SimpleNamespace(budget=100., min_success_probability=.8), evaluate, 2*n)
    assert result['pareto_front'] == []
    assert result['pareto_front_all_valid']
    assert all(r['metrics']['status'] == 'ok' for r in result['pareto_front_all_valid'])
    assert all(r['metrics']['satisfaction_probability'] < .8 for r in result['pareto_front_all_valid'])


def test_max_error_matching_uses_infinity_fit_not_least_squares(tree):
    target = np.zeros_like(tree.paths)
    target[:, -1] = 1.
    basis = np.zeros_like(tree.paths)
    basis[:, -1] = np.where(np.arange(len(basis)) % 2, 2., 1.)
    result = match_known_behaviors(tree, target, [{'family': 'fixed', 'table': basis}],
                                  max_legs=1, abs_tol=.35, rel_tol=0.)
    assert result['matched']
    assert result['match']['max_absolute_error'] <= .35


def test_finite_fit_budget_exhaustion_is_inconclusive_not_catalog_nonmatch(tree):
    target = np.zeros_like(tree.paths)
    target[:, -1] = 1.
    wrong = np.zeros_like(tree.paths)
    wrong[:, 1] = 1.
    result = match_known_behaviors(tree, target, [{'family': 'wrong', 'table': wrong},
        {'family': 'right', 'table': target}], max_legs=1, abs_tol=1e-8, max_fit_attempts=1)
    assert not result['matched']
    assert result['fits_evaluated'] == 1
    assert result['novelty_status'] == 'fit_budget_inconclusive'
    assert not result['search_complete']


def test_explicit_cash_controls_include_zero_allocation_and_funded_dated_coupons(tree):
    assert {'cash_only', 'cash_bond'} <= set(FAMILY_NAMES)
    cash = family_candidate(tree, 'cash_only', {}, notional=100.)
    assert np.all(cash['table'] == 0.) and cash['price'] == 0.
    bond = family_candidate(tree, 'cash_bond', {'capital_ratio': .5, 'coupon': .01}, notional=100.)
    assert np.all(bond['table'][:, 1:-1] == 1.)
    assert tree.price(bond['table']) == pytest.approx(50.)
    assert np.all(np.ptp(bond['table'], axis=0) == 0.)
    empty = family_candidate(tree, 'cppi', {'capital_ratio': 0.}, notional=100.)
    assert np.all(empty['table'] == 0.)


def test_cash_bond_initial_analytic_fraction_has_no_oracle_dependency(tree):
    fraction = .4
    terminal_ratio = 1. + fraction * (np.exp(tree.rate * tree.months * tree.dt) - 1.)
    demand = SimpleNamespace(budget=100., terminal_total_ratio=terminal_ratio,
        withdrawal_ratios={}, exit_scenarios=({'month': None, 'probability': 1.},), min_success_probability=.8,
        funding_mode='fixed_budget_leg_control')
    result = optimize_known_families(tree, demand, lambda x: {
        'cost': tree.price(x), 'satisfaction_probability': 0., 'drawdown_loss': 0., 'feasible': False},
        3*len(FAMILY_NAMES))
    initial = next(row for row in result['attempts'] if row['family'] == 'cash_bond')
    assert initial['params']['capital_ratio'] == pytest.approx(fraction)
    assert initial['params']['coupon'] == 0.
    assert result['evaluations'] == 3*len(FAMILY_NAMES)
    assert all(row['params'] == {} for row in result['attempts'] if row['family'] == 'cash_only')


def test_minimum_service_capital_cash_bond_sizes_absolute_delivery(tree):
    desired_fraction = .9
    terminal_ratio = desired_fraction * np.exp(tree.rate * tree.months * tree.dt)
    demand = SimpleNamespace(budget=100., terminal_total_ratio=terminal_ratio,
        withdrawal_ratios={}, exit_scenarios=({'month': None, 'probability': 1.},),
        funding_mode='minimum_service_capital', min_success_probability=.8)
    result = optimize_known_families(tree, demand, lambda x: {
        'cost': tree.price(x), 'satisfaction_probability': 0., 'drawdown_loss': 0., 'feasible': False},
        2*len(FAMILY_NAMES))
    initial = next(row for row in result['attempts'] if row['family'] == 'cash_bond')
    assert initial['params']['capital_ratio'] == pytest.approx(desired_fraction)
    assert initial['price'] == pytest.approx(90.)
    assert result['funding_mode'] == 'minimum_service_capital'


def test_calendar_payout_mirrors_only_public_dated_liabilities_and_exact_q_cost(tree):
    demand = SimpleNamespace(budget=100., terminal_total_ratio=1.1, withdrawal_ratios={1: .1, 3: .2})
    row = family_candidate(tree, 'calendar_payout', {}, notional=100., demand=demand)
    expected = np.zeros_like(tree.paths)
    expected[:, 1] = 10.
    expected[:, 3] = 20.
    expected[:, 4] = 80.
    assert np.allclose(row['table'], expected)
    assert row['price'] == pytest.approx(10*np.exp(-tree.rate*tree.dt) +
        20*np.exp(-tree.rate*3*tree.dt) + 80*np.exp(-tree.rate*4*tree.dt))
    tree.node_cashflows(row['table'])
    assert not row['calendar_diagnostics']['withdrawals_exceed_target_delivery']
    scaled = family_candidate(tree, 'calendar_payout', {'delivery_scale': 1.2}, notional=100., demand=demand)
    assert np.allclose(scaled['table'], 1.2*expected)


def test_calendar_payout_declares_overdelivery_when_withdrawals_exceed_terminal_goal(tree):
    demand = SimpleNamespace(terminal_total_ratio=.1, withdrawal_ratios={1: .2})
    row = family_candidate(tree, 'calendar_payout', {}, notional=100., demand=demand)
    assert np.all(row['table'][:, -1] == 0.)
    assert row['calendar_diagnostics']['withdrawals_exceed_target_delivery']
