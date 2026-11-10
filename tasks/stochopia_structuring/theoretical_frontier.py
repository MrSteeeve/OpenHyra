"""Evaluation-only finite-tree LP/MILP reference for service funding.

There is one nonnegative payment variable per *market* history, not per
exogenous demand event. Conditional Q values are imposed by linear backward
equations. LP enforces every positive-P case; MILP uses a single binary per
leaf/event to gate the entire conjunction. No result is a continuous GBM bound.

This module must not be imported into candidate generation or its feedback.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np


MODEL_SCOPE = (
    'Exact finite binomial market, bounded nonnegative adapted node cashflows, '
    'minimum initial service capital K=C+R with endogenous zero-interest reserve R, '
    'K at most B, outside cash excluded from service wealth, monthly withdrawals and '
    'flow-adjusted wealth drawdown, exogenous full exit at ex-dividend Q value '
    'minus haircut, no partial sales; not a continuous GBM optimum.'
)
CONTROL_SCOPE = MODEL_SCOPE.replace(
    'minimum initial service capital K=C+R with endogenous zero-interest reserve R, '
    'K at most B, outside cash excluded from service wealth',
    'regression-only fixed-budget contract-leg cost control, reserve B minus Q cost')


def _report(target, **values):
    result = dict(status='error', solver_status=None, solver_kind=None,
                  target_success=float(target), primal_cost=None,
                  lower_cost_bound=None, gap=None, absolute_gap=None,
                  proven_optimal=False, attained_success=None, payment_table=None,
                  contract_purchase_cost=None, cash_reserve=None, initial_funded_wealth=None,
                  model_scope=MODEL_SCOPE, evaluation_only=True)
    result.update(values)
    return result


class _Rows:
    def __init__(self, columns, *, deduplicate=False, limit=400_000):
        self.columns = columns
        self.indices = []
        self.rows = []
        self.data = []
        self.lower = []
        self.upper = []
        self.seen = set() if deduplicate else None
        self.limit = limit

    def add(self, coefficients, lower=-np.inf, upper=np.inf):
        items = tuple(sorted((k, float(v)) for k, v in coefficients.items() if v != 0.))
        if self.seen is not None:
            key = (items, float(lower), float(upper))
            if key in self.seen:
                return
            self.seen.add(key)
        if len(self.lower) >= self.limit:
            raise OverflowError('reference constraint resource limit exceeded')
        row = len(self.lower)
        for column, value in items:
            self.rows.append(row)
            self.indices.append(column)
            self.data.append(value)
        self.lower.append(lower)
        self.upper.append(upper)

    def matrix(self):
        from scipy.sparse import coo_matrix
        return coo_matrix((self.data, (self.rows, self.indices)),
                          shape=(len(self.lower), self.columns)).tocsr()


def _difference(left, right, scale):
    result = left.copy()
    for key, value in right.items():
        result[key] = result.get(key, 0.) - scale * value
    return result


def _success(evaluation):
    for name in ('joint_success_probability', 'satisfaction_probability', 'success_probability', 'demand_success_probability'):
        if name in evaluation:
            return float(evaluation[name])
    raise KeyError('evaluator lacks joint success probability')


def _coalesce_cases(tree, cases):
    """Combine financially identical cases, without merging market states.

    Once full exit has occurred, later stock moves cannot change available
    cash or flow-adjusted wealth. Equal exit prefixes and *complete* withdrawal
    vectors therefore imply identical success constraints for every payment
    table. Both event measures can be summed exactly. No-exit cases still use
    their full market history; exogenous events never index payment variables.
    """
    grouped = {}
    for case in cases:
        month = case['exit_month']
        stop = tree.months if month is None else int(month)
        prefix = tuple(int(node) for node in tree.path_nodes[int(case['leaf_index']), :stop + 1])
        key = (month, prefix, tuple(float(value) for value in case['withdrawals']))
        if key not in grouped:
            grouped[key] = {**case, 'weight_p': float(case['weight_p']),
                            'weight_q': float(case['weight_q']), 'source_case_count': 1}
        else:
            grouped[key]['weight_p'] += float(case['weight_p'])
            grouped[key]['weight_q'] += float(case['weight_q'])
            grouped[key]['source_case_count'] += 1
    return list(grouped.values())


def solve_frontier_point(tree, demand, target_success=1., time_limit=60.,
                         *, max_constraints=400_000, max_binary_cases=16_000,
                         compress_cases=True):
    """Minimize service capital (or the explicit legacy control objective).

    A proven optimum is claimed only after solver optimality *and* independent
    evaluator acceptance. On limits, attainable incumbent and certified dual
    bound remain separate. Unsupported sale permissions explicitly block use.
    """
    target = float(target_success)
    if not math.isfinite(target) or not 0. <= target <= 1.:
        raise ValueError('target_success must lie in [0, 1]')
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit must be finite and positive')
    funding_mode = demand.funding_mode
    if funding_mode not in ('minimum_service_capital', 'fixed_budget_leg_control'):
        raise ValueError('unsupported funding mode')
    primary = funding_mode == 'minimum_service_capital'

    def report_point(**kwargs):
        return _report(target, model_scope=MODEL_SCOPE if primary else CONTROL_SCOPE,
                       funding_mode=funding_mode,
                       cost_definition='minimum initial service capital C+R' if primary else 'contract purchase price C; fixed-funded control only',
                       **kwargs)

    if demand.partial_sales:
        return report_point(status='unsupported', reason='partial-sales theory is not implemented; no-sale bound cannot certify this branch')
    from scipy import optimize
    from stochopia.research.path_demand import demand_cases, evaluate_cashflows

    started = time.monotonic()
    cases = [case for case in demand_cases(tree, demand) if case['weight_p'] > 0.]
    original_case_count = len(cases)
    if compress_cases:
        cases = _coalesce_cases(tree, cases)
    compression_info = {'original_case_count': original_case_count,
                        'compressed_case_count': len(cases),
                        'case_compression': 'exact exit-prefix and full-withdrawal-vector equivalence' if compress_cases else 'disabled'}
    soft = target < 1.
    count = len(tree.node_times)
    horizon = tree.months
    nominal_budget = float(demand.budget)
    # Monetary optimization is homogeneous: divide continuous decisions and
    # every cash requirement by B. This keeps the exact economic problem while
    # avoiding currency-size effects in the solver's feasibility tolerances.
    budget = 1.
    cap = float(demand.max_payment_ratio) * budget
    discount = math.exp(-tree.rate * tree.dt)
    prices = np.asarray(tree.q_mass) * np.exp(-tree.rate * tree.dt * np.asarray(tree.node_times))
    if np.any(prices <= 0.) or not np.all(np.isfinite(prices)):
        return report_point(status='unsupported', reason='reference requires strictly positive finite Q node prices')
    if soft and len(cases) > max_binary_cases:
        return report_point(status='resource_limit', reason='binary case resource limit exceeded', case_count=len(cases), **compression_info)
    # x[0:count], ex-div V[count:2*count], purchase cost C, optional R, then z.
    cost_index = 2 * count
    reserve_index = cost_index + 1 if primary else None
    binary_offset = cost_index + 2 if primary else cost_index + 1
    variables = binary_offset + (len(cases) if soft else 0)
    upper_bounds = np.zeros(variables)
    upper_bounds[:count] = np.minimum(cap, budget / prices)
    upper_bounds[0] = 0.  # No inception payment in this research protocol.
    for node, t in enumerate(tree.node_times):
        remaining_cap = cap * sum(discount ** k for k in range(1, horizon - int(t) + 1))
        upper_bounds[count + node] = min(remaining_cap, budget / prices[node])
    upper_bounds[cost_index] = budget
    if primary:
        upper_bounds[reserve_index] = budget
    if soft:
        upper_bounds[binary_offset:] = 1.
    equalities = _Rows(variables, limit=max_constraints)
    inequalities = _Rows(variables, deduplicate=not soft, limit=max_constraints)
    objective = np.zeros(variables)
    objective[cost_index] = 1.
    if primary:
        objective[reserve_index] = 1.
        inequalities.add({cost_index: -1., reserve_index: -1.}, lower=-budget)
    price_row = {node: -price for node, price in enumerate(prices)}
    price_row[cost_index] = 1.
    equalities.add(price_row, 0., 0.)
    for node, t in enumerate(tree.node_times):
        if t == horizon:
            continue  # Terminal V is fixed to zero by its bound.
        row = {count + node: 1.}
        for child in tree.children[node]:
            child = int(child)
            q_conditional = tree.q_mass[child] / tree.q_mass[node]
            row[child] = -discount * q_conditional
            row[count + child] = -discount * q_conditional
        equalities.add(row, 0., 0.)

    def requirement(row, rhs, binary):
        if soft:
            # Finite box bounds imply row*u >= -sum negative magnitude*upper.
            # This M makes z=0 redundant even before using pricing equalities.
            minimum = sum(value * upper_bounds[key] for key, value in row.items() if value < 0.)
            big_m = max(0., rhs - minimum)
            row = row.copy()
            row[binary] = -big_m
            inequalities.add(row, lower=rhs - big_m)
        else:
            inequalities.add(row, lower=rhs)

    try:
        for case_index, case in enumerate(cases):
            nodes = tree.path_nodes[int(case['leaf_index'])]
            exit_month = case['exit_month']
            final_payment_month = horizon if exit_month is None else int(exit_month)
            binary = binary_offset + case_index
            withdrawals = np.cumsum(np.asarray(case['withdrawals'], dtype=float)) / nominal_budget
            # Primary expressions are absolute funded-service wealth. Legacy
            # control expressions are W-B. External withdrawals are added back.
            wealth = [{reserve_index: 1., cost_index: 1.}] if primary else [{}]
            for month in range(1, horizon + 1):
                stop = min(month, final_payment_month)
                paid = {int(n): 1. for n in nodes[1:stop + 1]}
                reserve_terms = {reserve_index: 1.} if primary else {cost_index: -1.}
                cash = {**paid, **reserve_terms}
                if exit_month is not None and month > exit_month:
                    cash[count + int(nodes[exit_month])] = 1. - demand.exit_discount
                if withdrawals[month] > 0.:
                    requirement(cash, withdrawals[month] if primary else withdrawals[month] - budget, binary)
                marked = {**paid, **reserve_terms}
                if exit_month is not None and month >= exit_month:
                    marked[count + int(nodes[exit_month])] = 1. - demand.exit_discount
                else:
                    marked[count + int(nodes[month])] = 1.
                wealth.append(marked)
            requirement(wealth[-1], budget * (demand.terminal_total_ratio if primary else demand.terminal_total_ratio - 1.), binary)
            if demand.max_drawdown < 1.:
                for later in range(1, final_payment_month + 1):
                    for earlier in range(later):
                        row = _difference(wealth[later], wealth[earlier], 1. - demand.max_drawdown)
                        requirement(row, 0. if primary else -demand.max_drawdown * budget, binary)
            if time.monotonic() - started >= time_limit:
                return report_point(status='limit', reason='time limit during model construction', solver_kind='milp' if soft else 'lp', **compression_info)
        if soft:
            inequalities.add({binary_offset + i: float(case['weight_p']) for i, case in enumerate(cases)}, lower=target)
    except OverflowError as exc:
        return report_point(status='resource_limit', reason=str(exc), solver_kind='milp' if soft else 'lp', constraint_count=len(inequalities.lower), **compression_info)

    remaining = max(.001, time_limit - (time.monotonic() - started))
    eq_matrix = equalities.matrix()
    ineq_matrix = inequalities.matrix()
    if soft:
        from scipy.sparse import vstack
        constraints = optimize.LinearConstraint(
            vstack([eq_matrix, ineq_matrix], format='csc'),
            np.asarray(equalities.lower + inequalities.lower),
            np.asarray(equalities.upper + inequalities.upper))
        integrality = np.zeros(variables, dtype=int)
        integrality[binary_offset:] = 1
        result = optimize.milp(c=objective, integrality=integrality,
                               bounds=optimize.Bounds(np.zeros(variables), upper_bounds),
                               constraints=constraints,
                               options={'time_limit': remaining, 'mip_rel_gap': 1e-9})
    else:
        result = optimize.linprog(c=objective, A_ub=-ineq_matrix,
                                  b_ub=-np.asarray(inequalities.lower),
                                  A_eq=eq_matrix, b_eq=np.asarray(equalities.lower),
                                  bounds=list(zip(np.zeros(variables), upper_bounds)),
                                  method='highs', options={'time_limit': remaining,
                                      'primal_feasibility_tolerance': 1e-9,
                                      'dual_feasibility_tolerance': 1e-9})
    status = {0: 'optimal', 1: 'limit', 2: 'infeasible', 3: 'unbounded', 4: 'solver_error'}.get(result.status, 'solver_error')
    # HiGHS LP results may carry a placeholder mip_dual_bound=0; it is not
    # the LP dual optimum and must never override a proven LP objective.
    lower = getattr(result, 'mip_dual_bound', None) if soft else None
    if lower is None and result.status == 0:
        lower = result.fun
    if lower is not None and not math.isfinite(lower):
        lower = None
    if lower is not None:
        lower = float(lower) * nominal_budget
    report = report_point(status=status, solver_status=int(result.status),
                     solver_kind='milp' if soft else 'lp', solver_message=str(result.message),
                     lower_cost_bound=None if lower is None else float(lower),
                     variable_count=variables, case_count=len(cases),
                     constraint_count=len(equalities.lower) + len(inequalities.lower),
                     elapsed_seconds=time.monotonic() - started,
                     normalization={'optimization_budget': 1., 'reported_budget': nominal_budget,
                         'monetary_variables': 'budget ratios during optimization; original currency in results',
                         'original_currency_reevaluation': True,
                         'added_economic_slack_or_fees': 0.},
                     solver_tolerances=({'relative_mip_objective_gap': 1e-9} if soft else
                         {'primal_feasibility_budget_ratio': 1e-9, 'dual_feasibility': 1e-9}),
                     **compression_info)
    if result.x is not None and np.all(np.isfinite(result.x)):
        node_payments = np.clip(result.x[:count], 0., cap)
        node_payments[0] = 0.
        node_payments *= nominal_budget
        table = node_payments[np.asarray(tree.path_nodes)]
        # Reserve selection is part of each frontier point: do not silently
        # refit its reserve to a different primary probability requirement.
        evaluation = evaluate_cashflows(tree, replace(demand, min_success_probability=target), table)
        attained = _success(evaluation)
        purchase_cost = float(tree.price(table))
        primal = evaluation.get('cost') if primary else purchase_cost
        reserve = evaluation.get('cash_reserve') if primary else nominal_budget - purchase_cost
        # A solver's incumbent is not an attainable reference until the common
        # evaluator confirms *joint* success, including every cash constraint.
        success_verified = attained + 1e-10 >= target
        if not soft:
            success_verified = bool(evaluation.get('case_results')) and all(
                case['success'] for case in evaluation['case_results'] if case['weight_p'] > 0.)
        accepted = (success_verified and evaluation.get('status') == 'ok'
                    and primal is not None and math.isfinite(primal)
                    and primal <= nominal_budget + 1e-9 * nominal_budget)
        summary = {key: value for key, value in evaluation.items() if key != 'case_results'}
        report.update(attained_success=attained, independent_evaluation=summary,
                      incumbent_accepted=accepted,
                      validation_tolerances={'joint_probability': 1e-10,
                          'monetary_original_currency': evaluation.get('rules', {}).get('monetary_comparison_tolerance'),
                          'hard_success': 'every positive-P case must pass common evaluator'})
        if accepted:
            report.update(primal_cost=primal, payment_table=table.tolist(),
                          node_payments=node_payments.tolist(),
                          contract_purchase_cost=purchase_cost, cash_reserve=reserve,
                          initial_funded_wealth=purchase_cost + reserve,
                          solver_primal_cost=float(result.fun) * nominal_budget,
                          solver_cash_reserve=float(result.x[reserve_index]) * nominal_budget if primary else reserve)
            if lower is not None:
                absolute_gap = max(0., primal - float(lower))
                report.update(absolute_gap=absolute_gap,
                              gap=absolute_gap / max(abs(primal), 1e-12))
            cost_agreement_tolerance = nominal_budget * 1e-9
            report['validation_tolerances']['objective_original_currency'] = cost_agreement_tolerance
            agrees_with_bound = lower is not None and abs(primal - float(lower)) <= cost_agreement_tolerance
            report['proven_optimal'] = result.status == 0 and agrees_with_bound
            if result.status == 0 and not agrees_with_bound:
                report.update(status='validation_failed', reason='common evaluator funding cost disagrees with optimal solver bound')
        elif result.status == 0:
            report.update(status='validation_failed', reason='solver solution fails common evaluator; no attainable optimum certified')
    return report


def theoretical_frontier(tree, demand, targets=(.8, .9, .95, 1.), time_limit=60., **kwargs):
    points = [solve_frontier_point(tree, demand, target, time_limit, **kwargs) for target in targets]
    return {'schema': 'path-theoretical-frontier.v2',
            'model_scope': MODEL_SCOPE if demand.funding_mode == 'minimum_service_capital' else CONTROL_SCOPE,
            'funding_mode': demand.funding_mode,
            'evaluation_only': True, 'points': points,
            'all_proven_optimal': all(point['proven_optimal'] for point in points)}


def preflight_gate(theory_report, baseline_report, *, cost_tolerance=1e-6,
                   success_tolerance=1e-8):
    """A necessary research headroom gate, never permission to spend on APIs."""
    if not math.isfinite(cost_tolerance) or cost_tolerance < 0 or not math.isfinite(success_tolerance) or success_tolerance < 0:
        raise ValueError('gate tolerances must be finite and nonnegative')
    reasons = []
    points = theory_report.get('points', [])
    theory_modes = {point.get('funding_mode') for point in points if point.get('funding_mode')}
    if theory_report.get('funding_mode'):
        theory_modes.add(theory_report['funding_mode'])
    baseline_mode = baseline_report.get('funding_mode')
    if len(theory_modes) > 1 or (baseline_mode and theory_modes and theory_modes != {baseline_mode}):
        reasons.append('model/evaluator mismatch: theoretical and catalog funding objectives differ')
    if not points:
        reasons.append('theoretical frontier is missing')
    # The catalog's primary success threshold must not discard valid comparators
    # for lower points on the theoretical frontier. Prefer its unfiltered data.
    baselines = baseline_report.get('pareto_front_all_valid',
        baseline_report.get('attempts', baseline_report.get('pareto_front',
        baseline_report.get('points', baseline_report.get('records', [])))))
    if not baselines:
        reasons.append('optimized catalog frontier is missing')
    valid_baselines = []
    for record in baselines:
        metrics = record.get('metrics', record)
        if record.get('status') in ('rejected', 'error', 'invalid') or metrics.get('status') in ('rejected', 'error', 'invalid'):
            continue
        success = metrics.get('satisfaction_probability', metrics.get('success_probability', metrics.get('joint_success_probability')))
        cost = metrics.get('cost', metrics.get('q_cost'))
        if (isinstance(success, (int, float, np.number)) and not isinstance(success, (bool, np.bool_))
                and isinstance(cost, (int, float, np.number)) and not isinstance(cost, (bool, np.bool_))
                and math.isfinite(success) and math.isfinite(cost)
                and 0. <= success <= 1. + success_tolerance and cost >= -cost_tolerance):
            valid_baselines.append((float(success), float(cost)))
    for point in points:
        target = point.get('target_success')
        if not isinstance(target, (int, float, np.number)) or not math.isfinite(target) or not 0. <= target <= 1.:
            reasons.append('theoretical frontier point lacks a valid success target')
            continue
        meeting = [cost for success, cost in valid_baselines if success + success_tolerance >= target]
        if not meeting:
            reasons.append(f'target {target}: catalog lacks optimized feasible comparator at target')
        if not point.get('proven_optimal') or point.get('status') != 'optimal':
            reasons.append(f"target {point.get('target_success')} lacks a validated proven optimum")
            continue
        lower = point.get('lower_cost_bound')
        if lower is None or not math.isfinite(lower):
            reasons.append(f'target {target} has no certified cost bound')
            continue
        if meeting and min(meeting) < lower - cost_tolerance:
            reasons.append(f'target {target}: model/evaluator mismatch; valid catalog cost lies below proven theoretical bound')
        elif meeting and min(meeting) <= lower + cost_tolerance:
            reasons.append(f'target {target}: optimized catalog already reaches the reference frontier')
    return {'allowed': not reasons, 'api_allowed': False, 'reasons': reasons,
            'scope': 'necessary offline research headroom check; API budget authorization is separate'}
