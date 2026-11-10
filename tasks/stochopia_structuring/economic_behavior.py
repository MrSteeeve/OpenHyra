"""Numerical economic audits on a fixed, finite monthly market tree.

Cashflow matching is evidence only on the supplied tree. Equivalent ASTs are
not counted as new mechanisms, and a finite-catalog nonmatch is provisional.
Pricing and hedge deltas supplement the cashflows; they do not prove equality
in continuous markets or outside the sampled support.
"""
from __future__ import annotations

from itertools import combinations
import math
import numpy as np


def _table(tree, table):
    x = np.asarray(table, dtype=float)
    if x.shape != np.asarray(tree.paths).shape or np.any(~np.isfinite(x)):
        raise ValueError('cashflows must be a finite full monthly table')
    if np.any(np.abs(x[:, 0]) > 1e-10):
        raise ValueError('initial cashflow must be zero')
    tree.node_cashflows(x)
    return x


def numerical_q_audit(tree, table):
    """Ex-dividend values and one-step stock units on each history node."""
    x = _table(tree, table)
    values = np.asarray(tree.conditional_values(x), dtype=float)
    spot = np.asarray(tree.paths, dtype=float)
    nodes = np.asarray(tree.path_nodes)
    deltas = []
    for t in range(int(tree.months)):
        child_ids, idx = np.unique(nodes[:, t + 1], return_index=True)
        parent = nodes[idx, t]
        order = np.argsort(parent, kind='stable')
        if len(order) % 2:
            raise ValueError('expected binary market tree')
        pairs = idx[order].reshape(-1, 2)
        parent_pairs = parent[order].reshape(-1, 2)
        if np.any(parent_pairs[:, 0] != parent_pairs[:, 1]):
            raise ValueError('expected two children per history')
        price_difference = spot[pairs[:, 1], t + 1] - spot[pairs[:, 0], t + 1]
        next_claim = values[:, t + 1] + x[:, t + 1]
        difference = next_claim[pairs[:, 1]] - next_claim[pairs[:, 0]]
        # A deterministic/degenerate node has no identified stock hedge.
        dividend_growth = math.exp(float(getattr(tree, 'config', {}).get('dividend', 0.)) * float(tree.dt))
        delta = np.divide(difference, price_difference * dividend_growth,
                          out=np.zeros_like(difference), where=np.abs(price_difference) > 1e-14)
        for node, d, denom in zip(parent_pairs[:, 0], delta, price_difference):
            deltas.append({'node': int(node), 'time': t, 'stock_units': float(d),
                           'identified': bool(abs(denom) > 1e-14)})
    return {'price': float(tree.price(x)), 'conditional_values': values,
            'deltas': deltas,
            'scope': 'exact Q values on supplied tree; discrete replication deltas only'}


def behavior_signature(tree, table, abs_tol=.01, rel_tol=1e-6, program_complexity=None):
    """Classify observed behavior, with a same-terminal history witness."""
    if not math.isfinite(abs_tol) or not math.isfinite(rel_tol) or abs_tol < 0 or rel_tol < 0:
        raise ValueError('tolerances must be nonnegative and finite')
    x = _table(tree, table)
    audit = numerical_q_audit(tree, x)
    scale = float(np.max(np.abs(x)))
    tolerance = abs_tol + rel_tol * scale
    witness = None
    # CRR paths share terminal prices despite having distinct full histories.
    _, groups = np.unique(np.round(np.log(np.asarray(tree.paths)[:, -1]), 12), return_inverse=True)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        first = indices[0]
        errors = np.max(np.abs(x[indices] - x[first]), axis=1)
        changed = np.flatnonzero(errors > tolerance)
        if len(changed):
            other = int(indices[changed[0]])
            witness = {'path_a': int(first), 'path_b': other,
                       'terminal_spot': float(np.asarray(tree.paths)[first, -1]),
                       'max_cashflow_difference': float(errors[changed[0]]),
                       'path_a_cashflows': x[first].tolist(),
                       'path_b_cashflows': x[other].tolist()}
            break
    times = np.arange(x.shape[1]) * float(tree.dt)
    grown_payouts = np.cumsum(x * np.exp(-float(tree.rate) * times), axis=1) * np.exp(float(tree.rate) * times)
    wealth = audit['conditional_values'] + grown_payouts
    high = np.maximum.accumulate(wealth, axis=1)
    drawdown = np.divide(high - wealth, high, out=np.zeros_like(high), where=high > tolerance)
    price = audit['price']
    terminal_comparable = grown_payouts[:, -1]
    riskless_terminal = price * math.exp(float(tree.rate) * times[-1])
    delta_values = [r['stock_units'] for r in audit['deltas'] if r['identified']]
    early = x[:, 1:-1]
    flags = {
        'history_dependent_on_tree': witness is not None,
        'intermediate_payouts': bool(np.any(np.abs(early) > tolerance)),
        'contingent_intermediate_payouts': bool(early.size and np.any(np.ptp(early, axis=0) > tolerance)),
        'negative_stock_delta_on_tree': bool(delta_values and min(delta_values) < -tolerance),
        'can_underperform_same_cost_cash_on_tree': bool(np.any(terminal_comparable < riskless_terminal - tolerance)),
        'mark_to_market_drawdown_on_tree': bool(np.max(drawdown) > rel_tol + 1e-10),
        'negative_payment_on_tree': bool(np.any(x < -tolerance)),
    }
    return {**flags, 'risk_flags': flags, 'price': price,
            'maximum_drawdown_fraction': float(drawdown.max()),
            'minimum_terminal_accumulated_cashflows': float(terminal_comparable.min()),
            'delta_range': [float(min(delta_values)), float(max(delta_values))] if delta_values else None,
            'same_terminal_different_history_witness': witness,
            'program_complexity': complexity_stratum(program_complexity),
            'scope': 'sampled economic behavior on the supplied finite tree; no AST novelty inference'}


def complexity_stratum(complexity=None):
    """Descriptive implementation strata, never an economic novelty score."""
    if complexity is None:
        return {'stratum': 'unreported', 'reported': None, 'used_for_novelty': False}
    if isinstance(complexity, dict):
        count = int(complexity.get('node_count', complexity.get('operation_count', 0)))
        reported = dict(complexity)
    else:
        count = int(complexity)
        reported = {'node_count': count}
    if count < 0:
        raise ValueError('program complexity must be nonnegative')
    return {'stratum': 'small' if count <= 20 else 'medium' if count <= 100 else 'large',
            'reported': reported, 'used_for_novelty': False}


def _nnls_from_gram(gram, cross, support):
    """Tiny active-set NNLS on a prespecified support (at most three legs)."""
    best = np.zeros(len(support))
    best_objective = 0.
    for active_size in range(1, len(support) + 1):
        for active in combinations(range(len(support)), active_size):
            selected = [support[j] for j in active]
            g = gram[np.ix_(selected, selected)]
            c = cross[selected]
            coef = np.linalg.lstsq(g, c, rcond=None)[0]
            if np.any(coef < -1e-10):
                continue
            coef = np.maximum(coef, 0.)
            objective = float(coef @ g @ coef - 2 * coef @ c)
            if objective < best_objective:
                best_objective = objective
                best[:] = 0
                best[list(active)] = coef
    return best


def _infinity_fit(matrix, target):
    """Minimize the same maximum cashflow error used for acceptance."""
    from scipy.optimize import linprog
    scale = max(float(np.max(np.abs(matrix))), float(np.max(np.abs(target))), 1.)
    # Duplicate node equations do not change the uniform-error problem.
    equations = np.unique(np.column_stack((matrix, target)) / scale, axis=0)
    a, y = equations[:, :-1], equations[:, -1]
    constraints = np.vstack((np.column_stack((a, -np.ones(len(y)))),
                             np.column_stack((-a, -np.ones(len(y))))))
    result = linprog(np.r_[np.zeros(matrix.shape[1]), 1.], A_ub=constraints,
                     b_ub=np.r_[y, -y], bounds=[(0., None)] * (matrix.shape[1] + 1),
                     method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                               'dual_feasibility_tolerance': 1e-9})
    return (result.x[:-1] if result.success else None), str(result.message)


def match_known_behaviors(tree, table, candidates, *, max_legs=2, abs_tol=.01,
                          rel_tol=1e-6, program_complexity=None, max_fit_attempts=256):
    """Fit finite templates / nonnegative combinations with a fixed leg limit.

    No additional legs, coefficients of opposite sign, date remappings or
    after-the-fact tolerance relaxations are introduced. `candidates` may be
    the optimized catalog's full attempts. NNLS proposes a coefficient vector;
    if it fails the uniform tolerance, a nonnegative minimax LP checks the
    same fixed support. The fit is a finite-tree check,
    not an economic proof, and its numerical fit budget is separate from the
    demand evaluator call budget.
    """
    if isinstance(max_legs, bool) or max_legs not in (1, 2, 3):
        raise ValueError('max_legs must be one of 1, 2, 3')
    if isinstance(max_fit_attempts, bool) or int(max_fit_attempts) != max_fit_attempts or max_fit_attempts < 1:
        raise ValueError('max_fit_attempts must be a positive integer')
    max_fit_attempts = int(max_fit_attempts)
    if not math.isfinite(abs_tol) or not math.isfinite(rel_tol) or abs_tol < 0 or rel_tol < 0:
        raise ValueError('tolerances must be finite and nonnegative')
    target = _table(tree, table)
    templates = list(candidates)
    target_nodes = np.asarray(tree.node_cashflows(target), dtype=float).reshape(-1)
    matrices = []
    tables = []
    for row in templates:
        x = _table(tree, row['table'])
        tables.append(x)
        matrices.append(np.asarray(tree.node_cashflows(x), dtype=float).reshape(-1))
    match = best = None
    tested = 0
    solver_failures = []
    total_supports = sum(math.comb(len(templates), k)
                         for k in range(1, min(max_legs, len(templates)) + 1))
    if matrices:
        basis = np.column_stack(matrices)
        gram = basis.T @ basis
        cross = basis.T @ target_nodes
        target_scale = float(np.max(np.abs(target_nodes)))
        tolerance = abs_tol + rel_tol * target_scale
        for count in range(1, min(max_legs, len(templates)) + 1):
            for support in combinations(range(len(templates)), count):
                if tested >= max_fit_attempts:
                    break
                coefficients = _nnls_from_gram(gram, cross, support)
                fitted = basis[:, support] @ coefficients
                error = float(np.max(np.abs(fitted - target_nodes)))
                tested += 1
                method = 'NNLS witness satisfies uniform tolerance'
                if error > tolerance:
                    minimax, status = _infinity_fit(basis[:, support], target_nodes)
                    method = 'nonnegative minimax LP on fixed support'
                    if minimax is None:
                        solver_failures.append({'support': list(support), 'status': status})
                    else:
                        coefficients = minimax
                        error = float(np.max(np.abs(basis[:, support] @ coefficients - target_nodes)))
                actual = [i for i, c in enumerate(coefficients) if c > 0.]
                result = {'legs': [{'family': templates[support[j]].get('family', 'unnamed'),
                                     'params': templates[support[j]].get('params', {}),
                                     'coefficient': float(coefficients[j]),
                                     'template_index': int(support[j])} for j in actual],
                          'max_absolute_error': error, 'tolerance': tolerance, 'fit_method': method}
                if best is None or error < best['max_absolute_error']:
                    best = result
                if error <= tolerance:
                    match = result
                    break
            if match is not None or tested >= max_fit_attempts:
                break
    auxiliary = None
    if match:
        fitted_table = sum((leg['coefficient'] * tables[leg['template_index']]
                            for leg in match['legs']), np.zeros_like(target))
        target_audit = numerical_q_audit(tree, target)
        fit_audit = numerical_q_audit(tree, fitted_table)
        auxiliary = {
            'target_q_price': target_audit['price'], 'fitted_q_price': fit_audit['price'],
            'maximum_conditional_q_value_difference': float(np.max(np.abs(
                target_audit['conditional_values'] - fit_audit['conditional_values']))),
            'maximum_stock_delta_difference': max((abs(a['stock_units'] - b['stock_units'])
                for a, b in zip(target_audit['deltas'], fit_audit['deltas'])), default=0.),
            'scope': 'auxiliary finite-tree numerical evidence'}
    complete = tested == total_supports and not solver_failures
    novelty = ('sampled_behavior_matched' if match else 'fit_budget_inconclusive'
               if tested < total_supports else 'fit_solver_inconclusive' if solver_failures
               else 'provisional_unfamiliarity_to_finite_catalog')
    return {'matched': match is not None, 'match': match, 'best_sample_fit': best,
            'settings': {'max_legs': max_legs, 'abs_tol': abs_tol, 'rel_tol': rel_tol,
                         'coefficient_constraint': 'nonnegative', 'templates': len(templates),
                         'max_fit_attempts': max_fit_attempts},
            'fits_evaluated': tested, 'total_fixed_supports': total_supports,
            'search_complete': complete, 'solver_failures': solver_failures, 'economic_proof': False,
            'auxiliary_q_and_delta_evidence': auxiliary,
            'behavior': behavior_signature(tree, target, abs_tol, rel_tol, program_complexity),
            'novelty_status': novelty,
            'primary_endpoint': 'distance to the same discrete-model theory frontier',
            'scope': 'successful sample match is not economic proof; nonmatch is not financial originality'}
