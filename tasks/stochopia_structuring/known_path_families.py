"""Finite, demand-specific catalog on the shared monthly market tree.

All payments are CNY cashflows at the fixed service notional, with no initial
payment. Monthly observation is part of each definition: neither these note
payoffs nor the insurance strategies claim continuous-market optimality.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np

FAMILY_NAMES = (
    'cash_only', 'cash_bond', 'calendar_payout',
    'vanilla_call', 'vanilla_put', 'digital_call', 'down_and_out_put',
    'geometric_asian_call', 'arithmetic_asian_call', 'lookback_call',
    'autocallable', 'snowball', 'cliquet', 'range_accrual', 'call_spread',
    'two_way', 'cppi', 'tipp', 'obpi', 'drawdown_cushion', 'ratchet',
    'highwater_protection', 'periodic_payout', 'quantile_hedge', 'average_trigger',
)

# Bounds are public protocol choices, not inferred global economic limits.
_SPECS = {
    'strike': (.7, 1.3, 1.), 'participation': (.25, 2., 1.),
    'principal': (0., 1., 1.), 'barrier': (.5, .95, .75),
    'ko': (.9, 1.2, 1.05), 'ki': (.5, .95, .7),
    'coupon': (0., .03, .005), 'floor': (0., .01, 0.),
    'cap': (.01, .15, .05), 'lower': (.6, .99, .8),
    'upper': (1.01, 1.5, 1.2), 'width': (.05, .5, .2),
    'multiplier': (0., 5., 3.), 'guarantee': (0., 1., .8),
    'capital_ratio': (0., 1., 1.), 'equity_fraction': (0., 1., .5),
    'quantile': (.05, 1., .8), 'payout_interval': (1., 12., 1.),
    'delivery_scale': (.8, 1.2, 1.),
}
_KEYS = {
    'cash_only': (),
    'cash_bond': ('capital_ratio', 'coupon', 'payout_interval'),
    'calendar_payout': ('delivery_scale',),
    'vanilla_call': ('strike', 'participation'),
    'vanilla_put': ('strike', 'participation'),
    'digital_call': ('strike', 'participation'),
    'down_and_out_put': ('strike', 'barrier', 'participation'),
    'geometric_asian_call': ('strike', 'participation'),
    'arithmetic_asian_call': ('strike', 'participation'),
    'lookback_call': ('participation',),
    'autocallable': ('ko', 'coupon', 'principal'),
    'snowball': ('ko', 'ki', 'coupon', 'principal'),
    'cliquet': ('floor', 'cap', 'participation', 'principal'),
    'range_accrual': ('lower', 'upper', 'coupon', 'principal'),
    'call_spread': ('strike', 'width', 'participation', 'principal'),
    'two_way': ('strike', 'participation', 'principal'),
    'obpi': ('strike', 'participation', 'guarantee'),
    'ratchet': ('participation', 'principal'),
    'highwater_protection': ('guarantee', 'participation'),
    'quantile_hedge': ('quantile', 'principal'),
    'average_trigger': ('strike', 'participation'),
}
for _name in ('cppi', 'tipp', 'drawdown_cushion', 'periodic_payout'):
    _KEYS[_name] = ('multiplier', 'guarantee', 'coupon', 'capital_ratio',
                     'equity_fraction', 'payout_interval')


def parameter_boundaries():
    return {f: {k: {'lower': _SPECS[k][0], 'upper': _SPECS[k][1],
                    'default': 0. if f == 'cash_bond' and k == 'coupon' else _SPECS[k][2],
                    'integer': k == 'payout_interval'}
                for k in _KEYS[f]} for f in FAMILY_NAMES}


def _parameters(family, params):
    if family not in FAMILY_NAMES:
        raise ValueError(f'unknown family: {family}')
    unknown = set(params) - set(_KEYS[family])
    if unknown:
        raise ValueError(f'unknown parameters for {family}: {sorted(unknown)}')
    result = {}
    for key in _KEYS[family]:
        lo, hi, default = _SPECS[key]
        if family == 'cash_bond' and key == 'coupon':
            default = 0.
        x = float(params.get(key, default))
        if not math.isfinite(x) or not lo <= x <= hi:
            raise ValueError(f'{key} must be finite and in [{lo}, {hi}]')
        if key == 'payout_interval':
            if x != int(x):
                raise ValueError('payout_interval must be an integer')
            x = int(x)
        result[key] = x
    return result


def _strategy(tree, family, p, notional):
    """Stock/cash ledger; coupons are deducted before each rebalance."""
    s = np.asarray(tree.paths, dtype=float)
    n = int(tree.months)
    growth = math.exp(float(tree.rate) * float(tree.dt))
    dividend_growth = math.exp(float(getattr(tree, 'config', {}).get('dividend', 0.)) * float(tree.dt))
    wealth = np.full(len(s), notional * p['capital_ratio'])
    high = wealth.copy()
    paid_value = np.zeros(len(s))
    table = np.zeros_like(s)
    stock = np.zeros(len(s))
    cash = wealth.copy()
    error = min_cash = min_stock = 0.
    for t in range(n + 1):
        if t:
            marked = stock * s[:, t] / s[:, t - 1] * dividend_growth + cash * growth
            paid_value *= growth
            wealth = marked
            high = np.maximum(high, wealth + paid_value)
            if t == n:
                table[:, t] = wealth
                wealth = np.zeros_like(wealth)
            elif t % p['payout_interval'] == 0:
                table[:, t] = np.minimum(p['coupon'] * notional, wealth)
                wealth = wealth - table[:, t]
                paid_value += table[:, t]
            error = max(error, float(np.max(np.abs(marked - wealth - table[:, t]))))
        if t == n:
            break
        base_floor = p['guarantee'] * notional * p['capital_ratio'] * math.exp(
            -float(tree.rate) * float(tree.dt) * (n - t))
        # A withdrawal must reduce the protected remaining-account amount.
        floor = np.maximum(0., base_floor - paid_value)
        if family in ('tipp', 'drawdown_cushion'):
            floor = np.maximum(floor, p['guarantee'] * high - paid_value)
        if family == 'periodic_payout':
            desired = p['equity_fraction'] * wealth
        else:
            desired = p['multiplier'] * np.maximum(wealth - floor, 0.)
        stock = np.minimum(wealth, np.maximum(desired, 0.))
        cash = wealth - stock
        min_cash = min(min_cash, float(cash.min()))
        min_stock = min(min_stock, float(stock.min()))
    return table, {'initial_strategy_wealth': notional * p['capital_ratio'],
                   'maximum_self_financing_error': error,
                   'minimum_cash_holding': min_cash,
                   'minimum_stock_value': min_stock,
                   'coupon_source': 'withdrawal from strategy wealth',
                   'stock_dividends': 'reinvested at the declared tree yield',
                   'leverage': False,
                   'gap_risk': 'monthly floor can be breached between rebalances'}


def family_candidate(tree, family, params, notional=100_000., *, demand=None):
    """Build one adapted cashflow table and price it on exactly tree.Q."""
    notional = float(notional)
    if not math.isfinite(notional) or notional <= 0:
        raise ValueError('notional must be positive and finite')
    p = _parameters(family, params)
    s = np.asarray(tree.paths, dtype=float)
    if s.ndim != 2 or s.shape[1] != int(tree.months) + 1 or np.any(s <= 0):
        raise ValueError('tree.paths must contain positive full monthly spot paths')
    s = s / s[:, :1]
    n = int(tree.months)
    table = np.zeros_like(s)
    terminal = s[:, -1]
    scope = 'monthly discrete payoff; exact Q price on this finite tree'
    diagnostics = None
    calendar_diagnostics = None
    if family == 'cash_only':
        scope = 'no purchased contract; evaluator determines operational cash reserve under the declared funding mode'
    elif family == 'calendar_payout':
        # Literal public task liabilities are an ordinary known control.
        # Client event scenarios never enter the payment information set.
        withdrawals = _field(demand, 'withdrawal_ratios', {t: .01 for t in range(1, n)})
        target_ratio = float(_field(demand, 'terminal_total_ratio', 1.))
        if not math.isfinite(target_ratio) or target_ratio < 0:
            raise ValueError('terminal_total_ratio must be finite and nonnegative')
        withdrawal_total = 0.
        for date, ratio in withdrawals.items():
            t, amount = int(date), float(ratio) * notional
            if t != float(date) or not 1 <= t <= n or not math.isfinite(amount) or amount < 0:
                raise ValueError('calendar liabilities must be nonnegative and on the monthly grid')
            table[:, t] += amount
            withdrawal_total += amount
        target = notional * target_ratio
        terminal_remaining = max(0., target - withdrawal_total)
        table[:, -1] += terminal_remaining
        table *= p['delivery_scale']
        calendar_diagnostics = {'public_withdrawal_total': withdrawal_total,
                                'target_total_delivery': target,
                                'additional_terminal_payment': terminal_remaining,
                                'withdrawals_exceed_target_delivery': withdrawal_total > target,
                                'payment_information': 'public dated requirements; no client-event conditioning'}
        scope = 'deterministic public liability mirror; ordinary known control, not a theory solver oracle'
    elif family == 'cash_bond':
        wealth = notional * p['capital_ratio']
        for t in range(1, n + 1):
            wealth *= math.exp(float(tree.rate) * float(tree.dt))
            payment = (wealth if t == n else min(p['coupon'] * notional, wealth)
                       if t % p['payout_interval'] == 0 else 0.)
            table[:, t] = payment
            wealth -= payment
        scope = 'deterministic cash/bond account; dated coupons withdrawn from funded account'
    elif family in ('cppi', 'tipp', 'drawdown_cushion', 'periodic_payout'):
        table, diagnostics = _strategy(tree, family, p, notional)
        if family == 'drawdown_cushion':
            scope = ('monthly drawdown-cushion discrete approximation; NOT the true '
                     'Grossman-Zhou continuous-time optimal strategy')
    else:
        if family in ('vanilla_call', 'vanilla_put', 'digital_call', 'down_and_out_put'):
            payoff = np.maximum(terminal - p['strike'], 0.)
            if family in ('vanilla_put', 'down_and_out_put'):
                payoff = np.maximum(p['strike'] - terminal, 0.)
            if family == 'digital_call':
                payoff = (terminal > p['strike']).astype(float)
            if family == 'down_and_out_put':
                payoff *= np.min(s, axis=1) > p['barrier']
            payoff *= p['participation']
        elif family in ('geometric_asian_call', 'arithmetic_asian_call', 'average_trigger'):
            avg = (np.exp(np.mean(np.log(s[:, 1:]), axis=1))
                   if family == 'geometric_asian_call' else np.mean(s[:, 1:], axis=1))
            payoff = ((avg > p['strike']).astype(float) if family == 'average_trigger'
                      else np.maximum(avg - p['strike'], 0.)) * p['participation']
        elif family == 'lookback_call':
            payoff = p['participation'] * (terminal - s.min(axis=1))
        elif family in ('autocallable', 'snowball'):
            alive = np.ones(len(s), dtype=bool)
            for t in range(1, n):
                knocked_out = alive & (s[:, t] > p['ko'])
                table[knocked_out, t] = notional * (
                    p['principal'] + p['coupon'] * t)
                alive[knocked_out] = False
            if family == 'autocallable':
                payoff = alive * p['principal']
            else:
                knocked_in = s.min(axis=1) <= p['ki']
                # Loss participation is bounded by funded principal.
                redemption = p['principal'] * np.minimum(terminal, 1.)
                payoff = alive * np.where(knocked_in, redemption,
                                          p['principal'] + p['coupon'] * n)
        elif family == 'cliquet':
            returns = s[:, 1:] / s[:, :-1] - 1.
            payoff = p['principal'] + p['participation'] * np.clip(returns, p['floor'], p['cap']).sum(axis=1)
        elif family == 'range_accrual':
            in_range = (s[:, 1:] >= p['lower']) & (s[:, 1:] <= p['upper'])
            payoff = p['principal'] + p['coupon'] * in_range.sum(axis=1)
        elif family == 'call_spread':
            payoff = p['principal'] + p['participation'] * (
                np.maximum(terminal - p['strike'], 0.) -
                np.maximum(terminal - p['strike'] - p['width'], 0.))
        elif family == 'two_way':
            payoff = p['principal'] + p['participation'] * np.abs(terminal - p['strike'])
        elif family == 'obpi':
            payoff = p['guarantee'] + p['participation'] * np.maximum(terminal - p['strike'], 0.)
        elif family == 'ratchet':
            increments = np.maximum(s[:, 1:] / s[:, :-1] - 1., 0.)
            payoff = p['principal'] + p['participation'] * increments.sum(axis=1)
        elif family == 'highwater_protection':
            payoff = np.maximum(p['guarantee'] * s.max(axis=1),
                                1. + p['participation'] * (terminal - 1.))
        elif family == 'quantile_hedge':
            # Terminal-only monotone quantile gate. It is a baseline, not a
            # Neyman-Pearson optimum for arbitrary path or withdrawal demand.
            weights = np.asarray(tree.p_leaf, dtype=float)
            levels, inv = np.unique(np.round(terminal, 12), return_inverse=True)
            masses = np.bincount(inv, weights=weights)
            survival = np.cumsum(masses[::-1])[::-1]
            eligible = np.flatnonzero(survival >= p['quantile'] - 1e-12)
            cutoff = levels[eligible[-1]] if len(eligible) else levels[0]
            payoff = p['principal'] * (np.round(terminal, 12) >= cutoff)
        else:  # all branches enumerated above
            raise ValueError(family)
        table[:, -1] = notional * payoff
    if np.any(~np.isfinite(table)) or np.any(table < -1e-10):
        raise ValueError('family produced an invalid cashflow')
    tree.node_cashflows(table)
    row = {'family': family, 'params': p, 'notional': notional,
           'table': table, 'price': float(tree.price(table)),
           'definition_scope': scope, 'strategy_diagnostics': diagnostics,
           'calendar_diagnostics': calendar_diagnostics}
    return row


def _field(demand, name, default):
    return demand.get(name, default) if isinstance(demand, Mapping) else getattr(demand, name, default)


def _metric_rank(row):
    m = row['metrics']
    return (-float(m.get('satisfaction_probability') or 0.),
            float(m['drawdown_loss']) if m.get('drawdown_loss') is not None else math.inf,
            float(m['cost']) if m.get('cost') is not None else math.inf)


def _pareto(rows):
    return [row for row in rows if not any(
        all(b <= a for b, a in zip(_metric_rank(other), _metric_rank(row))) and
        any(b < a for b, a in zip(_metric_rank(other), _metric_rank(row)))
        for other in rows)]


def optimize_known_families(tree, demand, evaluate, budget, seed=0):
    """Balanced incumbent-centred random/coordinate search; exactly budget calls.

    No theory answer is used. Each family receives a default and a bounded
    numeric proposal before additional rounds mutate its current incumbent.
    This is finite local search, not a global optimum or blind-grid claim.
    """
    if isinstance(budget, Mapping):
        budget = budget['max_evaluations']
    if isinstance(budget, bool) or int(budget) != budget:
        raise ValueError('budget must be an integer')
    budget = int(budget)
    if budget < 2 * len(FAMILY_NAMES):
        raise ValueError('budget needs at least two evaluations per family')
    rng = np.random.default_rng(seed)
    notional = float(_field(demand, 'service_notional', _field(demand, 'budget', 100_000.)))
    funding_mode = _field(demand, 'funding_mode', 'minimum_service_capital')
    bounds = parameter_boundaries()
    attempts = []
    incumbents = {f: None for f in FAMILY_NAMES}
    best_by_family = {f: None for f in FAMILY_NAMES}
    for index in range(budget):
        family = FAMILY_NAMES[index % len(FAMILY_NAMES)]
        iteration = index // len(FAMILY_NAMES)
        params = {k: v['default'] for k, v in bounds[family].items()}
        if iteration == 0 and family == 'cash_bond':
            scenarios = _field(demand, 'exit_scenarios', ({'month': None},))
            no_withdrawals = (not any(_field(demand, 'withdrawal_ratios', {}).values()) and
                              not any(any(s.get('withdrawal_ratios', {}).values()) for s in scenarios))
            no_exit = all(s.get('month') is None for s in scenarios)
            carry_gain = math.expm1(float(tree.rate) * float(tree.dt) * int(tree.months))
            desired_gain = float(_field(demand, 'terminal_total_ratio', 1.)) - 1.
            if no_withdrawals and no_exit:
                # Ordinary closed-form bond sizing using only public demand
                # and bank rate. No theoretical solver output is consulted.
                if funding_mode == 'minimum_service_capital':
                    delivery = float(_field(demand, 'terminal_total_ratio', 1.))
                    params['capital_ratio'] = float(np.clip(delivery / (1. + carry_gain), 0., 1.))
                elif carry_gain > 0:
                    params['capital_ratio'] = float(np.clip(desired_gain / carry_gain, 0., 1.))
        if iteration == 1:
            for key, spec in bounds[family].items():
                params[key] = (int(rng.integers(spec['lower'], spec['upper'] + 1))
                               if spec['integer'] else float(rng.uniform(spec['lower'], spec['upper'])))
        elif iteration > 1 and params:
            params = incumbents[family]['params'].copy()
            key = list(params)[(iteration - 2) % len(params)]
            spec = bounds[family][key]
            span = (spec['upper'] - spec['lower']) * .5 ** min(iteration - 1, 5)
            value = np.clip(params[key] + rng.uniform(-span, span), spec['lower'], spec['upper'])
            params[key] = int(round(value)) if spec['integer'] else float(value)
        candidate = family_candidate(tree, family, params, notional=notional, demand=demand)
        metrics = dict(evaluate(candidate['table']))
        for key in ('cost', 'satisfaction_probability', 'drawdown_loss', 'feasible'):
            if key not in metrics:
                raise ValueError(f'evaluate must return {key}')
        rejected = metrics.get('status', 'ok') in ('rejected', 'error', 'invalid')
        numeric = [float(metrics[k]) for k in ('cost', 'satisfaction_probability', 'drawdown_loss')
                   if metrics[k] is not None]
        if not all(math.isfinite(x) for x in numeric):
            raise ValueError('evaluation metrics must be finite')
        if not rejected and any(metrics[k] is None for k in ('cost', 'satisfaction_probability', 'drawdown_loss')):
            raise ValueError('nonrejected evaluation metrics must be numeric')
        if (metrics['satisfaction_probability'] is not None and not 0 <= metrics['satisfaction_probability'] <= 1
                or any(metrics[k] is not None and metrics[k] < 0 for k in ('cost', 'drawdown_loss'))):
            raise ValueError('invalid evaluation metrics')
        row = {**candidate, 'table': candidate['table'].tolist(), 'metrics': metrics,
               'evaluation': index + 1, 'eligible': bool(metrics['feasible']) and not rejected}
        attempts.append(row)
        previous = incumbents[family]
        if previous is None or (not row['eligible'], _metric_rank(row)) < (not previous['eligible'], _metric_rank(previous)):
            incumbents[family] = row
        if row['eligible'] and (best_by_family[family] is None or _metric_rank(row) < _metric_rank(best_by_family[family])):
            best_by_family[family] = row
    eligible = [r for r in attempts if r['eligible']]
    valid = [r for r in attempts if r['metrics'].get('status', 'ok') not in
             ('rejected', 'error', 'invalid') and all(r['metrics'][k] is not None
             for k in ('cost', 'satisfaction_probability', 'drawdown_loss'))]
    target = float(_field(demand, 'min_success_probability', .95))
    meeting = [r for r in eligible if r['metrics']['satisfaction_probability'] >= target - 1e-12]
    return {'budget': budget, 'evaluations': len(attempts), 'seed': seed,
            'notional': notional, 'attempts': attempts,
            'funding_mode': funding_mode,
            'cost_definition': 'Q contract purchase cost + minimum operational cash reserve'
                if funding_mode == 'minimum_service_capital' else 'Q contract purchase cost; fixed service budget reserve control',
            'best': min(eligible, key=_metric_rank) if eligible else None,
            'best_by_family': best_by_family, 'pareto_front': _pareto(eligible),
            'pareto_front_all_valid': _pareto(valid),
            'best_cost_meeting_target': min(meeting, key=lambda r: r['metrics']['cost']) if meeting else None,
            'target_success_probability': target,
            'parameter_boundaries': bounds, 'feed_to_generator': False,
            'scope': 'best observed within stated parameter boundaries and finite budget; not global optimum',
            'primary_endpoint': 'distance to the same discrete-model theory frontier',
            'unmatched_catalog_status': 'provisional unfamiliarity to a finite catalog, not financial novelty'}
