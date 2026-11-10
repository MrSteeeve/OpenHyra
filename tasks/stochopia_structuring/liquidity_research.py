"""Development/holdout diagnostics for common equity liquidity mechanisms."""
from copy import deepcopy
import json
import numpy as np

from stochopia.research.liquidity_market import generate_panel, reference_metrics
from stochopia.research.liquidity_execution import execute_strategy, contract_cashflows

GOAL = 'equity-liquidity-mechanism-discovery.v1'
PAIRS = ('spread', 'depth', 'recovery', 'common')
SCENARIOS = ('normal', *(f'{k}_{s}' for k in PAIRS for s in ('bad', 'good')),
             'price_up', 'price_down', 'vol_high', 'vol_low')


def pilot_config(direction='bearish'):
    if direction not in ('bearish', 'bullish'):
        raise ValueError('direction must be bearish or bullish')
    return {'schema': GOAL, 'phase': 'diagnostic-pilot', 'direction': direction,
        'prompt_revision': '2026-10-13-r2',
        'market': {'days': 20, 'steps_per_day': 78, 'n_stocks': 10, 'rate': 0., 'drift': 0., 'sigma': .2, 'depth': .025},
        'risk': {'buyer_capital': 1_000_000., 'seller_capital': 1_000_000.,
                 'es_fraction': .10, 'stress_fraction': .20, 'fee_bps': 1.,
                 'borrow_rate': .03, 'gross_limit': 2., 'risk_fee': .10},
        'paths': {'development': 64, 'quote': 128, 'holdout': 256},
        'seeds': {'development': 20261013, 'quote': 771013, 'holdout': 881013},
        'rounds': 2, 'evaluations_per_proposal': 8, 'evaluation_budget_per_arm': 16,
        'model': 'deepseek-v4.1-flash', 'max_tokens': 393216,
        'thresholds': {'response_capital_fraction': .001, 'coverage': .8, 'nuisance': .25},
        'view_weights': {'common_bad': .5 if direction=='bearish' else .1,
                         'normal': .4, 'common_good': .1 if direction=='bearish' else .5},
        'decisions': 'User approved capital, ES/stress limits, 8 API calls, mean+0.1 residual ES95 quote',
        'scope': 'Synthetic reduced-form 5-minute market; market-order strategies; diagnostic, not a statistical discovery claim'}


def expected_shortfall(loss, alpha=.95):
    """Empirical tail mean, including fractional mass at the tail boundary."""
    values = np.sort(np.asarray(loss, dtype=float))[::-1]
    mass = len(values) * (1-alpha)
    count = int(np.floor(mass))
    return float((values[:count].sum() + (mass-count)*values[min(count, len(values)-1)]) / mass)


def parse_liquidity_generation(text):
    raw = text.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
    artifact = json.loads(raw)
    if artifact['track'] not in ('synthetic', 'contract'):
        raise ValueError('track must be synthetic or contract')
    params = artifact.get('params', {})
    if not isinstance(params, dict) or any(type(v) not in (int,float) or not np.isfinite(v) for v in params.values()):
        raise ValueError('params must be an object of finite numeric scalars')
    if artifact['track']=='synthetic':
        if not isinstance(artifact['policy']['targets'], dict):
            raise ValueError('policy targets must be an object')
    else:
        if not isinstance(artifact['contract_sexpr'], str) or float(artifact['notional'])<=0:
            raise ValueError('contract text and positive notional required')
    json.dumps(artifact, allow_nan=False)
    return artifact


def liquidity_prompt(request):
    steps = request['steps']
    return f"""Design one executable research mechanism to trade COMMON equity liquidity.
The primary goal is a distinct economically useful mechanism, not maximal raw return,
program length, syntax novelty or leverage. Liquidity deterioration should increase
net P&L for bearish exposure; improvement should increase P&L for bullish exposure.
The evaluator separately tests spread, depth, recovery and common shocks, and tests
price/volatility changes with liquidity fixed. Follow the track, capital and fees.
Return ONE JSON object with track, rationale and the following track-specific fields.
Track synthetic: policy={{"targets":{{instrument: "S-expression", ...}},
"rebalance_every": integer}}, params={{numeric scalars}}. Targets are fractions of
INITIAL buyer capital; short positions are allowed under the supplied capital limits.
Tradable instrument names: basket, call, put, s0..s9. basket is the equal-weight equity
ETF proxy; call/put are its synthetic ATM European options expiring at the horizon.
They have bid/ask spreads and capacity limits. No limit orders. Terminal liquidation
is an actual capacity-limited trade. Express each target with existing DSL expressions:
numbers, (spot basket), (spot s0), (external spread), (external depth),
(external recovery), (external liquidity), (external spread_s0), (external depth_s0),
(lin a x b), (clip x lo hi), (step x level [up/down]), (smooth_step x level width),
(add x y), (mul_const k x), (mul x y), (min x y), (max x y),
(avg_arith x daily), (running_min x), (running_max x), (count_above x level daily),
(occupation x level up daily). Other published observable names are in the request.
Policy expressions must use only current/past data: no terminal, ret, hit_time,
var_clock, discretionary exercise, arbitrary Python or new grammar. A constant target
is a valid expression. Position targets at the last step are forced to zero.
Track contract: contract_sexpr (full contract), params (numeric), notional (currency),
and OPTIONAL hedge_policy in the same policy format. The hedge policy is an executable
seller hedge proposal. It is fitted/quoted on independent pricing paths, never holdout.
Use the SAME existing StructurerDSL: (contract (meta (underlyings basket)
(maturity {steps}) (schedules)) (premium body)).
The built-in daily schedule means EVERY 5-minute step, not calendar days. Avoid spelling
out the full grid; use daily directly in functionals/coupon_on. Never put daily inside
meta schedules: (schedules (obs daily)) is INVALID. Named schedules must contain
integer grid indices, e.g. (schedules (obs 78 156)). Only declared spot names may be referenced. Existing
external series can condition payments. Allowed bodies: pay, both, give, scale, when,
coupon_on, until, activate_if, terminate_if. (pay expression maturity) pays at expiry;
(coupon_on daily expression) pays each step. (at obs i) is zero-based schedule index.
Expressions also allow terminal, avg, extrema, count, occupation, ret, pl and clip.
ret is a RATIO with EXACTLY three arguments: (ret (spot basket) (at obs 0) (at obs 1)).
Subtract 1 for a return. (ret (spot basket)) is INVALID. The current DSL has no rolling
previous-step return operator. Do not invent it. Whole-program depth <=12, nodes <=64;
use flat multi-argument add/mul rather than deeply nested binary add.
Clip external data BEFORE aggregation when necessary for finite bounds. clip alone
does not create novelty. No funded principal needed: use premium wrapper, price is
set by evaluator, and negative future payments require funded cash. t=0 payoff is zero.
Cashflow expressions are fractions: external notional multiplies them ONCE. Do not
multiply by the currency notional again inside the expression. A dimensionless 0.001
pay expression means 0.1% of notional, not 0.001 currency units.
The observations are FULL relative spread (initial approx .001), depth in units
of executable stock notional / INITIAL capital / step, recovery fraction (larger =
faster recovery, hence BETTER liquidity), liquidity (public auxiliary DIFFICULTY
index: HIGHER = WORSE, NOT higher quality, and NOT the ground-truth objective).
liquidity=(spread/initial_spread + initial_depth/depth + initial_recovery/recovery)/3.
These are observables, not tradables. Price and liquidity innovations in this synthetic
pilot are independent. Pure spread/depth/recovery/common interventions keep stock
paths and option midprice paths IDENTICAL within pairs. Never assume that illiquidity
automatically makes equities fall or implied volatility rise. Pure price/volatility
interventions conversely keep liquidity observables fixed. Contract design may use
joint conditions, but price co-movement is not donated to the liquidity task.
Quotation = discounted mean cashflows + 0.1 * residual seller ES95 about mean; the
evaluator compares no hedge, fitted static basket, fitted basket+put, and your optional
hedge policy, choosing the lowest feasible quote. Residual risk and funding are gates.
Known swaps, calls/puts, Asians, barriers, occupation/range accrual, price proxies and
reversal rules are reference families; no one reference is a generation template.
Changing parameters or names is not a new mechanism. Explain the concrete behavior,
why it isolates liquidity, costs, risks, and what ablation would remove its advantage.
Use development history only when present. Output a complete artifact, not a patch.
"""


class LiquidityEvaluator:
    def __init__(self, config=None, phase='development'):
        self.config = deepcopy(config or pilot_config())
        self.phase = phase
        cfg = self.config
        self.panels = {s: generate_panel(seed=cfg['seeds'][phase], n_paths=cfg['paths'][phase],
                          config=cfg['market'], scenario=s) for s in SCENARIOS}
        self.quote_panel = generate_panel(seed=cfg['seeds']['quote'], n_paths=cfg['paths']['quote'],
                                         config=cfg['market'], scenario='normal')
        # Labels come from fixed execution-service probes, not from the public
        # auxiliary liquidity index or a candidate's own trades.
        self.reference = {s: reference_metrics(self.panels[s], cfg['risk']['buyer_capital'])
                          for s in SCENARIOS if s.endswith(('_bad','_good'))}

    def reference_labels(self, name):
        bad, good = self.reference[f'{name}_bad'], self.reference[f'{name}_good']
        cost = bad['cost'][:,0,0] >= 1.25*good['cost'][:,0,0]
        depth = bad['unfinished'][:,2,0]-good['unfinished'][:,2,0] >= .25
        recovery = bad['recovery_half_life'] >= 1.25*good['recovery_half_life']
        return {'spread':cost, 'depth':depth, 'recovery':recovery,
                'common':cost & depth & recovery}[name]

    def reference_report(self):
        return {'quantity_unit':'fraction of initial reference capital, not ADV',
            'deadlines_minutes':[5,30,120], 'quantity_fractions':[.01,.1,.5],
            'measurement':'Frozen-state symmetric execution probes; no candidate influence',
            'severity':'cost +25%, unserved fraction +25 percentage points, recovery half-life +25%',
            'labelled_pairs': {name:int(self.reference_labels(name).sum()) for name in PAIRS},
            'scenarios': {s:{'mean_cost':r['mean_cost'].tolist(),
                'mean_unfinished':r['mean_unfinished'].tolist(),
                'mean_recovery_half_life':float(r['recovery_half_life'].mean())} for s,r in self.reference.items()}}

    def public_request(self, track, mode, round_index, feedback):
        cfg = self.config
        p = self.panels['normal']
        external = p.paths.external
        return {'research_goal': GOAL, 'prompt_revision': cfg['prompt_revision'],
            'track': track, 'search_mode': mode, 'round_index': round_index,
            'steps': p.paths.n_steps, 'market': cfg['market'], 'risk': cfg['risk'],
            'direction': cfg['direction'], 'thresholds': cfg['thresholds'],
            'root_observables': {name: float(series[0, 0]) for name, series in external.items()},
            'search_objective': 'Discover distinct feasible mechanisms with coverage/cross-risk/cost advantages.',
            'feedback': feedback if mode == 'feedback' else [],
            'evaluation_budget': cfg['evaluation_budget_per_arm'],
            'quote_scope': 'Finite hedge collection, synthetic reference measure, no calibrated market quote',
            'sampling': 'Model and public assumptions only; no sampled paths or holdout results supplied'}

    def hedge_config(self):
        cfg = dict(self.config['risk'])
        cfg['market_capital'] = cfg['buyer_capital']
        cfg['buyer_capital'] = cfg['seller_capital']
        cfg['trace'] = True
        return cfg

    def contract_funding_failure(self, flows, price, hedge):
        cfg = self.config['risk']
        paid = np.cumsum(flows,axis=1)
        buyer_cash = cfg['buyer_capital']-price+paid
        seller_draw = hedge['trace']['funding']+paid-price
        return ((buyer_cash.min(1)<-1e-6) | (seller_draw.max(1)>cfg['seller_capital']+1e-6)
                | hedge['funding_violation'] | hedge['unfinished'])

    def _static_hedges(self, artifact, fair, liability):
        p = self.quote_panel
        cfg = self.hedge_config()
        base = .05
        policies = [('none', {'targets': {}, 'rebalance_every': p.paths.n_steps})]
        gains = []
        for instrument in ('basket', 'put'):
            result = execute_strategy({'track': 'synthetic', 'policy': {'targets': {instrument: str(base)},
                                      'rebalance_every': p.paths.n_steps}}, p, cfg)
            gains.append(result['pnl'])
        for k, label in ((1, 'static-basket'), (2, 'static-basket-put')):
            x = np.column_stack(gains[:k])
            # Fit only independent quotation samples; cap ex-ante position size.
            coef = np.linalg.lstsq(x-x.mean(0), liability-fair, rcond=None)[0]
            weights = np.clip(coef*base, -.5, .5)
            policies.append((label, {'targets': {name: str(float(w)) for name, w in
                zip(('basket', 'put'), weights)}, 'rebalance_every': p.paths.n_steps}))
        if artifact.get('hedge_policy'):
            policies.append(('proposed-hedge', artifact['hedge_policy']))
        return policies

    def quote(self, artifact, risk_fee=None):
        p, cfg = self.quote_panel, self.config['risk']
        coefficient = cfg['risk_fee'] if risk_fee is None else risk_fee
        table = contract_cashflows(artifact, p)
        discount = np.exp(-p.paths.rate*p.paths.t_years.detach().numpy())
        liability = table @ discount
        fair = float(liability.mean())
        trials = []
        for label, policy in self._static_hedges(artifact, fair, liability):
            result = execute_strategy({'track': 'synthetic', 'policy': policy, 'params': artifact.get('params', {})}, p, self.hedge_config())
            residual = liability-fair-result['pnl']
            risk_base = max(0., expected_shortfall(residual))
            price = fair + coefficient*risk_base
            seller_loss = liability-price-result['pnl']
            feasible = (not np.any(self.contract_funding_failure(table, price, result))
                and expected_shortfall(seller_loss) <= cfg['seller_capital']*cfg['es_fraction']
                and price <= cfg['buyer_capital'])
            trials.append({'label': label, 'policy': policy, 'price': price,
                'fair_mean': fair, 'risk_base_es95': risk_base,
                'risk_charge': coefficient*risk_base, 'risk_fee': coefficient, 'seller_es95': expected_shortfall(seller_loss),
                'hedge_cost_mean': float(np.mean(result['costs'])), 'feasible': bool(feasible)})
        available = [q for q in trials if q['feasible']]
        selected = min(available or trials, key=lambda q: q['price'])
        return {**selected, 'all_hedge_trials': trials, 'quote_feasible': bool(available),
                'pricing_paths': len(liability), 'pricing_seed': self.config['seeds']['quote']}

    def evaluate(self, artifact, quote=None, *, keep_behavior=False):
        cfg = self.config['risk']
        capital = cfg['buyer_capital']
        track = artifact['track']
        if track not in ('synthetic', 'contract'):
            raise ValueError('unknown track')
        if track == 'contract':
            quote = quote or self.quote(artifact)
        outcomes = {}
        for scenario, panel in self.panels.items():
            if track == 'synthetic':
                execution = execute_strategy(artifact, panel, cfg)
                pnl = execution['pnl']
                seller_loss = np.zeros_like(pnl)
                violation = execution['funding_violation'] | execution['unfinished']
                costs = execution['costs']
            else:
                flows = contract_cashflows(artifact, panel)
                discount = np.exp(-panel.paths.rate*panel.paths.t_years.detach().numpy())
                total = flows @ discount
                pnl = total-quote['price']
                hedge = execute_strategy({'track': 'synthetic', 'policy': quote['policy'], 'params': artifact.get('params', {})}, panel, self.hedge_config())
                seller_loss = total-quote['price']-hedge['pnl']
                violation = self.contract_funding_failure(flows, quote['price'], hedge)
                costs = np.full(len(pnl), quote['risk_charge']) + hedge['costs']
            outcomes[scenario] = {'pnl': pnl, 'seller_loss': seller_loss, 'violation': violation, 'costs': costs}
        sign = 1. if self.config['direction'] == 'bearish' else -1.
        threshold = capital*self.config['thresholds']['response_capital_fraction']
        response, coverage, response_se = {}, {}, {}
        for name in PAIRS:
            differences = sign*(outcomes[f'{name}_bad']['pnl']-outcomes[f'{name}_good']['pnl'])
            labels = self.reference_labels(name)
            if not labels.any():
                raise ValueError('no execution-probe-labelled pairs for '+name)
            differences = differences[labels]
            response[name] = float(differences.mean())
            response_se[name] = float(differences.std(ddof=1)/np.sqrt(len(differences)))
            coverage[name] = float(np.mean(differences>threshold))
        target_size = max(1e-9, min(response.values()))
        nuisance = {}
        for name, a, b in (('price', 'price_up', 'price_down'), ('volatility', 'vol_high', 'vol_low')):
            nuisance[name] = float(np.mean(np.abs(outcomes[a]['pnl']-outcomes[b]['pnl']))/target_size)
        normal = outcomes['normal']
        view_mean = sum(w*float(outcomes[s]['pnl'].mean()) for s,w in self.config['view_weights'].items())
        buyer_es = expected_shortfall(-normal['pnl'])
        seller_es = expected_shortfall(normal['seller_loss'])
        buyer_stress = max(float(np.max(-o['pnl'])) for o in outcomes.values())
        seller_stress = max(float(np.max(o['seller_loss'])) for o in outcomes.values())
        gates = {'buyer_es': buyer_es<=capital*cfg['es_fraction'],
                 'seller_es': seller_es<=cfg['seller_capital']*cfg['es_fraction'],
                 'buyer_stress': buyer_stress<=capital*cfg['stress_fraction'],
                 'seller_stress': seller_stress<=cfg['seller_capital']*cfg['stress_fraction'],
                 'funding': not any(np.any(o['violation']) for o in outcomes.values()),
                 'quote': track=='synthetic' or quote['quote_feasible'],
                 'direction': min(coverage.values())>=self.config['thresholds']['coverage'],
                 'nuisance': max(nuisance.values())<=self.config['thresholds']['nuisance'],
                 'net_view_gain': view_mean>threshold}
        metrics = {'status': 'ok', 'phase': self.phase, 'track': track,
            'response_means': response, 'response_se': response_se, 'coverage_by_factor': coverage,
            'coverage': min(coverage.values()), 'nuisance_by_factor': nuisance, 'nuisance': max(nuisance.values()),
            'cost': float(normal['costs'].mean()), 'view_net_mean': view_mean,
            'normal_net_mean': float(normal['pnl'].mean()), 'buyer_es95': buyer_es, 'seller_es95': seller_es,
            'buyer_stress_loss': buyer_stress, 'seller_stress_loss': seller_stress,
            'funding_failure_paths': {s: int(o['violation'].sum()) for s,o in outcomes.items()},
            'gates': {k: bool(v) for k,v in gates.items()}, 'feasible': bool(all(gates.values())), 'quote': quote,
            'scenario_net_means': {s: float(o['pnl'].mean()) for s,o in outcomes.items()}}
        behavior = np.concatenate([outcomes[s]['pnl'] for s in SCENARIOS])
        if keep_behavior:
            return metrics, behavior
        return metrics


def scaled_variant(artifact, index):
    """Same local parameter optimizer for both arms and baseline families."""
    value = deepcopy(artifact)
    scales = (.25, .5, .75, 1., 1.25, 1.5, 1.75, 2.)
    factor = scales[index]
    # A transparent local parameter perturbation, not a claimed global optimizer.
    # Identical proposal and catalog routines; every evaluation is charged.
    params = value.get('params', {})
    if not isinstance(params, dict):
        raise ValueError('params must be an object')
    for key, val in params.items():
        value['params'][key] = float(val)*(0.9 if index%2==0 else 1.1)
    if value['track'] == 'contract':
        value['notional'] *= factor
    else:
        targets = value['policy']['targets']
        value['policy']['targets'] = {k: f'(mul_const {factor} {v})' for k,v in targets.items()}
    return value


def known_families(track, steps):
    def policy(name, targets, every=78):
        return {'family': name, 'artifact': {'track': 'synthetic', 'policy': {
            'targets': targets, 'rebalance_every': every}, 'params': {}}}
    if track == 'synthetic':
        return [policy('cash', {}), policy('basket-long', {'basket': '.1'}, steps),
            policy('basket-short', {'basket': '-.1'}, steps), policy('protective-put', {'put': '.01'}, steps),
            policy('straddle', {'put': '.01', 'call': '.01'}, steps),
            policy('reversal', {'basket': '(clip (lin -2 (spot basket) 2) -.1 .1)'}),
            policy('trend', {'basket': '(clip (lin 2 (spot basket) -2) -.1 .1)'}),
            policy('liquidity-threshold', {'basket': '(lin -.1 (step (external liquidity) 1 up) .05)'})]
    l = '(external liquidity)'
    expressions = [('liquidity-exchange', f'(clip (lin 1 {l} -1) -.2 .2)', {}),
        ('liquidity-call', f'(clip (lin 1 {l} -1) 0 .2)', {}),
        ('liquidity-put', f'(clip (lin -1 {l} 1) 0 .2)', {}),
        ('asian-liquidity', f'(clip (lin 1 (avg_arith (clip {l} 0 2) daily) -1) -.2 .2)', {}),
        ('depth-digital', '(mul_const .1 (step (external depth) $threshold down))', {'threshold': .025}),
        ('recovery-call', '(clip (lin -1 (external recovery) $threshold) 0 .2)', {'threshold': .2}),
        ('occupation-liquidity', f'(mul_const .1 (occupation {l} $threshold up daily))', {'threshold': 1.2}),
        # Input spread is FULL spread; B-J reference c is HALF spread and
        # normalized H=1/2*(sum(c-c0)+ -K)+, hence factor steps/4 at K=0.
        ('bj-spread', f'(mul_const {steps/4} (avg_arith (clip (lin 1 (external spread) -.001) 0 .01) daily))', {})]
    return [{'family': name, 'artifact': {'track': 'contract', 'notional': 100_000., 'params': params,
        'contract_sexpr': f'(contract (meta (underlyings basket) (maturity {steps}) (schedules)) (premium (pay {expr} maturity)))'}}
        for name,expr,params in expressions]


def diagnostic_rank(metrics):
    # Representatives only. The archive retains every valid proposal and the
    # complete three-dimensional report; raw economic return is not selection.
    return (int(metrics.get('feasible', False)), metrics.get('coverage', -1.),
            -metrics.get('nuisance', 1e30), -metrics.get('cost', 1e30))


def compare_catalog(metrics, behavior, catalog):
    rows = [r for r in catalog if r['metrics']['status']=='ok']
    matching = []
    for row in rows:
        b = row['behavior']
        x = np.column_stack((b, np.ones(len(b))))
        coeff = np.linalg.lstsq(x, behavior, rcond=None)[0]
        error = float(np.max(np.abs(behavior-x@coeff)))
        if coeff[0] > 0 and error <= max(.01, 1e-6*float(np.max(np.abs(behavior)))):
            matching.append(row['family'])
    eligible = [r['metrics'] for r in rows if r['metrics']['feasible']]
    dominated = any(r['coverage']>=metrics['coverage'] and r['nuisance']<=metrics['nuisance']
                    and r['cost']<=metrics['cost'] for r in eligible)
    # Compare to the entire constrained catalog envelope, not an arbitrary point.
    coverage_refs = [r['coverage'] for r in eligible if r['nuisance']<=metrics['nuisance'] and r['cost']<=metrics['cost']]
    nuisance_refs = [r['nuisance'] for r in eligible if r['coverage']>=metrics['coverage'] and r['cost']<=metrics['cost']]
    cost_refs = [r['cost'] for r in eligible if r['coverage']>=metrics['coverage'] and r['nuisance']<=metrics['nuisance']]
    improvement = ((bool(coverage_refs) and metrics['coverage']>=max(coverage_refs)+.05) or
        (bool(nuisance_refs) and metrics['nuisance']<=min(nuisance_refs)-.05) or
        (bool(cost_refs) and metrics['cost']<min(cost_refs)*.95))
    return {'matched_parameterized_controls': sorted(set(matching)),
        'economic_advantage_diagnostic': bool(metrics['feasible'] and not dominated and improvement),
        'catalog_feasible_points': len(eligible), 'catalog_weakly_dominates': dominated,
        'novelty_status': 'unresolved',
        'matching_scope': 'Affine single-control fit on sampled net P&L only; no complete cashflow/strategy equivalence or expert review',
        'new_mechanism_confirmed': False}
