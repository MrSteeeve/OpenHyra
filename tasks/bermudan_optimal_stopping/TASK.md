# Task: Bermudan optimal stopping — discover a transferable LSMC feature program

Submit a bounded, typed feature program that improves a fixed Ridge LSMC
stopping policy under an evaluator-owned risk-neutral Black--Scholes model.
The candidate never submits an option price, stopping decision, confidence
interval, path, random seed, or dual process.

This is the bounded Feature IR Phase 1 plus Phase 4 primal--dual audit MVP. It
is deliberately not arbitrary Python policy search and not full-algorithm
search: candidates cannot alter LSMC, Ridge, simulation, payoff, stopping, or
dual construction.

## Run it

Start a new run, perform search iterations, invoke the one-time private audit,
and export the run bundle with:

```bash
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-v1 --init
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-v1 --iterations 4
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-v1 --final-audit
python3 harness.py --task bermudan_optimal_stopping --run-id bermudan-v1 --export-bundle artifacts/bermudan-v1
```

Final audit requires the parent process to provide a fresh seed; its result
must not be fed back into further proposal or repair rounds.

## Search objective

The public search suite contains one-dimensional Bermudan puts, a multi-asset
max-call, and a basket put. For every instance and repeat, candidate and frozen
baseline use the same policy-fit paths and the same independent pricing paths
(Common Random Numbers). The score is the 95% lower confidence bound of the
paired, strike-normalized candidate-minus-baseline lower-bound improvement.
Higher is better.

The estimand is the equally weighted mean over this fixed suite and its fixed
repeat cells—not a randomly sampled population of market instances. Each
cell's pathwise paired standard error is aggregated as
`sqrt(sum(cell_se^2)) / number_of_cells`; cross-instance dispersion is not
mislabelled as Monte Carlo standard error.

The baseline uses `underlying`, its square and cube, and `intrinsic`. A score of
zero means exact reproduction of that baseline under the fixed evaluator.

## Candidate artifact

You may modify only `feature_program.json`. `solve.sh` copies it to the required
`solution.json`; do not add reported prices or metrics. The exact top-level
schema is:

```json
{
  "schema": "openhyra-feature-program.v1",
  "features": [
    {"op": "underlying"},
    {"op": "square", "arg": {"op": "underlying"}}
  ]
}
```

The program has at most 16 scalar features, 128 total AST nodes, and depth 8.
All unknown fields and operations are rejected. Constants must be finite and
lie in `[-10, 10]`. Asset indices lie in `[0, 3]` and must exist in every suite
instance on which the program is evaluated. Outputs are finite and clipped to
`[-1e6, 1e6]` by the defined interpreter semantics.

### Terminals

- `{"op":"constant","value": NUMBER}`
- `{"op":"time"}`: current time divided by maturity
- `{"op":"time_to_maturity"}`
- `{"op":"spot","asset": INDEX}`: selected spot divided by strike
- `{"op":"mean_spot"}`, `max_spot`, `min_spot`
- `{"op":"basket_spot"}`: contract basket weights, or equal weights when none
- `{"op":"underlying"}`: first spot for put, maximum for max-call, weighted
  basket for basket put, divided by strike
- `{"op":"intrinsic"}`: current payoff divided by strike

### Unary expressions

Each has the form `{"op": OP, "arg": EXPR}`. Supported operations are:

`abs`, `square`, `cube`, `sqrt_abs`, `log1p_abs`, `exp_neg_abs`, and
`reciprocal_one_plus_abs`.

### Binary expressions

Each has the form `{"op": OP, "left": EXPR, "right": EXPR}`. Supported
operations are:

`add`, `subtract`, `multiply`, `divide_safe`, `minimum`, and `maximum`.

`divide_safe` replaces a denominator whose magnitude is below `1e-8` with a
signed `1e-8`. The interpreter and the fixed standardization make these
semantics deterministic.

## Trusted financial protocol

The evaluator fixes correlated geometric Brownian motion under the
risk-neutral measure, exercise dates, strike, rates, dividends, volatility,
correlation, contract payoff, discounting, path counts, and Ridge penalty.
It fits backward only on policy-fit paths, freezes the policy, then generates
an independent pricing sample. A decision at time `m` receives only `S_m`,
the time index, and public instance parameters. Exercise occurs at most once
and maturity settlement is mandatory for paths not stopped earlier.

All rewards used by the dynamic program and audit are discounted to time zero:

```text
Z_m = exp(-r * t_m) * payoff(S_m).
```

## Private primal--dual audit

Final audit is separate from search feedback. A sealed request seed derives a
new hidden multi-product suite and domain-separated policy-fit, pricing, dual
outer, and nested inner random streams. For the evaluator-owned adapted value
proxy `f_m`, the dual process uses `M_0=0` and

```text
Delta M_m = f_m(S_m outer)
            - mean_b f_m(S_m inner,b | S_(m-1) outer).
```

Outer and inner successors are conditionally iid, so every finite-inner-sample
increment is conditionally mean zero. The resulting sample mean of
`max_m(Z_m-M_m)` estimates a mathematically valid dual upper bound. The audit
score is the mean normalized confidence gap plus `0.25 * Q90`; lower is better.
Raw `upper-lower` diagnostics are reported without clipping, including any
finite-sample bound-order reversal.

For a configured overall confidence level `1-alpha`, the gap expands the upper
and lower endpoints with one-sided component error `alpha/2` each. Thus
`z = Phi^-1(1-alpha/2)` (1.96 at 95%) and Bonferroni's union bound gives at
least the stated simultaneous coverage without assuming the two estimators are
independent. This is not a claim that each endpoint has a separate two-sided
95% interval.

The configured audit repeats each hidden instance under independently derived
fit and evaluation streams to expose algorithm-training variability. The
reported lower/upper path standard errors remain conditional on each frozen
fitted policy; they are not an unconditional confidence interval for the
entire randomized learning algorithm. This MVP is a primal--dual acceptance
scaffold. Its evaluator-owned value proxy and finite nested budget are not
claimed to be a tight or competitively searched dual construction.

## Claim boundary

A public search gain is evidence only for a paired improvement on the frozen
development suite. Passing private audit supports a reproducible feature
component under this exact Black--Scholes/LSMC protocol. It does not by itself
establish a universally better stopping algorithm or a production price.
