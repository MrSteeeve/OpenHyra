# Proposal: direct_r0_0

- Scope/operator: `whole_program` / `whole_program_restart`
- Materialized mechanism: replace the blank/inherited continuation-regression
  structure with a complete direct-decision program. It performs backward
  empirical policy optimization over monotone discounted-payoff exercise
  boundaries and uses bootstrap-median aggregation at every exercise date.
- Target slice: `instance:public-put-atm`.
- Falsifiable prediction: positive mean paired discounted-payoff difference on
  `instance:public-put-atm` relative to the matched control.
- Falsifier: the upper endpoint of the 95% paired confidence interval is below
  zero.
- Matched control: unchanged parent under identical evaluator inputs, seed, and
  compute budget.

The expected effect is structural: for the one-dimensional ATM put, the
stopping region is represented directly by a low-variance boundary instead of
indirectly by a global continuation-value regression. The evaluator remains
the sole owner of paths, payoff computation, policy application, and score.
