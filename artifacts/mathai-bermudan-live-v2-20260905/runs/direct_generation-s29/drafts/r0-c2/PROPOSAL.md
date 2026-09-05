# Proposal: cross-fitted spline state transition

- Mechanism: `direct_r0_1`
- Operator: `ast_mutation`
- Scope: `whole_program`
- Target slice: `instance:public-put-atm`
- Materialized change: preserve the parent's normalized ridge continuation
  program, but replace its in-sample backward stopping update with deterministic
  five-fold continuation estimates and replace duplicate linear columns with a
  compact polynomial/payoff-hinge basis.
- Falsifiable prediction: positive mean paired discounted-payoff improvement on
  `instance:public-put-atm` under the evaluator's identical seed and budget.
- Falsifier: the upper endpoint of the 95% paired confidence interval is below
  zero.
- Matched control: the unchanged parent program, evaluated with identical input,
  seed, and compute budget.
