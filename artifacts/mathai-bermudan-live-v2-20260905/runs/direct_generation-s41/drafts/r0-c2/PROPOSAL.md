# Proposal: cross-fitted residual hybrid

- Mechanism: `direct_r0_1`
- Scope/operator: `whole_program` / `ast_mutation`
- Target slice: `instance:public-put-atm`
- Materialized mutation: retain the parent ridge backbone and trained tanh
  residual, repair its invalid feature dimension and lost intercept, express
  features in moneyness, use two-fold out-of-fold continuation predictions in
  the backward stopping recursion, and admit the residual branch through an
  out-of-fold shrinkage coefficient.
- Falsifiable prediction: positive mean paired discounted-payoff difference on
  `instance:public-put-atm` relative to the unchanged parent.
- Matched control: the unchanged parent program, evaluated with identical
  input paths, seeds, pricing paths, and budget.
- Falsifier: the upper endpoint of the 95% paired confidence interval is below
  zero.
