# Proposal: `direct_r1_1`

Rewrite only `fit(...)` as seeded, fold-selected ridge Longstaff--Schwartz
backward induction. Training concentrates on in-the-money observations, adds a
small near-the-money support set when that slice is sparse, preserves an
unpenalized intercept, and treats evaluator payoffs as already discounted.

- Target slice: `instance:public-put-atm`
- Falsifiable prediction: positive mean paired discounted payoff on the target
  slice versus the unchanged parent under identical inputs and budget.
- Matched control: the unchanged parent program, evaluated with the identical
  instance paths, seeds, and evaluator budget.
- Falsifier: the upper endpoint of the 95% paired confidence interval is below
  zero.
