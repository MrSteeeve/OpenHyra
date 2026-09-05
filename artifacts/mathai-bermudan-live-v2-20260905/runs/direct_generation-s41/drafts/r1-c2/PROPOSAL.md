# Proposal: cross-fitted nonlinear continuation training

The rewritten `fit(...)` performs backward induction with out-of-fold continuation predictions for every training-path stopping update. It saves a full-sample standardized polynomial ridge model augmented by a seeded tanh random-feature residual at each non-terminal exercise date. The intended mechanism is lower policy-selection bias with enough smooth nonlinear interaction capacity for the public at-the-money put.

- Target slice: `instance:public-put-atm`
- Falsifiable prediction: positive mean paired discounted-payoff change on the target slice.
- Falsifier: the upper endpoint of the 95% paired confidence interval is below zero.
- Matched control: the unchanged parent program evaluated with identical inputs, seeds, and budget.
