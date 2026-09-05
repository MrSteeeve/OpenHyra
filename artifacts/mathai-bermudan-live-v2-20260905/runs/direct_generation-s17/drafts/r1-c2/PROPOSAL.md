# Proposal: cross-fitted nonlinear LSM fit

Mechanism `direct_r1_1` rewrites only the top-level `fit(...)` body. It uses
out-of-fold continuation estimates for backward stopping decisions, focuses
training on paths with current or future positive value, and replaces iterative residual
optimization with a ridge-solved random-feature residual. The prediction
interface and evaluator-owned financial problem remain unchanged.

- Target slice: `instance:public-put-atm`
- Falsifiable prediction: positive mean paired discounted-payoff difference
  versus the unchanged parent on the target slice under identical inputs and
  budget.
- Matched control: unchanged parent program, evaluated with the same instance,
  simulation paths, seed policy, and request budget.
- Falsifier: the upper endpoint of the 95% paired confidence interval is below
  zero.
