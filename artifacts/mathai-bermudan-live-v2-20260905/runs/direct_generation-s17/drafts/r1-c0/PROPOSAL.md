# Proposal: robust two-parent continuation crossover

The intervention is an AST-level replacement of only
`_combine_predictions(left, right)`. It robustly aligns the pure-MLP parent's
location and dispersion to the ridge-plus-residual parent, clips extreme
cross-parent disagreements, and blends the aligned prediction at 25% weight.

- Target slice: `instance:public-put-atm`
- Falsifiable prediction: positive mean paired discounted-payoff difference
  on the target slice.
- Falsifier: the upper endpoint of the 95% paired confidence interval is below
  zero.
- Matched control: the unchanged parent program under identical evaluator
  inputs, seeds, paths, and budget.
