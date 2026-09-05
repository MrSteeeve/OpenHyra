# Proposal: disagreement-gated parent crossover

- Mechanism: replace the equal continuation-value average with a smooth, per-path crossover that gives the standalone MLP 50% weight under agreement and decreases its weight toward 12% as relative parent disagreement grows. The ridge-plus-residual parent remains the stability anchor, and the result remains inside the two-parent convex hull.
- Target slice: `instance:public-put-atm`.
- Falsifiable prediction: positive mean paired discounted-payoff difference on the target slice versus the unchanged parent/control under identical evaluator inputs and budget.
- Falsifier: the upper endpoint of the 95% paired confidence interval is below zero.
- Matched control: unchanged parent program with identical input paths, seeds, and evaluator budget.
