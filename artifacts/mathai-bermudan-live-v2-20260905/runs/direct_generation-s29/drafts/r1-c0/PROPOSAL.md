# Proposal: disagreement-gated AST crossover

The crossover uses parent A's ridge continuation prediction as an anchor and
adds up to 35% of parent B's nonlinear correction. The correction is smoothly
suppressed as scale-normalized parent disagreement grows. Non-finite outputs
fall back to the anchor and continuation values are projected to the
nonnegative payoff domain.

- Target slice: `instance:public-put-atm`
- Falsifiable prediction: positive mean paired payoff on the declared slice.
- Falsifier: the upper 95% paired confidence bound is below zero.
- Matched control: the unchanged parent program under identical evaluator
  inputs, seeds, and budget.
