# Proposal: local-state Bellman restart

Mechanism `direct_r0_0` is instantiated as a whole-program restart using
leave-one-out adaptive-neighbour Bellman regression blended with a regularized
global polynomial surface. The local component is deliberately dominant for
the one-dimensional ATM put and uses local-linear inference to represent its
curved exercise boundary.

- Target slice: `instance:public-put-atm`
- Falsifiable prediction: positive mean paired discounted-payoff difference on
  the target slice versus the unchanged parent under identical inputs and budget.
- Falsifier: the upper endpoint of the 95% paired confidence interval is below zero.
- Matched control: unchanged parent with identical input and evaluator budget.
