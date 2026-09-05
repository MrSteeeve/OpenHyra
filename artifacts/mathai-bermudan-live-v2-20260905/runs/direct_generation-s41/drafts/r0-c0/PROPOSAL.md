# Proposal: cross-fitted boundary splines

Mechanism `direct_r0_0` is instantiated as a whole-program restart using
cross-fitted backward induction and a compact hinge-spline continuation
surface. Out-of-fold continuation estimates determine recursive training-path
stops, while full-data fits serve independent prediction queries. The intended
effect is less in-sample stopping bias near the ATM put exercise boundary.

Falsifiable prediction: positive mean paired discounted payoff on
`instance:public-put-atm` relative to the unchanged parent under identical
inputs and budget.

Matched control: the unchanged parent with the identical input and evaluator
budget.

Falsifier: the upper endpoint of the 95% paired confidence interval is below
zero.
