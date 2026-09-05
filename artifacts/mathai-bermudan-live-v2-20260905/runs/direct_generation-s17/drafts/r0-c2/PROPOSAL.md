# Proposal: validation-guarded nonlinear continuation residual

## Intervention

- Mechanism: `direct_r0_1`
- Family: `open_program_generation`
- Scope: `whole_program`
- Operator: `ast_mutation`
- Target slice: `instance:public-put-atm`

The materialized child preserves the parent program's backward cash-flow
targets, causal log/polynomial/payoff features, ridge continuation component,
and trained tanh residual. It repairs the inert `if True` child guard into a
per-time validation control-flow gate. The nonlinear residual receives a
held-out least-squares shrinkage weight in `[0, 1]` and is disabled unless its
guarded validation MSE beats ridge alone by a nontrivial margin. Predictions
also enforce the financial nonnegativity invariant and a finite fallback.

## Falsifiable hypothesis

Prediction: this mutation has positive mean paired discounted payoff versus
the unchanged parent on `instance:public-put-atm`, under identical evaluator
inputs and budget.

Falsifier: the upper endpoint of the 95% paired confidence interval for the
payoff difference is below zero.

Matched control: the unchanged parent program, evaluated with identical input
and budget.
