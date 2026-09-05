# Bermudan Math AI workshop pilot (2026-09-05)

This bundle runs 3 seeds x 2 public rounds for both a deterministic
Context-to-Proposal path and a direct-generation path. Every round
executes two operator classes and two guided/control pairs. Each
control uses the same baseline source, candidate seed, data request,
and compute cap. One private hidden audit follows each mode/seed.

The four executed operators are whole_program_restart, ast_mutation,
ast_crossover, and subsystem_rewrite. The Context path uses observed
round-one effects to choose its round-two composition parent. The
direct path follows a frozen parent schedule.

This is a deterministic pilot of the protocol, not an estimate of an
LLM search effect. The full metrics preserve source/model digests,
training path and payoff hashes, evaluator target-stream hashes,
fit time, independent replay and causal lookahead probes. Candidate
internal continuation targets are explicitly unobserved.

Rows: 54; public pairs: 24; failures: 1. The recorded failure is the
context seed 29 round-two subsystem rewrite: replacing the fit model under
an inherited MLP predict function was rejected at prediction time. The
failure remains in the ledger and that seed's private audit selected the best
successful guided candidate.
The claim remains evaluator-guided open Python program search on Bermudan.
