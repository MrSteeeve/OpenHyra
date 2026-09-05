# Historical deterministic Bermudan protocol fixture (2026-09-05)

This bundle is superseded as workshop acceptance evidence by
`../mathai-bermudan-live-v2-20260905/`. It used a deterministic repository
generator, not real Context and Proposal model calls. Its historical isolation
behavior is not the current native-sandbox validation. Retain these rows as
development history; do not pool them into the real-model comparison.

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

Rows: 54; public pairs: 24; failures: 1.
The claim remains evaluator-guided open Python program search on Bermudan.
manifest.json records source digests, request matrix, and rebuild command.
