# OpenHyra 1.111814562869239 (legacy run)

This immutable artifact makes the reported SimpleTES-compatible set directly
verifiable without an Experience Bank or LLM backend.

The set was first recorded as `sol_0005`, iteration 4, candidate 3 of 4, in the
historical `seed_bestof4_20` run. The run used commit `2ae8b791...` with a dirty
working tree and the former winner-only policy: 80 candidates were evaluated,
but only one solution directory per Context was retained. It therefore verifies
the mathematical result, not an end-to-end execution of the current harness.

Run:

```bash
python3 verify.py
```

Expected output includes:

```json
{"score": 1.111814562869239, "n": 405, "sums": 2395, "diffs": 2003}
```

`solver.py` and `solve.sh` are the saved winner implementation. Running them
would repeat a 168-second stochastic search and is not required to verify the
published set. The packaged `solution.json` has one trailing newline added for
source-control hygiene; `source_record.json` retains the historical byte hash,
while `verification.json` and `manifest.json` distinguish both hashes.
