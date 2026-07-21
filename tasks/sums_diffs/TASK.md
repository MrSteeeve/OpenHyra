# Task: sums_diffs — maximize the sum-vs-difference exponent C(A)

Construct a finite set A of distinct integers maximizing

    C(A) = log(|A+A| / |A|) / log(|A-A| / |A|)

where A+A = {a+b : a,b in A} and A-A = {a-b : a,b in A}. HIGHER IS BETTER.
For "most" sets C(A) < 1 (differences outnumber sums since addition commutes);
sum-dominant constructions push it above 1. The best published value on this
benchmark is 1.15971 (Hyra) over 1.14489 (SimpleTES).

## Protocol

- You may ONLY modify `solver.py`. It must write `solution.json` in its own
  directory containing `{"A": [list of integers]}`.
- Constraints (checked by a trusted evaluator you cannot see or touch):
  distinct integers, 2 <= |A| <= 500000, |a| <= 1000000 for every element.
- The evaluator recomputes |A+A| and |A-A| itself (FFT over the indicator
  function). Your reported numbers are ignored — only the emitted set matters.
- `solver.py` must finish within ~150 seconds (the sandbox kills the run at
  200s). Use the time for search: constructions, local search, annealing, etc.
- Python standard library + numpy only. No network access.
