#!/usr/bin/env python3
"""Trusted evaluator for sums_diffs. Runs OUTSIDE the candidate's control.

Usage: evaluator.py <sandbox_dir>
Reads <sandbox_dir>/solution.json {"A": [ints]}, validates constraints, and
recomputes C(A) = log(|A+A|/|A|) / log(|A-A|/|A|) via FFT over the indicator
function. Prints a single JSON line: {"score": ..., "metrics": {...}} or
{"error": "..."}. Candidate-reported scores are never trusted.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

MAX_N = 500_000
MAX_ABS = 1_000_000


def fail(msg):
    print(json.dumps({"error": msg}))
    sys.exit(0)


def main():
    sandbox = Path(sys.argv[1])
    sol_path = sandbox / "solution.json"
    if not sol_path.exists():
        fail("solution.json not found")
    try:
        data = json.loads(sol_path.read_text())
    except ValueError as e:
        fail(f"solution.json is not valid JSON: {e}")
    A = data.get("A")
    if not isinstance(A, list) or not A:
        fail('solution.json must contain a non-empty list "A"')
    if not all(isinstance(a, int) for a in A):
        fail("all elements must be integers")
    if len(set(A)) != len(A):
        fail("elements must be distinct")
    n = len(A)
    if not (2 <= n <= MAX_N):
        fail(f"|A| must be in [2, {MAX_N}], got {n}")
    if any(abs(a) > MAX_ABS for a in A):
        fail(f"|a| must be <= {MAX_ABS}")

    arr = np.array(sorted(A), dtype=np.int64)
    lo = int(arr.min())
    L = int(arr.max()) - lo + 1
    f = np.zeros(L)
    f[arr - lo] = 1.0
    size = 2 * L - 1
    nfft = 1 << (size - 1).bit_length()
    F = np.fft.rfft(f, nfft)
    conv = np.fft.irfft(F * F, nfft)[:size]        # indicator conv -> A+A support
    corr = np.fft.irfft(F * np.conj(F), nfft)[:L]  # autocorrelation -> A-A support (>=0 half)
    sums = int((conv > 0.5).sum())
    diffs = 2 * int((corr[1:] > 0.5).sum()) + 1
    if diffs <= n:
        fail("degenerate set: |A-A| <= |A|")
    score = math.log(sums / n) / math.log(diffs / n)
    print(json.dumps({
        "score": round(score, 6),
        "metrics": {"n": n, "sums": sums, "diffs": diffs,
                    "span": int(arr.max() - arr.min())},
    }))


if __name__ == "__main__":
    main()
