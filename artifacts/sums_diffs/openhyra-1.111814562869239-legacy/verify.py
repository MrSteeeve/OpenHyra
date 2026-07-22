#!/usr/bin/env python3
"""Standalone verifier for the published legacy OpenHyra set."""

import hashlib
import json
import math
from pathlib import Path

EXPECTED = {
    "artifact_sha256": "0085bb547ab7e07c29fd89e02fa3f7660df1ed01f03e84ab01ca8942a7a615d2",
    "set_hash": "579ce595e244f06c7da990b506bdd4de6e5edf7c88b6d05c1df7f6e98c13df2d",
    "score": 1.111814562869239,
    "n": 405,
    "sums": 2395,
    "diffs": 2003,
    "span": 1198,
}


def canonical_hash(values):
    shifted = [value - values[0] for value in values]
    scale = math.gcd(*shifted[1:]) or 1
    normalized = [value // scale for value in shifted]
    reflected = [normalized[-1] - value for value in reversed(normalized)]
    canonical = min(normalized, reflected)
    raw = json.dumps(canonical, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    path = Path(__file__).with_name("solution.json")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED["artifact_sha256"]:
        raise SystemExit(f"artifact SHA-256 mismatch: {digest}")
    payload = json.loads(raw)
    values = payload.get("A")
    if not isinstance(values, list) or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in values):
        raise SystemExit("A must be a list of integers")
    values = sorted(set(values))
    if not 2 <= len(values) <= 512:
        raise SystemExit("SimpleTES size constraint failed")
    if values[0] < -1_000_000 or values[-1] > 1_000_000:
        raise SystemExit("SimpleTES element-range constraint failed")
    sums = len({a + b for a in values for b in values})
    diffs = len({a - b for a in values for b in values})
    score = math.log(sums / len(values)) / math.log(diffs / len(values))
    result = {
        "score": score,
        "n": len(values),
        "sums": sums,
        "diffs": diffs,
        "span": values[-1] - values[0],
        "set_hash": canonical_hash(values),
        "artifact_sha256": digest,
    }
    for key in ("n", "sums", "diffs", "span"):
        if result[key] != EXPECTED[key]:
            raise SystemExit(f"{key} mismatch: {result[key]}")
    if not math.isclose(score, EXPECTED["score"], rel_tol=0.0, abs_tol=1e-15):
        raise SystemExit(f"score mismatch: {score}")
    if result["set_hash"] != EXPECTED["set_hash"]:
        raise SystemExit(f"set_hash mismatch: {result['set_hash']}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
