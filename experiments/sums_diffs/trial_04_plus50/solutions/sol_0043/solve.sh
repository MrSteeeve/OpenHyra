#!/bin/bash
# Solution entry point: run the candidate solver, which must write solution.json.
# Scoring happens OUTSIDE this sandbox in the trusted evaluator.
cd "$(dirname "$0")"
PY="${OPENHYRA_PYTHON:-python3}"
exec "$PY" solver.py
