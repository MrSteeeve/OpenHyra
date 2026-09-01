#!/bin/bash
# The candidate is data-only. The trusted evaluator, outside the sandbox,
# owns all simulation, fitting, pricing, confidence intervals, and dual work.
set -eu
cd "$(dirname "$0")"
cp feature_program.json solution.json
