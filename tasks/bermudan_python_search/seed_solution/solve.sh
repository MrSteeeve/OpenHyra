#!/bin/bash
# ``solution.json`` transports the program interface declaration.  The
# evaluator invokes algorithm.py fit/predict separately for each instance.
set -eu
cd "$(dirname "$0")"
cp manifest.json solution.json
