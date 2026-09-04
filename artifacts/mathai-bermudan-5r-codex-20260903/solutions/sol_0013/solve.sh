#!/bin/bash
# ``solution.json`` is only the manifest transport envelope.  The trusted
# evaluator executes train.py separately for each instance and loads the
# resulting data-only policy artifact through its registered MLP runner.
set -eu
cd "$(dirname "$0")"
cp manifest.json solution.json
