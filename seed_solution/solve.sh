#!/bin/bash
# Solution entry point (mirrors the Hyra-results artifact format).
cd "$(dirname "$0")"
PY="${OPENHYRA_PYTHON:-python3}"
exec "$PY" train.py
