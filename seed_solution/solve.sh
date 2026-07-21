#!/bin/bash
# Solution entry point (mirrors the Hyra-results artifact format:
# run train.py, then emit a machine-readable solution.json).
cd "$(dirname "$0")"
PY="${OPENHYRA_PYTHON:-python3}"

set -e
set -o pipefail
"$PY" train.py 2>&1 | tee train.log

"$PY" - <<'PYEOF'
import re, json, sys
log = open("train.log", encoding="utf-8", errors="replace").read()
def grab(key):
    m = re.search(rf"^{key}:\s+([0-9.]+)\s*$", log, re.MULTILINE)
    return m.group(1) if m else None
vb = grab("val_bpb")
if vb is None:
    json.dump({"error": "no val_bpb in training log"}, open("solution.json", "w"))
    print("ERROR: could not parse val_bpb from training log", file=sys.stderr)
    sys.exit(1)
result = {"val_bpb": float(vb),
          "training_seconds": float(grab("training_seconds") or 0),
          "num_steps": int(float(grab("num_steps") or 0))}
json.dump(result, open("solution.json", "w"))
print("solution.json:", json.dumps(result))
PYEOF
