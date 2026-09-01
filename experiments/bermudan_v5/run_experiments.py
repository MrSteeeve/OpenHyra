#!/usr/bin/env python3
"""Generate harness commands for the Bermudan v5 experiment suite."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    config_path = Path(__file__).parent / "experiment_config.json"
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    config = json.loads(config_path.read_text())
    seeds = config["seeds"]
    runs = config["run_configurations"]
    task = config["task"]

    commands = []
    for run_config in runs:
        for seed in seeds:
            run_id = f"{run_config['name']}_seed{seed}"
            flags = run_config["flags"].replace("{seed}", str(seed))
            cmd = f"python3 harness.py --task {task} --run-id {run_id} --init {flags}"
            commands.append({
                "run_id": run_id,
                "config": run_config["name"],
                "seed": seed,
                "command": cmd,
                "hypotheses": run_config["hypotheses_tested"],
            })

    print(f"Experiment: {config['experiment_name']}")
    print(f"Task: {task}")
    print(f"Total runs: {len(commands)}")
    print(f"Seeds: {seeds}")
    print(f"Estimated compute: {config['compute_budget']['total_estimated_hours']}h")
    print()

    for i, cmd_info in enumerate(commands, 1):
        print(f"# Run {i}/{len(commands)}: {cmd_info['run_id']} (tests {', '.join(cmd_info['hypotheses'])})")
        print(cmd_info["command"])
        print()

    script_path = Path(__file__).parent / "run_all.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\nset -e\n\n")
        f.write(f"# OpenHyra Bermudan v5 Experiment Suite\n")
        f.write(f"# {len(commands)} runs across {len(seeds)} seeds\n\n")
        for cmd_info in commands:
            f.write(f"# {cmd_info['run_id']}\n")
            f.write(f"{cmd_info['command']}\n\n")
    script_path.chmod(0o755)
    print(f"Shell script written to: {script_path}")


if __name__ == "__main__":
    main()
