#!/bin/bash
set -e

# OpenHyra Bermudan v5 Experiment Suite
# 12 runs across 3 seeds

# baseline_v5_full_seed42
python3 harness.py --task bermudan_optimal_stopping --run-id baseline_v5_full_seed42 --init --v5 --iterations 30 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 42

# baseline_v5_full_seed123
python3 harness.py --task bermudan_optimal_stopping --run-id baseline_v5_full_seed123 --init --v5 --iterations 30 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 123

# baseline_v5_full_seed7
python3 harness.py --task bermudan_optimal_stopping --run-id baseline_v5_full_seed7 --init --v5 --iterations 30 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 7

# ablation_single_island_seed42
python3 harness.py --task bermudan_optimal_stopping --run-id ablation_single_island_seed42 --init --v5 --iterations 30 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 42

# ablation_single_island_seed123
python3 harness.py --task bermudan_optimal_stopping --run-id ablation_single_island_seed123 --init --v5 --iterations 30 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 123

# ablation_single_island_seed7
python3 harness.py --task bermudan_optimal_stopping --run-id ablation_single_island_seed7 --init --v5 --iterations 30 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 7

# ablation_no_v5_seed42
python3 harness.py --task bermudan_optimal_stopping --run-id ablation_no_v5_seed42 --init --iterations 30 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 42

# ablation_no_v5_seed123
python3 harness.py --task bermudan_optimal_stopping --run-id ablation_no_v5_seed123 --init --iterations 30 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 123

# ablation_no_v5_seed7
python3 harness.py --task bermudan_optimal_stopping --run-id ablation_no_v5_seed7 --init --iterations 30 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 7

# analogy_transfer_seed42
python3 harness.py --task bermudan_optimal_stopping --run-id analogy_transfer_seed42 --init --v5 --iterations 40 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 42

# analogy_transfer_seed123
python3 harness.py --task bermudan_optimal_stopping --run-id analogy_transfer_seed123 --init --v5 --iterations 40 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 123

# analogy_transfer_seed7
python3 harness.py --task bermudan_optimal_stopping --run-id analogy_transfer_seed7 --init --v5 --iterations 40 --workers 2 --candidates-per-context 2 --agent-stop --trial-seed 7

