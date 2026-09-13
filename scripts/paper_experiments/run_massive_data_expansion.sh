#!/bin/bash
set -e

REPO_ROOT=$(cd $(dirname $0)/../.. && pwd)
RESULTS_DIR="$REPO_ROOT/results/massive_pde_database_v1"

echo "================================================================"
echo "Starting Massive Data Expansion for AICO-Intellig"
echo "Target: 8 Physics Families x 500 Instances x 18 Algos x 3 Seeds"
echo "Total expected trajectories: ~216,000"
echo "================================================================"

python "$REPO_ROOT/scripts/generate_classification_data.py" \
    --config "$REPO_ROOT/scripts/paper_experiments/massive_sweep_config.json" \
    --out_dir "$RESULTS_DIR"

echo "Expansion Sweep Initiated/Completed. Check results at: $RESULTS_DIR"
