#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL

# Submit from the project root (sbatch kyanite.sh). Uses the project's .venv,
# built once on the cluster with:
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# SLURM runs a spooled copy of this script, so locate the project via the
# submit directory rather than the script's own path.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR" || exit 1

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
source .venv/bin/activate

python kyanite_rf_shap.py
