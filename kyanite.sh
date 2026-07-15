#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
source ~/kyanite_env/bin/activate

~/kyanite_env/bin/python3 kyanite_rf_shap.py