#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=fit_time
#SBATCH --output=fit_time_%A_%a_%j.out
#SBATCH --time=00-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80

SCRATCH="$(readlink -f "$SCRATCH")"
PROJECT="$SCRATCH/model_variance"
RUN_SCRIPT="$PROJECT/run_python.sh"

PY_SCRIPTS="$PROJECT/scripts"
PY_SCRIPT="$(readlink -f "$PY_SCRIPTS/testing/estimate_fit_times.py")"

bash "$RUN_SCRIPT" "$PY_SCRIPT"