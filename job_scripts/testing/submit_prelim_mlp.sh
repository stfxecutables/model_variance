#!/bin/bash
#SBATCH --account=rrg-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=mlp_hps
#SBATCH --output=/scratch/dberger/model_variance/slurm_logs/mlp_hps_%A_%a_%j.out
#SBATCH --array=0-9
#SBATCH --time=00-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-node=1

module load nixpkgs/16.09 intel/2018.3 fsl/6.0.1

SCRATCH="$(readlink -f "$SCRATCH")"
PROJECT="$SCRATCH/model_variance"
RUN_SCRIPT="$PROJECT/run_hperturbs.sh"

PY_SCRIPTS="$PROJECT/scripts"
PY_SCRIPT="$(readlink -f "$PY_SCRIPTS/script.py")"

# bash "$RUN_SCRIPT" "$PY_SCRIPT"
bash "$RUN_SCRIPT"