#!/bin/bash
THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
PROJECT="$THIS_SCRIPT_DIR"
APPTAINER="$PROJECT/apptainer/model_variance.sif"

cd "$PROJECT" || exit 1
module load apptainer/1.0
echo "Running $1 with container $APPTAINER:"
apptainer run --nv --bind "$(readlink -f "$PROJECT")" --app python "$APPTAINER" "$1"
