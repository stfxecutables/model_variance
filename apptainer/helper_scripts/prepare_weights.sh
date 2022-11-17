#!/usr/bin/env bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
APPTAINER_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT="$(dirname "$APPTAINER_ROOT")"
# WEIGHTS="seeded_inits"
PREP="$PROJECT/prepare"
MODELS="torchmodels"

cd "$PROJECT" || exit
# tar --exclude="__pycache__" -cvf seeded_inits.tar "$WEIGHTS"
if [ -d "$MODELS" ]; then
    tar --exclude="__pycache__" -cvf torchmodels.tar "$MODELS"
    mv torchmodels.tar "$PREP"
fi