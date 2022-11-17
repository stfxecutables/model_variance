#!/usr/bin/env bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
APPTAINER_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT="$(dirname "$APPTAINER_ROOT")"
DATA="data"
PREP="$PROJECT/prepare"

cd "$PROJECT" || exit
tar --exclude="__pycache__" -cvf data.tar "$DATA"
mv data.tar "$PREP"
