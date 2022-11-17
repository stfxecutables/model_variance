#!/usr/bin/env bash
set -e
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
APPTAINER_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT="$(dirname "$APPTAINER_ROOT")"
PREP="$PROJECT/prepare"
CODE_FILES=("run_scripts.tar" "scripts.tar" "src.tar" "test.tar" "experiments.tar")
DATA="data.tar"
MODELS="torchmodels.tar"

cd "$PREP" || exit
echo "Moving data and code into overlay"
for TAR in "${CODE_FILES[@]}";
do
    rm -rf "/home/project/${TAR/.tar/""}"
    tar -xf "$TAR" -C /home/project
    echo "Un-archived $TAR"
done;

if [ ! -f "/home/project/${DATA/.tar/""}" ]; then
    if [ -f "$DATA" ]; then
        tar -xf "$DATA" -C /home/project
        echo "Un-archived $DATA"
    else
        echo "Data missing in container /home/project and data.tar missing in parent at $(readlink -f $DATA)"
        exit 1
    fi
fi

if [ -f "$MODELS" ]; then
    tar -xf "$MODELS" -C /home/torchmodels
    echo "Un-archived $MODELS"
fi