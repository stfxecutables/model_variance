#!/usr/bin/env bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
APPTAINER_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT="$(dirname "$APPTAINER_ROOT")"
PREP="$PROJECT/prepare"
RUN_SCRIPTS="run_scripts"
SCRIPTS="scripts"
SRC="src"
TEST="test"
EXPERIMENTS="experiments"

cd "$PROJECT" || exit
rm -rf run_scripts.tar scripts.tar src.tar test.tar experiments.tar
tar --exclude="__pycache__" -cvf run_scripts.tar "$RUN_SCRIPTS"
tar --exclude="__pycache__" -cvf scripts.tar "$SCRIPTS"
tar --exclude="__pycache__" -cvf src.tar "$SRC"
tar --exclude="__pycache__" -cvf test.tar "$TEST"
tar --exclude="__pycache__" -cvf experiments.tar "$EXPERIMENTS"
mv run_scripts.tar "$PREP"
mv scripts.tar "$PREP"
mv src.tar "$PREP"
mv test.tar "$PREP"
mv experiments.tar "$PREP"
