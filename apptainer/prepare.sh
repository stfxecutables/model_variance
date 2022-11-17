#!/usr/bin/env bash

# To be run in login node, NOT in a job script

APPTAINER_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
PROJECT="$(dirname "$APPTAINER_ROOT")"
SCRIPT_DIR="$APPTAINER_ROOT/helper_scripts"
DOWNLOAD_DATA="$APPTAINER_ROOT/manual_download.sh"
PREP="$PROJECT/prepare"
PREP_DATA="$SCRIPT_DIR/prepare_data.sh"
PREP_CODE="$SCRIPT_DIR/prepare_code.sh"
PREP_WEIGHTS="$SCRIPT_DIR/prepare_weights.sh"
DATA="$PREP/data.tar"
# WEIGHTS="$PROJECT/seeded_inits.tar"
MODELS="$PREP/torchmodels.tar"
CODE_FILES=("run_scripts.tar" "scripts.tar" "src.tar" "test.tar" "experiments.tar")

if [ ! -d "$PREP" ]; then
    mkdir -p "$PREP"
fi
cd "$PROJECT" || exit


echo "......................................."
echo "Downloading data"
echo "......................................."
bash "$DOWNLOAD_DATA"

if [ ! -f "$DATA" ]; then
    echo "......................................."
    echo "Compressing data"
    echo "......................................."
    bash "$PREP_DATA"
    echo "Finished compressing data."
fi

if [ ! -f "$MODELS" ]; then
    echo "......................................."
    echo "Compressing model weights"
    echo "......................................."
    bash "$PREP_WEIGHTS"
    echo "Finished compressing weights."
fi

# always update the code
echo "......................................."
echo "Compressing code files"
echo "......................................."
bash "$PREP_CODE"
echo "Done compressing code"


echo "......................................."
echo "Summary:"
echo "......................................."
for FNAME in "${CODE_FILES[@]}";
do
    TAR="$PREP/$FNAME"
    echo "$TAR is ready for copying into overlay"
done;
echo "$DATA is ready for copying into overlay"
echo "$MODELS is ready for copying into overlay"


