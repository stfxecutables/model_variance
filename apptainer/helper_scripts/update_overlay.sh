#!/usr/bin/env bash
set -e

# To be run in login node, NOT in a job script

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
APPTAINER_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT="$(dirname "$APPTAINER_ROOT")"
UNTAR="$SCRIPT_DIR/untar_to_container.sh"
PREP_CODE="$SCRIPT_DIR/prepare_code.sh"

OVERLAY="$APPTAINER_ROOT/$1"
# https://stackoverflow.com/a/39296572
if [ -z "$CC_CLUSTER" ]; then
    CONTAINER="$APPTAINER_ROOT/centos"
else
    CONTAINER="$APPTAINER_ROOT/rmt.sif"
fi

echo "Overlay:   $OVERLAY"
echo "Project:   $PROJECT"
echo "Container: $CONTAINER"
echo "Untar:     $UNTAR"

bash "$PREP_CODE"

if [ -f "$CONTAINER" ]; then
    echo "ERROR: No container found at $CONTAINER."
    exit 1
fi

if [ ! -f "$OVERLAY" ]; then
    echo "ERROR: Persistent overlay with name $OVERLAY does not exist."
    exit 1
fi

apptainer run --overlay "$OVERLAY" "$CONTAINER" "$UNTAR"
