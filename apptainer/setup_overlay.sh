#!/usr/bin/env bash

# To be run in login node, NOT in a job script

APPTAINER_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
HELPER_SCRIPTS="$APPTAINER_ROOT/helper_scripts"
PROJECT="$(dirname "$APPTAINER_ROOT")"
UNTAR="$HELPER_SCRIPTS/untar_to_container.sh"

OVERLAY="$APPTAINER_ROOT/$1"
CONTAINER="$APPTAINER_ROOT/rmt.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: No container found at $CONTAINER."
    exit 1
fi

if [ -f "$OVERLAY" ]; then
    echo "ERROR: Persistent overlay with name $OVERLAY already exists."
    exit 1
fi


# Seeded weights, when generated, are about 901 MB, if generated for all seeds
# this will add up quickly so it is best we just generate for the seed at hand.
# EffnetV2-S is about 90M, and WideResNet-50-2 is about 265M.
# All data is 761MB. Data generated from one run
# is about 2G. We should probably reduce this, but that is what it is now.
# However, this means a 4-5GB overlay should be plenty for our needs.

cd "$APPTAINER_ROOT" || exit
echo -n "Creating overlay at $OVERLAY... "
apptainer overlay create --size 4096 --create-dir /home/project --create-dir /home/torchmodels "$OVERLAY"
echo "done"

cd "$PROJECT" || exit
apptainer run --overlay "$OVERLAY" --home "$PROJECT" "$CONTAINER" "$UNTAR"
