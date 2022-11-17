#!/usr/bin/env bash

# To be run in login node, NOT in a job script

APPTAINER_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
HELPER_SCRIPTS="$APPTAINER_ROOT/helper_scripts"
PROJECT="$(dirname "$APPTAINER_ROOT")"
APP_LOGS="$PROJECT/app_logs"
EXTRACT="$HELPER_SCRIPTS/extract_data.sh"
OVERLAY="$APPTAINER_ROOT/$1"

if [ ! -d "$APP_LOGS" ]; then
    mkdir -p "$APP_LOGS"
fi

CONTAINER="$APPTAINER_ROOT/rmt.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: No container found at $CONTAINER."
    exit 1
fi

if [ ! -f "$OVERLAY" ]; then
    echo "ERROR: Persistent overlay with name $OVERLAY does not exist."
    exit 1
fi

cd "$PROJECT" || exit
export PROJECT
if [ -n "${CC_CLUSTER}" ]; then  # if on cluster, ~ points to login, which we don't want
    echo "On cluster $CC_CLUSTER. Running apptainer with \`--home \"$(readlink -f .)\" \`"
    apptainer run --overlay "$OVERLAY" --home "$(readlink -f .)" "$CONTAINER" "$EXTRACT"
else
    echo "On local machine. Running apptainer without \`--home\' argument."
    apptainer run --overlay "$OVERLAY" "$CONTAINER" "$EXTRACT"
fi
