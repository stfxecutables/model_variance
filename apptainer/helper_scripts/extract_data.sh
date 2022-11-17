#!/usr/bin/env bash
set -e

HELPER_SCRIPTS="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
APPTAINER_ROOT="$(dirname "$HELPER_SCRIPTS")"
PROJECT="$(dirname "$APPTAINER_ROOT")"
rsync -amv --include='*.tar.gz' --include='*/' --exclude='*' /home/project/logs "$PROJECT/app_logs"