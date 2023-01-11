#!/usr/bin/env bash

THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
CC_RESULTS=$THIS_SCRIPT_DIR/cc_results
CEDAR_RESULTS=$CC_RESULTS/cedar
mkdir -p "$CEDAR_RESULTS"

cd "$CEDAR_RESULTS" || exit
rsync -chavz \
  --partial \
  --info=progress2 \
  --no-inc-recursive \
  --include='*.json' \
  --include='*/' \
  --exclude='*.ckpt' \
  --prune-empty-dirs \
  'cedar:/scratch/dberger/model_variance/results' . && \
echo 'Successfully downloaded results from Cedar'