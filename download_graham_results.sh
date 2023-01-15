#!/usr/bin/env bash

THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
CC_RESULTS=$THIS_SCRIPT_DIR/cc_results
GRAHAM_RESULTS=$CC_RESULTS/graham
mkdir -p "$GRAHAM_RESULTS"

cd "$GRAHAM_RESULTS" || exit
rsync -chavz \
  --partial \
  --info=progress2 \
  --no-inc-recursive \
  --include='*.json' \
  --include='*/' \
  --exclude='*.ckpt' \
  --prune-empty-dirs \
  'graham:/scratch/dberger/model_variance/results' . && \
echo 'Successfully downloaded results from Graham'