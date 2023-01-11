#!/usr/bin/env bash

THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
CC_RESULTS=$THIS_SCRIPT_DIR/cc_results
NIAGARA_RESULTS=$CC_RESULTS/niagara
mkdir -p "$NIAGARA_RESULTS"

cd "$NIAGARA_RESULTS" || exit
rsync -chavz \
  --partial \
  --info=progress2 \
  --no-inc-recursive \
  --include='*.json' \
  --include='*/' \
  --exclude='*.ckpt' \
  --prune-empty-dirs \
  'niagara:/gpfs/fs0/scratch/j/jlevman/dberger/model_variance/results' . && \
echo 'Successfully downloaded results from Niagara'