#!/bin/bash
THIS_SCRIPT_PARENT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd "$THIS_SCRIPT_PARENT" || exit 1
PILFER="$THIS_SCRIPT_PARENT/pilfer_sandbox_files.sh"

sudo apptainer build --sandbox model_variance/ build_centos.def && bash "$PILFER"
