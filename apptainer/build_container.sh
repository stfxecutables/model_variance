#!/bin/bash
THIS_SCRIPT_PARENT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd "$THIS_SCRIPT_PARENT" || exit 1
PILFER="$THIS_SCRIPT_PARENT/pilfer_container_files.sh"
# sudo apptainer build rmt.sif build_centos.def
# sudo apptainer build --sandbox centos/ build_centos.def
#
# sudo apptainer build --sandbox model_variance/ build_alpine.def
sudo apptainer build model_variance.sif build_alpine.def && bash "$PILFER"
