#!/usr/bin/env bash
# Grab pre-compiled wheels and .pyenv folder from existing .sif container and move them into
# local files for faster build times
APPTAINER_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
CONTAINER="$APPTAINER_ROOT/model_variance"
BUILD_FILES="$APPTAINER_ROOT/build_files"
BUILT_WHEELS="$BUILD_FILES/wheels"

if [ ! -d "$BUILD_FILES" ]; then
    mkdir -p "$BUILD_FILES"
fi
if [ ! -d "$BUILT_WHEELS" ]; then
    mkdir -p "$BUILT_WHEELS"
fi

if [ ! -d "$CONTAINER" ]; then
    echo "ERROR: No container found at $CONTAINER."
    exit 1
fi

sudo echo ""  # printing...
echo -n "Copying cached built wheels from previous container to $BUILT_WHEELS..."
sudo apptainer exec --bind $(readlink -f $BUILD_FILES) --no-home $CONTAINER cp -r /root/.cache/pip $BUILT_WHEELS
echo "done"

echo -n "Copying previous .pyenv directory from previous container to $BUILD_FILES..."
sudo apptainer exec --bind $(readlink -f $BUILD_FILES) --no-home $CONTAINER cp -r /app/.pyenv $BUILD_FILES
rm -rf $BUILD_FILES/.pyenv/.git
echo "done"

echo -n "Fixing permissions..."
sudo chown -R $USER:$USER "$BUILD_FILES"  # needed to make wheels addable in git
echo "done"

