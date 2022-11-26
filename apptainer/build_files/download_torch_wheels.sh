#!/usr/bin/env bash
THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd "$THIS_SCRIPT_DIR" || exit 1

# wget -N https://download.pytorch.org/whl/cu113/torchaudio-0.12.1%2Bcu113-cp310-cp310-linux_x86_64.whl
# wget -N https://download.pytorch.org/whl/cu113/torchvision-0.13.1%2Bcu113-cp310-cp310-linux_x86_64.whl
# wget -N https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp310-cp310-linux_x86_64.whl


pip download --python-version 3.10.8 --no-deps torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 --dest .