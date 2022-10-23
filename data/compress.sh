#!/bin/bash
THIS_SCRIPT_PARENT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
JSON="$THIS_SCRIPT_PARENT/json"
cd "$JSON" || exit 1

for json in *.json; do
    tar -czf "${json/json/tar.gz}" "$json"
done;