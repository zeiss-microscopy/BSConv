#!/bin/bash

# robust bash scripting
set -o errexit
set -o nounset

# get absolute path of this script
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

cd "$SCRIPT_DIR" && cd .. && python3 setup.py bdist_wheel
