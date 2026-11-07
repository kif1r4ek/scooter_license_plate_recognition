#!/usr/bin/env bash
# Read scooter numbers with the Docker image. Paths must be inside the current folder.
#   scripts/recognize.sh photo.jpg other.jpg
#   scripts/recognize.sh some/folder --json
set -euo pipefail
if [ $# -eq 0 ]; then set -- --help; fi
tty=(); if [ -t 1 ]; then tty=(-t); fi
exec docker run --rm "${tty[@]}" -v "$PWD:/photos:ro" -w /photos whoosh-lpr "$@"
