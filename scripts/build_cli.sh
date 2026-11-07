#!/usr/bin/env bash
# Build the CLI image (compiles, runs the unit tests, packs the models from models/).
#   scripts/build_cli.sh
set -euo pipefail
cd "$(dirname "$0")/.."
docker build --target build -t whoosh-lpr:build .
docker build -t whoosh-lpr .
echo "built: docker run --rm -v \"\$PWD/photos:/photos:ro\" whoosh-lpr ."
