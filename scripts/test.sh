#!/usr/bin/env bash
# Check everything that ships, inside Docker:
#   1. C++ unit tests and the C++-vs-Python comparison on the test photos
#   2. accuracy of the whole pipeline (Python, same models) on the test photos
#   3. the CLI image on a couple of photos
#
#   scripts/test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
tty=(); if [ -t 1 ]; then tty=(-t); fi

echo "== 1. C++ tests"
docker build -q --target build -t whoosh-lpr:build . > /dev/null
docker run --rm "${tty[@]}" -v "$PWD:/work:ro" -e WHOOSH_LPR_ROOT=/work whoosh-lpr:build \
  /build/whoosh_lpr_tests --gtest_brief=1

echo "== 2. accuracy on the test photos"
docker build -q -f docker/train-cpu.Dockerfile -t whoosh-lpr-train:cpu . > /dev/null
docker run --rm "${tty[@]}" --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work whoosh-lpr-train:cpu \
  python training/eval_e2e.py --split test --min-conf 0.3

echo "== 3. CLI"
docker build -q -t whoosh-lpr . > /dev/null
docker run --rm "${tty[@]}" -v "$PWD/data/raw:/photos:ro" whoosh-lpr "AG2938.jpg" "AN7307(2).jpg"
echo "all checks passed"
