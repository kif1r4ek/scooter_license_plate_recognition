#!/usr/bin/env bash
# Train both models from the photos in data/raw, inside Docker.
#
#   scripts/train.sh cpu   [options]   # this machine's CPU
#   scripts/train.sh gpu   [options]   # local NVIDIA GPU (driver + NVIDIA Container Toolkit)
#   scripts/train.sh modal [options]   # Modal cloud GPU (needs ~/.modal.toml from `modal setup`)
#
# Options go to training/run_all.py, e.g. --smoke, --run NAME, --ocr-steps 3000.
# Results: runs/<run>/; with a full run the models are also copied into models/.
set -euo pipefail
cd "$(dirname "$0")/.."

backend=${1:-}
case "$backend" in
  cpu|modal) tag=whoosh-lpr-train:cpu; dockerfile=docker/train-cpu.Dockerfile; extra=() ;;
  gpu)       tag=whoosh-lpr-train:gpu; dockerfile=docker/train-gpu.Dockerfile; extra=(--gpus all) ;;
  *) echo "usage: scripts/train.sh cpu|gpu|modal [run_all.py options]" >&2; exit 2 ;;
esac
shift

if [ ! -d data/raw ] || [ -z "$(ls -A data/raw)" ]; then
  echo "data/raw is empty: the photos are needed for training" >&2; exit 1
fi

if [ "$backend" = modal ]; then
  if [ -f "$HOME/.modal.toml" ]; then
    extra+=(-v "$HOME/.modal.toml:/tmp/home/.modal.toml:ro")
  elif [ -z "${MODAL_TOKEN_ID:-}" ]; then
    echo "no Modal token: run 'pip install modal && modal setup' or set MODAL_TOKEN_ID/MODAL_TOKEN_SECRET" >&2; exit 1
  fi
  for v in MODAL_TOKEN_ID MODAL_TOKEN_SECRET MODAL_PROFILE; do
    if [ -n "${!v:-}" ]; then extra+=(-e "$v"); fi
  done
fi

echo "== building $tag"
docker build -f "$dockerfile" -t "$tag" .

tty=(); if [ -t 1 ]; then tty=(-t); fi
echo "== training on $backend"
docker run --rm "${tty[@]}" --init \
  --user "$(id -u):$(id -g)" \
  --shm-size=4g \
  -v "$PWD:/work" -w /work \
  "${extra[@]}" \
  "$tag" python training/run_all.py --backend "$backend" "$@"
