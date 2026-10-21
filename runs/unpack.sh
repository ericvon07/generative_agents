#!/usr/bin/env bash
# Put the runs where read_valentine.py looks for them: the traces in logs/llm_trace/, the
# interviews in logs/interviews/ and the checkpoint subset in environment/frontend_server/storage/.
#
#   runs/unpack.sh               both runs
#   runs/unpack.sh vp9-jev       one of them
set -euo pipefail
cd "$(dirname "$0")/.."
runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(vp9-llm vp9-jev)
mkdir -p logs/llm_trace logs/interviews
for run in "${runs[@]}"; do
  for f in runs/"$run"/trace/*.jsonl.gz; do
    gunzip -c "$f" > logs/llm_trace/"$(basename "${f%.gz}")"
  done
  cp runs/"$run"/interviews/*.json logs/interviews/
  tar -xzf runs/"$run"/checkpoints.tar.gz -C environment/frontend_server/storage
  echo "unpacked $run"
done
