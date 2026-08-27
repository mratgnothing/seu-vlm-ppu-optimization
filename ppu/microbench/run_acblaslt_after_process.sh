#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 WAIT_PID REPO_ROOT" >&2
  exit 2
fi

wait_pid="$1"
repo_root="$2"
while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 30
done

# The probe library carries an SDK rpath and the competition image already
# exposes its CUDA-compatibility runtime to the project virtual environment.
# Do not source the top-level vendor envsetup here: in this image it exits a
# non-interactive caller shell before the benchmark command can run.
python_bin=/mnt/workspace/seu/envs/seu-vlm-ppu-20260826/bin/python
probe="$repo_root/ppu/microbench/acblaslt_matmul_sweep.py"
library="$repo_root/ppu/microbench/build/libseu_acblaslt_matmul_probe.so"
results_dir="$repo_root/results"

"$python_bin" "$probe" --library "$library" \
  --n 12288 --k 2048 --matrix-copies 8 \
  >"$results_dir/acblaslt-12288x2048-20260828.jsonl" 2>&1
"$python_bin" "$probe" --library "$library" \
  --n 2048 --k 6144 --matrix-copies 8 \
  >"$results_dir/acblaslt-2048x6144-20260828.jsonl" 2>&1
"$python_bin" "$probe" --library "$library" \
  --n 2048 --k 2048 --matrix-copies 16 \
  >"$results_dir/acblaslt-2048x2048-20260828.jsonl" 2>&1
"$python_bin" "$probe" --library "$library" \
  --n 6144 --k 2048 --matrix-copies 16 \
  >"$results_dir/acblaslt-6144x2048-20260828.jsonl" 2>&1
date -Is >"$results_dir/acblaslt-sweep-done-20260828.txt"
