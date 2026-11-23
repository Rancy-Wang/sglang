#!/usr/bin/env bash
# Run inside this task's A10 tmux session. No processes are killed by this script.
set -euo pipefail
if [[ $# != 5 ]]; then
  echo "Usage: bash a10_throughput_server.sh bf16|loopvq common-shape|paper-long GPU PORT LABEL" >&2
  exit 2
fi
method=$1 protocol=$2 gpu=$3 port=$4 label=$5
[[ $method == bf16 || $method == loopvq ]]
[[ $protocol == common-shape || $protocol == paper-long ]]
[[ $gpu =~ ^[0-7]$ && $port =~ ^[0-9]+$ && $label =~ ^[a-zA-Z0-9_-]+$ ]]
[[ -n ${TMUX:-} && $(hostname -s) == a10 ]]
source "$(dirname "${BASH_SOURCE[0]}")/a10_target_env.sh"
cd "$LOOPVQ_TARGET_REPO"
[[ $(git branch --show-current) == loopspec-test ]]
[[ -z $(git status --porcelain) ]]
[[ -z ${SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS:-} ]]
[[ -z $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader) ]]
"$LOOPVQ_PYTHON" - "$port" <<'PY'
import socket, sys
with socket.socket() as sock:
    assert sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0, "Port is occupied"
PY
export CUDA_VISIBLE_DEVICES="$gpu" TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
prefix=64 recent=256 capacity=41984 splits=8
if [[ $protocol == paper-long ]]; then
  prefix=8 recent=64 capacity=40960
  [[ $method == bf16 ]] || capacity=557568
fi
[[ $method == bf16 ]] || splits=48
command=("$LOOPVQ_PYTHON" -u -m sglang.launch_server
  --model-path "$LOOPVQ_MODEL" --host 127.0.0.1 --port "$port"
  --trust-remote-code --dtype bfloat16 --attention-backend triton --tp-size 1
  --page-size 8 --max-running-requests 16 --max-total-tokens "$capacity"
  --max-prefill-tokens 8192 --context-length 40960 --mem-fraction-static .90
  --disable-radix-cache --chunked-prefill-size -1
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 4 8 16
  --cuda-graph-backend-prefill disabled --enable-metrics --decode-log-interval 1
  --incremental-streaming-output --ouro-decode-profile optimized
  --triton-attention-num-kv-splits "$splits")
if [[ $method == loopvq ]]; then
  command+=(--loopvq-codebook-dir "$LOOPVQ_TASK_ROOT/artifacts/ouro_2p6b_think_loopvq8b_g16_lvgqe2e"
    --loopvq-prefix-tokens "$prefix" --loopvq-recent-tokens "$recent")
fi
log="$LOOPVQ_TASK_ROOT/logs/$label"
[[ ! -e $log.log && ! -e $log.launch.txt && ! -e $log.exit ]]
{
  date -u
  git rev-parse HEAD
  printf '%q ' "${command[@]}"
  printf '\n'
  nvidia-smi -i "$gpu" --query-gpu=uuid,name,memory.free,driver_version --format=csv
  free -h
  "$LOOPVQ_PYTHON" -c 'import sys,sglang; print(sys.executable); print(sglang.__file__)'
} > "$log.launch.txt"
set +e
"${command[@]}" 2>&1 | tee "$log.log"
serve_rc=${PIPESTATUS[0]}
printf '%s\n' "$serve_rc" > "$log.exit"
exit "$serve_rc"
