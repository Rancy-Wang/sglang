#!/usr/bin/env bash
# Run inside this task's tmux on A13 after checking GPU ownership.
set -euo pipefail
vq_repo=/homes/ruoxi07/repo/sglang-vqplus-kernel
vq_scripts="$vq_repo/research/loopspec/experiments/2026-11-27-sglang-accuracy"
vq_root=/homes/ruoxi07/experiments/loopvq/2026-11-27-sglang-accuracy
source "$vq_repo/research/loopspec/experiments/2026-11-23-loopvq-reproduction/a10_target_env.sh"
export PYTHONPATH="$vq_repo/python" CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
export TORCH_CUDA_ARCH_LIST=8.0 TORCH_EXTENSIONS_DIR="$vq_root/extensions-a13" MAX_JOBS=2
cd "$vq_repo"
trap 'echo $? > "$vq_root/pipeline.exit"' EXIT
[[ -n "${TMUX:-}" ]]
[[ -z "$(nvidia-smi -i 2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)" ]]
[[ "$(git branch --show-current)" == loopspec-test ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
vq_previous_sha=$(cat "$vq_root/pipeline.sha" 2>/dev/null || true)
if [[ "${1:-}" == --resume-serving ]]; then
    [[ -n "$vq_previous_sha" ]]
    git diff --quiet "$vq_previous_sha" HEAD -- python "$vq_scripts/check_context.py"
    [[ "$(cat "$vq_root/context-memcheck.exit")" == 0 ]]
    "$LOOPVQ_PYTHON" -c 'import json,sys; assert all(json.load(open(p))["passed"] for p in sys.argv[1:])' "$vq_root/context-validation.json" "$vq_root/context-memcheck.json"
    echo "Reusing unchanged context/memcheck evidence from $vq_previous_sha"
else
    [[ -z "${1:-}" ]]
fi
git rev-parse HEAD > "$vq_root/pipeline.sha"
if [[ "${1:-}" != --resume-serving ]]; then
printf 'context\n' > "$vq_root/pipeline.phase"
CUDA_VISIBLE_DEVICES=6 "$LOOPVQ_PYTHON" -u "$vq_scripts/check_context.py" --write-reference "$vq_root/context-oracle.pt" --output "$vq_root/context-validation.json" > "$vq_root/context-r4.log" 2>&1
printf 'memcheck\n' > "$vq_root/pipeline.phase"
set +e
CUDA_VISIBLE_DEVICES=6 compute-sanitizer --tool memcheck --error-exitcode 97 "$LOOPVQ_PYTHON" -u "$vq_scripts/check_context.py" --read-reference "$vq_root/context-oracle.pt" --output "$vq_root/context-memcheck.json" > "$vq_root/context-memcheck.log" 2>&1
vq_rc=$?
set -e
echo "$vq_rc" > "$vq_root/context-memcheck.exit"
[[ "$vq_rc" == 0 ]]
fi
printf 'tp2-acceptance\n' > "$vq_root/pipeline.phase"
"$LOOPVQ_PYTHON" -u "$vq_scripts/accept.py" --root "$vq_root" --model "$LOOPVQ_MODEL" --book "$LOOPVQ_TASK_ROOT/artifacts/ouro_2p6b_think_loopvq8b_g16_lvgqe2e" --gpus 6,7 > "$vq_root/acceptance.log" 2>&1
printf 'greedy-accuracy\n' > "$vq_root/pipeline.phase"
export VQ_ACCURACY_ROOT="$vq_root" VQ_ACCURACY_SCRIPTS="$vq_scripts"
"$LOOPVQ_PYTHON" - <<'PY'
import os, shlex, subprocess, json
from pathlib import Path
r=Path(os.environ['VQ_ACCURACY_ROOT']);s=Path(os.environ['VQ_ACCURACY_SCRIPTS'])
common=['--root',str(r),'--loopvq','/homes/ruoxi07/repo/LoopVQ','--original-root',os.environ['LOOPVQ_TASK_ROOT'],
        '--training-root',str(r.parent/'2026-11-26-residual'),'--model',os.environ['LOOPVQ_MODEL'],
        '--eval-python','/homes/ruoxi07/anaconda3/envs/loopvq-eval/bin/python']
for name,gpus,port,arms in [('bf16','2,3','30942',['bf16']),('plain','4,5','30944',['loopvq']),('best-kl','6,7','30946',['residual-ft'])]:
    cmd=[os.environ['LOOPVQ_PYTHON'],'-u',str(s/'worker.py')]+common+['--worker',name,'--gpus',gpus,'--port',port,'--serving-revision',json.loads((r/'acceptance.json').read_text())['sglang_sha'],'--arms']+arms
    # tmux server environments do not inherit exports made in this pane.
    env={k:v for k,v in os.environ.items() if k in ('PATH','PYTHONPATH','LD_LIBRARY_PATH','VIRTUAL_ENV','CUDA_HOME','CC','CXX','TORCH_EXTENSIONS_DIR','TORCH_CUDA_ARCH_LIST','MAX_JOBS','OMP_NUM_THREADS','TMPDIR','HF_HOME','XDG_CACHE_HOME','SGLANG_CACHE_DIR','SGLANG_JIT_CACHE_DIR','TRITON_CACHE_DIR','WANDB_DIR','WANDB_BASE_URL')}
    command=shlex.join(['env']+[f'{k}={v}' for k,v in env.items()]+cmd)+' > '+shlex.quote(str(r/(name+'-full.log')))+' 2>&1; echo $? > '+shlex.quote(str(r/(name+'-full.exit')))
    (r/(name+'-full-command.txt')).write_text(command+'\n')
    subprocess.run(['tmux','new-window','-t','vq-accuracy','-n',name+'-full',command],check=True)
print('All greedy workers launched after successful kernel and TP2 acceptance gates',flush=True)
PY
