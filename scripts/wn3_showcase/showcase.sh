#!/usr/bin/env bash
# Release showcase (release-docs PLAN R3+R4) inside ONE GPU lock on the FINAL tree, QUIET held by us. Every R3/R4 forecast runs
# through the FINAL tree's scripts/run_parallel_cases.sh (README recipe == what is measured). Arg: base name (dirs W7/<base>_*).
# idle 30 s dmon -> sizing run 0227 1 h (cold compile; records the C-auto plan + host peak) -> launcher --dry-run over the 6
# cases = N -> R3 N x 72 h (--compress) -> R4 N=1..N x 4 h -> R07 restart (harness checkpoint path, W7/r07_pair.sh).
# Env SHOWCASE_MPS=1 adds --mps (only if the SI41 MPS A/B is positive and the FINAL launcher has the option).
set -uo pipefail
W=<USER_HOME>/wrf_gpu2_lanes/wn3/W7; base=$W/${1:?}; C=<USER_HOME>/wrf_gpu2_lanes/wn3/cases
own=0; [[ -e /tmp/wrf_gpu2_quiet ]] || { echo "wn3-W7 $(date -u +%FT%TZ)" > /tmp/wrf_gpu2_quiet; own=1; }
cleanup() { [[ $own == 1 ]] && grep -q "^wn3-W7" /tmp/wrf_gpu2_quiet 2>/dev/null && rm -f /tmp/wrf_gpu2_quiet; }
trap cleanup EXIT; trap "cleanup; exit 143" TERM INT
export GPUWRF_JAX_CACHE_DIR=$W/cache/jax JAX_COMPILATION_CACHE_DIR=$W/cache/jax GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$W/cache/autotune
export CUDA_CACHE_PATH=$W/cache/cuda TRITON_CACHE_DIR=$W/cache/triton GPUWRF_WRF_ROOT=<DATA_ROOT>/wrf_gpu2/v025/tenerife_b4/wrf_root
export GPUWRF_CENSUS=0 TF_CPP_MIN_LOG_LEVEL=1 LC_ALL=C OMP_NUM_THREADS=4 PYTHON=<USER_HOME>/miniconda3/bin/python3
if [[ -n "${FLAGS_FILE:-}" ]]; then set -a; source "$FLAGS_FILE"; set +a; fi
PC=$W/source/scripts/run_parallel_cases.sh
opt=(--gpu-log-interval 1 --compress --compress-cpus 8,9,24,25); [[ "${SHOWCASE_MPS:-0}" == 1 ]] && opt+=(--mps)
cli=(--max-dom 3 --emit-initial-history)
dirs=(); for c in 20260227_18z_a1 20260502_18z_a1 20260614_18z_a1 20260220_18z_a1 20260608_18z_a1 20260120_18z_a1; do dirs+=($C/$c); done
M=${base}_meta; mkdir -p $M
echo "start $(date -u +%FT%TZ) cpus=$(taskset -pc $$ | awk -F': ' '{print $2}') launcher=$(sha256sum $W/source/scripts/parallel_cases.py | cut -c1-16)" > $M/log.txt
env | grep -E '^(GPUWRF_|JAX_|XLA_|OMP_|CUDA_)' | sort > $M/env.txt
nvidia-smi dmon -s pucm -d 1 -o T -c 30 > $M/idle_dmon.log 2>&1
bash $PC --out-root ${base}_cold --gpu-log-interval 1 ${dirs[0]} -- "${cli[@]}" --hours 1 >> $M/log.txt 2>&1 || { echo "sizing FAILED" >> $M/log.txt; exit 1; }
bash $PC --dry-run --out-root ${base}_r3 "${dirs[@]}" -- "${cli[@]}" --hours 72 > $M/admission.json 2>> $M/log.txt
n=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["would_start_now"])' $M/admission.json)
echo "admission n=$n $(date -u +%T)" >> $M/log.txt
(( n >= 1 )) || exit 1
bash $PC --out-root ${base}_r3 "${opt[@]}" "${dirs[@]:0:$n}" -- "${cli[@]}" --hours 72 >> $M/log.txt 2>&1; echo "R3 rc=$? $(date -u +%T)" >> $M/log.txt
for k in $(seq 1 $n); do
  bash $PC --out-root ${base}_r4_n$k "${opt[@]}" "${dirs[@]:0:$k}" -- "${cli[@]}" --hours 4 >> $M/log.txt 2>&1; echo "R4 n=$k rc=$? $(date -u +%T)" >> $M/log.txt
done
# R07 restart on the FINAL tree: 0227 6 h, SIGKILL after verified root_step=201, resume; compared by VALUES against R3 0227 h0-h6.
mkdir -p ${base}_r07 && bash $W/r07_pair.sh $W ${base}_r07/stream 20260227_18z_a1 6 201 >> $M/log.txt 2>&1; echo "R07 rc=$? $(date -u +%T)" >> $M/log.txt
echo "end $(date -u +%FT%TZ)" >> $M/log.txt
