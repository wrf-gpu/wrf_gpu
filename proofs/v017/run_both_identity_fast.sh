#!/usr/bin/env bash
# v0.17 BOTH identity 72h reruns (FAST-compile) + CPU postprocess, under ONE GPU
# lock acquisition. Switzerland first (single-domain replay, segmented fast path),
# then Canary (nested, already bounded). Postprocess (CPU-only) runs inline after
# each GPU forecast. Designed to be invoked WRAPPED in scripts/with_gpu_lock.sh.
set -uo pipefail   # not -e: we want to attempt both cases + postprocess even if one GPU step is imperfect

ROOT=/home/enric/src/wrf_gpu2/.wt-rc
cd "$ROOT"

echo "################# SWITZERLAND ##################"
bash proofs/v017/run_switzerland_identity_fast.sh
SWRR="$(cat /tmp/v017_switz_identity_rr.txt 2>/dev/null)"
echo "[switz] postprocess RR=$SWRR"
bash proofs/v017/finalgates/run_switzerland_postprocess.sh "$SWRR" || echo "[switz] postprocess returned nonzero"

echo "################# CANARY #######################"
bash proofs/v017/run_canary_identity_fast.sh
CARR="$(cat /tmp/v017_canary_identity_rr.txt 2>/dev/null)"
echo "[canary] postprocess RR=$CARR"
bash proofs/v017/finalgates/run_canary_postprocess.sh "$CARR" || echo "[canary] postprocess returned nonzero"

echo "################# BOTH DONE ####################"
echo "SWITZ_RR=$SWRR"
echo "CANARY_RR=$CARR"
echo "ALL IDENTITY 72H GPU+POSTPROCESS DONE"
