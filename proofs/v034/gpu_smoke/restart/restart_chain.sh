#!/bin/bash
# GPU release-defaults Swiss d01 restart check on snapshot 14253f8ca, ONE shared cache, run under with_gpu_lock.
# B: --hours 1 checkpoint every 20 root steps, SIGKILL after root_step=20 VERIFIED; D: fresh-process resume to 1 h;
# A: uninterrupted 1 h control; compare wrfout streams byte-wise. Deadline guard 18:56Z.
L=<USER_HOME>/wrf_gpu2_lanes/o1-gpusmoke; S=$L/snap_main; X=$L/restart; mkdir -p $X
SWISS=$S/examples/switzerland_d01; C=$X/cache
export PYTHONPATH=$S/src GPUWRF_JAX_CACHE_DIR=$C/jax JAX_COMPILATION_CACHE_DIR=$C/jax GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$C/autotune
unset GPUWRF_FAST_DEFAULTS
CLI="python $L/restart_tools/gpu_cli.py run --input-dir $SWISS --scratch-dir $X/scratch"
left() { echo $(( $(date -d "today 18:56:00Z" +%s) - $(date -u +%s) )); }
log() { echo "$(date -u +%FT%TZ) $*" >> $X/chain.log; }
log "rev $(git -C $S rev-parse HEAD) left=$(left)s"
# B (cold compile + kill)
setsid bash -c "exec $CLI --output-dir $X/stream --hours 1 --checkpoint-dir $X/ck --checkpoint-interval-steps 20 > $X/B.out 2> $X/B.err" &
until grep -q "VERIFIED .* root_step=20" $X/B.err 2>/dev/null; do
  sleep 2; [ $(left) -lt 200 ] && { log "B: deadline before root_step=20"; P=$(grep -o "GPU_CLI_PID [0-9]*" $X/B.err | awk '{print $2}'); [ -n "$P" ] && kill -9 $P; exit 2; }
  P=$(grep -o "GPU_CLI_PID [0-9]*" $X/B.err 2>/dev/null | awk '{print $2}')
  [ -n "$P" ] && ! ps -p $P > /dev/null && { log "B exited before root_step=20"; exit 1; }; done
P=$(grep -o "GPU_CLI_PID [0-9]*" $X/B.err | awk '{print $2}')
[ "$(cat /proc/$P/comm)" = python ] && kill -9 $P && log "SIGKILL pid=$P after root_step=20 VERIFIED"
sleep 3; ps -p $P > /dev/null && log "pid $P still alive"
G=$(grep -o "VERIFIED [^ ]* root_step=20" $X/B.err | awk '{print $2}'); log "G20=$G left=$(left)s"
[ $(left) -lt 300 ] && { log "skip D/A: deadline"; exit 2; }
timeout $(( $(left) - 150 )) $CLI --output-dir $X/stream --hours 1 --checkpoint-dir $X/ck --checkpoint-interval-steps 20 --resume-checkpoint $G > $X/D.out 2> $X/D.err; log "D rc=$? left=$(left)s"
[ $(left) -lt 170 ] && { log "skip A: deadline"; exit 2; }
timeout $(( $(left) - 10 )) $CLI --output-dir $X/control --hours 1 > $X/A.out 2> $X/A.err; log "A rc=$? left=$(left)s"
log "ARMS DONE"
