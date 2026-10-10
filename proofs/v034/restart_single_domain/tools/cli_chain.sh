#!/bin/bash
# Real Swiss d01 CLI restart chain on core 8 (sequential). usage: cli_chain.sh ROOT
X=$1; mkdir -p $X
SWISS=<USER_HOME>/src/wrf_gpu2_wt/o1-restart/examples/switzerland_d01
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=<USER_HOME>/src/wrf_gpu2_wt/o1-restart/src \
  GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" \
  CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false GPUWRF_FAST_DEFAULTS=0
CLI="taskset -c 8 nice -n 19 python <USER_HOME>/wrf_gpu2_lanes/o1-restart/tools/cli_cpu.py run --input-dir $SWISS"
quiet() { while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done; }
arm() { name=$1; shift; quiet; echo "$(date -u +%FT%TZ) START $name" >> $X/chain.log
  /usr/bin/time -v timeout 5400 $CLI "$@" > $X/$name.out 2> $X/$name.err; rc=$?
  echo "$(date -u +%FT%TZ) END $name rc=$rc" >> $X/chain.log; return $rc; }
arm B_first --output-dir $X/stream --hours 1 --checkpoint-dir $X/ck --checkpoint-interval-steps 100 --scratch-dir $X/scratch
G200=$(grep -o "VERIFIED [^ ]* root_step=200" $X/B_first.err | awk '{print $2}')
echo "G200=$G200" >> $X/chain.log
arm C0_refuse --output-dir $X/stream --hours 2 --checkpoint-dir $X/ck --checkpoint-interval-steps 100 --resume-checkpoint $G200 --scratch-dir $X/scratch
# C: extension, killed externally after the step-300 generation is VERIFIED
quiet; echo "$(date -u +%FT%TZ) START C_extend_killed" >> $X/chain.log
setsid bash -c "exec $CLI --output-dir $X/stream --hours 2 --checkpoint-dir $X/ck --checkpoint-interval-steps 100 --resume-checkpoint $G200 --extend-run --scratch-dir $X/scratch > $X/C_extend_killed.out 2> $X/C_extend_killed.err" &
until grep -q "VERIFIED .* root_step=300" $X/C_extend_killed.err 2>/dev/null; do
  sleep 2; grep -q "Traceback\|error:" $X/C_extend_killed.err 2>/dev/null && { echo "C failed early" >> $X/chain.log; exit 1; }
  CP=$(grep -o "CLI_CPU_PID [0-9]*" $X/C_extend_killed.err 2>/dev/null | awk '{print $2}')
  [ -n "$CP" ] && ! ps -p $CP > /dev/null && { echo "C exited before root_step=300" >> $X/chain.log; exit 1; }; done
PID=$(grep -o "CLI_CPU_PID [0-9]*" $X/C_extend_killed.err | awk '{print $2}')
[ "$(cat /proc/$PID/comm)" = python ] && kill -9 $PID && echo "$(date -u +%FT%TZ) SIGKILL pid=$PID after root_step=300 VERIFIED" >> $X/chain.log
sleep 3; ps -p $PID > /dev/null && echo "pid $PID still alive" >> $X/chain.log
G300=$(grep -o "VERIFIED [^ ]* root_step=300" $X/C_extend_killed.err | awk '{print $2}')
echo "G300=$G300" >> $X/chain.log
arm D_resume --output-dir $X/stream --hours 2 --checkpoint-dir $X/ck --checkpoint-interval-steps 100 --resume-checkpoint $G300 --scratch-dir $X/scratch
arm A_control --output-dir $X/control --hours 2 --scratch-dir $X/scratch
quiet; JAX_PLATFORMS=cpu taskset -c 8 nice -n 19 python <USER_HOME>/wrf_gpu2_lanes/o1-restart/tools/compare_streams.py $X/control $X/stream $X/comparison.json >> $X/chain.log 2>&1
echo "$(date -u +%FT%TZ) CHAIN_DONE" >> $X/chain.log
