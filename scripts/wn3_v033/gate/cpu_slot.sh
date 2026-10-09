#!/usr/bin/env bash
# CPU orchestration only. Frozen scorers, manifests and gates remain unchanged.
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0
wn3_ready() {
  while [[ -e /tmp/wrf_gpu2_quiet ]]; do sleep 10; done
}
wn3_slot() {
  local s sc
  exec 8>/dev/null
  while :; do
    wn3_ready
    if (( $(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo) >= 12 )); then
      while read -r s sc; do
        [[ $s =~ ^[0-9]+$ && -n $sc ]] || continue
        exec 8>"$G/.d6slot.$s"
        if flock -n 8; then
          # Never admit a scorer to a PU currently used by any wrf.exe.
          if <USER_HOME>/miniconda3/bin/python3 - "$sc" <<'PY'
import os, sys
from pathlib import Path
cores = {int(x) for x in sys.argv[1].split(',')}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        if (p / 'comm').read_text().strip() == 'wrf.exe' and cores & os.sched_getaffinity(int(p.name)):
            sys.exit(1)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
PY
          then CPUS=$sc; D6CPUS=$sc; return; fi
          flock -u 8
        fi
      done < "$G/d6_slots.conf"
    fi
    sleep 20
  done
}
