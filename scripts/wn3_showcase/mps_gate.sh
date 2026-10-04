#!/usr/bin/env bash
# Inside the lock: yield at once (rc 75) if a fid-q2/fid-cloud wrapper waits (manager 13:5xZ); else the A/B under QUIET.
source <USER_HOME>/wrf_gpu2_lanes/wn3/W6/fid_waiting.sh
if fid_waiting; then echo "gate: fid-* waiting -> yield $(date -u +%T)" >> <USER_HOME>/wrf_gpu2_lanes/wn3/W6/mps_ab/queued.txt; exit 75; fi
exec bash <USER_HOME>/wrf_gpu2_lanes/wn3/W6/quiet_wrap.sh bash <USER_HOME>/wrf_gpu2_lanes/wn3/W6/mps_ab.sh
