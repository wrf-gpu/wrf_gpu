# sourced: fid_waiting -> 0 if a real with_gpu_lock.sh wrapper with --label fid-* is waiting or holding (argv-checked, not text)
fid_waiting() {
  local p a
  for p in $(pgrep -f 'with_gpu_lock\.sh --label fid-'); do
    a=$(tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | sed -n 2p)
    [[ $a == */with_gpu_lock.sh ]] && return 0
  done
  return 1
}
