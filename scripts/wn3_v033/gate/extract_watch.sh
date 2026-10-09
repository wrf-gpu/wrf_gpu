#!/usr/bin/env bash
# Touch V33/gate/extract_ok/<case> when release-docs posts 'extraction DONE <case>' on MAILBOX (manager 2026-10-06T09:49:33Z: per-case
# frame deletion needs that line). A case counts only if its token (full name, YYYYMMDD or MMDD of the issue) appears within 60
# characters after 'extraction DONE' (cut at ; ( : | and at pending/not/waiting/next/queued/later/still/failed) on a line from release-docs; the matched line is stored in the marker (audit). Cases = every
# rc33g_* cases.txt (IC-perturbed members excluded: they need no extraction). Stops at 2026-10-08T00:00Z or on extract_watch.stop.
set -uo pipefail
G=<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate; MB=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; mkdir -p $G/extract_ok
end=$(date -d 2026-10-08T00:00:00Z +%s)
while (( $(date +%s) < end )) && [[ ! -e $G/extract_watch.stop ]]; do
  grep -E '^[^ ]+ release-docs[ :-].*extraction DONE' $MB | while IFS= read -r line; do
    for c in $(cat $G/runs/rc33g_*/cases.txt 2>/dev/null | awk 'NF{print $1}' | grep -v _icp | sort -u); do
      [[ -e $G/extract_ok/$c ]] && continue
      issue=${c%_a[0-9]}; ymd=${issue%_18z}; mmdd=${ymd:4:4}
      tail=$(printf '%s' "$line" | grep -o 'extraction DONE[^;(:|]\{0,48\}' | head -1 | sed -E 's/[ ,]+(pending|not|NOT|waiting|next|queued|TODO|later|tbd|still|failed|FAIL).*//')
      if printf '%s' "$tail" | grep -qE "(^|[^0-9])($c|$ymd|$mmdd)([^0-9]|$)"; then
        printf '%s\n' "$line" > $G/extract_ok/$c.tmp && mv $G/extract_ok/$c.tmp $G/extract_ok/$c
        echo "$(date -u +%FT%TZ) $c <- $(printf '%s' "$line" | cut -c1-200)" >> $G/extract_watch.log
      fi
    done
  done
  sleep 30
done
