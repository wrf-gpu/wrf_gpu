#!/usr/bin/env bash
# Isolate the base-vs-s0s1 divergence (1 hour, cheap):
#   (1) base@tempdir vs base@tempdir  -> path/tempdir-confound control
#   (2) base@tempdir vs s0s1@worktree -> the real S0+S1 effect
# HEAD-vs-HEAD @1h was already 0/56, so the forecast is deterministic at 1h;
# if (1)==0 and (2)>0, the diff is REAL code. If (1)>0, it's a path confound.
set -euo pipefail
ROOT="<USER_HOME>/src/wrf_gpu2/.claude/worktrees/agent-a9a81584cb39b24b6"
PROBE="proofs/perf/v015/probe_ab_identity.py"
OUT="$ROOT/proofs/perf/v016"; mkdir -p "$OUT"; cd "$ROOT"

BASE_SRC="$(mktemp -d /tmp/v016_base_src.XXXXXX)"
git archive 499d3724 src | tar -x -C "$BASE_SRC"
for entry in "$ROOT"/* "$ROOT"/.[!.]*; do
  name="$(basename "$entry")"; case "$name" in src|.git|.git*) continue;; esac
  [ -e "$entry" ] && ln -sfn "$entry" "$BASE_SRC/$name"
done

run () { PYTHONPATH="$2" python "$PROBE" --tag "$1" --hours 1 --dump-state; cp -f "proofs/perf/v015/ab_$1.json" "$OUT/"; }
cmp () { PYTHONPATH="$ROOT/src" python - "$1" "$2" "$3" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))["hashes"]; b=json.load(open(sys.argv[2]))["hashes"]; lab=sys.argv[3]
mm=[]; tot=0
for h in sorted(set(a)|set(b)):
    al=a.get(h,{});bl=b.get(h,{})
    for k in sorted(set(al)|set(bl)):
        tot+=1
        if al.get(k)!=bl.get(k): mm.append(k)
print(f"[{lab}] {len(mm)}/{tot} mismatches; leaves={sorted(mm)}")
PY
}

run base_iso_a "$BASE_SRC/src"
run base_iso_b "$BASE_SRC/src"
run s0s1_iso   "$ROOT/src"
echo "----- ISOLATION RESULTS (1h) -----"
cmp "$OUT/ab_base_iso_a.json" "$OUT/ab_base_iso_b.json" "CONTROL base@tmp vs base@tmp"
cmp "$OUT/ab_base_iso_a.json" "$OUT/ab_s0s1_iso.json"   "EFFECT  base@tmp vs s0s1@worktree"
rm -rf "$BASE_SRC"
echo "----- done -----"
