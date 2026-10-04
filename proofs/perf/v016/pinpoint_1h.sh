#!/usr/bin/env bash
# Pinpoint WHERE strict bitwise breaks, with a CLEAN isolated compile cache per
# run (eliminates the persistent-cache + tempdir-path confounds). 1 production
# hour each. Trees:
#   base   = 499d3724           (pre-S0)
#   s0     = 12ed00be           (S0 only: inert carry BaseState leaf)
#   s0s1   = worktree HEAD      (S0 + S1 base seam)
# Comparisons:
#   base-vs-base2 : determinism/cache CONTROL (same tree twice)
#   base-vs-s0    : does ADDING the carry leaf alone flip bits? (XLA scan codegen)
#   base-vs-s0s1  : the full refactor
#   s0-vs-s0s1    : does the S1 seam alone (given S0) flip bits?
set -euo pipefail
ROOT="<USER_HOME>/src/wrf_gpu2/.claude/worktrees/agent-a9a81584cb39b24b6"
PROBE="proofs/perf/v015/probe_ab_identity.py"
OUT="$ROOT/proofs/perf/v016"; mkdir -p "$OUT"; cd "$ROOT"

mktree () {  # ref -> echoes a tempdir with that ref's src + asset symlinks
  local ref="$1" d; d="$(mktemp -d /tmp/v016_tree.XXXXXX)"
  git archive "$ref" src | tar -x -C "$d"
  for entry in "$ROOT"/* "$ROOT"/.[!.]*; do
    local name; name="$(basename "$entry")"; case "$name" in src|.git|.git*) continue;; esac
    [ -e "$entry" ] && ln -sfn "$entry" "$d/$name"
  done
  echo "$d"
}

BASE_T="$(mktree 499d3724)"
S0_T="$(mktree 12ed00be)"
WORK_SRC="$ROOT/src"

run () {  # tag srcdir  -- each run gets its OWN empty cache dir
  local tag="$1" src="$2" cache; cache="$(mktemp -d /tmp/v016_cache.XXXXXX)"
  echo "[pp] run $tag (src=$src cache=$cache)" >&2
  JAX_COMPILATION_CACHE_DIR="$cache" PYTHONPATH="$src" python "$PROBE" --tag "$tag" --hours 1
  cp -f "proofs/perf/v015/ab_$tag.json" "$OUT/"
  rm -rf "$cache"
}
cmp () { PYTHONPATH="$WORK_SRC" python - "$1" "$2" "$3" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))["hashes"]; b=json.load(open(sys.argv[2]))["hashes"]; lab=sys.argv[3]
mm=[];tot=0
for h in sorted(set(a)|set(b)):
    al=a.get(h,{});bl=b.get(h,{})
    for k in sorted(set(al)|set(bl)):
        tot+=1
        if al.get(k)!=bl.get(k): mm.append(f"{h}:{k}")
print(f"[{lab}] {len(mm)}/{tot} mismatches"+("" if not mm else f"  e.g. {mm[:8]}"))
PY
}

run base_pp  "$BASE_T/src"
run base2_pp "$BASE_T/src"
run s0_pp    "$S0_T/src"
run s0s1_pp  "$WORK_SRC"
echo "================ PINPOINT (1h, isolated cache) ================"
cmp "$OUT/ab_base_pp.json"  "$OUT/ab_base2_pp.json" "CONTROL base-vs-base2 (same tree)"
cmp "$OUT/ab_base_pp.json"  "$OUT/ab_s0_pp.json"    "S0 carry-leaf  base-vs-s0"
cmp "$OUT/ab_base_pp.json"  "$OUT/ab_s0s1_pp.json"  "S0+S1 full     base-vs-s0s1"
cmp "$OUT/ab_s0_pp.json"    "$OUT/ab_s0s1_pp.json"  "S1 seam only   s0-vs-s0s1"
rm -rf "$BASE_T" "$S0_T"
echo "================ done ================"
