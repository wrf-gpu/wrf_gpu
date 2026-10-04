#!/usr/bin/env bash
# ADR-031 S0+S1 Tier-S bit-identity gate.
# Runs the v0.15 AB identity probe (3 production hours, sha256 per state leaf)
# on (A) the base commit 499d3724 source and (B) the S0+S1 HEAD source, back to
# back under ONE GPU lock acquisition, then compares the per-hour leaf hash maps.
# PASS = 0 mismatching leaf hashes (the FP64_DEFAULT forecast graph is unchanged).
set -euo pipefail

ROOT="<USER_HOME>/src/wrf_gpu2/.claude/worktrees/agent-a9a81584cb39b24b6"
BASE_REF="499d3724"
PROBE="proofs/perf/v015/probe_ab_identity.py"
OUT="$ROOT/proofs/perf/v016"
mkdir -p "$OUT"

cd "$ROOT"

# Materialize the base-commit source into a temp dir WITHOUT mutating the worktree.
BASE_SRC="$(mktemp -d /tmp/v016_base_src.XXXXXX)"
echo "[ab] materializing base ($BASE_REF) src -> $BASE_SRC" >&2
git archive "$BASE_REF" src | tar -x -C "$BASE_SRC"
# The package resolves repo-relative assets/scripts as parents[3] of the module
# (= repo root) at import time (Thompson .npz, scripts/extract_rrtmg_tables.py, ...).
# This refactor touches ONLY src/gpuwrf, so every other top-level entry is
# byte-identical to base -- symlink them all (except src, which must stay the base
# checkout) into the base-src root so the base import finds every asset/script.
for entry in "$ROOT"/* "$ROOT"/.[!.]*; do
  name="$(basename "$entry")"
  case "$name" in
    src|.git|.git*) continue;;
  esac
  [ -e "$entry" ] && ln -sfn "$entry" "$BASE_SRC/$name"
done

run_probe () {  # $1 = tag, $2 = PYTHONPATH src dir
  local tag="$1" srcdir="$2"
  echo "[ab] === run tag=$tag (src=$srcdir) ===" >&2
  PYTHONPATH="$srcdir" python "$PROBE" --tag "$tag" --hours 3
  cp -f "proofs/perf/v015/ab_${tag}.json" "$OUT/ab_${tag}.json"
}

# (A) base-commit graph, (B) S0+S1 HEAD graph -- same process pool, same GPU.
run_probe base "$BASE_SRC/src"
run_probe s0s1 "$ROOT/src"

echo "[ab] comparing hash maps -> $OUT/s0s1_identity_verdict.json" >&2
PYTHONPATH="$ROOT/src" python - "$OUT/ab_base.json" "$OUT/ab_s0s1.json" "$OUT/s0s1_identity_verdict.json" <<'PY'
import json, sys
base = json.load(open(sys.argv[1]))["hashes"]
cand = json.load(open(sys.argv[2]))["hashes"]
mismatches = []
total = 0
hours = sorted(set(base) | set(cand))
for h in hours:
    bl = base.get(h, {}); cl = cand.get(h, {})
    for leaf in sorted(set(bl) | set(cl)):
        total += 1
        if bl.get(leaf) != cl.get(leaf):
            mismatches.append({"hour": h, "leaf": leaf,
                               "base": bl.get(leaf), "s0s1": cl.get(leaf)})
verdict = {
    "schema": "ADR031_S0S1_TierS_BitIdentity",
    "base_ref": "499d3724",
    "candidate": "worker/opus/v016-fp32-s0s1 HEAD",
    "leaf_hash_comparisons": total,
    "mismatches": len(mismatches),
    "pass": len(mismatches) == 0,
    "mismatch_detail": mismatches[:50],
    "base_wall_s": None,
    "s0s1_wall_s": None,
}
json.dump(verdict, open(sys.argv[3], "w"), indent=2)
print(json.dumps({k: v for k, v in verdict.items() if k != "mismatch_detail"}, indent=2))
print(f"\nADR-031 S0+S1 Tier-S: {len(mismatches)}/{total} leaf-hash mismatches "
      f"-> {'PASS (byte-identical)' if not mismatches else 'FAIL'}")
PY

rm -rf "$BASE_SRC"
echo "[ab] done" >&2
