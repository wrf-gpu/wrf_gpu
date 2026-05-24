#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATCH_ROOT="$ROOT/external/wrf_savepoint_patch"
SOURCE_COPY="$PATCH_ROOT/source_copy"
SPRINT="$ROOT/.agent/sprints/2026-05-24-m6b0r-relink-completion"
CANONICAL="/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF"
ENV_SCRIPT="/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh"
STABLE="/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe"
EXPECTED_STABLE_SHA="1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37"
EXPECTED_WRF_HEAD="115e5756f98ee2370d62b6709baac6417d8f7338"

mkdir -p "$SPRINT"

stable_before="$(sha256sum "$STABLE" | awk '{print $1}')"
if [[ "$stable_before" != "$EXPECTED_STABLE_SHA" ]]; then
  echo "FATAL: operational wrf.exe sha changed before relink: $stable_before" >&2
  exit 1
fi
printf '%s  %s\n' "$stable_before" "$STABLE" > "$SPRINT/proof_source_copy_sha256.txt"

canonical_head="$(git -C "$CANONICAL" rev-parse HEAD)"
if [[ "$canonical_head" != "$EXPECTED_WRF_HEAD" ]]; then
  echo "FATAL: canonical WRF source drifted: $canonical_head" >&2
  exit 1
fi

rm -rf "$SOURCE_COPY"
mkdir -p "$SOURCE_COPY"
rsync -a "$CANONICAL/" "$SOURCE_COPY/"

if [[ ! -x /bin/csh ]]; then
  tcsh_path="$(command -v tcsh || true)"
  if [[ -z "$tcsh_path" ]]; then
    echo "FATAL: WRF compile scripts need csh/tcsh, and neither /bin/csh nor tcsh is available" >&2
    exit 7
  fi
  while IFS= read -r script; do
    sed -i "1s|^#!/bin/csh|#!$tcsh_path|" "$script"
  done < <(find "$SOURCE_COPY" -type f -exec grep -Il '^#!/bin/csh' {} +)
fi

(
  cd "$SOURCE_COPY"
  echo "source_copy=$SOURCE_COPY"
  echo "canonical=$CANONICAL"
  echo "source_head=$canonical_head"
  find . -type f \
    ! -path './.git/*' \
    ! -path './main/wrf.exe' \
    -print0 | sort -z | xargs -0 sha256sum
) >> "$SPRINT/proof_source_copy_sha256.txt"

(
  cd "$SOURCE_COPY"
  echo "source_head_before_patch=$canonical_head"
  patch -p1 < "$PATCH_ROOT/solve_em.F.patch"
  patch -p1 < "$PATCH_ROOT/configure.wrf.patch"
  rm -f configure.wrf.orig dyn_em/solve_em.F.orig

  # The committed wrapper artifact also contains the standalone shim program.
  # Full WRF relink needs only the module to avoid a duplicate main symbol.
  awk 'BEGIN{copy=0} /^module savepoint_wrapper/{copy=1} copy{print}' \
    "$PATCH_ROOT/dyn_em/savepoint_wrapper.F90" > dyn_em/savepoint_wrapper.F90

  if ! grep -q 'savepoint_wrapper.o' dyn_em/Makefile; then
    sed -i 's/module_small_step_em.o[[:space:]]*\\/module_small_step_em.o \t\t\\\n        savepoint_wrapper.o          \\/' dyn_em/Makefile
  fi
  if ! grep -q '^savepoint_wrapper.o:' dyn_em/Makefile; then
    sed -i '/^module_bc_em.o:/i savepoint_wrapper.o: savepoint_wrapper.F90\n\t$(RM) $@ savepoint_wrapper.mod\n\t$(FC) -o $@ -c $(FCFLAGS) $(OMP) $(MODULE_DIRS) $(PROMOTION) $(FCSUFFIX) $<\n' dyn_em/Makefile
  fi

  # The canonical tree carries a known-good small_step_gpu2.o. Rebuilding it
  # with NVHPC 26.3 leaves an unresolved device ieee_is_finite helper at final
  # wrf.exe link, unrelated to the savepoint patch. Keep the copied object
  # newer than its source so this relink rebuilds only the changed WRF units.
  touch dyn_em/small_step_gpu2.o main/small_step_gpu2.mod

  echo "source_head_after_patch=$canonical_head"
  diff -qr "$CANONICAL" "$SOURCE_COPY" | sort || true
  if find . -name '*.rej' -print -quit | grep -q .; then
    echo "FATAL: rejected patch hunks found" >&2
    find . -name '*.rej' -print >&2
    exit 2
  fi
) 2>&1 | tee "$SPRINT/proof_patch_apply.txt"

# shellcheck source=/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
source "$ENV_SCRIPT"

(
  cd "$SOURCE_COPY"
  ./compile em_real -j 4
) 2>&1 | tee "$SPRINT/proof_compile.txt"

if [[ ! -x "$SOURCE_COPY/main/wrf.exe" ]]; then
  echo "FATAL: relinked WRF executable missing: $SOURCE_COPY/main/wrf.exe" >&2
  exit 3
fi
if ! grep -q -- '-DWRF_SAVEPOINT' "$SOURCE_COPY/configure.wrf"; then
  echo "FATAL: configure.wrf does not contain -DWRF_SAVEPOINT" >&2
  exit 4
fi

relinked_sha="$(sha256sum "$SOURCE_COPY/main/wrf.exe" | awk '{print $1}')"
if [[ "$relinked_sha" == "$EXPECTED_STABLE_SHA" ]]; then
  echo "FATAL: relinked wrf.exe unexpectedly matches operational sha" >&2
  exit 5
fi
printf '%s  %s\n' "$relinked_sha" "$SOURCE_COPY/main/wrf.exe" > "$SPRINT/proof_relinked_sha256.txt"

stable_after="$(sha256sum "$STABLE" | awk '{print $1}')"
printf '%s  %s\n' "$stable_after" "$STABLE" > "$SPRINT/proof_operational_unchanged_post_relink.txt"
if [[ "$stable_after" != "$EXPECTED_STABLE_SHA" ]]; then
  echo "FATAL: operational wrf.exe sha changed after relink: $stable_after" >&2
  exit 6
fi

echo "relink_complete=true"
echo "source_copy=$SOURCE_COPY"
echo "relinked_wrf=$SOURCE_COPY/main/wrf.exe"
echo "relinked_sha256=$relinked_sha"
echo "operational_sha256_before=$stable_before"
echo "operational_sha256_after=$stable_after"
