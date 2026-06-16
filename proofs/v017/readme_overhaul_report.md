# v0.17 Public README Overhaul — Report

**Worker:** Opus doc worker (`worker/opus/v017-readme`)
**Date:** 2026-06-16
**Base:** `worker/opus/v017-release` (the exact shipped v0.17 tree, HEAD 99c1b83c)
**Objective:** Bring the stale, v0.15-centric public README to current v0.17 state —
leaner, honest, with the 4 principal-directed fixes. CPU-only (no GPU used).

README: 606 → 475 lines. All local links and embedded images verified to resolve;
all internal anchors match headings.

---

## FIX 1 — Headline + first paragraph (DONE)

Rewrote the lead to state plainly **what the project is** (GPU-native,
WRF-compatible regional model; clean JAX rewrite, not a Fortran port), **what it
can/is-good-for** (real ARW forecasts on a GPU; capability the CPU stack can't
reach on one box — 1 km fits one card MEASURED, cluster PROJECTED; transparent
forkable artifact), and **what it is NOT** (not universal WRF v4; not proven for
24/72 h forecast-skill equivalence; not a single-card speedup story; no
DFI/FDDA/WRF-Chem/etc.). Removed the two stacked v0.15/v0.17 narrative blocks that
opened the old file.

Added a prominent **"First run is slow on purpose, then fast"** callout box up top:
the first run JIT-compiles the GPU kernels (**~8–12 min one-time cold compile, no
output**); the **persistent on-disk cache** makes later runs a fast read
(`cold ~147 s → cache-hit ~29 s`, bit-identical); the opt-in fused fast-mode
(`GPUWRF_NESTED_FUSE=1`) carries a **separate ~38 min one-time compile (cached)**.
This is repeated in the Quickstart and the resource-profile table.

## FIX 2 — WRF-v4 identity proof section (DONE — 3 proofs built/embedded)

Replaced the old **v015** embeds (`docs/assets/v015/...`, 2 regions) with **three
current v0.17 dashboards** pointing at `docs/assets/v017/...`:

1. **Switzerland d01 72 h** — already present (v017).
2. **Canary L2 d02 72 h (nested)** — already present (v017).
3. **Canary L2 d01 72 h (parent)** — **BUILT this sprint** with the CPU-only
   builder (`scripts/build_identity_proof_plots.py`), from retained paired data
   (CPU truth `…/wrf_l2_backfill_output/20260501_18z_l2_72h_…` d01 + GPU
   `…/v017_canary_d02_72h_identity_fast_…/gpu_output/…` d01). Manifest +
   5 PNGs committed under `proofs/v017/identity_proof/canary_l2_d01/` and
   `docs/assets/v017/identity_proof/canary_l2_d01/`.

### The "4 plots, all green" investigation — precise result

**Important honesty correction:** the retained data supports **3** distinct
identity dashboards, and **none is 10/10 "all green"** — each is **9/10 within
frozen tolerance** with the **dynamics/thermodynamics core cell-for-cell identical
(r ≈ 0.99–1.00)** and **exactly one bounded diagnostic per region drawn RED**
(never painted green). This matches the builder's honesty contract and the v0.15
gate history.

| Region | Built | Headline | Bounded miss (drawn red) |
|---|---|---|---|
| Switzerland d01 72 h | yes (v017, pre-existing) | 9/10 within | RAINNC 5.08 mm vs 1.0 mm (5.08× limit) |
| Canary L2 d02 72 h (nested) | yes (v017, pre-existing) | 9/10 within | QVAPOR 1.44×10⁻³ vs 1.0×10⁻³ (1.44× limit) |
| Canary L2 d01 72 h (parent) | **yes — built this sprint** | 9/10 within | QVAPOR 1.23×10⁻³ vs 1.0×10⁻³ (1.23× limit) |

**Why not a 4th.** I searched all retained `/mnt/data/wrf_gpu_validation/v017*`
GPU outputs and `/mnt/data/canairy_meteo/runs/` CPU truth. The only additional
GPU/CPU pairs with retained 72 h history are the three above. Specifically:

- **Switzerland d02:** no GPU d02 retained — the Switzerland identity_fast run is
  d01-only (73 files, d01).
- **Canary d03 (1 km):** no retained GPU d03 72 h `wrfout` (the d03 7 h replay dir
  retained only proof JSON, no per-hour history).
- **"Big Switzerland" larger domain:** CPU16 history is retained, but the GPU bench
  dir retained **no** per-hour `wrfout` (only a benchmark JSON) → not pairable.
- **all-7 9-domain run (`v017_all7_OPUS`, 20260428_18z):** retains d01–d09 GPU
  history but **only 4 leads (19–22 h)**, and its nest geometry does **not match**
  any retained CPU truth (e.g. GPU d03 102×69 vs L3 CPU d03 93×75) → not pairable.

**Manager decision needed (low stakes):** the prompt's framing ("there should be 4
plots, all green") is not achievable on retained data without fabrication. I
embedded the **3 honest 9/10 proofs** and labelled them precisely. If a true 4th
proof is wanted, it requires a **new GPU run** (no GPU was used this sprint), e.g.
a Switzerland d02 72 h GPU run paired to a CPU d02 truth, or a Canary d03 1 km 72 h
GPU run paired to matching-geometry CPU truth. Recommend shipping the 3 as-is — the
identity story (dynamics/thermo cell-for-cell, bounded diagnostics drawn red) is
fully made by 3 regions across 2 independent cases and both parent + nest domains.

## FIX 3 — TOST → cell-identity framing (DONE)

Removed the TOST-centric "Statistical honesty" paragraph and the "TOST scoring path
unblocked" validation bullet. The README now frames the **cell-identity proof**
(the grid-delta-atlas / per-cell, per-lead, per-variable identity proof against a
frozen tolerance manifest) as the **primary fidelity gate**, and states explicitly
that the cell-identity proof **supersedes** the earlier TOST framing. Kept the
honesty intact: the **dynamics/thermo core is proven cell-for-cell identical**;
the broader **24 h/72 h forecast-skill equivalence (T2/U10/V10) is the open
credibility gate (KI-9)** — preserved verbatim in the lead, the identity section,
and Honest boundaries. KI-5 reworded to "superseded by cell-identity as the primary
gate; no TOST PASS claimed."

## FIX 4 — Version notes → lean table (DONE)

Replaced ~10 long prose version paragraphs (the entire "Current status",
"Historical — v0.15.0", and the cumulative-capability prose) with a single
**Version history table** (columns: version | one-line headline | key proof/link).
v0.17 and v0.16 rows are slightly fuller; v0.15 medium; v0.14→v0.1.0 one-liners.
The detailed per-release evidence prose (hundreds of lines under "Validation
(v0.14.0)" / carried-evidence bullets) was cut — it lives in the `RELEASE_NOTES_v*`
files and `proofs/`, which the table links.

## General cleanups (DONE)

- Consolidated the two near-duplicate Quickstart/Run blocks into one Quickstart
  (single-domain + nested) — removed the redundant "## Run" section.
- Merged the resource-profile + performance content; added a dedicated, honest
  **Performance** section (default ~parity bit-identical; opt-in fuse ~1.27–1.30×
  not bitwise; launch/occupancy ceiling; capability headline MEASURED/PROJECTED).
- Kept: "Use the manager" quickstart, MEASURED/PROJECTED labels throughout, the
  capability headline (1 km MEASURED / cluster PROJECTED), whole-Earth@1km
  PROJECTED note, the fail-closed scope table, the GPU-operational physics menu,
  the roadmap table, core goals, "where to look first", known issues, layout.
- Updated section titles to v0.17 ("Known issues (v0.17.0)", "Honest boundaries —
  what is NOT claimed", roadmap "post-v0.14.0" → current).

### Stale links fixed (were already broken in the shipped README)

- `PROJECT_PLAN.md`, `PROJECT_SCOPE.md`, `PROJECT_SPEC.md` — **do not exist** in the
  shipped tree (3+ broken refs in the old README). Repointed to
  `PROJECT_CONSTITUTION.md` / `CHANGELOG.md` / `docs/GPU_PORT_GAPS_TODO.md`.
- `.agent/reviews/` — **not shipped publicly** (the public `.agent/` tree contains
  only `skills/`; this matches the v0.17 release-assembly "FIX1 .agent tree"). The
  2 old refs were broken; removed and repointed to in-repo docs.

After the fixes, a full link sweep reports **ALL LOCAL LINKS RESOLVE** and all 4
internal anchors match their headings.

---

## Files changed / added

- `README.md` (rewritten, 606 → 475 lines)
- `docs/assets/v017/identity_proof/canary_l2_d01/` (5 PNGs — NEW, CPU-built)
- `proofs/v017/identity_proof/canary_l2_d01/identity_proof_manifest.json` (NEW)
- `proofs/v017/readme_overhaul_report.md` (this report)

## Commands run (CPU-only)

```bash
taskset -c 0-3 python3 scripts/build_identity_proof_plots.py \
  --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
  --gpu-dir /mnt/data/wrf_gpu_validation/v017_canary_d02_72h_identity_fast_20260615T024626Z/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z \
  --domain d01 --init "2026-05-01T18:00:00+00:00" \
  --case-id canary_l2_d01_72h --region-label "Canary L2 d01 72h (2026-05-01 18Z, v0.17)" \
  --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
  --proof-dir proofs/v017/identity_proof/canary_l2_d01 \
  --asset-dir docs/assets/v017/identity_proof/canary_l2_d01
# → 9/10 within, worst QVAPOR 1.23× limit, 72 leads, 5 plots
```

## Unresolved risks / decision needed

- **The "4 green plots" expectation is not met by retained data** (see FIX 2). 3
  honest 9/10 proofs are embedded. A 4th requires a new GPU run — manager to decide
  whether to commission one or ship 3. (Recommend: ship 3.)
- No GPU was touched; the identity builder is CPU-only and deterministic. The
  manager should still eyeball the 3 embedded dashboards before re-publishing.
