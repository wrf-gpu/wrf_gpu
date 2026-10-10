# Scheme support matrix — v0.3.4

This page states what `gpuwrf run` does with each physics code in `namelist.input`. From v0.3.4 every option is either bound and run, or refused before any compute with a named reason. Nothing is silently replaced by a default; wrf_gpu ≤ 0.3.3 did that, see [KNOWN_ISSUES.md](../../KNOWN_ISSUES.md).

Operational preflight is authoritative. The machine-readable catalog is [`src/gpuwrf/io/scheme_catalog.py`](../../src/gpuwrf/io/scheme_catalog.py); the per-key binding rules are in [namelist compatibility](../namelist-compatibility.md#cli-binding-v034).

## Status levels

| Status | Meaning |
|---|---|
| **Release-validated** | Part of the configuration compared with original CPU-WRF in full forecasts on the GPU (v0.3.3 eight-arm validation, reused for v0.3.4 because the default program is unchanged) |
| **Experimental, GPU-smoked** | Faithful port matched an unmodified-WRF oracle on CPU; the GPU compiled it and ran a few finite steps of a small real case. No forecast-level comparison |
| **Experimental, CPU-oracle** | Faithful port matched an unmodified-WRF oracle on CPU and runs in the CPU coupled path. GPU compile/run not yet qualified |
| **Catalog, unvalidated** | Older oracle-tested port that the CLI now binds and runs. Never compared with CPU-WRF in a coupled forecast; treat results as unvalidated |
| **Refused** | Recognized WRF option that the port cannot run faithfully. `gpuwrf run` stops before compute and names the reason |

## Release-validated configuration

| Key | Code | Scheme |
|---|---|---|
| `mp_physics` | 8 | Thompson |
| `bl_pbl_physics` | 5 | MYNN-EDMF (`icloud_bl=1`, WRF default sub-options) |
| `sf_sfclay_physics` | 5 | MYNN surface layer |
| `ra_lw_physics` / `ra_sw_physics` | 4 / 4 | RRTMG |
| `sf_surface_physics` | 4 | Noah-MP (WRF default `opt_*`) |
| `cu_physics` | 1 / 0 | Kain–Fritsch on outer 9 km domains; none on convection-permitting nests |
| `&dynamics` | — | as set in the validation namelists: `diff_opt=1`/`km_opt=4`, epssm 0.5, damp_opt 3, zdamp 5000, dampcoef 0.2, w_damping 1, diff_6th 2/0.12. From v0.3.4 an *omitted* key takes the WRF Registry default (e.g. epssm 0.1, w_damping 0, diff_6th_opt 0), not these values |

## Experimental schemes added in v0.3.4

GPU smoke = Swiss 42×42 d01 case with only that option changed from the release defaults, on an RTX 5090: cold compile, three model steps, all fields finite, static kernel check (E41: no per-thread stack above 256 B) [M, 2026-10-10]. It is a compile-and-run check, not a forecast comparison.

| Key = code | Scheme | Status | Scheme-specific gaps |
|---|---|---|---|
| `bl_pbl_physics=9` | CAM-UW (Bretherton–Park) | Experimental, GPU-smoked | Surface term uses the total radiative heating where WRF uses the LW part (daytime difference); no coupled forecast comparison |
| `cu_physics=5` / `93` | Grell-3D ensemble / Grell–Devenyi | Experimental, GPU-smoked | Positive `cudt` refused; the downdraft-origin search and GD negative-check branches are not exercised by the oracle; no forecast comparison |
| `cu_physics=4` | Scale-aware SAS | Experimental, GPU-smoked, **programmatic API only** | `gpuwrf run` refuses it, as WRF ARW itself does (`check_a_mundo`); bitwise vs the pristine WRF REAL build on the oracle columns; WRF STEPCU held-rate cadence; no forecast comparison |
| `mp_physics=18` | NSSL two-moment | Experimental, CPU-oracle (GPU not run) | fp64 port matches the pristine-WRF oracle to 1e-12 on 17 cases; root-boundary extra species follow WRF's flow-dependent boundary; combination with any cumulus scheme is refused; not compiled or run on the GPU; no forecast comparison |
| `km_opt=3` (with `diff_opt=2`) | 3-D Smagorinsky | Experimental, GPU-smoked | `isfflx`, `tke_drag_coefficient`, `tke_heat_flux` are not read (WRF defaults 1/0/0 apply); vertical diffusion only with `bl_pbl_physics=0` |
| `ra_lw_physics=3` / `ra_sw_physics=3` | CAM radiation | Experimental, GPU-run finite — fails the static stack check (2,144 B per thread spill, a performance/robustness issue; fix planned for 0.4.0) | Sea-ice fraction treated as 0; uniform CAM aerosol climatology; absorptivity/emissivity recomputed at WRF `cam_abs_freq_s` cadence (default 6 h) |
| `sf_surface_physics=3` | RUC land surface | Experimental, GPU-smoked, **programmatic API only** | `gpuwrf run` still refuses it; WRF SFCDIAGS 2 m diagnostics not ported; sea-ice/lake points refused |

## Catalog schemes, unvalidated

These codes are bound by the CLI and their production step builds on a real case (measured CPU trace through the real CLI loader, [namelist compatibility](../namelist-compatibility.md#cli-binding-v034)). That is not a forecast or GPU qualification. Codes marked *fp64 island* crashed at trace time under the fp32 release carry in earlier versions; their scheme now runs in an explicit fp64 island inside the fp32 model, which is slower than a native fp32 kernel.

| Key | Codes |
|---|---|
| `mp_physics` | 0 (none); *fp64 island*: 1 Kessler, 2 Lin, 3 WSM3, 4 WSM5, 6 WSM6, 10 Morrison, 13 SBU-YLin, 14 WDM5, 16 WDM6, 97 Goddard |
| `bl_pbl_physics` (+ partner `sf_sfclay_physics`) | 0 (+0); 1 YSU (+1); 2 MYJ (+2); 3 GFS (+3); 7 ACM2 (+1), 8 BouLac (+1), 12 GBM (+1) — *fp64 island*; 11 Shin-Hong (+1); 99 MRF (+1) |
| `sf_sfclay_physics` | 1, 2, 3, 7, 91 with a compatible PBL (pairing violations are refused) |
| `cu_physics` | 1 Kain–Fritsch outside the validated outer-domain use, 2 BMJ, 3 Grell–Freitas, 6 Tiedtke, 16 New Tiedtke |
| `ra_lw_physics` / `ra_sw_physics` | LW 1 RRTM, 31 Held–Suarez (SW 0); SW 1 Dudhia, 2 Goddard (*fp64 island*) |
| `sf_surface_physics` | 0 (prescribed bulk surface) |
| `diff_opt` / `km_opt` | `diff_opt=0` (no explicit diffusion) |

## Refused

| Selection | Reason |
|---|---|
| `mp_physics=24/26/28/40` (WSM7, WDM7, aerosol-aware Thompson, Morrison-aerosol) | The root lateral-boundary update does not represent their extra species (hail, aerosol numbers, Ns/Ng/Nc) — demoted in v0.3.4; the column kernels remain oracle-tested |
| `sf_surface_physics` other than 0/4 under `gpuwrf run` | The native driver builds only the bulk surface and Noah-MP; slab (1), RUC (3, programmatic API only) and Pleim–Xiu (7) are refused by the CLI |
| PBL/surface-layer pairs WRF does not allow | e.g. `bl_pbl_physics=7` with `sf_sfclay_physics=7`; `bl_pbl_physics=1` with 91 |
| Unbound keys set away from the WRF default | e.g. `rk_ord`, `smdiv`, `non_hydrostatic=.false.`, Noah-MP `dveg`/`opt_*`, `hypsometric_opt` ≠ 2 |
| `km_opt=2` / `km_opt=5` | Older TKE/SMS scaffolds, not WRF-qualified; non-finite in unit tests under the fp32 release carry — refused in v0.3.4 |
| `diff_opt=2` with `km_opt=1` (constant K), and any other pair without an operational path | ≤0.3.3 ran these with *no* explicit diffusion; now refused |
| Per-domain selections WRF rejects or overrides | physics keys that must match on all domains; `mp_physics` that differs between domains |

Every other WRF code is refused. The preflight message names the reason and, where one exists, a supported alternative.
