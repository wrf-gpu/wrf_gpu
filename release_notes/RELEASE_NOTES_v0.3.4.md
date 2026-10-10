# v0.3.4 — namelist fix release

v0.3.4 is a **fix release**. It makes `gpuwrf run` honour, or loudly refuse, every physics and dynamics option in `namelist.input`. It also fixes a CPU solver hang and adds standalone restart and opt-in experimental schemes. The validated default forecast program is unchanged from v0.3.3.

## Action required: check runs that changed physics options

**wrf_gpu 0.3.3 and earlier silently ignored non-default `mp_physics`, `bl_pbl_physics`, `sf_sfclay_physics`, `ra_lw_physics` and `ra_sw_physics`.** The `&dynamics` damping and sixth-order-filter settings were ignored too. The namelist check accepted the requested scheme, but the forecast ran the release suite: Thompson, MYNN, MYNN surface layer and RRTMG. **Re-run any forecast whose namelist selected something else:** its output came from the default suite.

Runs that used the release configuration asked for what actually ran. They are unaffected. Details and the full key list: [KNOWN_ISSUES.md](../KNOWN_ISSUES.md#-v033-gpuwrf-run-silently-ignored-non-default-physics-options-fixed-in-v034).

## What changed

- **Every scheme option is bound or refused (P0).** Both CLI drivers now bind the scheme and dynamics keys per domain. Before any compute, `gpuwrf run` refuses any explicit value it would not run. That covers unknown keys, unbound keys set away from the WRF default, unsupported land options, and per-domain combinations WRF itself rejects. The [support matrix](../docs/release/SUPPORT_MATRIX_v0.3.4.md) lists each code's status.
- **Older catalog schemes that crashed now run, or are refused.** Several catalog microphysics, PBL and radiation codes failed to build under the fp32 release carry (they were never actually reachable from the CLI before). They now run their scheme in an explicit fp64 island: `mp_physics` 1–16 and 97, `bl_pbl_physics` 7/8/12, `ra_sw_physics=2`. These are unvalidated catalog schemes, not forecast-qualified configurations. `mp_physics` 24/26/28/40 are refused: the root lateral-boundary update does not carry their extra species. Constant-K diffusion (`diff_opt=2`, `km_opt=1`), which previously ran with no explicit diffusion, and the unqualified `km_opt=2`/`5` closures are refused too.
- **CPU tridiagonal solver no longer hangs.** On the CPU backend, jaxlib's LAPACK tridiagonal solve could deadlock XLA's thread pool when many independent solves ran at once, particularly with few pinned cores. CPU runs now use an equivalent Thomas-algorithm scan that agrees with LAPACK within round-off; the GPU lowering is byte-identical [M].
- **Standalone checkpoint/resume.** Native single-domain d01 runs can now write verified checkpoint generations, resume after a crash in the same output stream, and continue a finished run to a later end with `--extend-run`. On the GPU with release defaults, a one-hour Swiss run killed with SIGKILL and resumed in a fresh process matched the uninterrupted run byte for byte in all 754 history variables [M]; a two-hour CPU run on the legacy path did the same (3/3 files) [M]. Checkpoint overhead on the GPU is not measured. Nested checkpoint/resume is unchanged. CPU-WRF replay runs still refuse these flags.
- **Runs past the boundary data are refused.** A native run whose end lies beyond the `wrfbdy_d01` coverage now stops before it starts, instead of holding the last boundary record.
- **Experimental opt-in schemes.** New faithful ports run only when the namelist selects them. Each matched an unmodified-WRF oracle on CPU; GPU status is in the [support matrix](../docs/release/SUPPORT_MATRIX_v0.3.4.md). They are **not** validated forecast configurations.
  Added: CAM-UW PBL (`bl_pbl_physics=9`), Grell-3D and Grell–Devenyi cumulus (`cu_physics=5`, `93`), 3-D Smagorinsky (`km_opt=3`), CAM longwave/shortwave radiation (`ra_lw_physics=3`, `ra_sw_physics=3`), the RUC land model (`sf_surface_physics=3`) scale-aware SAS cumulus (`cu_physics=4`) and NSSL two-moment microphysics (`mp_physics=18`, CPU only; not yet run on the GPU); RUC and SAS are reachable only through the programmatic API (`gpuwrf run` refuses them; WRF ARW itself refuses `cu_physics=4`).
  On an RTX 5090, CAM-UW, Grell-3D, Grell–Devenyi, 3-D Smagorinsky, RUC and SAS each compiled and ran three finite steps of the Swiss case and passed the static kernel stack check [M]. CAM radiation ran finite but still fails that check (2,144 B per-thread stack spill); the fix is planned for 0.4.0.

## Validation

The release default program (Thompson / MYNN / MYNN surface layer / RRTMG / Noah-MP, with Kain–Fritsch where configured) traces to the same program as v0.3.3: the production step traced through the real CLI loader on the Swiss case is identical at tag v0.3.3 and in v0.3.4 (location-stripped jaxpr), and the PROD Canary d01 program is identical before and after the namelist-binding change [M, CPU trace]. Its v0.3.3 original-CPU validation therefore carries over unchanged; v0.3.4 adds no new forecast comparison. See the [v0.3.3 release notes](RELEASE_NOTES_v0.3.3.md) for those results and disclosed deviations.

Experimental schemes carry CPU oracle evidence and, for most, a GPU smoke run. Compiling and running a few finite GPU steps is not a forecast validation. Speed and energy figures retain their dated v0.3.2 measurements.

## Known issues

- All v0.3.3 disclosed deviations remain: 0115 near-threshold rain, late coastal V10/wind-speed excess, 0408 trace graupel, 0115 cirrus lifetime and brief WRF-order negative moisture. See [KNOWN_ISSUES.md](../KNOWN_ISSUES.md).
- Open v0.3.3 follow-ups, not addressed here: coastal 10 m wind-speed excess, nine unpinned E41 loop-layout invariants, unproven negative-moisture amplitude versus WRF, and the small-grid firewall cost (about 12% more kernels on grids with ≤4,096 columns).
- Experimental schemes have scheme-specific gaps listed in the support matrix. For example, CAM-UW uses the total radiative heating in its surface term, and RUC is reachable only through the programmatic API.

## Upgrading

The executable cache is keyed by version: the **first v0.3.4 run on each installation recompiles once** (cold compile time), even for the unchanged default program. Namelists that set options the port does not run now stop with `NamelistNotHonouredError` and a named reason, instead of silently running defaults. Remove or correct those keys. **Omitted `&dynamics` keys now take WRF's defaults** (for example `epssm` 0.1, `w_damping` 0, `diff_6th_opt` 0), as WRF does; ≤0.3.3 silently used 0.5, 1 and 2. Namelists that set these keys explicitly, like all validation cases, run the unchanged program. See the [user guide](https://wrf-gpu.github.io/wrf_gpu/) for installation.

**Release identity:** v0.3.4 · the commit tagged `v0.3.4`; default forecast program identical to v0.3.3 (`be0dd543c9f7f6d619a8b504348b11267ffe913f`).
