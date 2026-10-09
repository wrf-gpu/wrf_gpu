# W3 release documentation preparation — 2026-10-07 09:00Z

W2 packet `5b87589d7` stays available for its current critic. Its 113 figures
and 657 paired domain-hours remain W2 diagnostics; no acceptance is inherited
by the W3 numerical source. A7/A8 and the corrected 0115 comparator definition
are preserved. The W3 source commit is pending. A9 is frozen at `ae2ac4381879` (2026-10-07 09:01:30Z); it keeps A5/A7/A8 inputs and adds the pre-data W3 clear-sky key-in-effect check.

## Named shortwave defect and proposed correction

Original WRF RRTMG `reftra_sw` uses `kmodts=2`, PIFM:
`gamma1=(8−omega*(5+3*g))/4`, `gamma2=3*omega*(1−g)/4`.
The port's retained `_reftra_eddington` and native band routine instead use
the Eddington choice (`kmodts=1`). A second difference clips single-scattering
albedo at 0.999999, below WRF's conservative-branch threshold 0.9999995.
The manager accepted this named cause on 2026-10-07 08:57:12Z.

[M, same-input diagnostic] Sixteen original N36 columns show a clear-sky
surface-downward deficit about 1.0–1.6 W/m². A local PIFM+SSA intervention
reduces the maximum discrepancy to 0.00592041015625 W/m². This is diagnosis,
not a reviewed product fix or a whole-forecast improvement claim.

Proposed key: `GPUWRF_RRTMG_SW_REFTRA_WRF`, initially OFF. W3 requires the
reviewed source plus its reviewed default-ON packet. The owner must cover
all-sky cloudy columns, including optically thick stratocumulus, and report
the legacy f64 scope. The current prototype preserves the historical Eddington f64 path when fused REAL is disabled (owner 09:01:29Z); its final scope remains under review. Neither that legacy path nor forecast gates are cleared
by the sixteen-column clear-sky result.

## Binding wording rules for the W3 template

- Preserve the A7/A8 reference chain, carried unchanged into A9: 0227
  G-PD uses FINAL V0227/V0227M, F1'' uses FINAL's printed limits with
  conservative half-last-digit exclusion. They read no regression vs
  FINAL; FINAL itself passed with original RC/BISECT references.
- Report aggregate station clearance with its member-level qualification.
  W2 example [M]: R32-65 is 8/8 CLEAR, yet B/T2 gate A is DAMAGE on IC R
  (+0.01476 K versus S 0.01088, CI [0.00654,0.02342]); D_P −0.00384 is
  NOT REPLICATED → CLEAR, one realization per IC. W3 numbers are pending;
  never carry the W2 8/8 reading over as W3 clearance.
- On 0115's retained full-grid fields, the grey older-source Tb4−Tb2 RMSE
  **IS the frozen A3 C comparator (member minus twin)**. A3 uses C>limit
  or C/X≥0.75 on 72-hour breach frames; ratio guides 1/2 are empirical
  context. One IC pair is not a confidence interval. Use the original
  CPU-WRF as truth for every blue GPU−CPU error curve.

## Source binding and data refresh

Conditional amendment A10 (`74413c8aeab4`, 2026-10-07 09:11Z) admits a
reviewed E41 fix into W3 before the defaults/source freeze. W2 Monica's
two-domain fused d01 program has three hard STACK failures: 664/520/648 B
against the 256 B limit. All W2 three-nest programs pass at 32 B; the older
FINAL Monica program is 64 B. Attribution and fix remain pending. Runtime
shares 0.34/0.35/0.25% do not waive the frozen hard rule. The composed W3
source, key set and all executable-stack coverage must be bound before
calling its release packet accepted.

1. Receive the manager's immutable W3 source and pre-data A9; verify source
   tree, actual executed revisions, effective controls and compression hashes.
2. Capture 0227 W3R/W3P, 0408, and 0115 plus its genuine replicate on the
   original CPU-WRF references. Preserve each realization's identity; a copied
   autotune pin replay is not independent evidence. Keep every raw breach.
3. Capture Monica's two domains against RF02's original/restart joined truth,
   retaining the documented continuity check. The first 102 CPU files on
   scratch are still live inputs and cannot be purged yet.
4. Capture the shipped Swiss example for all 25 frames against RD11 original
   CPU-WRF. W2's Swiss waiter is cancelled; no W2 Swiss result exists.
5. Use unchanged finite-first native-array statistics: RMSE, bias, spatial
   |error| p25–75/p5–95, heatmap, frozen D6 limit and empirical pair context;
   cross-case mean ± sample SD. Keep one-frame memory use. Package PNG+HTML.
6. Record source-specific gate/classification results only after complete
   domains/hours and independent verdicts. Preserve older galleries as dated
   diagnostics. Update README/notes and publish a new ref for the same critic.

Repository-history maps and verified remote hashes come from compile. The
approved cleanup remains separate from forecast numerical changes. No push,
rewrite, new GPU arm or CPU-WRF reference run is owned by this preparation.

## Actual final composition and byte status

The release source is versioned `168fcdb9107bc408153b26c40179d13bad4b8cd0` (version-only successor of composed `9a531cc62`). The barrier changes are intended to preserve values, but the actual whole-program Monica probe found **104/340 output leaves different**, all finite; magnitude is not established here. Therefore no bitwise numeric/long-forecast inheritance is claimed for that program. Changed compiled E41 and necessary timing are checked separately; affected numeric gates run once on the final composition. Closed REFTRA/oracle/OFF/E115 evidence is reused. No intermediate W2 or source-prefix forecast is repeated.
