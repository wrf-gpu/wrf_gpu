# BLOCKER - M5-S1.x strict Thompson parity and HLO table-gather regression

## Decision

M5-S1.x table extraction is complete and reproducible, but the sprint cannot honestly claim the strict ADR-005 gate.

The exported table asset is `data/fixtures/thompson-tables-v1.npz`, pinned in `fixtures/manifests/analytic-thompson-column-v1.yaml`. The JAX hot path wires the small active WRF table paths for `t_Efrw`, `tps_iaus`, `tni_iaus`, `tpi_ide`, and Field snow moments. The large rain-freezing tables are extracted and pinned, but not left in the jitted timestep body because the dynamic 4-D gathers caused an HLO/launch regression.

## Evidence

Post-table Tier-1 max absolute residuals in `artifacts/m5/tier1_thompson_parity.json`:

| Field | Max abs |
|---|---:|
| `qv` | `4.7608781466789915e-06` |
| `qc` | `1.2657008724556353e-07` |
| `qr` | `4.760876436193939e-06` |
| `qi` | `1.269506099959173e-07` |
| `qs` | `9.2685395199886e-11` |
| `qg` | `2.9509294563467847e-06` |
| `Ni` | `126975.12500000079` |
| `Nr` | `67300.453125` |
| `T` | `0.011792500929288963` |

These still violate strict ADR-005 tolerances (`abs=1e-10, rel=1e-8` for hydrometeors; `abs=1e-3, rel=1e-6` for `Ni/Nr`).

Performance/HLO evidence:

- Current small-table hot path: `artifacts/m5/thompson_profile.json` reports `kernel_launches_per_step=5`, `temporary_bytes_per_step=0`, `host_to_device_bytes_post_init=0`.
- Current full HLO at `data/scratch/m5/thompson_column_production_full.txt` is `343407` bytes; tracked truncated HLO at `artifacts/m5/hlo_dump/thompson_column_production.txt` is below git hygiene limits.
- Direct dynamic 4-D rain-freezing table reads produced `kernel_launches_per_step=23`.
- Packed rain-freezing table reads still produced `kernel_launches_per_step=9`.
- Removing the rain-freezing table from the hot path still leaves `kernel_launches_per_step=5`, so even the small active table gathers need a follow-up fusion design before claiming the contract's 1-launch/no-HLO-regression AC.

## Diagnosis

The table export removed the largest table/proxy residuals:

- `qc`: `1.519672247928526e-04 -> 1.2657008724556353e-07`
- `qi`: `1.3714003800346464e-04 -> 1.269506099959173e-07`
- `qs`: `1.447943623500526e-04 -> 9.2685395199886e-11`
- `T`: `4.250859614040792e-02 -> 1.1792500929288963e-02`

Remaining strict misses are dominated by process paths outside the now-wired small table substitutions:

- warm precipitating cells retain WRF `qr/Nr` while the JAX rain evaporation path removes nearly all rain;
- graupel cells retain WRF `qg` while the JAX graupel sublimation/melting path depletes it;
- the largest `Ni` miss points at additional WRF nucleation/freezing/number-balance behavior, not the `tps/tni_iaus` lookup itself;
- the large rain-freezing lookup tables need a different representation or staging strategy because naive JAX gathers violate the HLO/launch acceptance criteria.

## Required Next Work

Open a follow-up fix cycle with two parallel tracks:

1. Table-gather/fusion design: find a representation for `t_Efrw`, `iaus`, and `qrfz` that keeps HLO under the contract limit and restores `kernel_launches_per_step=1`.
2. Physics residual closure: compare the remaining max-residual cells against WRF per-process tendencies for rain evaporation, graupel sublimation/melting, cloud-water freezing/nucleation, and number-balance finalization.

Per the bug-fix parallel-pair rule, the confirmed HLO/table-gather regression and remaining physics residual diagnosis should get manager-dispatched parallel review, including Gemini when quota is available.
