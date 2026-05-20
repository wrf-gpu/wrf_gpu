# BLOCKER - Attempt 4 Thompson strict Tier-1 tolerance gap

Date: 2026-05-20
Sprint: 2026-05-20-m5-s1-thompson-microphysics-column
Class: physics-parity

Evidence: attempt 4 implemented the diagnosis-prescribed WRF checkpoint ordering, gated cloud-ice number changes to the sublimation branch only, replaced the harness `dz=1e30` workaround with a locally patched WRF Thompson object that zeroes terminal-velocity arrays before the sedimentation flux loops, and restored ADR-005 strict fixture tolerances.

Tier-1 still fails the restored tolerances against the compiled WRF harness. Current max absolute errors after the order/Ni/sedimentation fixes are:

- `qv`: `1.4304079020558032e-05`
- `qc`: `1.517228938283358e-04`
- `qr`: `4.760876436193939e-06`
- `qi`: `1.3708094759935232e-04`
- `qs`: `1.447943623500527e-04`
- `qg`: `1.5218435328806104e-05`
- `Ni`: `126975.12500000044`
- `Nr`: `67300.453125`
- `T`: `0.040290844661740266 K`

The order fix did reduce the main temperature error below the attempt-4 backstop threshold (`0.3186 K` to `0.0403 K`), so the order-refactor diagnosis is confirmed. The remaining gap is dominated by the deferred WRF lookup-table and moment parity items named in the attempt-4 contract: `t_Efrw`, `tps_iaus`, `tni_iaus`, rain/freezing lookup tables, snow/graupel moments, and related number/mass balance details.

Manager's attempted mitigation: not applicable; this is the worker handoff blocker requested by the attempt-4 contract after strict tolerances could not be met without table-export work.

Recommended human/manager action: dispatch M5-S1.x for exact WRF table/moment export and replacement, or explicitly amend ADR-005 if the project wants a narrower non-final Thompson parity claim before table parity.
