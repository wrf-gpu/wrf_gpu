# FINAL output-integrity exceptions

The FINAL model is `1fece48dabe6805a624f86d169c036cd39f267a5`. Its PROD 24 h run has
**zero frozen D6 numerical-bound failures in either domain [M]**. The separate integrity
report remains **FAIL**: five fields are zero on the GPU where CPU-WRF contains the
following traces. The manager accepted the five-field trace exception class at
05:37Z on 2026-10-04. The requested all-frame audit, completed at 05:50Z, increased
the d01 frame counts and QSNOW maximum below; neither the FINAL model nor the
integrity checker was changed. The raw all-frame result is still **FAIL**.

| Domain | Field | Affected frames / 24 | Output hours (UTC) | GPU value | Maximum CPU trace [M] | Unit |
|---|---|---|---|---|---|---|
| d01 | QICE | 15 | 2026-07-26 08–16Z, 19Z–2026-07-27 00Z | 0 | 1.863837978676201e-15 | kg/kg |
| d01 | QSNOW | 15 | Same hours | 0 | 8.739286997251714e-15 | kg/kg |
| d02 | CANLIQ | 1 | 2026-07-26 13Z | 0 | 2.235258079963387e-6 | kg/m² |
| d02 | CANWAT | 1 | 2026-07-26 13Z | 0 | 2.235258079963387e-6 | kg/m² |
| d02 | FWET | 1 | 2026-07-26 13Z | 0 | 0.0005369787104427814 | dimensionless |

The follow-up checks variable degeneracy and metadata at all 24 hourly frames
per domain, h1–h24. No CPU variables or CPU global attributes are missing. The
initial output is absent from this benchmark arm; it is not covered by this
audit. The earlier three-frame sample (01Z, 13Z and the final 00Z) found the same
five fields but counted only two events for each d01 field. Exact file times and
all **33 field-frame events** are in the evidence receipt, including the d01
step-alignment offsets of 18 or 36 seconds. The six WN3 cases have separate checks
of every paired frame; any exception there needs its own approval and disclosure.

The canopy traces have a measured forcing explanation. At d02 cells (46,86) and
(46,87), CPU-WRF receives approximately 0.003 mm of drizzle during 12–13Z, leaving
2.24e-6 and 2.05e-6 kg/m² of canopy water. The GPU receives that light rain earlier
and has a dry canopy at 13Z. The NF07 audit finds no frozen precipitation and canopy
temperatures at least 282.8 K across the affected wet-canopy cells, so the snow45
ground-rain and frozen-canopy changes are inactive there. This is light-rain timing,
with Noah-MP responding to its own forcing; it is not a missing writer field.

These exceptions cover the PROD numerical illustration, whose older WN2 input
path does not govern the operational release validation. A future amplitude floor
for the degeneracy rule is planned for v0.3.1 and has not been applied to FINAL.

Exact values, coverage, report hashes and the approval are preserved in
[the evidence receipt](evidence/final_prod_integrity_exception.json). Original
artifacts: `integrate/FINAL/scoring/final_prod_integrity_allframes.json`,
the earlier `final_prod_integrity.json`,
`final_d01_vs_d5_24h.json`, `final_d02_vs_d5_24h.json`, and
`b-thompson/NF07/canopy_13z{,_cells}.txt` under the lane artifact roots.
