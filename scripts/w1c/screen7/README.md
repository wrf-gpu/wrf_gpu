# SCREEN7 launcher (0530 × {C_new, alp1_low, alp1_high}, FULL selection, cold, no resume)

Authoritative sealed copy: `<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/S7_SEALED_01` (this directory mirrors its new files).
ALISIOS decision `cross/TO_WRF_GPU2_FROM_ALISIOS_MGR_SCREEN7_DECISION_20261031T0658Z.md`; IDs confirmed
`..._SCREEN7_0530_IDS_CONFIRMED_20261031T0031Z.md` (bindings `SCREEN7_0530_MEMBER_BINDINGS_PROPOSED_FULL_20261031.json` da1f13b8).

## What is reused unchanged (byte-identical, checked by make_seal.py)
- SCREEN6_SEALED_03: `screen4.py` (run_member / full_history), `screen4_member.py` (backend-compile refusal),
  `screen4_inputs.py` (native cold-input preflight), `w1a_*`, `gpu_gate.py`, `night_seq_v10.sh`.
- ALP1_J3_REFERENCE_SEALED_01: `alp1_contract.py` (FULL parameter tree, CAPS/phase names, wave total 3600 s).
- W1C_SEALED_01: `w1c_rows.py` (signed-GO rows), `w1c_disk.py` (per-member 437 GiB floor, bounded wait,
  STOP_ON_FIRST_FAIL rows), `icvariant.py` (product key function for R10).

## New
| file | role |
|---|---|
| `s7.py` | launcher: rows = confirmed bindings (+ signed GO identity-equal), prime ALP1_NEW_PRIME (1873 == live layer), seams `registry` + `c.parameters` (FULL operand gate), **member cap 450 s incl. post**, admission of each member only if its full cap fits entry/window/wave budget/lease |
| `s7_budget.py` | ALP1 wave ledger (fork of alp1_budget.py): START before the official GPU lock, END after it, whole entry elapsed charged; phase `SCREEN7_0530`, entry cap min(1500, remaining − 120) |
| `s7_inputs.py` | per-arm native input preflight with the arm's exact FULL parameter file |
| `s7_keys.py` | R10 CPU key proof: 0530×C3_prime == 0219×C3_prime (case independence), 0530×{J3a,J3b} == 0530×C3_prime (alp1 traced) |
| `s7_metrics.py` | ALISIOS decision metrics (their analyse_compact.py 36abc15f definitions, cloud.py binding bec74fe3, masks d1e99270) |
| `s7_raw_release.py` | W1c release tool + metrics export/readback BEFORE the first unlink, journal `RAW_DELETE_JOURNAL_SCREEN7.jsonl` |
| `LAYOUT.json` | output_root, member paths, journal, metrics path (JSON pointer for the ALISIOS GO/standing) |

## Caps, budget, disk
- Member: 450 s from registry begin to the end of the post checks (model process killed at the member deadline;
  `screen4.run_member` keeps its own 360 s model cap). Measured basis: W1c attempts 270–280 s, J3a 389 s (incl. a J3-only
  response check SCREEN7 does not have).
- Entry: wave ledger remainder 1637.6 s → entry cap 1500 s + 120 s release. Member k starts only if
  `prior_used + entry_elapsed + 450 ≤ 3600 − 120` and `now + 450 ≤ entry deadline` (= cap − 30 s reserve).
  Expected: 3 × ~280 s + 2 ALISIOS extract/release waits (~180 s each) ≈ 1200 s; member 3 must start by ≈ 1020 s.
- Disk: entry and every member need DATA ≥ 437 GiB (430 + 5.16 raw + 1), bounded wait for the previous member's
  release; raw per member 5.16 GiB [M W1c], private peak ≤ 24 GiB.
