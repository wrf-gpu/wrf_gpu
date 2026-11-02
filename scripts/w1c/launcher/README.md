# W1c-1L sealed launcher (lane v46-icvariant) — sources

Sealed dir: `<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/W1C_SEALED_01` = these three files (`w1c.py`, `w1c_inputs.py`,
`make_seal.py`) + `scripts/w1c/icvariant.py` (v3.1) + the SCREEN6_SEALED_03 modules copied byte-identical
(`screen4.py screen4_member.py screen4_inputs.py w1a_*.py gpu_gate.py night_seq_v10.sh`; `make_seal.py` refuses any drift).

What is new vs SCREEN4/6: the W1c manager admission (`wrf_gpu2.w1c.admission.v1`): rows = signed GO `members`
(order C0530 → SIC_1L0530 → C0219 → SIC_1L0219, row_id `W1C_1L_V1.<base>.<arm>.det0`); C entries = per-case base authority
(names/SHA/clocks must equal the paid SCREEN6 v2 entry; 0219 may come from the issued supplement with sizes); SIC entries =
icvariant `stage` entry + its pinned variant admission + pinned R10 key receipt (`verify_key_receipt`, frozen context);
same prime 1ad80617 / cache layer / C parameters for both arms; registry issue = base cycle, tags W1C_1L_V1/case/arm/det0/variant.
Budget (GO draft 028240e5): entry deadline 1800 s, SIC pair ≤ 900 s, member cap 360 s (SCREEN4), private peak ≤ 24 GiB,
DATA ≥ 454 GiB at entry and ≥ 430 GiB during writes; STOP_ON_FIRST_FAIL, halt after member 4, no retry/resume; raw KEEP until a
separate manager disposition. C reruns: `C_identity_vs_SCREEN4` (54 wrfout name+SHA vs SCREEN4 member 1/2), evidence only.

Steps once ALISIOS has ISSUED both variant admissions (+ 0219 base authority with sizes) — CPU, current core plan, private TMPDIR:
1. per case: `icvariant.py stage` (≈ 5 s) → STAGE.json; 2. per case: `icvariant.py keys` (2 native loads, 6–20 min) → KEYS.json;
3. manager writes the W1c admission (shape = `synth_build.py` output) + signed GO; 4. `make_seal.py --admission … --source-commit …`
→ SEAL/entry/inputs/PLAN; 5. sequencer SEQ smoke (E243) + critic; 6. window: inputs.sh (4 native preflights) → entry.sh.
Synthetic CPU evidence (`synth_*.py`, NOT ALISIOS authority, fabricated key receipts): validate PASS, registry schema smoke rc 0
(private atlas), C-identity self-check/mutant, 11 admission mutants 0 false accepts.

## Disk plan v2 (ALISIOS decision (b) 2026-10-31T03:12Z, supersedes the 454 GiB plan)
Per member [M, all 10 W1a members]: raw 54 wrfout = **5.16 GiB**; scratch/proofs/registry < 0.1 GiB; ALISIOS extract ≤ 0.5 GiB
(on <DATA_ROOT>, kept). 0219 facade +0.11 GiB at staging (0530 already on DATA). Cache: shared paid layer, no copy.
Gate (w1c_disk.py): entry AND before every member free ≥ **437 GiB** (430 + 5.16 + 1, rounded up); below it the launcher WAITS
(poll 20 s, holding the GPU lock), bounded by the 1800 s entry deadline and window end − 300 s; after any wait the window,
deadline and GPU lease are re-checked; if a member no longer fits: HALT, this and later rows NOT_ATTEMPTED
(status W1C_1L_HALTED_DISK_FLOOR, no registry start), no waiver, no retry. Raw is freed per member by `w1c_raw_release.py`
(below) as soon as ALISIOS releases that member.
Free space, F = free at entry, assuming each member's raw is released before the next member's gate:
| step | free (GiB) | F = 437 | F = 439 |
|---|---|---|---|
| 0219 facade staged (before entry) | F − 0.1 | — | — |
| m1 C0530 done | F − 5.3 | 431.7 | 433.7 |
| m1 released, extract kept | F − 0.6 | 436.4 → **m2 waits/halts** | 438.4 ✓ |
| m2 SIC0530 done → released | F − 5.9 → F − 1.2 | | 433.1 → 437.8 ✓ |
| m3 C0219 done → released | F − 6.5 → F − 1.8 | | 432.5 → 437.2 ✓ |
| m4 SIC0219 done → released | F − 7.1 → F − 2.4 | | 431.9 → 436.6 |
Each extract stays on <DATA_ROOT>, so member k needs F ≥ 437 + 0.6·(k−1): **all four start without halting only if F ≥ 438.8 ≈
439 GiB at entry** (at 437 the run halts before m2 unless ≥ 0.6 GiB is freed elsewhere). Extract reserve: the launcher's hard
floor stays 437 GiB per member (430 + 5.16 + 1); the ALISIOS extract reserve (≤ 0.5 GiB + ~0.1 GiB overhead per finished
member) is planned in the ENTRY condition: the manager gate should only arm W1c at **F ≥ 439 GiB** (recommended), otherwise
a later member may wait/halt by design. Private peak ≈ 6 GiB (one member raw).
Time: members ≈ 4 × (213–253 s + ≈ 60 s setup) ≈ 1100–1250 s of the 1800 s entry deadline → ≈ 550–700 s total for the three
ALISIOS extract + QA + release + delete cycles between members (≈ 3–4 min each); a slower cycle halts the run (rows recorded,
re-admission needed). 14:00Z ≥ 415 GiB rule: unaffected (worst F − 7.1).

## Raw release tool (`w1c_raw_release.py`, manager runs it as a watcher; ALISIOS Addendum10/11 format)
`w1c_raw_release.py --admission <W1c admission> --admission-sha <pin> --standing <ISSUED standing disposition JSON, 0444>
--standing-sha <pin from the GO>` (without --standing: draft mode, checks all but the standing binding, never deletes).
Per member (only the four our_ids of the admission; old SCREEN6 directives of other IDs are never touched):
- directive `cross/TO_WRF_GPU2_W1A_SCREEN6_RAW_DISPOSITION_<our_id>.md` must contain row_id, our_id, the standing SHA and
  the SHAs of EXTRACT_DONE.json / READER_PROOF.json / SHA_READBACK.json / RAW_WRFOUT_RELEASE_INVENTORY.json;
- evidence in the GO row's extract_output_dir: EXTRACT_DONE status EXTRACT_DONE, SHA_READBACK status SHA_READBACK_PASS,
  READER_PROOF all18new_fields_pass true, POST_GATE_RESULT state PASS_METADATA_ONLY;
- inventory {row_id, our_id, raw_root, only_files[54 {path,bytes,sha256}], total_bytes, authorization == standing ref,
  no_directories_or_metadata_may_be_unlinked true}; raw_root == GO row raw_wrfout_dir == `<output_root>/<our_id>/wrfout`
  (hard rule); only_files == the native-PASS member receipt's wrfout_files; every path lexically `<root>/<id>/wrfout/<name>`;
- standing JSON `alisios.w1c.standing_raw_disposition.v1`: ISSUED by alisios_mgr, inside its window, members == admission
  rows (our_id order + row_id), 0444, SHA pinned.
Deletion: root and every component down to wrfout real directories (no symlink), one O_NOFOLLOW dirfd, hash from an
O_NOFOLLOW fd, dir + file inode re-checked right before unlinkat, size + SHA before each unlink, a journal line fsync'd to
`<output_root>/RAW_DELETE_JOURNAL_W1C_1L.jsonl` (O_NOFOLLOW) BEFORE each unlink; never directories/root/receipts/logs/inputs/
namelists/ledgers/compacts/extracts/journal. Output root literal for the GO: `<DATA_ROOT>/wrf_gpu2/scratch/v46-icvariant/W1C_1L_20261031`.
