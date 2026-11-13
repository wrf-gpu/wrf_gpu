# v0.5.0 decisions and evidence

This is a compact public-facing summary of the decisions recorded during the
release candidate. The original full reports and raw verdicts remain separate.

- **Compact-AOT scanner rule for all v0.5.0 GPU checks**, 11 November 2026: an
  INCOMPLETE compiled-resource scan is accepted only when it comes solely from
  missing metadata of compact AOT executables, every other gate passes and each
  compact executable matches a scanned one (rule of 7 November 2026, SHA256
  `2d0cc72e659f32080cfec2680124abb1fdc4b3f8c3fa0c2eb5b7e18019d0a922`). Scope
  decision SHA256:
  `4e41a9d1ac5a6558e547c3760f01cd91251c4724b82cfc32bd052a667e93ea6d`.
- **Nine-nest nest program, two known input-transpose fusions**, carried from
  the nine-nest v0.4.6 runs (564 and 522 optimized HLO instructions, stack at
  most 160 B, no large non-leaf callee): accepted and disclosed; the in-run
  extract check and the research build carry the same program. SHA256
  `036220b0fc6d6824991c1c5f664736d2e175dcc321e8f71d0f6bb38b78aa0232`,
  `3e998f4f2fcb789c6511fa65928c3f50d884fed5bf2837bb4070c45811c39d60`,
  `9598f1cc5ed09f5b4e2a79ef36db986b29ba1edbd3a5e49f3725f0f5bb15faac`.
- **Stochastic-on program, one input-transpose fusion with 600 instructions**
  (rule 500), stack at most 32 B: accepted and disclosed for this opt-in program
  only; the same fusion appears under a second cache name after the follow-up
  check (amendment: the acceptance is bound to the fusion class, program, size
  and at most one flagged entry, not to a cache file name). Splitting it is
  deferred to the performance release.
  SHA256 `9bef1a221d16d55f6c99133b87b307c7921d00cd78a216b75adc7b491227a128`,
  amendment `e88b02a8d324cb8bc07e19265f75b6e71131538e38266d6bb072f2d526a0e630`.
- **AFWA-on program, one input-transpose fusion with 664 instructions** in the
  domain-1 step program (rule 500), stack at most 32 B: accepted and disclosed
  for this opt-in program only; at most one such entry per cache. GPU check of
  the option (two-domain 9/3 km Canary case, 3 hours, three on/off pairs): final
  PASS, GPU kernel equal to the WRF code, warm cost on/off 0.998; on/off field
  differences up to 0.87 K (2 m temperature) and 0.78 m/s (10 m wind), inside the
  check's gross-error bound (2 K, 2 m/s) but larger than the cold-compile noise
  band measured on a 1 km case (0.7 K, 0.2 m/s). The operational user's advice is
  to compare AFWA-on runs only with an AFWA-on twin of the same program. SHA256
  `4e08b9235fcf28e8627edca585f513bf1503691602864f686237e6ebac4e7a32`; result
  SHA256 `a43914949720431ed286193142f72621cca13d30d715e3295912c7bd9b7407dc`.
- **Shared compile store (experimental)**, 11 November 2026: ships off by
  default; the failed cross-build byte-identity check is disclosed, not a
  release blocker (decision of the operational user, 11 November 2026).
- **85 vertical levels**, 11 November 2026: research setting, disclosed limit,
  not release-qualified.
- **Final-frame graupel trace, decided before the run**, 11 November 2026: if the
  72-hour WN3 0227 check shows the same d03 QGRAUP trace class as v0.4.5/v0.4.6
  (CPU-WRF a few cells around 1e-12 kg/kg, GPU zero) and it is the only
  output-integrity failure, it is disclosed as in v0.4.6; any other deviation
  stops the release. Thresholds, D6 limits and floor rules unchanged. SHA256
  `b3cbf726e39ff09715d18542d21117b45552a276d966ee6d0976ed05110802c5`.
- **Compile proxy over the 1.10 check**, 12 November 2026: cold-minus-warm WN3
  three-hour proxy 592.65 s (v0.5.0) against 473.13 s (v0.4.6), ratio 1.253.
  Ruled cross-session host load, not a code regression: lowered default programs
  and the loaded program count (1446) are unchanged, pre-forecast time is equal
  (83.0 s against 79.0 s), and warm runs without compilation that overlapped the
  same host load were 1.2–1.6× slower. Disclosed, with a same-session cold
  repeat as the condition. SHA256
  `81ead56923f03634fa6ac990337c7d19603749a95e1fd178af11f1c27e6f379d`.
  **Withdrawn and replaced on 13 November 2026:** the repeat from an empty cache
  gave 624.42 s (ratio 1.32 against 473.13 s), so the ruling was withdrawn. The
  two figures had different starting caches: the v0.4.6 reference ran after the
  Swiss and two-domain checks, which had already compiled most small start-up
  programs and autotuning results of the WN3 run. A v0.4.6 control on the same
  empty basis, same host and day, gave 623.17 s: ratio **1.002**, no regression;
  the program set is identical. Decision SHA256
  `49089a456a52b934a1c8a587768a74c0ec53669441c387628fd7c5783b215ad0`.
- **RW1, RW2, RW4 and Swiss 24 h**, 12 November 2026: PASS without a decision
  (short-check verdict SHA256
  `23e47181f26b135cccf181296f67a6a34e17d49c21f4e4dd5dc6c739cbb5226e`).
- **RW3 Tenerife 72 h and speed/memory**, 13 November 2026: speed and memory
  PASS without a decision (d03 stepping 0.9992, VRAM 1.006, RSS 0.996;
  whole-run time inconclusive because of data-disk latency, disclosed). Fidelity
  PASS, disclosed: the only output-integrity failure is the d03 QGRAUP trace at
  hour 72 (CPU-WRF up to 6.6e-12 kg/kg, GPU zero), the class fixed before the
  run; frozen floor rule PASS with 0 real failures and 6 floor-limited d03
  `RAINNC` rows at hours 67–72 (v0.4.6: 5). SHA256
  `0e7a21ba9cfedf1cff3ccfe027a0b90d5efdf08160dda7a3d6d1778a174d89ca`; RW
  verdict SHA256
  `d89a4c6be8ac21ff8d2cbe026095ab8001f85f85c595dab216e72f19ae4ceb27`.
- **Block 7, land-use patch on the GPU**, 13 November 2026: PASS (rc 0,
  finite output, new compile keys only for the programs that contain the patched nest, compiled-resource gate
  PASS, sealed bundle unchanged).

The numerical [release notes](../../../../release_notes/RELEASE_NOTES_v0.5.0.md)
identify the default-program identity, RW results, speed and memory separately.
