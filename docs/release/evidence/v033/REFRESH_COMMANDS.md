# Public identity evidence inventory — v0.3.3 preparation

**This source-dated candidate was held.** Final composed W3 numerical source is `9a531cc625c47805bebea4fdf30d735709533a08`,
`src/` tree `7ca41d8bc6a41711746365036451d69c225618f9`; versioned run source is `168fcdb9107bc408153b26c40179d13bad4b8cd0`. Its affected forecast
gates are pending. Every older gallery retains its executed source label.

| Source group | Ready | Publication work / missing |
|---|---|---|
| Base `136c6e51c` / forecast tree `ccf8b7cf2` | 14 primaries, 3,066 domain-hours, 33,684 rows; 322 case PNG + 44 summary; separate 0227 GPU IC context, 23 PNG | Copy all retained complete sets into the curated public tree; base campaign remains PARTIAL 14/15, no base Monica run |
| Pre-W2 `69d8a9a7d` | 0227/0408/0115 + Monica: 803 domain-hours, 8,822 rows; 92 case PNG + 44 summary; Swiss 25 frames/3 figures | Already packaged, source-dated; standalone V0227M versus original CPU P now captured/rendered (23 PNG) |
| W2 executed `5a2adb6e1` + A6 controls / `471efc475` | Three primaries: 657 domain-hours, 7,218 rows; 69 case PNG + 44 summary, including full-footprint 0115 C curves | Already packaged; W0115 remains 32 REAL wind rows. Standalone W2bP history was released/deleted: pair-spread context survives, full native member quantiles cannot be reconstructed |
| Monica W2 `471efc475` | 146 domain-hours, 1,604 rows; 23 PNG packaged; 2,920 RMSE/bias + 5,840 count checks match complete D6 | Package as diagnostic: D6/integrity annex pass, E41 HARD FAIL; no transfer of release acceptance |
| W2 Z0115 R/P | Finished model and scorer receipts, genuine R distinction | ZP/ZR/YN/YM now captured/rendered against original CPU R: 4 × 23 PNG; their frozen verdicts remain source-specific |
| Final W3 primaries, pairs, Monica, Swiss | Frozen source and unchanged CPU references; reusable component proofs | Affected trajectory outputs and verdicts pending; preserve each primary/replicate identity. Swiss W2 was cancelled, no result exists |

The plots use every native cell without masks. Spatial |error| p25–75/p5–95
is a spatial distribution, not a confidence interval. Full retained IC fields
provide the grey RMSE curve. For 0115's frozen A3 proxy, C is older-source
GPU member minus twin; truncated 16-level T/U/V/QV pair fields are omitted.
Primary deviations always compare to original CPU-WRF, not an older GPU run.

## Reused versus needed

| Item | Reuse | Needed |
|---|---|---|
| REFTRA `5ab322a76` source/oracles/OFF/E115 | Closed exact-source receipts, including original clear/cloudy columns and inherited fixture classification | No component rerun absent new source or failure |
| Bit-identical E41 barriers/layout | Numeric/oracle/long results after minimal whole-program same-input byte proof | Changed compiled-program E41 and necessary timing; no barrier-only long forecast |
| REFTRA ON trajectory | Original CPU inputs, frozen comparators and existing baseline receipts | Affected gates once on final composed W3; strict 24 h from those same 72 h comparisons |
| Current galleries | Completed finite-first extraction, numeric audits and source-labelled images | Copy/package/link checks; render retained metrics only where images are missing |
| Final W3 plots | Same reviewed statistical definitions and one-frame memory workflow | Extract new affected-source frames, bind receipt/compression/source, render once; no model rerun for plotting |
| Guide / history cleanup | Read-only live HTTP/version evidence and initial guide audit; approved rewrite decision | Correct triad/build/search once; final review only after composition; cleanup includes release then sealed maps and remote verification |

Owned obsolete GPU waiters: **zero**, hence zero cancellations. Lane work
uses cores 12/13, nice 19, JAX CPU and QUIET. Observed extraction peaks are
458,740 KiB (three-nest) and 358,888 KiB (Monica); they are analysis RSS,
not forecast admission sizes. Estimated refresh after frame readiness:
5–10 minutes per case on the shared two cores [I]. Final model/scorer ETA
belongs to integrate/wn3; it is not inferred from image-render timing.

## Exact refresh commands

Run from the immutable final docs/tool checkout. `RD_W3_REV` must be the
actually executed full frozen revision; `RD_RUN` contains `receipt.json`,
`compress_receipt.jsonl` and `wrfout/`. Use a fresh `RD_OUT`. A three-nest
primary uses original `wg_CASE/run/run`; 0227's IC-P member uses original
CPU P0227. An 0115 IC-P comparison against R must disclose the IC difference.

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; taskset -c 12,13 nice -n 19 env JAX_PLATFORMS=cpu timeout 1800 python scripts/release_plots/stream_release_case.py --run "$RD_RUN" --cpu "$RD_CPU" --revision "$RD_W3_REV" --repository "$RD_REPOSITORY" --case "$RD_CASE" --domains d01 d02 d03 --manifest "$RD_MANIFEST" --scope "$RD_SCOPE" --out "$RD_OUT"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; taskset -c 12,13 nice -n 19 env JAX_PLATFORMS=cpu MPLCONFIGDIR="$RD_MPL" timeout 240 python scripts/release_plots/hourly_deviation_bands.py --render-only --manifest "$RD_MANIFEST" --out "$RD_OUT"
```

For Monica, use `--domains d01 d02` and original RF02 joined CPU truth.
For a full-frame CPU IC curve add `--pair "$RD_CPU_PAIR" --pair-reference
cpu --pair-label "$RD_PAIR_SCOPE"` to extraction. Source/hash guards refuse
dirty, failed, mismatched or incomplete compressed histories before extraction.
The renderer refuses incomplete/nonfinite metrics before plotting. Remove
the named MPL cache by its **exact literal path** after the last render.

For the complete, declared primary set (never mix numerical source groups):

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; taskset -c 12,13 nice -n 19 env JAX_PLATFORMS=cpu MPLCONFIGDIR="$RD_SUMMARY_MPL" timeout 300 python scripts/release_plots/summarize_hourly_bands.py --case "0227=$RD_0227/metrics.json" --case "0408=$RD_0408/metrics.json" --case "0115=$RD_0115/metrics.json" --case "Monica=$RD_MONICA/metrics.json" --expected-cases 0227 0408 0115 Monica --title "$RD_SUMMARY_SCOPE" --out "$RD_SUMMARY"
```

Keep separate pages for every IC member and genuine replicate; the primary
case weights in the cross-case summary do not change because more replicas
were run. Swiss 24 h uses the existing dedicated 25-frame comparator and
figure builder, not the 73-hour extractor. No long model run is required
solely to regenerate images.

Current public-ready inventory: 776 source-labelled identity PNG + 3 Swiss figures, 65 HTML pages including 20 guide pages, 2,480 local links checked. Five missing retained diagnostics were extracted once; Y0 uses the recorded 219/219-byte replay proof. W3 fresh affected-source outputs remain pending.
