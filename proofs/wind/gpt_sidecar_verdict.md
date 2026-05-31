# GPT-5.5 (gpt-5.5) sidecar independent critique — case3 wind residual

**Q1**

Mostly sound. Tiny CPU-WRF L2-vs-L3 spread relative to persistence is strong evidence that this is not a calm/no-signal metric artifact. Shared lateral BCs weaken the claim of “intrinsic predictability,” but not the validation conclusion: your target is CPU-WRF behavior on this d02 grid, and CPU-WRF is essentially deterministic here.

Caveat: the committed diagnostic only directly overlaps case3 self-spread through lead 19 h for L2 and 8 h for the second L3 run, while the failing score is 24 h. That is still persuasive, but “decisive at 24 h” needs either direct 24 h CPU spread or explicit extrapolation.

**Q2**

Yes, A dominates and is not a 10 m diagnostic problem. Applying the WRF ratio to the GPU k0 wind recovers only about `(0.933 - 0.871) * 6.11 ~= 0.38 m/s` of a `~2.0 m/s` 10 m speed deficit. Case3 V10 would almost certainly remain below persistence.

But I would not say A is proven outside your whole ownership if `mynn_pbl.py` is in scope. The k0 wind is prognostic after dycore plus PBL momentum. It is outside `surface_layer.py`, but not exonerated from MYNN without a MYNN-off, PBL-tendency budget, or surface-drag sensitivity proof.

Also, B’s interpretation is underdetermined. WRF ratio `0.933` being above the GPU neutral ratio `0.896` does not only imply “WRF is less stable.” WRF MYNN with `ISFTCFLX=0` dynamically updates ocean roughness, while GPU water `znt=0.00285` is a surrogate. At `za ~= 25.8 m`, neutral ratio `0.933` corresponds to an effective `z0 ~= 1.8e-5 m`. So roughness/provenance can explain much of the ratio gap. Do not tune stability based on GPU `zol` and WRF diagnostic ratio alone.

**Q3**

Argument for changing: if you can show a source-backed parity bug, for example exact MYNN ocean roughness / `sf_mynn` diagnostic behavior, then a narrow fix is justified even if the immediate gain is small.

Argument against changing: a generic “lift stable-marine ratio” is a case-tuned empirical correction. It risks case2, where water V10 is already near balanced and U10 is already positive-biased, and it still does not close case3. That is a bad trade unless a multi-case gate proves net improvement.

Recommendation: leave the diagnostic alone for this close. Document case3 as a real low-level prognostic wind deficiency, with a minor and currently confounded surface-diagnostic contribution. Next proof should isolate dycore vs MYNN PBL momentum, not tune `surface_layer.py`.

**VERDICT:** No sound, low-regression `surface_layer.py` lever. Honest close: case3 residual is dominated by prognostic k0 wind deficiency outside surface-diagnostic scope; `mynn_pbl.py` remains a possible B2-owned suspect until budgeted out.
tokens used
