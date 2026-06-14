# v0.16 fp32 verdict — evidence index

The make-or-break full-working-set (fp32) investigation for v0.16 is **complete and
double-confirmed** (Opus implementation + independent GPT reproduction). The
valid-numerics fp32 speed ceiling on this RTX 5090 is **~1.1×** (NOT ~4× or ~6×),
with **0% VRAM-peak reduction from precision alone**. This folder carries the
primary evidence so the public release is self-contained.

## The verdict in one line

fp64 GPU ≈ CPU-WRF parity (GeForce fp64 = 1/64 fp32 hardware law) is unchanged. The
**genuine** fp32 win is a real but small **~1.1×** (acoustic + safe-compute; oracles
pass). The ~4× / ~2×-VRAM headline is **PROVEN unreachable with valid numerics**:

1. Demoting the **whole persistent State** to fp32 (−700 MiB carried fp64 arrays at
   65k) moves the VRAM peak by **0 GiB** — the peak is **transient** working memory,
   not persistent State.
2. The base absolutes `p_total`/`ph_total` (~1e5) **cannot be stored fp32** — doing so
   corrupts the geopotential/PGF gradients **27× / 127×** beyond the gated-fp32 budget
   (bits are lost at *storage*, so an in-loop fp64 island is powerless). They are
   conservation-pinned to fp64 **and** are the large arrays.
3. The transient peak is **precision-insensitive** (XLA `temp_size` 5305→5379 MiB,
   unchanged) — dominated by fp64 cancellation islands + the qke-pinned MYNN work
   (qke goes non-finite in fp32 at 1 km: 3036 cells).

The 4.3× "cost-proxy" is a **numerically-invalid** global-fp32 artifact (x64 off;
corrupts conservation/cancellation). Double-single recovery costs fp64-equivalent
storage + ~16× time. 6× always exceeded the RTX 5090 roofline.

**The real wins shipped in 0.16:** the genuine ~1.1× fp32 lane **plus** the
1 km-unlock (MYNN BouLac dense→O(nz)/chunked — *orthogonal* to fp32; a 1 km Canary now
fits one RTX 5090). The boundary-forced long-horizon fixture is **built** (fp64 stable
under LBC; fp64-vs-fp64 control = 0.000 RMSE).

## Reports

- `opus-fullws-fp32-verdict.md` — the Opus implementation + measured verdict (outcome (b)).
- `gpt-fullws-fp32-crosscheck.md` — independent GPT reproduction + adversarial refutation
  attempt. Verdict: **IMPOSSIBILITY CONFIRMED** (reproduced 1.105× / 1.111×, VRAM ratio
  1.000; could find no valid-numerics lever toward >2×).

## Measured proof objects (`proofs/perf/v016/`)

| File | What it proves |
|---|---|
| `fullws_fp32_km_bench.json` | GPU bench, real Switzerland d01: 16k **1.107×** / 65k **1.110×**, VRAM ratio **1.000**; 147k OOM (fp32 alone does not unlock). |
| `fullws_safe_km_bench.json` | The numerically-defensible `safe` lane (keeps `p_total`/`ph_total` fp64): 16k **1.108×**, VRAM ratio **1.000** — confirms even the valid lane is ~1.1× / 0% VRAM. |
| `fullws_base_absolute_oracle.{json,py}` | The conservation-pin proof: storing the base absolutes fp32 corrupts the geopotential/PGF differences 27× / 127× the gated-fp32 floor; `GATE_PASS=False`. |
| `fullws_boulac_onz_147k.json` | The **orthogonal** 1 km-unlock: full-ws + BouLac-O(nz) runs the 1 km / 147k grid to finite completion at 21.31 GiB. |
| `fullws_transient_memory_probe.py` | XLA `memory_analysis().temp_size` probe (the transient high-water mark is precision-insensitive). |
| `gpt_fullws_reproduce.json` | Independent GPT GPU reproduction (matches: 1.105× / 1.111×, VRAM 1.000, `[]`-unlock). |
| `gpt_fullws_transient_memory_probe.json` | Independent GPT direct-GPU XLA memory analysis (temp arena does not shrink under fp32). |
| `gpt_safe_alias_probe.json` | Why the `safe` field set must exclude `p`/`ph` (the `State.replace` alias-sync would otherwise re-demote the totals). |
| `gpt_double_single_probe.json` | Double-single recovery costs fp64-equivalent storage + ~16× time — not a path to the ceiling. |

See also the honest performance panel: `proofs/v016/dashboard/HONEST_PERF_PANEL.md`.
