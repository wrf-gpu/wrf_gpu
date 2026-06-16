# v0.17 BouLac ONZ Bench

## Verdict

`GPUWRF_MYNN_BOULAC_ONZ=1` is now a production opt-in for the bit-identical
O(nz)-working-set BouLac path.  The default dense/chunked behavior is unchanged
when the flag is off.

The legacy pure scan is still rejected.  It is available only with
`GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN=1` for pathology reproduction.

## Bit Identity

Artifact: `proofs/perf/v017/boulac_onz_oracle.json`

Result: PASS.

- Dispatched `GPUWRF_MYNN_BOULAC_ONZ=1` vs dense: `max_abs == 0.0`
- Production helper vs dense: `max_abs == 0.0`
- Regimes: 8 inherited BouLac stratification cases
- No clamps or tolerance changes

## 1 km VRAM

Case: real Switzerland d01 cost proxy tiled to 384 x 384 x 44, 147,456 columns,
fp64, 8 steps, clean process per mode.

| mode | result | warm ms/step | est compile s | peak VRAM GiB | finite |
|---|---:|---:|---:|---:|---:|
| dense | OOM, 18.80 GiB allocation | n/a | n/a | n/a | n/a |
| production ONZ | FIT | 583.98 | 135.73 | 18.57 | true |
| legacy scan | OOM, CUDA graph command-buffer | n/a | n/a | 20.33 before failure | n/a |

The production ONZ result is within the prior target band (~18.25-21.31 GiB)
and makes the 147k-column fp64 case run where dense fails.

## Canary

Case: real Canary d03 domain from
`/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z`,
75 x 93 x 44, 6,975 columns, fp64, 8 steps.

| mode | warm ms/step | est compile s | peak VRAM GiB | finite |
|---|---:|---:|---:|---:|
| dense | 38.45 | 49.83 | 3.13 | true |
| production ONZ | 39.81 | 100.84 | 1.73 | true |

Canary d03 peak VRAM drops by 1.39 GiB.  Warm step time is essentially neutral
at this small grid; compile is longer because chunk=1 unrolls 44 source blocks.

## Pathology

The old pure scan implementation is still pathological in the full operational
JIT:

- 16,384 columns: fits, but estimated compile is 86.78 s.
- 147,456 columns: fails while instantiating CUDA graphs:
  command buffer with 71 entries, 20 alive graphs, CUDA out of memory.

Trigger: the legacy scan lowers the parcel search as a deep data-dependent
chain.  XLA keeps too many graph/command-buffer objects live at 1 km scale.  The
production ONZ route avoids this by preserving the dense cumsum/where/reduce
arithmetic order inside source chunks of size 1.

## Artifacts

- `proofs/perf/v017/boulac_onz_bench.json`
- `proofs/perf/v017/boulac_onz_oracle.json`
- `proofs/perf/v017/boulac_onz_swiss_147k_dense.json`
- `proofs/perf/v017/boulac_onz_swiss_147k_onz.json`
- `proofs/perf/v017/boulac_onz_swiss_16k_legacy_scan.json`
- `proofs/perf/v017/boulac_onz_swiss_147k_legacy_scan.json`
- `proofs/perf/v017/boulac_onz_canary_d03_dense.json`
- `proofs/perf/v017/boulac_onz_canary_d03_onz.json`
