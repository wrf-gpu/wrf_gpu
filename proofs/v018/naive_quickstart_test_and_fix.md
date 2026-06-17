# v0.18 naive-user quickstart acceptance test → 0.18.1 fix

## Method

After v0.18.0 was pushed public, a **naive-user agent** was given ONLY a fresh
public clone (`git clone https://github.com/wrf-gpu/wrf_gpu.git`), hard-isolated
from the developer tree, and told to run the README "Switzerland" quickstart
end-to-end and log every problem. It was allowed to set the documented data env
vars (`GPUWRF_WRF_ROOT`, `GPUWRF_CANAIRY_ROOT`) as a user on the reference system
would.

## Result: PASS end-to-end, with documentation gaps

The install + CLI + forecast all ran cleanly and produced a valid 113-variable
`wrfout` (all fields finite and physically sane). **The software was correct — no
code defect.** The findings were all documentation/discoverability:

| ID | Severity | Finding |
| --- | --- | --- |
| F1 | blocker (advertised path) | Quickstart showed only `--input-dir my_case` (a placeholder); the advertised Switzerland case was not discoverable from the public repo — the tester had to spelunk the filesystem to find inputs. |
| F2 | major | `docs/equivalence-switzerland.md` carried a stale `v0.13.0` status banner contradicting the v0.18 front-page identity proof, and was unlinked from the Quickstart. |
| F3 | minor | Quickstart default `--domain d02` (and its ≥26 GiB VRAM note) did not match a single-domain d01 case. |
| F4 | minor | Cold-compile estimate (`~8–12 min`) was far over for a small single domain (measured ~½–1 min). |
| F5 | nit | JIT-cache env var named inconsistently across docs (`JAX_COMPILATION_CACHE_DIR` vs `GPUWRF_JAX_CACHE_DIR`; both are honored). |

## Fix shipped in 0.18.1 (docs + bundled example; no model-code change)

- **Bundled a small real-data example** `examples/switzerland_d01/` (`wrfinput_d01`
  + `wrfbdy_d01` + `namelist.input`, ~13 MB). Provenance: derived from public-domain
  NCEP **GFS** analysis (2023-01-15 00Z, 42×42 @ 3 km, 44 levels) via WPS/`real.exe`
  — freely redistributable; scanned PII-clean before shipping. Resolves F1 with **no
  local path** in the docs.
- **Rewrote the Quickstart** (`README.md` + `docs/quickstart.md`) to a concrete,
  copy-pasteable command using the bundled case (`--input-dir examples/switzerland_d01
  --domain d01`), the required `GPUWRF_WRF_ROOT` (generic `/path/to/your/WRF`
  placeholder), domain-scaled cold-compile guidance (F3/F4), and a unified
  JIT-cache env-var description (F5).
- **Refreshed `docs/equivalence-switzerland.md`** (v0.13.0 → v0.18 status; bundled
  example noted) and **linked** the example + equivalence doc from the README
  "Where to look first" table (F2).

## Re-validation (the proof)

The exact documented command was run against the bundled example on the reference
GPU (RTX 5090), inputs supplied only via `GPUWRF_WRF_ROOT`:

```
python -m gpuwrf.cli run --input-dir examples/switzerland_d01 \
    --output-dir runs/switzerland_d01 --domain d01 --hours 1 \
    --scratch-dir <non-tmpfs scratch>
```

Result: **`verdict: PIPELINE_GREEN`**, `wrfout_inventory_status: PASS`, one
`wrfout_d01_2023-01-15_01:00:00` (~9.9 MB) written; ~165 s total wall incl. a cold
compile. (`speedup`/`station_score` are NOT_RUN/FAIL by design — a bare forecast
with no CPU reference attached, which the quickstart does not require.)
