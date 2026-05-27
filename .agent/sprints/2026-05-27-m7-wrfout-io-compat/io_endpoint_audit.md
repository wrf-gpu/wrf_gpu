# M7 I/O Endpoint Footprint Audit

Generated UTC: 2026-05-27T00:15:01+00:00
Reference run directory: `/mnt/data/canairy_meteo/runs/wrf_l3/20260525_18z_l3_24h_20260526T221207Z`

## Verdict

The current GPU forecast path consumes Gen2 `wrfinput_d02` plus d02 hourly `wrfout` history for replay, and writes only compact `wrfout_gpu_d02_p###h.npz` proof containers. It does not currently produce a drop-in NetCDF `wrfout`, does not consume or produce native `wrfbdy` in `build_replay_case`, and has no observed `wrfrst` restart endpoint.

## Endpoint Matrix

| Endpoint | Current GPU role | Evidence | M7 daily-pipeline implication | Action |
|---|---|---|---|---|
| `wrfinput_d02` | Consumed for grid metrics, static fields, land state, and initial state context. Not produced. | `build_replay_case` calls `run.wrfinput_file(domain)` and land/metric loaders. | Acceptable as a read-only Gen2 IC source for M7 if documented; not a WRF-compatible producer. | Keep read path; document that GPU v0 does not emit wrfinput. |
| `wrfbdy` | Not consumed by `build_replay_case`; validation code can decode `wrfbdy_d01` separately. | `boundary_replay.decode_wrfbdy`; `build_replay_case` uses `load_history_boundary_leaves` from d02 wrfout history. | Native WRF boundary compatibility is not satisfied by the forecast path. | Either implement wrfbdy consumption for forecast forcing or document d02-history replay as an explicit M7 deviation. |
| `wrfout` | Consumed from d02 hourly history for initial/boundary replay; produced as `.npz`, not NetCDF. | `run_to_output_leads` calls `write_wrfout_gpu(...wrfout_gpu_d02_p###h.npz)`. | Downstream raw-wrfout consumers cannot read GPU output unchanged. | Implement NetCDF wrfout writer or adapter. |
| `wrfrst` | No consumer or producer found in allowed audit surface. | No `wrfrst` references in `src/gpuwrf` endpoint code. | M7 restart-continuity gate remains structurally open. | Add WRF-compatible restart or explicit GPU checkpoint format plus continuity test and deviation document. |

## Static Evidence Snippets

### `d02_replay.py`

- `src/gpuwrf/integration/d02_replay.py:40: from gpuwrf.io.gen2_wrfout_loader import normalize_valid_time`
- `src/gpuwrf/integration/d02_replay.py:264: def load_history_boundary_leaves(`
- `src/gpuwrf/integration/d02_replay.py:271: """Build real lateral replay leaves from the Gen2 d02 hourly wrfout history."""`
- `src/gpuwrf/integration/d02_replay.py:274: history_count = len(run.history_files(domain))`
- `src/gpuwrf/integration/d02_replay.py:277: raise FileNotFoundError(f"{run.path} has fewer than two wrfout_{domain} history files")`
- `src/gpuwrf/integration/d02_replay.py:325: "source": "Gen2 d02 hourly wrfout side-history replay",`
- `src/gpuwrf/integration/d02_replay.py:335: def build_replay_case(run_dir: str \| Path = DEFAULT_REPLAY_RUN_DIR, *, domain: str = "d02") -> ReplayCase:`
- `src/gpuwrf/integration/d02_replay.py:338: _debug(f"build_replay_case start run_dir={run_dir} domain={domain}")`
- `src/gpuwrf/integration/d02_replay.py:347: metrics = load_wrfinput_metrics(run.wrfinput_file(domain))`
- `src/gpuwrf/integration/d02_replay.py:351: boundary_leaves, boundary_meta = load_history_boundary_leaves(run, grid, domain=domain)`
- `src/gpuwrf/integration/d02_replay.py:352: _debug("load_history_boundary_leaves complete")`
- `src/gpuwrf/integration/d02_replay.py:426: _debug("build_replay_case done")`
- `src/gpuwrf/integration/d02_replay.py:892: history = run.history_files(domain)`
- `src/gpuwrf/integration/d02_replay.py:895: raise FileNotFoundError(f"{run.path} has no wrfout_{domain} at lead {lead_hours:g}h")`
- `src/gpuwrf/integration/d02_replay.py:1125: case = build_replay_case(run_dir, domain=domain)`
- `src/gpuwrf/integration/d02_replay.py:1224: "build_replay_case",`
- `src/gpuwrf/integration/d02_replay.py:1228: "load_history_boundary_leaves",`

### `boundary_replay.py`

- `src/gpuwrf/io/boundary_replay.py:168: def decode_wrfbdy(`
- `src/gpuwrf/io/boundary_replay.py:213: def wrfbdy_path_for_run(run: Gen2Run, domain: str = "d01") -> Path:`
- `src/gpuwrf/io/boundary_replay.py:447: "decode_wrfbdy",`
- `src/gpuwrf/io/boundary_replay.py:449: "wrfbdy_path_for_run",`

### `gen2_accessor.py`

- `src/gpuwrf/io/gen2_accessor.py:33: WRFOUT_TIME_RE = re.compile(r"wrfout_d\d{2}_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")`
- `src/gpuwrf/io/gen2_accessor.py:182: source_wrfout: str`
- `src/gpuwrf/io/gen2_accessor.py:206: with Dataset(self.source_wrfout, "r") as dataset:`
- `src/gpuwrf/io/gen2_accessor.py:208: raise KeyError(f"{name!r} is not present in {self.source_wrfout}")`
- `src/gpuwrf/io/gen2_accessor.py:223: source_path=self.source_wrfout,`
- `src/gpuwrf/io/gen2_accessor.py:238: restart_compatible=True,`
- `src/gpuwrf/io/gen2_accessor.py:318: for file_path in self.path.glob("wrfout_d0*_*")`
- `src/gpuwrf/io/gen2_accessor.py:319: if (match := re.search(r"wrfout_d(?P<num>\d{2})_", file_path.name))`
- `src/gpuwrf/io/gen2_accessor.py:325: def history_files(self, domain: str) -> list[Path]:`
- `src/gpuwrf/io/gen2_accessor.py:327: files = sorted(self.path.glob(f"wrfout_{domain}_*"))`
- `src/gpuwrf/io/gen2_accessor.py:329: wrfinput = self.wrfinput_file(domain)`
- `src/gpuwrf/io/gen2_accessor.py:332: raise FileNotFoundError(f"no wrfout files for {domain} in {self.path}")`
- `src/gpuwrf/io/gen2_accessor.py:335: def wrfinput_file(self, domain: str) -> Path:`
- `src/gpuwrf/io/gen2_accessor.py:345: with Dataset(self.wrfinput_file(domain), "r") as dataset:`
- `src/gpuwrf/io/gen2_accessor.py:350: for path in self.history_files(domain):`
- `src/gpuwrf/io/gen2_accessor.py:361: first = self.history_files(domain)[0]`
- `src/gpuwrf/io/gen2_accessor.py:399: source_wrfout=str(first),`
- `src/gpuwrf/io/gen2_accessor.py:407: first = self.history_files(domain)[0]`
- `src/gpuwrf/io/gen2_accessor.py:424: path = self.wrfinput_file(domain)`
- `src/gpuwrf/io/gen2_accessor.py:434: file_patterns = ("wrfout_d0*_*", "wrfinput_d0*", "wrfbdy_*", "namelist.input", "namelist.output")`
- `src/gpuwrf/io/gen2_accessor.py:484: files = self.history_files(domain)`
- `src/gpuwrf/io/gen2_accessor.py:496: raise FileNotFoundError(f"no {domain} wrfout history file for time {time!r}")`
