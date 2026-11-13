"""In-run extract collector: per-frame host compute, staging, NetCDF4 + manifest.

Data flow (no step-program change):
  writer materialize stage (step or materialize thread)
    -> prepare_wrfout_payload (the existing single device->host batch; under a
       variable subset the extract's RAW_NAMES are added to the fetch set)
    -> ExtractCollector.submit(name, own_step, prepared, state)
         * device_get of State.roughness_m (one 2-D field; ZNT is not a history field)
         * enqueue references to the immutable host arrays (bounded queue)
  extract worker thread
    -> v1_fields.frame_fields -> atomic per-frame staging file (crash/resume safe)
  run end: finalize() -> one NetCDF4 per requested domain (tmp, full readback,
    rename) -> manifest.json written atomically after every readback.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gpuwrf.extract import points as PT
from gpuwrf.extract import v1_fields as V
from gpuwrf.extract.request import SEGMENT_PREFIX, ExtractRequest, published_chain

SCHEMA = "gpuwrf.extract.v1"
SPEC_DOC = "TO_WRF_GPU2_EXTRACT_SPEC_V1_20261110T_REEXTRACT.md"
SPEC_SHA256_PREFIX = "5dc0d04a"
STAGING = ".staging"
TIME_VARS = {"actual_time_unix_s": np.int64, "nominal_time_unix_s": np.int64,
             "time_offset_seconds": np.int64, "lead_h": np.float64, "tau_h": np.float64,
             "own_step": np.int64}
MANDATORY = (*(f for f in V.SURFACE if f not in V.PRODUCER_OPTIONAL), "ZNT", "u10_east", "v10_north", "wind_speed", "cloud_max", "LWP")
PBL_NAMES = {0: "none", 1: "YSU", 2: "MYJ", 3: "GFS", 4: "QNSE", 5: "MYNN2.5", 6: "MYNN3", 7: "ACM2",
             8: "BouLac", 9: "UW", 10: "TEMF", 11: "Shin-Hong", 12: "GBM", 16: "EEPS", 17: "KEPS", 99: "MRF"}


@dataclass(frozen=True)
class DomainPlan:
    name: str
    dt_s: float
    interval_s: float
    expected_steps: tuple[int, ...]
    bl_pbl_physics: int | None = None
    mp_physics: int | None = None
    nwp_diagnostics: int = 0
    afwa_severe_opt: int | None = None
    dx_m: float | None = None
    cu_physics: int | None = None


def _scalar(namelist: Any, key: str, default=None):
    if namelist is None:
        return default
    value = namelist.get(key, default) if isinstance(namelist, Mapping) else getattr(namelist, key, default)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def plan_for(name, namelist, dt_s, interval_minutes, alarm_steps, emit_initial, dx_m=None) -> DomainPlan:
    steps = tuple(sorted(set(([0] if emit_initial else []) + [int(s) for s in alarm_steps])))
    return DomainPlan(name=name, dt_s=float(dt_s), interval_s=float(interval_minutes) * 60.0,
                      expected_steps=steps, bl_pbl_physics=_scalar(namelist, "bl_pbl_physics"),
                      mp_physics=_scalar(namelist, "mp_physics"),
                      nwp_diagnostics=_scalar(namelist, "nwp_diagnostics", 0) or 0,
                      afwa_severe_opt=_scalar(namelist, "afwa_severe_opt"),
                      dx_m=None if dx_m is None else float(dx_m),
                      cu_physics=_scalar(namelist, "cu_physics"))


def history_alarm_steps(total_steps: int, dt_s: float, interval_s: float) -> tuple[int, ...]:
    """Absolute history alarm steps ``ceil(i * interval / dt)`` for i >= 1, coalesced -- the same rule as
    ``nested_pipeline._history_alarm_steps`` (equality is tested); used by the offline re-extract."""
    dt = Fraction(str(float(dt_s)))
    interval = Fraction(str(float(interval_s)))
    if dt <= 0 or interval <= 0 or int(total_steps) != total_steps or total_steps < 0:
        raise ValueError("history alarm schedule needs dt > 0, interval > 0 and a nonnegative step count")
    ratio = interval / dt
    out: list[int] = []
    for k in range(1, int((int(total_steps) * dt) // interval) + 1):
        a = k * ratio
        step = -(-a.numerator // a.denominator)
        if not out or out[-1] != step:
            out.append(step)
    return tuple(out)


# Writer routes for the extract's surface inputs: the base builder reads these State aliases and fills a
# default when none exists; the full/diagnostic routes look the WRF name up in diagnostics/state/land
# (wrfout_writer._full_source_aliases). A name none of them provides is a compatibility fallback.
PRODUCER_ALIASES = {
    "U10": ("U10", "u10"), "V10": ("V10", "v10"), "T2": ("T2", "t2"), "Q2": ("Q2", "q2"),
    "TSK": ("TSK", "tsk", "t_skin"), "PSFC": ("PSFC", "psfc"), "PBLH": ("PBLH", "pblh"),
    "UST": ("UST", "ustar"), "HFX": ("HFX", "hfx"), "LH": ("LH", "lh"),
    "RAINC": ("RAINC", "rainc", "rainc_acc"), "RAINNC": ("RAINNC", "rainnc", "rain_acc"),
    "SWDOWN": ("SWDOWN", "swdown"), "GLW": ("GLW", "glw"), "SST": ("SST", "sst", "t_sst"),
    "WSPD10MAX": ("WSPD10MAX", "wspd10max"),  # optional: nwp_diagnostics or AFWA severe (lane v50-gust)
}


def output_cadence(plan: DomainPlan) -> dict:
    """Per-nest output cadence + per-field time sampling (manifest)."""
    per = Fraction(str(plan.interval_s)) / Fraction(str(plan.dt_s))
    return {"history_interval_s": plan.interval_s, "dt_s": plan.dt_s,
            "own_steps_per_interval": str(per), "frame_own_steps": list(plan.expected_steps),
            "frame_rule": "lead 0 + WRF history alarms ceil(i*interval/dt); actual time = run start + own_step*dt, "
                          "nominal = floor(lead/interval)*interval (time_offset_seconds = actual - nominal)",
            "sampling": {**{f: V.SAMPLING[f] for f in V.SAMPLING},
                         "other_fields": "instantaneous at the frame's actual time (UST, U10, V10, ... every frame)"},
            "wspd10max_producers": {"nwp_diagnostics": plan.nwp_diagnostics, "afwa_severe_opt": plan.afwa_severe_opt},
            "wspd10max": wspd10max_cadence(plan)}


def wspd10max_producer(plan: DomainPlan) -> str | None:
    """The run's WSPD10MAX producer (gpuwrf refuses AFWA together with nwp_diagnostics)."""
    if (plan.afwa_severe_opt or 0) == 1:
        return "afwa_severe"
    if (plan.nwp_diagnostics or 0) == 1:
        return "nwp_diagnostics"
    return None


def wspd10max_window(plan: DomainPlan, own_step: int) -> tuple[int, int] | None:
    """Own steps [first, last] whose 10 m winds enter the WSPD10MAX written at ``own_step``; None at lead 0
    (no step completed) or without a known producer.
    afwa_severe (diagnostics/afwa.reset_due, module_diag_afwa.F is_after_history_dump): reset on step 1 and on
    the step after each history write -> previous alarm step + 1 .. own_step.
    nwp_diagnostics (diagnostics/nwp.py, WRF idump = INT(REAL(history_interval*60)/dt)): reset where
    (step-1) % idump == 0 -> 1 + idump*floor((own_step-1)/idump) .. own_step."""
    step = int(own_step)
    producer = wspd10max_producer(plan)
    if step < 1 or producer is None:
        return None
    if producer == "afwa_severe":
        ratio = Fraction(str(plan.interval_s)) / Fraction(str(plan.dt_s))
        k = max(int(step / ratio) - 1, 0)
        while -(-((k + 1) * ratio).numerator // ((k + 1) * ratio).denominator) < step:
            k += 1
        prev = -(-(k * ratio).numerator // (k * ratio).denominator) if k > 0 else 0
        return prev + 1, step
    idump = int(np.float32(np.float32(plan.interval_s / 60.0) * np.float32(60)) / np.float32(plan.dt_s))
    if idump < 1:
        raise ValueError(f"{plan.name}: nwp_diagnostics history interval shorter than dt")
    return 1 + idump * ((step - 1) // idump), step


def wspd10max_cadence(plan: DomainPlan) -> dict:
    producer = wspd10max_producer(plan)
    rules = {"afwa_severe": "reset on the step after each history write: window = previous history step + 1 .. "
                            "frame step (since the previous output)",
             "nwp_diagnostics": "reset every idump = trunc(interval*60/dt) own steps (WRF integer truncation): "
                                "window = 1 + idump*floor((step-1)/idump) .. frame step; NOT since the previous "
                                "output when interval/dt is not an integer",
             None: "no producer in this run: WSPD10MAX is NOT_AVAILABLE (a payload value without a producer is "
                   "refused, never a fallback)"}
    return {"producer": producer, "window_rule": rules[producer],
            "frame_window_own_steps": {str(s): (list(w) if (w := wspd10max_window(plan, s)) else None)
                                       for s in plan.expected_steps},
            "lead_0": "initial value before the first step (no window)"}


_SOURCE_NAMES = ("history diagnostics", "model State", "land-surface state")


def producer_provenance(sources) -> dict:
    """name -> where a producer was found in ``sources`` (diagnostics, state, land state), None if nowhere."""
    from gpuwrf.io import wrfout_writer as base
    out = {}
    for name, aliases in PRODUCER_ALIASES.items():
        names = tuple(dict.fromkeys((*aliases, *base._full_source_aliases(name))))
        out[name] = next((f"{_SOURCE_NAMES[i] if i < 3 else 'source'}: {a}" for i, src in enumerate(sources)
                          if src is not None for a in names if base._lookup(src, a, None) is not None), None)
    if out.get("SST") is not None:  # a producer exists: record what the value is (same text as the offline path)
        out["SST"] = V.SST_SOURCE
    return out


def unobserved_inputs(sources) -> tuple[str, ...]:
    """Surface names with no producer in any of ``sources`` (diagnostics, state, land state)."""
    return tuple(n for n, where in producer_provenance(sources).items() if where is None)


def _rss_kib() -> int | None:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def frame_times(plan: DomainPlan, own_step: int, run_start: datetime, issue: datetime) -> dict:
    lead = Fraction(str(plan.dt_s)) * int(own_step)
    interval = Fraction(str(plan.interval_s))
    nominal = (lead // interval) * interval
    if lead.denominator != 1 or Fraction(nominal).denominator != 1:
        raise ValueError(f"{plan.name}: history time step {own_step} is not an integral second")
    start = int(run_start.timestamp())
    if start != run_start.timestamp():
        raise ValueError("run start is not an integral UTC second")
    actual = start + int(lead)
    nom = start + int(nominal)
    return {"actual_time_unix_s": actual, "nominal_time_unix_s": nom, "time_offset_seconds": actual - nom,
            "lead_h": (nom - int(issue.timestamp())) / 3600.0, "tau_h": (nom - start) / 3600.0,
            "own_step": int(own_step)}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def code_sha256() -> dict:
    root = Path(__file__).resolve().parent
    return {p.name: _sha256_file(p) for p in sorted(root.glob("*.py"))}


def _coerce_raw(prepared_fields: Mapping[str, Any], dimensions: Mapping[str, Any],
                names: tuple[str, ...] = V.RAW_NAMES) -> tuple[dict, dict]:
    """(raw, shape_mismatch). Only arrays of the exact WRF shape (optionally with a leading Time axis
    of 1) are taken; anything else (e.g. a scalar the writer would broadcast) is refused, not coerced."""
    from gpuwrf.io import wrfout_writer as base
    raw, mismatch = {}, {}
    for name in names:
        if name not in prepared_fields:
            continue
        spec = base.WRFOUT_VARIABLE_SPECS[name]
        shape = tuple(base._shape_for_dimensions(spec.dimensions, dimensions))
        got = tuple(np.shape(prepared_fields[name]))
        if got not in (shape, (1, *shape)):
            mismatch[name] = {"shape": list(got), "expected": list(shape)}
            continue
        # Exactly the array the history writer writes for this variable.
        raw[name] = base._coerce_array(name, prepared_fields[name], shape,
                                       dtype=base._numpy_dtype_for_spec(spec))
    return raw, mismatch


def _grid_column_metrics(grid: Any, dimensions: Mapping[str, Any]) -> dict:
    """DNW/C1H/C2H exactly as the FULL history writer writes them (wrfout_writer._add_grid_coordinate_fields:
    ``_coerce_array(name, np.asarray(metrics.<attr>), (nz,))`` from the GridSpec DycoreMetrics).

    The reduced-subset builder has no producer for these names and the full-stream compatibility layer
    then fills them with zeros [M: PROD lead 0, all three vectors 0.0]; the extract therefore never takes
    them from a payload in-run. Absent metrics -> nothing is added (payload values or MISSING, then QC)."""
    if grid is None:
        return {}
    from gpuwrf.io import wrfout_writer as base
    metrics = base._lookup(grid, "metrics")
    nz = dimensions.get("bottom_top")
    if metrics is None or nz is None:
        return {}
    out = {}
    for name, attr in (("DNW", "dnw"), ("C1H", "c1h"), ("C2H", "c2h")):
        value = base._lookup(metrics, attr)
        if value is not None and np.asarray(value).shape == (int(nz),):
            out[name] = base._coerce_array(name, np.asarray(value), (int(nz),))
    return out


class ExtractCollector:
    """Host-only V1 extract; one per run (B=1)."""

    def __init__(self, request: ExtractRequest, *, run_start: datetime, plans: Mapping[str, DomainPlan],
                 input_dir: Path, resumed: bool = False, max_pending: int = 2) -> None:
        self.request = request
        self.run_start = run_start.astimezone(timezone.utc)
        self.input_dir = Path(input_dir)
        self.all_domains = tuple(plans)
        wanted = request.domains if request.domains is not None else self.all_domains
        unknown = sorted(set(wanted) - set(plans))
        if unknown:
            raise ValueError(f"extract requests domains not in this run: {unknown}")
        self.plans = {name: plans[name] for name in wanted}
        # Extension of a completed run: the published extract stays untouched; this run's new frames go to a
        # chained segment <extract_dir>/segment_NN (own manifest, parents bound by manifest SHA256).
        self.segment: dict | None = None
        self.replayed_published: list[str] = []
        if resumed and (request.extract_dir / "manifest.json").is_file():
            self.segment = self._extension_segment()
        for plan in self.plans.values():
            if self.segment is None and 0 not in plan.expected_steps:
                raise ValueError("extract V1 needs the lead-0 frame: run with --emit-initial-history")
            for step in plan.expected_steps:
                frame_times(plan, step, self.run_start, request.issue_utc)
        self.points: tuple[PT.Point, ...] = ()
        self.point_sources: tuple[PT.PointSource, ...] = ()
        self.max_distance_m: dict[str, float] = {}
        if request.points_csv:
            self.points, self.point_sources = PT.load_point_csvs(request.points_csv)
            if tuple(s.sha256 for s in self.point_sources) != tuple(request.points_sha256):
                raise ValueError("extract points_csv changed since the request was bound (SHA differs)")
            overrides = dict(request.max_distance_m)
            unknown = sorted(set(overrides) - set(self.plans))
            if unknown:
                raise ValueError(f"extract max_distance_m names domains not extracted: {unknown}")
            for name, plan in self.plans.items():
                if name in overrides:
                    self.max_distance_m[name] = overrides[name]
                elif plan.dx_m is not None and plan.dx_m > 0:
                    self.max_distance_m[name] = PT.DEFAULT_MAX_DISTANCE_CELLS * plan.dx_m
                else:
                    raise ValueError(f"{name}: grid spacing unknown; give max_distance_m for it")
        self.fetch_names = tuple(dict.fromkeys(V.RAW_NAMES + (PT.PROFILE_RAW if self.points else ())))
        # DNW/C1H/C2H come from grid.metrics in submit() (see _grid_column_metrics); a reduced history
        # payload carries zero vectors for them, so the reduced fetch set never asks for them.
        self.reduced_fetch_names = tuple(n for n in self.fetch_names if n not in V.COLUMN)
        self._mappings: dict[str, tuple[PT.Mapping1, ...]] = {}
        self.dir = self.segment["dir"] if self.segment is not None else request.extract_dir
        self.stage = self.dir / STAGING
        self._bind(resumed)
        self.conflicts: list[str] = []
        self.timing: dict[str, list[float]] = {"submit_s": [], "compute_s": [], "profile_s": [],
                                               "stage_write_s": []}
        self.rss_kib: list[int] = []
        self.rss_delta_kib: list[int] = []
        self.payload_bytes: dict[str, list[int]] = {}  # host bytes of each frame's fetched payload (= its D2H volume)
        self._queued_bytes = 0
        self.queued_bytes_max = 0
        self._bytes_lock = threading.Lock()
        self._submitted: set[tuple[str, int]] = set()
        self._error: BaseException | None = None
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max_pending)
        self._thread = threading.Thread(target=self._worker, name="gpuwrf-extract", daemon=True)
        self._thread.start()
        self._closed = False

    # ------------------------------------------------------------------ extension
    def _extension_segment(self) -> dict:
        """Plan the segment of a resumed extension: expected frames = the extended schedule minus every frame
        already published (base + READY segments). Refuses before any write when the published parts do not
        belong to this run or the extension does not add a frame to every extracted nest."""
        base = Path(self.request.extract_dir)
        chain = published_chain(self.request)
        published = {n: set() for n in self.plans}
        for part, manifest, _ in chain:
            if manifest.get("wrf_start_utc") != self.run_start.isoformat():
                raise ValueError(f"extract {part} belongs to a run starting {manifest.get('wrf_start_utc')}, "
                                 f"not {self.run_start.isoformat()}")
            domains = manifest.get("domains", {})
            # Compare the extracted nests only; the run's other nests are recorded NOT_REQUESTED (subset extract).
            extracted = {n: e for n, e in domains.items() if (e or {}).get("status") != "NOT_REQUESTED"}
            skipped = set(domains) - set(extracted)
            if set(extracted) != set(self.plans) or skipped != set(self.all_domains) - set(self.plans):
                raise ValueError(f"extract {part} extracted nests {sorted(extracted)} (not requested "
                                 f"{sorted(skipped)}); this run extracts {sorted(self.plans)} of {list(self.all_domains)}")
            for n, entry in extracted.items():
                steps = set(entry["output_cadence"]["frame_own_steps"]) - set(entry.get("missing_steps", []))
                published[n] |= {int(x) for x in steps}
        plans = {}
        for n, plan in self.plans.items():
            if not published[n] <= set(plan.expected_steps):
                raise ValueError(f"extension {n}: published frames {sorted(published[n] - set(plan.expected_steps))} "
                                 "are not in this run's schedule (an extension may only add frames)")
            rest = tuple(x for x in plan.expected_steps if x not in published[n])
            if not rest:
                raise ValueError(f"extension {n}: the resumed run adds no history frame beyond the published extract")
            plans[n] = replace(plan, expected_steps=rest)
        self.plans = plans
        index = len(chain)
        return {"index": index, "dir": base / f"{SEGMENT_PREFIX}{index:02d}",
                "parents": [{"dir": "." if part == base else part.name, "manifest_sha256": sha}
                            for part, _, sha in chain],
                "published_steps": {n: sorted(v) for n, v in published.items()}}

    # ------------------------------------------------------------------ binding
    def _binding(self) -> dict:
        out = {"schema": SCHEMA, "policy": self.request.policy(), "run_start_utc": self.run_start.isoformat(),
               **({"segment": {k: v for k, v in self.segment.items() if k != "dir"}} if self.segment else {}),
               "issue_utc": self.request.issue_utc.isoformat(),
               "domains": {n: {"dt_s": p.dt_s, "interval_s": p.interval_s, "expected_steps": list(p.expected_steps)}
                           for n, p in self.plans.items()}}
        if self.points:
            out["max_distance_m"] = self.max_distance_m
        return out

    def _bind(self, resumed: bool) -> None:
        finals = sorted(p.name for p in self.dir.iterdir()
                        if p.name in ("manifest.json", "points.csv") or p.name.startswith(("extract_", "profiles_"))
                        ) if self.dir.is_dir() else []
        if finals and resumed:
            raise ValueError(f"extract {self.dir} was already published by a completed run {finals}; a resumed "
                             "extension cannot append to published files -- extend with a fresh request/extract dir")
        self.dir.mkdir(parents=True, exist_ok=True)
        if finals:
            raise FileExistsError(f"extract dir {self.dir} already holds final outputs {sorted(finals)}; "
                                  "use a fresh --extract-dir per attempt")
        current = self._binding()
        binding = json.dumps(current, sort_keys=True, indent=1).encode()
        path = self.stage / "BINDING.json"
        if path.exists():
            if not resumed:
                raise FileExistsError(f"extract staging {self.stage} exists from another attempt; "
                                      "use a fresh --extract-dir (only a resume may continue it)")
            old = json.loads(path.read_bytes())

            def fixed(doc):  # everything except the per-domain step lists
                return {**doc, "domains": {n: {k: v for k, v in d.items() if k != "expected_steps"}
                                           for n, d in doc.get("domains", {}).items()}}
            if fixed(old) != fixed(current):
                raise ValueError("extract staging binding differs from this resumed run")
            for n, d in current["domains"].items():
                if not set(old["domains"][n]["expected_steps"]) <= set(d["expected_steps"]):
                    raise ValueError(f"extract resume {n}: the run may only extend the bound frame schedule")
            if path.read_bytes() != binding:  # same-policy extension: rebind the longer schedule
                _atomic_bytes(path, binding)
            return
        self.stage.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(path, binding)

    # --------------------------------------------------------------- run hooks
    @property
    def raw_history(self) -> bool:
        return self.request.raw_history

    def wants(self, name: str) -> bool:
        return name in self.plans

    def prepare_subset(self, subset, name: str | None = None):
        """Device fetch set of nest ``name``. RAW on: full history (None) or the written subset, plus the
        extract inputs only for extracted nests. RAW off: only the extract inputs (unrequested nests are
        skipped by the caller before any fetch, see ``wants``)."""
        if name is not None and name not in self.plans:
            return subset  # not extracted: exactly the history stream's own fetch
        if not self.request.raw_history:
            return self.reduced_fetch_names
        if subset is None:
            return None
        return tuple(dict.fromkeys((*subset, *self.reduced_fetch_names)))

    def submit(self, name: str, own_step: int, prepared: Any, state: Any, *, sources=None) -> None:
        """In-run entry. ``sources`` = (diagnostics, state, land state) of the payload build: surface
        names none of them provides are writer fallbacks and are never reported as observed."""
        self._raise_pending()
        if name not in self.plans:
            return
        t0 = time.perf_counter()
        znt = getattr(state, "roughness_m", None)
        if znt is not None:
            import jax
            znt = np.array(jax.device_get(znt), dtype=np.float32, copy=True)
        self.payload_bytes.setdefault(name, []).append(
            int(sum(getattr(v, "nbytes", 0) for v in prepared.fields.values())) + int(getattr(znt, "nbytes", 0)))
        # Only the extract inputs are kept (references, no copy): the queue never holds a full payload.
        fields = {k: prepared.fields[k] for k in self.fetch_names if k in prepared.fields}
        if hasattr(prepared, "grid"):
            # In-run DNW/C1H/C2H come ONLY from the grid metrics (a reduced payload zero-fills them);
            # an absent metric stays absent (MISSING_INPUT), never a payload value.
            for k in V.COLUMN:
                fields.pop(k, None)
            fields.update(_grid_column_metrics(prepared.grid, prepared.dimensions))
        prov = producer_provenance(sources) if sources is not None else {}
        unobserved = tuple(n for n, where in prov.items() if where is None)
        self.submit_host(name, own_step, fields, prepared.dimensions, znt, _t0=t0, unobserved=unobserved,
                         provenance={n: w for n, w in prov.items() if w is not None and n in V.PRODUCER_OPTIONAL})

    def submit_host(self, name: str, own_step: int, fields: Mapping[str, Any], dimensions: Mapping[str, Any],
                    znt: np.ndarray | None, *, _t0: float | None = None, unobserved=(),
                    provenance: Mapping[str, str] | None = None) -> None:
        """Host arrays (WRF names/shapes, as written) of one frame; also the offline re-extract entry."""
        self._raise_pending()
        if self._closed:
            raise RuntimeError("extract collector is closed; frame submitted after close")
        if name not in self.plans:
            return
        if self.segment is not None and int(own_step) in self.segment["published_steps"][name]:
            # Extension: a frame the parents already published (e.g. a history replay at the resume step) is
            # not restaged; it stays in its published part. Recorded in the segment manifest.
            self.replayed_published.append(f"{name}:{int(own_step)}")
            return
        key = (name, int(own_step))
        if key in self._submitted:
            # One process submits every history step once; a second submission is a schedule error.
            self.conflicts.append(f"{name} step {own_step}: submitted twice in one process")
            return
        self._submitted.add(key)
        t0 = time.perf_counter() if _t0 is None else _t0
        fields = {k: fields[k] for k in self.fetch_names if k in fields}
        nbytes = int(sum(np.asarray(v).nbytes for v in fields.values()))
        with self._bytes_lock:
            self._queued_bytes += nbytes
            self.queued_bytes_max = max(self.queued_bytes_max, self._queued_bytes)
        self._queue.put((name, int(own_step), fields, dict(dimensions), znt, tuple(unobserved),
                         dict(provenance or {}), nbytes))
        self.timing["submit_s"].append(time.perf_counter() - t0)

    def _raise_pending(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"extract worker failed: {type(self._error).__name__}: {self._error}") from self._error

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                if self._error is None:
                    self._process(*job[:7])
            except BaseException as exc:  # noqa: BLE001 - surfaced on the next submit/join
                if self._error is None:
                    self._error = exc
            finally:
                if job is not None:
                    with self._bytes_lock:
                        self._queued_bytes -= job[7]
                self._queue.task_done()

    def _mapping(self, name: str, static: Mapping[str, np.ndarray], ddir: Path) -> tuple[PT.Mapping1, ...]:
        if name not in self._mappings:
            mapping = PT.map_points(self.points, name, static, self.max_distance_m[name])
            doc = json.dumps([m.row() for m in mapping], sort_keys=True).encode()
            path = ddir / "points.json"
            if path.exists():
                if path.read_bytes() != doc:
                    self.conflicts.append(f"{name}: point mapping differs from the staged mapping")
            else:
                _atomic_bytes(path, doc)
            self._mappings[name] = mapping
        return self._mappings[name]

    def _process(self, name, own_step, fields, dimensions, znt, unobserved=(), provenance=None) -> None:
        plan = self.plans[name]
        rss0 = _rss_kib()
        t0 = time.perf_counter()
        raw, mismatch = _coerce_raw(fields, dimensions, self.fetch_names)
        out, status, qc = V.frame_fields(raw, znt, mp_physics=plan.mp_physics, cu_physics=plan.cu_physics,
                                         unobserved=unobserved, provenance=provenance)
        if mismatch:
            qc["shape_mismatch_inputs"] = mismatch
        if status.get("WSPD10MAX", {}).get("status") == "OK":
            window = wspd10max_window(plan, own_step)
            status["WSPD10MAX"]["producer"] = wspd10max_producer(plan) or "unknown (present in the input, no " \
                                                                           "producer declared in the plan)"
            status["WSPD10MAX"]["window_own_steps"] = list(window) if window else None
        static = {k: np.asarray(raw[k], np.float32) for k in V.STATIC if k in raw}
        static_sha = hashlib.sha256(b"".join(static[k].tobytes() for k in V.STATIC if k in static)).hexdigest()
        meta = {"domain": name, "times": frame_times(plan, own_step, self.run_start, self.request.issue_utc),
                "status": status, "qc": qc, "static_sha256": static_sha,
                "static_missing": [k for k in V.STATIC if k not in static]}
        t1 = time.perf_counter()
        ddir = self.stage / name
        ddir.mkdir(parents=True, exist_ok=True)
        staged = dict(out)
        if self.points:
            mapping = self._mapping(name, static, ddir)
            ok = [m for m in mapping if m.status == "OK"]
            if ok:
                jj = np.array([m.y for m in ok], dtype=np.intp)
                ii = np.array([m.x for m in ok], dtype=np.intp)
                ice = V.expected_ice_species(plan.mp_physics)
                cols, pstatus = PT.profile_columns(raw, jj, ii, qncloud_available=plan.mp_physics == 28,
                                                   qice_expected=None if ice is None else "QICE" in ice)
                nz = int(np.asarray(raw["T"]).shape[0]) if "T" in raw else 0
                for f in PT.PROFILE_FIELDS:
                    staged["prof__" + f] = cols.get(f, np.full((len(ok), nz), np.nan, np.float32))
                for f in V.GRID_FIELDS:
                    staged["psfc__" + f] = np.asarray(out[f])[jj, ii]
                meta["profile_status"] = pstatus
                meta["profile_qc"] = {f"negative_{f}": int((cols[f] < 0).sum())
                                      for f in ("QV_kg_kg", "QCLOUD_kg_kg", "QICE_kg_kg") if f in cols}
        t2 = time.perf_counter()
        spath = ddir / "static.npz"
        if not spath.exists():
            self._stage_npz(spath, static, {"static_sha256": static_sha})
        target = ddir / f"s{own_step:09d}.npz"
        if target.exists():
            with np.load(target) as old:
                same = set(old.files) == set(staged) | {"__meta__"} and all(
                    old[k].tobytes() == np.asarray(staged[k]).tobytes() for k in staged)
            if not same:
                self.conflicts.append(f"{name} step {own_step}: replayed frame differs from staged frame")
        else:
            self._stage_npz(target, staged, meta)
        self.timing["compute_s"].append(t1 - t0)
        self.timing["profile_s"].append(t2 - t1)
        self.timing["stage_write_s"].append(time.perf_counter() - t2)
        rss = _rss_kib()
        if rss is not None:
            self.rss_kib.append(rss)
            if rss0 is not None:
                self.rss_delta_kib.append(rss - rss0)

    @staticmethod
    def _stage_npz(path: Path, arrays: Mapping[str, np.ndarray], meta: dict) -> None:
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        payload = {k: np.asarray(v) for k, v in arrays.items()}
        payload["__meta__"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
        with open(tmp, "wb") as f:
            np.savez(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)  # a verified restart may follow (checkpoint drain joins this queue)

    def join(self) -> None:
        if not self._closed:
            self._queue.join()
        self._raise_pending()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()

    # ---------------------------------------------------------------- finalize
    def finalize(self, *, engine: Mapping[str, Any] | None = None) -> dict:
        self.join()
        self.close()
        self._raise_pending()
        t0 = time.perf_counter()
        domains: dict[str, Any] = {}
        artifacts: dict[str, Any] = {}
        for name in self.all_domains:
            if name not in self.plans:
                domains[name] = {"status": "NOT_REQUESTED"}
                continue
            entry, produced = self._finalize_domain(name)
            domains[name] = entry
            for file_name, info in produced:
                artifacts[file_name] = info
        points_block = None
        if self.points:
            points_block, csv_artifact = self._finalize_points(domains)
            artifacts[csv_artifact[0]] = csv_artifact[1]
        statuses = [d["status"] for d in domains.values() if d["status"] != "NOT_REQUESTED"]
        overall = "FAIL" if ("FAIL" in statuses or self.conflicts) else "PARTIAL" if "PARTIAL" in statuses else "READY"
        manifest = self._manifest(overall, domains, artifacts, engine or {})
        if points_block is not None:
            manifest["points"] = points_block
            manifest["deferred"].pop("points_profiles", None)
        import resource
        manifest["host_cost"] = {**{k: {"n": len(v), "sum_s": float(sum(v)), "max_s": float(max(v)) if v else 0.0,
                                        "mean_s": float(sum(v) / len(v)) if v else 0.0}
                                    for k, v in self.timing.items()},
                                 "finalize_s": time.perf_counter() - t0,
                                 "rss_kib_after_frame": {"n": len(self.rss_kib),
                                                         "min": min(self.rss_kib, default=None),
                                                         "max": max(self.rss_kib, default=None),
                                                         "last": self.rss_kib[-1] if self.rss_kib else None},
                                 "rss_delta_kib_per_frame": {"n": len(self.rss_delta_kib),
                                                             "max": max(self.rss_delta_kib, default=None),
                                                             "sum": int(sum(self.rss_delta_kib))},
                                 "queued_input_bytes_max": int(self.queued_bytes_max),
                                 "fetched_payload_bytes_per_frame": {
                                     n: {"n": len(v), "max": max(v), "mean": float(sum(v) / len(v))}
                                     for n, v in sorted(self.payload_bytes.items()) if v},
                                 "queue_bound": f"{self._queue.maxsize} pending + 1 active frame, extract inputs only",
                                 "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                 "note": "per-frame host work runs on the extract thread; submit_s is the "
                                         "step/materialize-thread cost (ZNT device_get + enqueue)"}
        manifest["raw_history"] = "on" if self.request.raw_history else "off (extract only; no wrfout written)"
        manifest["column_metrics_source"] = ("in-run: DNW/C1H/C2H from GridSpec DycoreMetrics, coerced as the full history "
                                             "writer does (never from a reduced payload, which carries zeros); "
                                             "offline: the wrfout variables")
        # Publication order: every fallible step (staging cleanup) runs BEFORE manifest.json exists;
        # the manifest is the last rename. A failed cleanup publishes FAIL, never READY.
        if overall == "READY":
            try:
                self._remove_staging()
            except Exception as exc:  # noqa: BLE001 - recorded and published as FAIL
                overall = manifest["status"] = "FAIL"
                manifest.setdefault("finalize_problems", []).append(
                    f"staging cleanup failed: {type(exc).__name__}: {exc}")
        overall = self._publish_manifest(manifest, overall)
        return {"status": overall, "manifest": str(self.dir / "manifest.json"),
                "files": sorted(artifacts), "domains": {n: d["status"] for n, d in domains.items()}}

    def _publish_manifest(self, manifest: dict, overall: str) -> str:
        """manifest.json strictly last: tmp + fsync, rename, directory fsync. If any step fails, no READY
        manifest remains: it is removed and a FAIL manifest is attempted (best effort); returns the status."""
        path = self.dir / "manifest.json"
        try:
            _atomic_bytes(path, (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
            return overall
        except Exception as exc:  # noqa: BLE001 - publication failure is a FAIL, never READY
            path.unlink(missing_ok=True)
            for p in self.dir.glob(".manifest.json.tmp-*"):
                p.unlink(missing_ok=True)
            manifest = {**manifest, "status": "FAIL"}
            manifest.setdefault("finalize_problems", []).append(f"manifest publication failed: {type(exc).__name__}: {exc}")
            try:
                _atomic_bytes(path, (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
            except Exception:  # noqa: BLE001
                path.unlink(missing_ok=True)
            return "FAIL"

    def _remove_staging(self) -> None:
        for name in self.plans:
            ddir = self.stage / name
            if ddir.is_dir():
                for p in ddir.iterdir():
                    if p.suffix in (".npz", ".json"):
                        p.unlink()
                ddir.rmdir()
        (self.stage / "BINDING.json").unlink(missing_ok=True)
        self.stage.rmdir()

    def _finalize_domain(self, name: str):
        import netCDF4

        plan = self.plans[name]
        ddir = self.stage / name
        frames = sorted(ddir.glob("s[0-9]*.npz")) if ddir.is_dir() else []
        steps = [int(p.stem[1:]) for p in frames]
        missing = sorted(set(plan.expected_steps) - set(steps))
        extra = sorted(set(steps) - set(plan.expected_steps))
        entry: dict[str, Any] = {"expected_frames": len(plan.expected_steps), "frames": len(frames),
                                 "missing_steps": missing, "unexpected_steps": extra,
                                 "dt_s": plan.dt_s, "history_interval_s": plan.interval_s,
                                 "pbl_scheme": {"bl_pbl_physics": plan.bl_pbl_physics,
                                                "name": PBL_NAMES.get(plan.bl_pbl_physics, "unknown")},
                                 "mp_physics": plan.mp_physics, "nwp_diagnostics": plan.nwp_diagnostics,
                                 "output_cadence": output_cadence(plan)}
        if not frames:
            entry["status"] = "FAIL"
            entry["reason"] = "no frames staged"
            return entry, []
        with np.load(ddir / "static.npz") as s:
            static = {k: s[k].copy() for k in s.files if k != "__meta__"}
            static_sha = json.loads(s["__meta__"].tobytes())["static_sha256"]
        ny, nx = static["HGT"].shape
        metas = []
        for p in frames:
            with np.load(p) as z:
                metas.append(json.loads(z["__meta__"].tobytes()))
        times = {k: np.array([m["times"][k] for m in metas], dtype=dt) for k, dt in TIME_VARS.items()}
        problems: list[str] = []
        if len(set(times["actual_time_unix_s"].tolist())) != len(frames) or np.any(np.diff(times["actual_time_unix_s"]) <= 0):
            problems.append("time axis not strictly ascending/unique")
        drift = sorted({m["static_sha256"] for m in metas} - {static_sha})
        if drift:
            problems.append("static fields changed between frames")
        absent_static = sorted({k for m in metas for k in m["static_missing"]})
        if absent_static:
            problems.append(f"static fields missing: {absent_static}")
        # Field status aggregation (per field, across frames).
        field_status: dict[str, Any] = {}
        for f in V.GRID_FIELDS:
            seen = {json.dumps(m["status"][f], sort_keys=True) for m in metas}
            field_status[f] = json.loads(seen.pop()) if len(seen) == 1 else {
                "status": "MIXED", "per_frame": [m["status"][f] for m in metas]}
        for f in MANDATORY:
            if field_status[f]["status"] != "OK":
                problems.append(f"mandatory field {f} status {field_status[f]['status']}")
        for f in V.PRODUCER_OPTIONAL:
            if field_status[f]["status"] not in ("OK", "NOT_AVAILABLE"):
                problems.append(f"field {f} status {field_status[f]['status']}")
        qc_frames = [m["qc"] for m in metas]
        range_fail: dict[str, int] = {}
        for q in qc_frames:
            for k, v in q.get("range_failures", {}).items():
                range_fail[k] = range_fail.get(k, 0) + v
            for k in ("missing_mandatory_inputs", "nonfinite_inputs", "unexplained_nonfinite",
                      "shape_mismatch_inputs", "missing_expected_species", "fill_value_inputs"):
                if q.get(k):
                    problems.append(f"{k}: {q[k]}")
        if range_fail:
            problems.append(f"range failures (not clipped): {range_fail}")
        if not all(q.get("z_w_monotone", False) for q in qc_frames):
            problems.append("z_w not monotone")
        if not all(q.get("mass_positive", False) for q in qc_frames):
            problems.append("non-positive column mass m_k")

        target = self.dir / self.request.file_name(name)
        tmp = self.dir / f".{target.name}.tmp-{os.getpid()}"
        stats: dict[str, Any] = {}
        chunk = (1, min(128, ny), min(128, nx))
        comp = dict(zlib=True, complevel=4, shuffle=True)
        try:
            with netCDF4.Dataset(tmp, "w", format="NETCDF4") as ds:
                ds.createDimension("time", len(frames))
                ds.createDimension("y", ny)
                ds.createDimension("x", nx)
                ds.setncattr("schema", SCHEMA)
                ds.setncattr("spec", f"{SPEC_DOC} (sha256 {SPEC_SHA256_PREFIX}...)")
                ds.setncattr("domain", name)
                ds.setncattr("run_id", self.request.run_id)
                ds.setncattr("member", self.request.member)
                ds.setncattr("issue_utc", self.request.issue_utc.isoformat())
                ds.setncattr("wrf_start_utc", self.run_start.isoformat())
                ds.setncattr("lead_axis", "lead_h=(nominal-issue)/3600; tau_h=(nominal-wrf_start)/3600 (WT1/Nightly)")
                for k, dt in TIME_VARS.items():
                    v = ds.createVariable(k, dt, ("time",))
                    v[:] = times[k]
                for k in V.STATIC:
                    v = ds.createVariable(k, "f4", ("y", "x"), fill_value=np.float32(np.nan), **comp,
                                          chunksizes=chunk[1:])
                    v.units = V.UNITS[k]
                    v[:] = static[k]
                for f in V.GRID_FIELDS:
                    is_status = f in V.STATUS_FIELDS
                    v = ds.createVariable(f, "i1" if is_status else "f4", ("time", "y", "x"),
                                          fill_value=np.int8(-1) if is_status else np.float32(np.nan),
                                          chunksizes=chunk, **comp)
                    v.units = V.UNITS[f]
                    if f in V.DEFINITIONS:
                        v.definition = V.DEFINITIONS[f]
                    v.field_status = json.dumps(field_status[f], sort_keys=True)
                    if is_status:
                        meaning = V.CLOUD_BASE_STATUS if f == "cloud_base_status" else V.INV_STATUS
                        v.flag_values = np.array(sorted(meaning), dtype=np.int8)
                        v.flag_meanings = " ".join(meaning[c] for c in sorted(meaning))
                    v.set_auto_maskandscale(False)
                for i, p in enumerate(frames):
                    with np.load(p) as z:
                        for f in V.GRID_FIELDS:
                            ds[f][i] = z[f]
            # Full readback before publication (each staged frame is opened once).
            nan_pattern_bad: set[str] = set()
            explained = {"cloud_base_msl": "cloud_base_status", "cloud_base_agl": "cloud_base_status",
                         "inv_base_msl": "inv_status", "inv_top_msl": "inv_status",
                         "inv_strength_K": "inv_status", "inv_gradient_K_m": "inv_status"}
            with netCDF4.Dataset(tmp) as ds:
                ds.set_auto_maskandscale(False)
                for k in TIME_VARS:
                    if ds[k][:].tobytes() != times[k].tobytes():
                        raise ValueError(f"readback mismatch {k}")
                for k in V.STATIC:
                    if np.asarray(ds[k][:]).tobytes() != static[k].astype(np.float32).tobytes():
                        raise ValueError(f"readback mismatch {k}")
                stats = {f: {"min": None, "max": None, "nonfinite": 0} for f in V.GRID_FIELDS}
                for i, p in enumerate(frames):
                    with np.load(p) as z:
                        frame = {f: z[f] for f in V.GRID_FIELDS}
                    for f, want in frame.items():
                        got = np.asarray(ds[f][i])
                        if got.dtype != want.dtype or got.tobytes() != want.tobytes():
                            raise ValueError(f"readback mismatch {f} frame {i}")
                        if f not in V.STATUS_FIELDS:
                            st = stats[f]
                            fin = np.isfinite(want)
                            st["nonfinite"] += int((~fin).sum())
                            if fin.any():
                                lo, hi = float(want[fin].min()), float(want[fin].max())
                                st["min"] = lo if st["min"] is None else min(st["min"], lo)
                                st["max"] = hi if st["max"] is None else max(st["max"], hi)
                    # NaN must be explained by a status.
                    for f, sf in explained.items():
                        if not np.array_equal(np.isnan(frame[f]), frame[sf] != 0):
                            nan_pattern_bad.add(f"{f} NaN pattern differs from {sf}")
                for f in V.STATUS_FIELDS:
                    stats[f] = {"min": None, "max": None, "nonfinite": 0}
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, target)
            _fsync_dir(self.dir)
        finally:
            tmp.unlink(missing_ok=True)
        for f in (*MANDATORY, *V.PRODUCER_OPTIONAL):
            if field_status[f]["status"] == "OK" and stats[f]["nonfinite"]:
                problems.append(f"non-finite mandatory {f}: {stats[f]['nonfinite']}")
        problems.extend(sorted(nan_pattern_bad))
        produced = [(target.name, {"bytes": target.stat().st_size, "sha256": _sha256_file(target),
                                   "domain": name, "kind": "grid", "readback": "PASS"})]
        partial_reasons = []
        if self.points:
            pentry, partifact, pproblems, ppartial = self._finalize_profiles(name, frames, metas, times)
            entry["profiles"] = pentry
            problems.extend(pproblems)
            partial_reasons.extend(ppartial)
            if partifact is not None:
                produced.append(partifact)
        offsets = times["time_offset_seconds"]
        if missing or extra:
            partial_reasons.append("missing or unexpected frames")
        entry["partial_reasons"] = partial_reasons
        entry.update({
            "status": "FAIL" if problems else ("PARTIAL" if partial_reasons else "READY"),
            "problems": problems, "shape_yx": [ny, nx], "file": target.name,
            "geometry_sha256": static_sha,
            "time": {"first_actual_unix_s": int(times["actual_time_unix_s"][0]),
                     "last_actual_unix_s": int(times["actual_time_unix_s"][-1]),
                     "time_offset_seconds_max_abs": int(np.abs(offsets).max()),
                     "lead_h": times["lead_h"].tolist(), "tau_h": times["tau_h"].tolist()},
            "field_status": field_status, "field_stats": stats,
            "status_counts_per_frame": [{"own_step": m["times"]["own_step"],
                                         "cloud_base": m["qc"]["cloud_base_status_counts"],
                                         "inversion": m["qc"]["inv_status_counts"]} for m in metas],
            "range_failures": range_fail,
            "negative_mixing_ratio_diagnosis": [
                {"own_step": m["times"]["own_step"], **m["qc"]["negative_mixing_ratio"]}
                for m in metas if m["qc"].get("negative_mixing_ratio")],
            "cloud_low_max_nan": int(sum(q.get("cloud_low_max_nan", 0) for q in qc_frames)),
            "calm_cells": int(sum(q.get("calm_cells", 0) for q in qc_frames)),
        })
        return entry, produced

    def _finalize_profiles(self, name, frames, metas, times):
        """profiles_<run>_<member>_dNN.nc: (time, point, level) native columns + (time, point) surface values."""
        import netCDF4

        mapping = self._mappings.get(name)
        ddir = self.stage / name
        if mapping is None:  # resumed process that staged nothing new for this domain
            rows = json.loads((ddir / "points.json").read_text())
            ok_rows = [r for r in rows if r["status"] == "OK"]
        else:
            ok_rows = [m.row() for m in mapping if m.status == "OK"]
            rows = [m.row() for m in mapping]
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        problems: list[str] = []
        partial: list[str] = []
        entry: dict[str, Any] = {"point_status_counts": counts, "points_in_file": len(ok_rows),
                                 "max_distance_m": self.max_distance_m[name]}
        mismatch = [r["id"] for r in rows if r["status"] == "GEOMETRY_MISMATCH"]
        if mismatch:
            partial.append(f"{len(mismatch)} grid-bound points do not match this run's geometry")
        if not ok_rows:
            entry["status"] = "NO_POINTS_IN_DOMAIN"
            return entry, None, problems, partial
        pstatus: dict[str, Any] = {}
        for f in PT.PROFILE_FIELDS:
            seen = {json.dumps(m.get("profile_status", {}).get(f), sort_keys=True) for m in metas}
            pstatus[f] = json.loads(seen.pop()) if len(seen) == 1 else {
                "status": "MIXED", "per_frame": [m.get("profile_status", {}).get(f) for m in metas]}
            if pstatus[f] is None or pstatus[f].get("status") not in ("OK", "NOT_AVAILABLE"):
                problems.append(f"profile field {f} status {pstatus[f]}")
        with np.load(frames[0]) as z:
            nz = z["prof__T_K"].shape[1]
        npt = len(ok_rows)
        target = self.dir / self.request.profiles_file_name(name)
        tmp = self.dir / f".{target.name}.tmp-{os.getpid()}"
        comp = dict(zlib=True, complevel=4, shuffle=True)
        pchunk = (1, min(64, npt), nz)
        point_vars = {
            "point_id": (str, [r["id"] for r in ok_rows]),
            "point_kind": (str, [r["kind"] for r in ok_rows]),
            "point_source": (str, [r["source"] for r in ok_rows]),
            "height_status": (str, [r["height_status"] for r in ok_rows]),
            "grid_y": (np.int32, np.array([r["grid_y"] for r in ok_rows], np.int32)),
            "grid_x": (np.int32, np.array([r["grid_x"] for r in ok_rows], np.int32)),
            "distance_m": (np.float64, np.array([r["distance_m"] for r in ok_rows], np.float64)),
            "point_lat": (np.float64, np.array([r["lat"] for r in ok_rows], np.float64)),
            "point_lon": (np.float64, np.array([r["lon"] for r in ok_rows], np.float64)),
            "cell_lat": (np.float32, np.array([r["cell_lat"] for r in ok_rows], np.float32)),
            "cell_lon": (np.float32, np.array([r["cell_lon"] for r in ok_rows], np.float32)),
            "model_HGT_m": (np.float32, np.array([r["model_HGT_m"] for r in ok_rows], np.float32)),
            "height_m": (np.float32, np.array([np.nan if r["height_m"] is None else r["height_m"]
                                               for r in ok_rows], np.float32)),
            "height_minus_model_HGT_m": (np.float32, np.array(
                [np.nan if r["height_minus_model_HGT_m"] is None else r["height_minus_model_HGT_m"]
                 for r in ok_rows], np.float32)),
        }
        stats: dict[str, Any] = {}
        try:
            with netCDF4.Dataset(tmp, "w", format="NETCDF4") as ds:
                ds.createDimension("time", len(frames))
                ds.createDimension("point", npt)
                ds.createDimension("level", nz)
                for key, value in (("schema", SCHEMA + ".profiles"), ("domain", name),
                                   ("run_id", self.request.run_id), ("member", self.request.member),
                                   ("issue_utc", self.request.issue_utc.isoformat()),
                                   ("wrf_start_utc", self.run_start.isoformat()),
                                   ("level_axis", "WRF mass level k, 0 = lowest; native, no interpolation"),
                                   ("point_mapping", "nearest mass cell (WT1 mapped_points), inside the "
                                                     "mass-point hull, distance <= max_distance_m"),
                                   ("max_distance_m", self.max_distance_m[name])):
                    ds.setncattr(key, value)
                for k, dt in TIME_VARS.items():
                    ds.createVariable(k, dt, ("time",))[:] = times[k]
                ds.createVariable("level", np.int16, ("level",))[:] = np.arange(nz, dtype=np.int16)
                for k, (dt, values) in point_vars.items():
                    v = ds.createVariable(k, dt, ("point",))
                    if dt is str:
                        for i, text in enumerate(values):
                            v[i] = text
                    else:
                        v[:] = values
                for f in PT.PROFILE_FIELDS:
                    v = ds.createVariable(f, "f4", ("time", "point", "level"), fill_value=np.float32(np.nan),
                                          chunksizes=pchunk, **comp)
                    v.units = PT.PROFILE_UNITS[f]
                    v.definition = PT.PROFILE_DEFINITIONS[f]
                    v.field_status = json.dumps(pstatus[f], sort_keys=True)
                    v.set_auto_maskandscale(False)
                for f in V.GRID_FIELDS:
                    is_status = f in V.STATUS_FIELDS
                    v = ds.createVariable(f, "i1" if is_status else "f4", ("time", "point"),
                                          fill_value=np.int8(-1) if is_status else np.float32(np.nan),
                                          chunksizes=(1, npt), **comp)
                    v.units = V.UNITS[f]
                    v.sampling = "value of the extract grid field at (grid_y, grid_x)"
                    v.set_auto_maskandscale(False)
                for i, p in enumerate(frames):
                    with np.load(p) as z:
                        for f in PT.PROFILE_FIELDS:
                            ds[f][i] = z["prof__" + f]
                        for f in V.GRID_FIELDS:
                            ds[f][i] = z["psfc__" + f]
            with netCDF4.Dataset(tmp) as ds:
                ds.set_auto_maskandscale(False)
                for k in TIME_VARS:
                    if ds[k][:].tobytes() != times[k].tobytes():
                        raise ValueError(f"profiles readback mismatch {k}")
                for k, (dt, values) in point_vars.items():
                    got = ds[k][:]
                    if dt is str:
                        if list(got) != list(values):
                            raise ValueError(f"profiles readback mismatch {k}")
                    elif np.asarray(got).tobytes() != np.asarray(values).tobytes():
                        raise ValueError(f"profiles readback mismatch {k}")
                for i, p in enumerate(frames):
                    with np.load(p) as z:
                        for key in [*("prof__" + f for f in PT.PROFILE_FIELDS), *("psfc__" + f for f in V.GRID_FIELDS)]:
                            f = key.split("__", 1)[1]
                            want = z[key]
                            got = np.asarray(ds[f][i])
                            if got.dtype != want.dtype or got.tobytes() != want.tobytes():
                                raise ValueError(f"profiles readback mismatch {f} frame {i}")
                            if key.startswith("prof__"):
                                st = stats.setdefault(f, {"min": None, "max": None, "nonfinite": 0})
                                fin = np.isfinite(want)
                                st["nonfinite"] += int((~fin).sum())
                                if fin.any():
                                    lo, hi = float(want[fin].min()), float(want[fin].max())
                                    st["min"] = lo if st["min"] is None else min(st["min"], lo)
                                    st["max"] = hi if st["max"] is None else max(st["max"], hi)
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, target)
            _fsync_dir(self.dir)
        finally:
            tmp.unlink(missing_ok=True)
        for f in PT.PROFILE_FIELDS:
            available = pstatus[f] is not None and pstatus[f].get("status") == "OK"
            if available and stats[f]["nonfinite"]:
                problems.append(f"non-finite profile values {f}: {stats[f]['nonfinite']}")
            if not available and stats[f]["nonfinite"] != len(frames) * npt * nz:
                problems.append(f"profile field {f} not available but holds values")
        entry.update({"status": "WRITTEN", "file": target.name, "levels": nz, "field_status": pstatus,
                      "field_stats": stats,
                      "negative_q_diagnosis": [m.get("profile_qc") for m in metas]})
        artifact = (target.name, {"bytes": target.stat().st_size, "sha256": _sha256_file(target),
                                  "domain": name, "kind": "profiles", "readback": "PASS"})
        return entry, artifact, problems, partial

    def _finalize_points(self, domains):
        """Run-level points.csv (every point x requested nest, with status) + manifest block."""
        rows = []
        for name in self.plans:
            mapping = self._mappings.get(name)
            if mapping is not None:
                rows.extend(m.row() for m in mapping)
            elif (self.stage / name / "points.json").is_file():
                rows.extend(json.loads((self.stage / name / "points.json").read_text()))
        header = list(dict.fromkeys(k for r in rows for k in r))
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in header})
        data = buf.getvalue().encode()
        path = self.dir / "points.csv"
        _atomic_bytes(path, data)
        if path.read_bytes() != data:
            raise ValueError("points.csv readback mismatch")
        no_coord = [p.id for p in self.points if p.lat is None]
        unknown_h = [p.id for p in self.points if p.height_status == "HEIGHT_UNKNOWN"]
        covered = {r["id"] for r in rows if r["status"] == "OK"}
        block = {
            "sources": [s.__dict__ for s in self.point_sources],
            "n_points": len(self.points),
            "max_distance_m": self.max_distance_m,
            "max_distance_rule": f"default {PT.DEFAULT_MAX_DISTANCE_CELLS} x dx per grid (WT1: 1.5 km on 1 km); "
                                 "request max_distance_m overrides per nest",
            "method": "nearest mass cell, unit-sphere chord on f32 XLAT/XLONG (WT1 mapped_points), "
                      "inside the mass-point hull (no extrapolation)",
            "status_values": list(PT.POINT_STATUS),
            "status_counts": {n: d.get("profiles", {}).get("point_status_counts") for n, d in domains.items()
                              if n in self.plans},
            "NO_COORDINATE": no_coord,
            "HEIGHT_UNKNOWN": unknown_h,
            "not_covered_by_any_requested_nest": [p.id for p in self.points
                                                  if p.lat is not None and p.id not in covered],
            "GEOMETRY_MISMATCH": sorted({r["id"] for r in rows if r["status"] == "GEOMETRY_MISMATCH"}),
            "points_csv": {"file": "points.csv", "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows)},
        }
        return block, ("points.csv", {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                      "kind": "points", "readback": "PASS"})

    def _manifest(self, overall, domains, artifacts, engine) -> dict:
        inputs = {}
        for pattern in ("namelist.input", "wrfbdy_d01", *(f"wrfinput_{n}" for n in self.all_domains)):
            p = self.input_dir / pattern
            if p.is_file():
                inputs[pattern] = _sha256_file(p)
        variables = {f: {"dtype": "int8" if f in V.STATUS_FIELDS else "float32", "dims": ["time", "y", "x"],
                         "units": V.UNITS[f]} for f in V.GRID_FIELDS}
        variables.update({k: {"dtype": "float32", "dims": ["y", "x"], "units": V.UNITS[k]} for k in V.STATIC})
        variables.update({k: {"dtype": np.dtype(t).name, "dims": ["time"]} for k, t in TIME_VARS.items()})
        return {
            "schema": SCHEMA, "status": overall,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "spec": {"document": SPEC_DOC, "sha256_prefix": SPEC_SHA256_PREFIX},
            "request": {"config_sha256": self.request.config_sha256, "config_path": str(self.request.config_path),
                        "run_id": self.request.run_id, "member": self.request.member, "arm": self.request.arm,
                        "seed": self.request.seed, "issue_utc": self.request.issue_utc.isoformat(),
                        "domains": list(self.plans)},
            "policy": self.request.policy(),
            **({"segment": {**{k: v for k, v in self.segment.items() if k != "dir"},
                            "name": self.segment["dir"].name,
                            "replayed_published_frames_dropped": list(self.replayed_published),
                            "rule": "extension of a completed run: this part holds only frames not published by its "
                                    "parents (base extract '.' + earlier segments, bound by manifest SHA256); the "
                                    "parents stay untouched"}} if self.segment else {}),
            "wrf_start_utc": self.run_start.isoformat(),
            "engine": dict(engine), "code_sha256": code_sha256(),
            "inputs_sha256": inputs,
            "geometry_sha256": {n: d.get("geometry_sha256") for n, d in domains.items() if "geometry_sha256" in d},
            "time_axes": {"actual_time_unix_s": "int64, actual history instant (run start + own_step*dt)",
                          "nominal_time_unix_s": "int64, history alarm time floor(lead/interval)*interval",
                          "time_offset_seconds": "int64, actual - nominal",
                          "lead_h": "float64, (nominal - issue)/3600", "tau_h": "float64, (nominal - wrf_start)/3600"},
            "encoding": {"format": "NETCDF4", "zlib": True, "complevel": 4, "shuffle": True,
                         "chunks": "(1, min(128,y), min(128,x))",
                         "fill_value": "NaN for float fields (meaning given by field_status / *_status), -1 for int8 status"},
            "variables": variables, "definitions": V.DEFINITIONS,
            "status_codes": {"cloud_base_status": V.CLOUD_BASE_STATUS, "inv_status": V.INV_STATUS,
                             **({"point_status": list(PT.POINT_STATUS),
                                 "height_status": ["OK", "HEIGHT_UNKNOWN", "NOT_A_STATION"]} if self.points else {})},
            **({"profile_variables": {
                **{f: {"dtype": "float32", "dims": ["time", "point", "level"], "units": PT.PROFILE_UNITS[f],
                       "definition": PT.PROFILE_DEFINITIONS[f]} for f in PT.PROFILE_FIELDS},
                **{f: {"dtype": "int8" if f in V.STATUS_FIELDS else "float32", "dims": ["time", "point"],
                       "units": V.UNITS[f]} for f in V.GRID_FIELDS}}} if self.points else {}),
            "deferred": {"points_profiles": "NOT_REQUESTED (milestone M2)",
                         "time_15m": "NOT_IMPLEMENTED (d03 15-min cadence; separate investigation)",
                         "zones": "NOT_IMPLEMENTED (consumer side, ZONE_CELLS_V1 unchanged)",
                         "QNCLOUD": "profiles only (M2, Addendum 1)"},
            "staging_conflicts": list(self.conflicts),
            "domains": domains, "artifacts": artifacts,
        }


def engine_identity() -> dict:
    out: dict[str, Any] = {}
    try:
        import gpuwrf
        out["gpuwrf_version"] = gpuwrf.__version__
    except Exception:  # noqa: BLE001
        out["gpuwrf_version"] = None
    try:
        from gpuwrf.runtime.aot_cheap_key import source_fingerprint_hash
        out["aot_source_fingerprint"] = source_fingerprint_hash()
    except Exception:  # noqa: BLE001
        out["aot_source_fingerprint"] = None
    try:
        root = Path(__file__).resolve().parents[3]
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30)
        out["git_head"] = head.stdout.strip() if head.returncode == 0 else None
        out["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except Exception:  # noqa: BLE001
        out["git_head"] = None
    return out
