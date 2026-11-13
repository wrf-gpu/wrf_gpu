"""Run N checkpoint branches of one parent generation, one after another (surface-control/state variants).

    python scripts/branch_series.py SERIES.json [--mode process|inprocess] [--dry-run]

SERIES.json (schema ``gpuwrf.branch_series.v1``)::

    {"schema": "gpuwrf.branch_series.v1",
     "input_dir": "CASE",                      # the parent's input directory (identity binds its path)
     "branch_from": "CKPT/generation-...",     # one verified parent generation
     "out_root": "runs/series",                # one new sub-directory per variant
     "end": {"hours": 18} | {"minutes": 1080} | {"end_time": "2026-02-28T12:00:00Z"},
     "parent_surface_controls": "neutral.npz", # the parent's --surface-controls file (null: parent ran without)
     "cli_args": ["--domains-from-namelist"],  # other `gpuwrf run` options the parent used
     "variants": [{"name": "null"},
                  {"name": "z0x2", "surface_controls": "z0x2.npz"},
                  {"name": "qv",   "manifest": "qv_manifest.json"}]}

Relative paths resolve against the directory of SERIES.json. A variant either names its own complete branch
manifest (``manifest``) or gets one written here: ``{"schema_version": 1}`` plus, for a variant with its own
``surface_controls``, ``parameters.surface_controls`` = the control-file digest of every domain whose digest
differs from the parent's recorded one. The CLI then authenticates the requested run against exactly that
declaration (``checkpoint_branch.branch_identity``); a domain whose consumer digest also covers CTOPO
(topo_wind) does not match the file digest and is refused there, never relaxed here.

Every branch keeps all CLI refusals (identity, schema, B=1, endpoint after the checkpoint, empty new output).
Modes:

* ``process`` (default): one ``python -m gpuwrf.cli run`` child per variant, sequential. Each child pays
  import, case load and executable load; warm executables come from the persistent caches.
* ``inprocess``: all variants in this one process through ``gpuwrf.cli.main``. Imports and already-loaded
  executables are reused (no retrace/reload when the step programs match). On CUDA with more than one variant
  the runner requires the verified ``scripts/with_gpu_lock.sh`` proof (lock fd + holder token,
  ``gpu_preflight.gpu_lock_status``): under a verified lock the nested preflight treats this process's own
  allocator pool as advisory, so no override is needed. The runner never forces the preflight
  (``--force-gpu-run`` / ``GPUWRF_FORCE_GPU_RUN`` are refused). The first failure stops the series (fail
  closed); later variants are reported NOT_RUN and can be re-run in process mode.

``cli_args`` are parsed by the real ``gpuwrf run`` parser with prefix abbreviations disabled. Only
domain-selection and AOT-prefetch options and ``--runtime-parameters`` / ``--emit-initial-history`` (which the
parent identity binds) may appear; every other option (output, scratch, checkpoint, branch, resume, manifest,
controls, end time, dry run, feedback, force, proof/compare/score) must stay at its default. Two-way feedback
parents/branches are refused (one-way nesting only).

Receipt: OUT_ROOT/series_receipt.json (per variant rc, wall, the CLI's own run payload summary, and JAX
compile/cache counters: inprocess from listeners in this process, process mode from each child's
scripts/jax_acct_run.py receipt OUT_ROOT/<variant>/acct.json, incl. the platform of the measured child). Wrap GPU use in ``scripts/with_gpu_lock.sh``; the lock fd is passed
to children.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = "gpuwrf.branch_series.v1"
REPO = Path(__file__).resolve().parents[1]
ACCT = Path(__file__).resolve().parent / "jax_acct_run.py"
_CLI_ARGS_ALLOWED = {"domain", "max_dom", "domains_from_namelist", "emit_initial_history", "aot_prefetch",
                     "runtime_parameters"}


def check_cli_args(cli_args: list[str]) -> None:
    """Parse ``cli_args`` with the real ``gpuwrf run`` parser (no abbreviations); refuse anything not allowlisted."""
    import argparse as _argparse

    from gpuwrf.cli import build_parser

    sub = next(a for a in build_parser()._actions if isinstance(a, _argparse._SubParsersAction))
    run = sub.choices["run"]
    run.allow_abbrev = False
    run.exit_on_error = False
    try:
        placeholder = ["--input-dir", "\0input", "--output-dir", "\0output"]  # required by the parser; set per variant
        defaults, _ = run.parse_known_args(placeholder)
        parsed, extra = run.parse_known_args(placeholder + list(cli_args))
    except (_argparse.ArgumentError, SystemExit) as exc:
        raise ValueError(f"cli_args rejected by the gpuwrf run parser: {exc}") from None
    if extra:
        raise ValueError(f"cli_args has unknown or abbreviated options {extra} (full option names only)")
    changed = sorted(k for k, v in vars(parsed).items() if v != getattr(defaults, k) and k not in _CLI_ARGS_ALLOWED)
    if changed:
        raise ValueError(f"cli_args may not set {changed}: the runner sets output/scratch/branch/manifest/controls/end "
                         "per variant; checkpoint, resume, dry-run, feedback, force and proof options are refused")


def _resolve(base: Path, value):
    return None if value is None else (base / value).resolve()


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    base = Path(path).resolve().parent
    if spec.get("schema") != SCHEMA:
        raise ValueError(f"series schema must be {SCHEMA}")
    unknown = set(spec) - {"schema", "input_dir", "branch_from", "out_root", "end", "parent_surface_controls",
                           "cli_args", "variants"}
    if unknown:
        raise ValueError(f"unknown series keys {sorted(unknown)}")
    end = spec.get("end")
    if not isinstance(end, dict) or len(end) != 1 or next(iter(end)) not in ("hours", "minutes", "end_time"):
        raise ValueError("end must be exactly one of {hours|minutes|end_time}")
    cli_args = list(spec.get("cli_args", []))
    if not all(isinstance(t, str) for t in cli_args):
        raise ValueError("cli_args must be strings")
    check_cli_args(cli_args)
    variants = spec.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants must be a non-empty list")
    names = [v.get("name") for v in variants]
    if any(not isinstance(n, str) or not n or n != Path(n).name or n in (".", "..") for n in names) or len(set(names)) != len(names):
        raise ValueError("variant names must be unique plain names")
    for v in variants:
        if set(v) - {"name", "surface_controls", "manifest"}:
            raise ValueError(f"{v['name']}: unknown variant keys")
        if "manifest" in v and "surface_controls" in v:
            raise ValueError(f"{v['name']}: give either a complete manifest (then add surface_controls to the series "
                             "parent default only) or surface_controls, not both")
    out = {"input_dir": _resolve(base, spec["input_dir"]), "branch_from": _resolve(base, spec["branch_from"]),
           "out_root": _resolve(base, spec["out_root"]), "end": end, "cli_args": cli_args,
           "parent_surface_controls": _resolve(base, spec.get("parent_surface_controls")),
           "variants": [{"name": v["name"], "surface_controls": _resolve(base, v.get("surface_controls")),
                         "manifest": _resolve(base, v.get("manifest"))} for v in variants]}
    if not (out["branch_from"] / "manifest.json").is_file():
        raise ValueError(f"{out['branch_from']}: not a checkpoint generation (no manifest.json)")
    for v in out["variants"]:
        for key in ("surface_controls", "manifest"):
            if v[key] is not None and not v[key].is_file():
                raise ValueError(f"{v['name']}: {key} {v[key]} does not exist")
    if out["parent_surface_controls"] is not None and not out["parent_surface_controls"].is_file():
        raise ValueError(f"parent_surface_controls {out['parent_surface_controls']} does not exist")
    if any(v["surface_controls"] is not None for v in out["variants"]) and out["parent_surface_controls"] is None:
        raise ValueError("variant surface controls need a parent that ran with surface controls "
                         "(parent_surface_controls); the family must already be enabled in the parent")
    return out


def _signature(fields: dict) -> dict:
    return {domain: {name: (list(array.shape), str(array.dtype)) for name, array in sorted(arrays.items())}
            for domain, arrays in sorted(fields.items())}


def declared_manifest(parent_identity: dict, controls: Path | None, parent_controls: Path | None = None) -> dict:
    """Empty manifest, plus the digests of every domain whose control values differ from the parent's.

    Pre-run guards (host only, no model import): the parent control file must reproduce the checkpoint's recorded
    per-domain digests, and a variant file must carry exactly the parent's control signature (domains, names,
    shapes, dtypes). A different signature is another program: the CLI would refuse it as an undeclared
    ``namelists`` change only after case initialization.
    """
    manifest: dict = {"schema_version": 1}
    if controls is None:
        return manifest
    from gpuwrf.surface_params.controls import controls_digest, load_controls
    fields, _ = load_controls(controls)
    recorded = parent_identity.get("surface_controls")
    if recorded is None:
        raise ValueError("the parent checkpoint records no surface controls; the family is not enabled there")
    unknown = sorted(set(fields) - set(recorded))
    if unknown:
        raise ValueError(f"control file names domains outside the parent run: {unknown}")
    if parent_controls is not None:
        parent_fields, _ = load_controls(parent_controls)
        stale = sorted(d for d in recorded if recorded[d] is not None and
                       (d not in parent_fields or controls_digest(parent_fields[d]) != recorded[d]))
        stale += sorted(d for d in parent_fields if recorded.get(d) is None)
        if stale:
            raise ValueError(f"parent_surface_controls does not reproduce the checkpoint's control digests on {stale}")
        if _signature(fields) != _signature(parent_fields):
            raise ValueError(f"{Path(controls).name}: control signature (domains/names/shapes/dtypes) differs from the "
                             f"parent's {_signature(parent_fields)}; only values may change in a branch series")
    changed = {name: controls_digest(arrays) for name, arrays in sorted(fields.items())
               if controls_digest(arrays) != recorded[name]}
    if changed:
        manifest["parameters"] = {"surface_controls": changed}
    return manifest


def variant_argv(spec: dict, variant: dict, manifest: Path) -> list[str]:
    key, value = next(iter(spec["end"].items()))
    end = {"hours": "--hours", "minutes": "--minutes", "end_time": "--end-time"}[key]
    # Scratch beside, not inside, the branch output: the CLI creates its scratch before the driver re-checks that
    # the branch output is empty (released v0.4.4-v0.4.6 refuse the default <output-dir>/.scratch there).
    out = spec["out_root"] / variant["name"]
    argv = ["run", "--input-dir", str(spec["input_dir"]), "--output-dir", str(out / "wrfout"),
            "--scratch-dir", str(out / "scratch"),
            "--branch-from", str(spec["branch_from"]), "--branch-manifest", str(manifest), end, str(value)]
    controls = variant["surface_controls"] or spec["parent_surface_controls"]
    if controls is not None:
        argv += ["--surface-controls", str(controls)]
    return argv + spec["cli_args"]


def _payload_summary(out: Path) -> dict:
    path = out / "wrfout" / "proofs" / "nested_pipeline_run.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"payload": None}
    keep = ("verdict", "device", "hours", "root_steps", "wall_clock_total_s", "wall_clock_forecast_only_s",
            "wall_s", "forecast_wall_s", "init_wall_s", "timings", "restart")
    return {"payload": str(path), "run": {k: payload[k] for k in keep if k in payload}}


def _lock_fds() -> tuple[int, ...]:
    fd = os.environ.get("GPUWRF_GPU_LOCK_FD", "")
    return (int(fd),) if fd.isdigit() and os.path.exists(f"/proc/self/fd/{fd}") else ()


class _Counters:
    KEYS = ("cache_requests", "cache_hits", "cache_misses", "cache_disabled", "backend_compile_events", "fresh_compiles")
    EVENTS = {"/jax/compilation_cache/compile_requests_use_cache": "cache_requests",
              "/jax/compilation_cache/cache_hits": "cache_hits",
              "/jax/compilation_cache/task_disabled_cache": "cache_disabled"}
    DURATIONS = {"/jax/core/compile/jaxpr_trace_duration": "traces",
                 "/jax/core/compile/jaxpr_to_mlir_module_duration": "lowerings",
                 "/jax/core/compile/backend_compile_duration": "backend_compile_events"}

    def __init__(self):
        self.counts: dict[str, int] = {}

    def event(self, name, **_):
        if name in self.EVENTS:
            self.counts[self.EVENTS[name]] = self.counts.get(self.EVENTS[name], 0) + 1

    def duration(self, name, _value, **_):
        if name in self.DURATIONS:
            self.counts[self.DURATIONS[name]] = self.counts.get(self.DURATIONS[name], 0) + 1

    def take(self) -> dict[str, int]:
        counts, self.counts = dict(self.counts), {}
        for key in ("cache_requests", "cache_hits", "cache_disabled", "backend_compile_events"):
            counts.setdefault(key, 0)
        counts["cache_misses"] = counts["cache_requests"] - counts["cache_hits"]
        # backend_compile_duration fires for every compile-or-load, hit or not: anything beyond the hits (incl.
        # cache-disabled compiles that never issue a cache request) is a fresh backend compile.
        counts["fresh_compiles"] = counts["backend_compile_events"] - counts["cache_hits"]
        return counts


def main(argv: list[str] | None = None) -> int:
    t_start = time.perf_counter()
    ap = argparse.ArgumentParser(prog="branch_series.py", description=__doc__.split("\n")[0])
    ap.add_argument("series", type=Path)
    ap.add_argument("--mode", choices=("process", "inprocess"), default="process")
    ap.add_argument("--dry-run", action="store_true", help="validate, write the manifests' plan as JSON, run nothing")
    a = ap.parse_args(argv)
    spec = load_spec(a.series)
    parent = json.loads((spec["branch_from"] / "manifest.json").read_text())
    if os.environ.get("GPUWRF_FORCE_GPU_RUN", "").strip().lower() not in ("", "0", "false", "no", "off"):
        print("branch_series: GPUWRF_FORCE_GPU_RUN is refused; the runner never forces the GPU preflight", file=sys.stderr)
        return 2
    if parent.get("identity", {}).get("feedback") or parent.get("identity", {}).get("feedback_schema") is not None:
        print("branch_series: the parent ran with two-way feedback; checkpoint branching supports one-way nesting only",
              file=sys.stderr)
        return 2
    reexec = os.environ.get("_GPUWRF_NESTED_ALLOC_REEXEC") == "1"  # the CLI re-executed this runner once (CUDA)
    if not a.dry_run and not reexec:
        busy = [str(spec["out_root"] / v["name"]) for v in spec["variants"] if (spec["out_root"] / v["name"]).exists()]
        if busy:
            print(f"branch_series: variant directories exist (never overwritten): {busy}", file=sys.stderr)
            return 2
    plan = []
    for v in spec["variants"]:
        manifest = v["manifest"]
        written = None
        if manifest is None:
            # A null variant re-checks the parent file itself (it must reproduce the recorded digests).
            written = declared_manifest(parent["identity"], v["surface_controls"] or spec["parent_surface_controls"],
                                        spec["parent_surface_controls"])
            manifest = spec["out_root"] / v["name"] / "branch_manifest.json"
        plan.append({"variant": v, "manifest": manifest, "written": written,
                     "argv": variant_argv(spec, v, manifest)})
    if a.dry_run:
        print(json.dumps([{"name": p["variant"]["name"], "manifest": str(p["manifest"]), "declared": p["written"],
                           "argv": p["argv"]} for p in plan], indent=1, default=str))
        return 0
    for p in plan:
        out = spec["out_root"] / p["variant"]["name"]
        out.mkdir(parents=True, exist_ok=True)
        if p["written"] is not None:
            p["manifest"].write_text(json.dumps(p["written"], indent=1, sort_keys=True) + "\n")
    receipt = {"schema": "gpuwrf.branch_series.receipt.v1", "mode": a.mode, "series": str(a.series.resolve()),
               "branch_from": str(spec["branch_from"]),
               "parent_manifest_sha256": hashlib.sha256((spec["branch_from"] / "manifest.json").read_bytes()).hexdigest(),
               "variants": []}
    counters = None
    if a.mode == "inprocess":
        from jax import monitoring
        import gpuwrf.cli as cli
        from gpuwrf.backends.platform import target_backend
        cuda = target_backend() == "cuda"
        if cuda and len(plan) > 1:
            from gpuwrf.runtime.gpu_preflight import gpu_lock_status
            lock = gpu_lock_status()
            if not lock["ok"]:
                print("branch_series: inprocess mode on CUDA needs the verified GPU lock (scripts/with_gpu_lock.sh: "
                      f"lock fd + holder token): {lock['reason']}", file=sys.stderr)
                return 2
            receipt["gpu_lock"] = {k: lock[k] for k in ("ok", "fd", "lock_file")}
        counters = _Counters()
        monitoring.register_event_listener(counters.event)
        monitoring.register_event_duration_secs_listener(counters.duration)
        receipt["import_s"] = time.perf_counter() - t_start
    stopped = None
    for index, p in enumerate(plan):
        name, out = p["variant"]["name"], spec["out_root"] / p["variant"]["name"]
        row = {"name": name, "manifest": str(p["manifest"]),
               "manifest_sha256": hashlib.sha256(Path(p["manifest"]).read_bytes()).hexdigest(), "argv": p["argv"]}
        if stopped is not None:
            receipt["variants"].append({**row, "status": "NOT_RUN", "reason": stopped})
            continue
        t0 = time.perf_counter()
        if a.mode == "process":
            # The child runs under scripts/jax_acct_run.py: same CLI entry, plus its own compile/cache counters
            # (cache misses, cache-disabled and fresh compiles) and the platform read in the measured process.
            acct = out / "acct.json"
            with open(out / "run.log", "ab") as log:
                proc = subprocess.run([sys.executable, "-B", str(ACCT), "--receipt", str(acct), "--", *p["argv"]],
                                      stdout=log, stderr=subprocess.STDOUT, pass_fds=_lock_fds(), cwd=str(REPO), check=False)
            rc = proc.returncode
            try:
                child = json.loads(acct.read_text())
                runs = child.get("runs") or []
                row["platform"] = child.get("platform")
                row["jax_counters"] = {k: sum(r.get(k, 0) for r in runs) for k in _Counters.KEYS} if runs else None
            except (OSError, ValueError):
                row["platform"], row["jax_counters"] = None, None
        else:
            counters.take()
            rc = int(cli.main(list(p["argv"])))
            row["jax_counters"] = counters.take()
        row.update(rc=rc, wall_s=time.perf_counter() - t0, status="OK" if rc == 0 else "FAILED", **_payload_summary(out))
        receipt["variants"].append(row)
        print(f"branch_series: {name} rc={rc} wall={row['wall_s']:.1f}s", file=sys.stderr, flush=True)
        if rc != 0 and a.mode == "inprocess":
            stopped = f"stopped after {name} failed (inprocess mode fails closed)"
    receipt["total_s"] = time.perf_counter() - t_start
    (spec["out_root"] / "series_receipt.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    return 0 if all(v.get("rc") == 0 for v in receipt["variants"]) else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)  # JAX worker threads must not keep a finished series alive (E146 class)
