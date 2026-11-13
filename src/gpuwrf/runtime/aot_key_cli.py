"""``gpuwrf aot-key``: per-program AOT cheap keys of a nested case, CPU only, no compile.

Loads the case like ``gpuwrf run`` (nested_pipeline._load_domains) in a CPU process and
prints the cheap key every step program would look up (aot_key_probe). With
``--compare-input-dir`` it loads a second case and reports, per program, whether both
cases share the program and which key component differs. A cache primed on one case
serves another case exactly when their programs match here.

The command never writes a persistent cache: what the case load compiles or serializes
(JAX persistent-cache entries, init AOT blobs, the init libm loop, Triton/CUDA caches) goes
to a private temporary directory that is deleted on exit, whatever the environment says
(incident 2026-10-18: a CPU aot-key after ``source <bundle>/bundle.env`` wrote 1564 CPU
entries into the sealed v0.4.1 bundle).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_COMPONENTS = ("program_config", "fn_identity", "source_fingerprint", "static_config", "carry_aval",
               "global_trace_env", "module_const_env", "baked_tables")

# Cache locations redirected into the private directory (subdirectory per variable). Only
# key-inert variables: the cheap key ignores the two JAX cache dirs (aot_cheap_key
# HLO_AFFECTING_ENV_DENYLIST / not GPUWRF_*) and the Triton/CUDA caches. JAX_COMPILATION_CACHE_DIR
# outranks GPUWRF_CACHE and the per-user default (compile_cache.resolve_cache_dir), so the keyed
# GPUWRF_CACHE and GPUWRF_XLA_AUTOTUNE_CACHE_DIR (E32) stay untouched; a CPU process writes no
# autotune results.
_PRIVATE_CACHE_ENV = {"JAX_COMPILATION_CACHE_DIR": "jax", "GPUWRF_JAX_CACHE_DIR": "jax",
                      "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda"}
_CACHE_ENV = (*_PRIVATE_CACHE_ENV, "GPUWRF_CACHE", "GPUWRF_XLA_AUTOTUNE_CACHE_DIR")


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "aot-key",
        help="print the AOT cheap keys a nested case would use (CPU only, no GPU, no compile)",
        description=(
            "Load a nested case on the CPU and print, per step program (root advance and "
            "fused cascades), the AOT cheap key it would look up. With --compare-input-dir, "
            "report which programs two cases share: a cache primed on one case serves the "
            "other exactly for the shared programs. Keys from this CPU process carry the CPU "
            "target, so compare cases with each other, not with a GPU cache listing. No persistent "
            "cache is written: what the case load compiles goes to a private temporary directory "
            "(also when JAX_COMPILATION_CACHE_DIR or GPUWRF_JAX_CACHE_DIR point elsewhere)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="case directory (as for gpuwrf run)")
    parser.add_argument("--namelist", type=Path, help="namelist.input to use instead of the one in --input-dir")
    parser.add_argument("--compare-input-dir", type=Path, help="second case directory to compare with")
    parser.add_argument("--compare-namelist", type=Path, help="namelist for --compare-input-dir")
    parser.add_argument("--surface-controls", type=Path, help="control file bound as in `gpuwrf run --surface-controls`")
    parser.add_argument("--compare-surface-controls", type=Path,
                        help="control file for the compare case (default: the compare case runs without controls); "
                        "with --compare-surface-controls alone the compare case is --input-dir itself, so two value "
                        "sets of one case are compared (checkpoint-branch series: same keys = no recompile)")
    parser.add_argument("--hours", type=float, default=3.0, help="forecast length the keys are computed for")
    parser.add_argument("--json", type=Path, help="also write the full report (incl. components) as JSON")
    parser.set_defaults(func=command)


def _case_dir(input_dir: Path, namelist: Path | None, scratch: Path) -> Path:
    """``input_dir`` itself, or a symlink view of it with ``namelist`` as namelist.input."""
    if namelist is None:
        return input_dir
    view = Path(tempfile.mkdtemp(prefix="case_", dir=scratch))
    for item in input_dir.iterdir():
        if item.name != "namelist.input":
            (view / item.name).symlink_to(item.resolve())
    (view / "namelist.input").symlink_to(namelist.resolve())
    return view


def case_keys(input_dir: Path, *, hours: float, scratch: Path, surface_controls: Path | None = None) -> dict[str, Any]:
    """Load one case and return its per-program key records (with components)."""
    from gpuwrf.cli import _namelist_max_dom
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime.aot_key_probe import case_program_keys
    from gpuwrf.runtime.domain_tree import DomainTree

    max_dom = _namelist_max_dom(input_dir / "namelist.input")
    if max_dom < 2:
        raise ValueError("aot-key covers nested runs (max_dom >= 2); single-domain runs use another path")
    names = pipeline.domain_names_for(max_dom)
    config = pipeline.NestedPipelineConfig(input_dir, scratch / "out", scratch / "proof", hours=hours,
                                           max_dom=max_dom, scratch_dir=scratch, surface_controls=surface_controls)
    hierarchy, bundles, meta, _start, _dt, carries = pipeline._load_domains(config, names)
    nesting = meta.get("nesting", {"feedback": bool(config.feedback), "spec_zone": 1, "smooth_option": 1})
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=nesting["feedback"],
                                   feedback_spec_zone=nesting["spec_zone"],
                                   feedback_smooth_option=nesting["smooth_option"])
    return case_program_keys(tree, carries, components=True)


def _compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            rows[name] = {"shared": False, "reason": "program only in one case"}
            continue
        if a[name].get("cheap_key") is None or b[name].get("cheap_key") is None:
            # A failed key computation proves nothing about the executables.
            rows[name] = {"shared": False, "unavailable": True, "reason": "no cheap key in "
                          + " and ".join(label for label, rec in (("case", a[name]), ("compare", b[name]))
                                         if rec.get("cheap_key") is None)}
            continue
        ca, cb = a[name].get("components", {}), b[name].get("components", {})
        rows[name] = {"shared": a[name]["cheap_key"] == b[name]["cheap_key"],
                      "differing_components": [c for c in _COMPONENTS if ca.get(c) != cb.get(c)]}
    return rows


def sealed_bundle_of(path: str | Path) -> Path | None:
    """Root of the sealed compile bundle (``<root>/sealed/BUNDLE_MANIFEST.json``) that holds ``path``."""
    try:
        path = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    for root in (path, *path.parents):
        if (root / "sealed" / "BUNDLE_MANIFEST.json").is_file():
            return root
    return None


def sealed_bundle_env() -> dict[str, str]:
    """Cache settings of this process that point into a sealed bundle: {setting: bundle root}."""
    import jax

    settings = {name: os.environ.get(name) for name in _CACHE_ENV}
    settings["jax_compilation_cache_dir"] = jax.config.jax_compilation_cache_dir
    found = {}
    for name, value in settings.items():
        bundle = sealed_bundle_of(value) if value else None
        if bundle is not None:
            found[name] = str(bundle)
    return found


@contextmanager
def removed_on_signal(path: Path) -> Iterator[None]:
    """Inside the block SIGTERM/SIGINT first delete ``path`` (its exact recorded name), then end the
    process with that signal (default disposition, exit status -N / 128+N); handlers restored on exit.

    A default SIGTERM kills the process without unwinding, so ``TemporaryDirectory`` cannot clean
    up the private cache (cr-w42-aotkey lifecycle probe). Ignored signals stay ignored; only the
    main thread can install handlers.
    """
    def handler(signum, _frame):
        for _ in range(2):  # a second pass catches a file another thread created meanwhile
            shutil.rmtree(path, ignore_errors=True)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            if signal.getsignal(signum) is not signal.SIG_IGN:
                previous[signum] = signal.signal(signum, handler)
    try:
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


@contextmanager
def private_compile_caches(root: Path) -> Iterator[Path]:
    """Inside the block every compile cache of this process lives under ``root`` (the caller deletes it).

    Redirects the key-inert cache variables (:data:`_PRIVATE_CACHE_ENV`), the JAX persistent
    cache directory (set from JAX_COMPILATION_CACHE_DIR at import even when GPUWRF_JAX_CACHE=0)
    and the status the AOT stores report; everything is restored on exit.
    """
    import jax
    from jax._src import compilation_cache as jax_cache

    from gpuwrf.runtime import compile_cache

    saved_env = {name: os.environ.get(name) for name in _PRIVATE_CACHE_ENV}
    saved_dir = jax.config.jax_compilation_cache_dir
    saved_status_dir = compile_cache.CACHE_STATUS.get("dir")
    jax_dir = root / "jax"
    for name, sub in _PRIVATE_CACHE_ENV.items():
        os.environ[name] = str(root / sub)
    jax.config.update("jax_compilation_cache_dir", str(jax_dir))
    jax_cache.reset_cache()  # the file cache is built once per process: re-open it at jax_dir
    if compile_cache.CACHE_STATUS.get("enabled"):
        compile_cache.CACHE_STATUS["dir"] = str(jax_dir)
    try:
        yield jax_dir
    finally:
        jax_cache.reset_cache()
        jax.config.update("jax_compilation_cache_dir", saved_dir)
        compile_cache.CACHE_STATUS["dir"] = saved_status_dir
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def command(args: argparse.Namespace) -> int:
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax

    try:  # the parser may already have imported jax; its backend is chosen on first use
        jax.config.update("jax_platforms", "cpu")
    except Exception:  # noqa: BLE001 - checked below
        pass
    if jax.default_backend() != "cpu":
        print("gpuwrf aot-key: error: JAX already runs on a non-CPU backend in this process", file=sys.stderr)
        return 2
    report: dict[str, Any] = {"schema": "gpuwrf.aot_key.v1", "platform": "cpu", "hours": args.hours,
                              "env_nested_aot": os.environ.get("GPUWRF_NESTED_AOT")}
    if os.environ.get("GPUWRF_NESTED_AOT", "1").strip().lower() in ("0", "false", "off", "no"):
        print("gpuwrf aot-key: note: GPUWRF_NESTED_AOT is off here, so a run in this environment loads no "
              "AOT executables; the keys below are those an AOT run would use", file=sys.stderr)
    report["sealed_bundle_env"] = sealed_bundle_env()
    for name, bundle in report["sealed_bundle_env"].items():
        print(f"gpuwrf aot-key: WARNING: {name} points into the sealed compile bundle {bundle}. aot-key does not "
              f"write it (private temporary cache); `gpuwrf run` in this environment writes it unless it runs inside "
              f"{bundle}/tools/bundle_consumer.sh --layer <dir outside the bundle>", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="gpuwrf_aot_key_") as tmp, removed_on_signal(Path(tmp)):
        scratch = Path(tmp)
        controls = getattr(args, "surface_controls", None)
        compare_controls = getattr(args, "compare_surface_controls", None)
        cases = [("case", args.input_dir, args.namelist, controls)]
        if args.compare_input_dir is not None or compare_controls is not None:
            cases.append(("compare", args.compare_input_dir or args.input_dir,
                          args.compare_namelist if args.compare_input_dir is not None else args.namelist,
                          compare_controls))
        with private_compile_caches(scratch / "cache"):
            for label, input_dir, namelist, surface_controls in cases:
                case_scratch = scratch / label
                case_scratch.mkdir()
                directory = _case_dir(input_dir, namelist, case_scratch)
                report[label] = {"input_dir": str(input_dir),
                                 "namelist": str(namelist or input_dir / "namelist.input"),
                                 "surface_controls": None if surface_controls is None else str(surface_controls),
                                 "programs": case_keys(directory, hours=args.hours, scratch=case_scratch,
                                                       **({} if surface_controls is None else
                                                          {"surface_controls": surface_controls}))}
    for label in ("case", "compare"):
        if label in report:
            print(f"{label}: {report[label]['input_dir']}")
            for name, record in report[label]["programs"].items():
                if record.get("cheap_key") is None:
                    print(f"  {name:12s} UNAVAILABLE ({record.get('unavailable', 'no cheap key')})")
                else:
                    print(f"  {name:12s} cheap_key={record['cheap_key']}")
    complete = all(record.get("cheap_key") is not None
                   for label in ("case", "compare") if label in report
                   for record in report[label]["programs"].values())
    if "compare" in report:
        report["comparison"] = _compare(report["case"]["programs"], report["compare"]["programs"])
        for name, row in report["comparison"].items():
            verdict = "UNAVAILABLE" if row.get("unavailable") else ("SHARED" if row["shared"] else "NOT SHARED")
            detail = row.get("reason") or ("" if row["shared"] else "differs in " + ", ".join(row["differing_components"]))
            print(f"  {name:12s} {verdict} {detail}".rstrip())
    report["complete"] = complete
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1, default=str) + "\n")
    if not complete:
        print("gpuwrf aot-key: error: some programs have no cheap key (see UNAVAILABLE)", file=sys.stderr)
        return 3
    return 0
