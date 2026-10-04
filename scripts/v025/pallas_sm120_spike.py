#!/usr/bin/env python3
"""Pallas-on-sm_120 viability spike: the real ``advance_w`` Thomas solve (§11).

Operator selection: contract §11 names "the production-shaped vertical
implicit/Thomas solve used by ``advance_w``" as the required real operator, and
allows the MYNN ``_condensation_edmf`` substitute **only** if the Thomas
interface cannot be isolated without production changes. It can:
``gpuwrf.dynamics.tridiag_solve.thomas_solve_scan(a, alpha, gamma, rhs)`` is a
pure, importable function with no production edit required. No substitution is
claimed and none is needed.

This module is the *whole* spike, but it runs in two separated modes:

``--interpret``  CPU / Pallas interpreter. Phase A. Establishes correctness
                 against the production XLA reference. Explicitly **not**
                 viability evidence: §11 says interpreter-only and
                 deprecated-backend results are diagnostic, never a verdict.
``--native``     Native Pallas Mosaic GPU on sm_120. Phase B only.  It first
                 consumes the same single-use Amendment-4 child handoff, then
                 loads the immutable real FAST advance-w savepoint. This is the
                 only mode that can emit a real verdict.

The verdict table is frozen in §11 and implemented verbatim in ``classify``.
Running ``--interpret`` can never return ``PALLAS_GREEN``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

# Production d01 geometry (FAST-v025): 120x70 mass columns, 45 w faces.
PROD_NZ = 45
PROD_NY = 70
PROD_NX = 120
PROD_COLUMNS = PROD_NY * PROD_NX
SAVEPOINT_SCHEMA = "wrf_gpu2.v025.m0.pallas_fast_savepoint.v1"
DEFAULT_SAVEPOINT_MANIFEST = REPO / "proofs/v025/m0/pallas_fast_savepoint_manifest.json"
RANGE_NAME = "GPUWRF_M0_PALLAS_TIMED_LOOP"


def production_reference():
    """The production XLA implementation, imported read-only (contract §3)."""
    from gpuwrf.dynamics.tridiag_solve import thomas_solve_scan

    return thomas_solve_scan


def make_state(seed: int = 20260727, nz: int = PROD_NZ, ny: int = PROD_NY, nx: int = PROD_NX):
    """Diagonally-dominant tridiagonal coefficients at production shape.

    NOTE ON PROVENANCE: these are *shape/conditioning* fixtures, not physics
    state. §11's real-savepoint requirement is satisfied by
    ``--savepoint``, which loads the FAST-v025 wrfout emitted by the CPU arm.
    This synthetic path exists only so the kernel can be developed and its
    correctness pinned before a savepoint extractor exists; any run that uses it
    records ``state_source_is_real: false`` and therefore cannot be a verdict.
    """
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    a = rng.uniform(-0.4, -0.1, shape)
    gamma = rng.uniform(-0.4, -0.1, shape)
    # alpha is the reciprocal of the pivot; keep the system well conditioned so a
    # correctness miss is a kernel bug, not an ill-posed problem.
    diag = 1.0 + np.abs(a) + np.abs(gamma)
    alpha = 1.0 / diag
    rhs = rng.uniform(-1.0, 1.0, shape)
    return (
        np.asarray(a, np.float64),
        np.asarray(alpha, np.float64),
        np.asarray(gamma, np.float64),
        np.asarray(rhs, np.float64),
    )


def thomas_numpy(a, alpha, gamma, rhs):
    """Independent NumPy oracle of WRF's own sweep (module_small_step_em.F).

    Deliberately a third implementation: comparing Pallas only against the JAX
    reference would make a shared misreading of WRF invisible.

    WRF-FAITHFUL BOUNDARY HANDLING -- this is not a generic Thomas solve. In
    ``advance_w`` the back-substitution covers only the **interior** faces
    ``1..nz-2``; the bottom face (terrain-following lower boundary) and the top
    face (rigid lid / Rayleigh damping) are boundary conditions and keep their
    forward-sweep values. A textbook back-substitution over all levels
    overwrites both and disagrees with the production reference by O(1) --
    which is exactly what the first version of this oracle did, and what the
    three-way comparison caught.
    """
    nz = rhs.shape[0]
    w = np.empty_like(rhs)
    w[0] = rhs[0]
    for k in range(1, nz):
        w[k] = (rhs[k] - a[k] * w[k - 1]) * alpha[k]
    out = np.empty_like(w)
    if nz <= 2:
        return w, w.copy()
    out[0] = w[0]
    out[nz - 1] = w[nz - 1]
    for k in range(nz - 2, 0, -1):
        out[k] = w[k] - gamma[k] * out[k + 1]
    return w, out


def build_pallas_kernel(nz: int, *, interpret: bool):
    """One program per column; the vertical sweep stays inside the kernel.

    This is the shape of the win the master plan is after: the whole dependent
    vertical chain resolves in one launch with the column resident, instead of
    ~2*nz dependent XLA kernels each paying the ~2-4 us device-side barrier the
    kernel report measured.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    def kernel(a_ref, alpha_ref, gamma_ref, rhs_ref, out_ref, fwd_ref):
        # Each program sees ONE column shaped (1, nz): the leading axis is the
        # block's column index, so every access is [0, k]. Indexing [k] instead
        # silently walks the wrong axis and still type-checks.
        def fwd_body(k, prev):
            w_k = (rhs_ref[0, k] - a_ref[0, k] * prev) * alpha_ref[0, k]
            fwd_ref[0, k] = w_k
            return w_k

        fwd_ref[0, 0] = rhs_ref[0, 0]
        jax.lax.fori_loop(1, nz, fwd_body, rhs_ref[0, 0])

        # Interior-only back-substitution, matching WRF's advance_w: the bottom
        # and top faces are boundary conditions and keep their forward values.
        def back_body(i, nxt):
            k = nz - 2 - i
            v = fwd_ref[0, k] - gamma_ref[0, k] * nxt
            out_ref[0, k] = v
            return v

        out_ref[0, 0] = fwd_ref[0, 0]
        out_ref[0, nz - 1] = fwd_ref[0, nz - 1]
        jax.lax.fori_loop(0, nz - 2, back_body, fwd_ref[0, nz - 1])

    def run(a, alpha, gamma, rhs):
        # (nz, ny, nx) -> (columns, nz): one contiguous column per program.
        def to_cols(x):
            return jnp.transpose(x.reshape(nz, -1), (1, 0))

        a_c, al_c, g_c, r_c = (to_cols(x) for x in (a, alpha, gamma, rhs))
        ncols = a_c.shape[0]
        spec = pl.BlockSpec((1, nz), lambda i: (i, 0))
        # Order follows the kernel signature's output refs: (out_ref, fwd_ref).
        out, fwd = pl.pallas_call(
            kernel,
            grid=(ncols,),
            in_specs=[spec] * 4,
            out_specs=[spec, spec],
            out_shape=[
                jax.ShapeDtypeStruct((ncols, nz), r_c.dtype),
                jax.ShapeDtypeStruct((ncols, nz), r_c.dtype),
            ],
            interpret=interpret,
        )(a_c, al_c, g_c, r_c)
        back = lambda x: jnp.transpose(x, (1, 0)).reshape(rhs.shape)  # noqa: E731
        return back(out), back(fwd)

    return run


def error_metrics(got, want) -> dict:
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    diff = np.abs(got - want)
    denom = np.maximum(np.abs(want), 1e-300)
    rel = diff / denom
    return {
        "max_abs": float(diff.max()),
        "max_rel": float(rel.max()),
        "median_rel": float(np.median(rel)),
        "p99_rel": float(np.percentile(rel, 99)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "all_finite": bool(np.isfinite(got).all()),
    }


def classify(*, native: bool, correctness_ok: bool, speedup, lower95) -> str:
    """Contract §11 verdict table, verbatim. Interpreter can never be GREEN."""
    if not native:
        return "DIAGNOSTIC_ONLY_NOT_A_VERDICT"
    if not correctness_ok:
        return "PALLAS_BLOCKED"
    if speedup is not None and lower95 is not None and speedup >= 1.30 and lower95 > 1.00:
        return "PALLAS_GREEN"
    return "PALLAS_CAPABLE_NOT_YET_FAST"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape)).encode("utf-8"))
    digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def load_real_savepoint(manifest_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load and hash-check the CPU-prepared real FAST Thomas inputs."""

    if not Path(manifest_path).is_file():
        raise RuntimeError(f"real FAST savepoint manifest is missing: {manifest_path}")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SAVEPOINT_SCHEMA
        or manifest.get("source", {}).get("state_source_is_real") is not True
        or manifest.get("native_pallas_verdict") != "MISSING"
    ):
        raise RuntimeError("real FAST savepoint manifest is not CPU-prepared/MISSING")
    payload = Path(str(manifest.get("payload_path", "")))
    if not payload.is_file() or _sha256_file(payload) != manifest.get("payload_sha256"):
        raise RuntimeError("real FAST savepoint payload is missing or hash-mismatched")
    with np.load(payload, allow_pickle=False) as archive:
        if set(archive.files) != {"a", "alpha", "gamma", "rhs"}:
            raise RuntimeError("real FAST savepoint array names changed")
        arrays = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in ("a", "alpha", "gamma", "rhs")
        }
    for name, array in arrays.items():
        record = (manifest.get("arrays") or {}).get(name) or {}
        if (
            list(array.shape) != record.get("shape")
            or str(array.dtype) != record.get("dtype")
            or _array_sha256(array) != record.get("sha256")
            or not np.isfinite(array).all()
        ):
            raise RuntimeError(f"real FAST savepoint array {name} failed identity")
    if tuple(arrays["rhs"].shape) != (PROD_NZ, PROD_NY, PROD_NX):
        raise RuntimeError(
            f"real FAST savepoint shape {arrays['rhs'].shape} is not production"
        )
    return arrays, manifest


def _compile_separately(jitted, arguments, *, clock=time.monotonic_ns):
    lower_start = clock()
    lowered = jitted.lower(*arguments)
    lower_end = clock()
    executable = lowered.compile()
    compile_end = clock()
    return executable, {
        "lower_seconds": (lower_end - lower_start) / 1e9,
        "compile_seconds": (compile_end - lower_end) / 1e9,
        "ready_monotonic_ns": compile_end,
    }


def _run_iterations(executable, arguments, iterations: int, jax) -> float:
    """Time only warmed compiled dispatches and one terminal synchronization."""

    started = time.monotonic_ns()
    result = None
    for _ in range(int(iterations)):
        result = executable(*arguments)
    jax.block_until_ready(result)
    finished = time.monotonic_ns()
    return (finished - started) / 1e9


def _bootstrap_median_lower95(values: list[float], *, samples: int = 20000) -> float:
    if len(values) < 2:
        raise ValueError("bootstrap requires at least two paired ratios")
    rng = np.random.default_rng(20260728)
    source = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(source), size=(samples, len(source)))
    medians = np.median(source[indices], axis=1)
    return float(np.quantile(medians, 0.025))


def _atomic_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise RuntimeError(f"refusing to replace native Pallas result: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_native(args, authorization: dict) -> dict:
    """Native Mosaic correctness, transfer, and paired timing on real FAST data."""

    import jax
    import jax.numpy as jnp

    from gpuwrf._x64_config import configure_jax_x64
    from gpuwrf.profiling.transfer_audit import count_transfer_bytes

    configure_jax_x64()
    platforms = sorted({device.platform for device in jax.devices()})
    if platforms != ["gpu"]:
        raise RuntimeError(f"native Pallas expected exactly GPU; observed {platforms}")
    if args.warm_iterations < 100:
        raise RuntimeError("--warm-iterations must be >=100")
    if args.alternating_pairs < 5:
        raise RuntimeError("--alternating-pairs must be >=5")

    arrays, manifest = load_real_savepoint(args.savepoint)
    dtype = np.float32 if args.dtype == "float32" else np.float64
    host_arguments = tuple(arrays[name] for name in ("a", "alpha", "gamma", "rhs"))
    device_arguments = tuple(jax.device_put(jnp.asarray(value, dtype=dtype))
                             for value in host_arguments)

    reference = jax.jit(production_reference())
    pallas_run = build_pallas_kernel(PROD_NZ, interpret=False)

    def candidate(a, alpha, gamma, rhs):
        solved, forward = pallas_run(a, alpha, gamma, rhs)
        return forward, solved

    candidate_jit = jax.jit(candidate)
    reference_executable, reference_compile = _compile_separately(
        reference, device_arguments
    )
    candidate_executable, candidate_compile = _compile_separately(
        candidate_jit, device_arguments
    )
    executable_ready_ns = max(
        reference_compile["ready_monotonic_ns"],
        candidate_compile["ready_monotonic_ns"],
    )

    reference_result = reference_executable(*device_arguments)
    candidate_result = candidate_executable(*device_arguments)
    jax.block_until_ready((reference_result, candidate_result))
    forward_ref, solved_ref = (
        np.asarray(jax.device_get(value)) for value in reference_result
    )
    forward_candidate, solved_candidate = (
        np.asarray(jax.device_get(value)) for value in candidate_result
    )
    forward_numpy, solved_numpy = thomas_numpy(*host_arguments)
    same_dtype_tolerance = 1e-12 if dtype is np.float64 else 1e-6
    candidate_vs_xla = {
        "forward": error_metrics(forward_candidate, forward_ref),
        "solved": error_metrics(solved_candidate, solved_ref),
    }
    candidate_vs_numpy = {
        "forward": error_metrics(forward_candidate, forward_numpy),
        "solved": error_metrics(solved_candidate, solved_numpy),
    }
    xla_vs_numpy = {
        "forward": error_metrics(forward_ref, forward_numpy),
        "solved": error_metrics(solved_ref, solved_numpy),
    }
    correctness_ok = bool(
        candidate_vs_xla["forward"]["max_rel"] < same_dtype_tolerance
        and candidate_vs_xla["solved"]["max_rel"] < same_dtype_tolerance
        and candidate_vs_xla["forward"]["all_finite"]
        and candidate_vs_xla["solved"]["all_finite"]
    )
    envelope = float(
        manifest["correctness_envelope"]["maximum"]
    )
    within_envelope = bool(
        candidate_vs_numpy["solved"]["p99_rel"] <= envelope
    )

    # Separate warmed transfer-audit loop. All host->device placement is above;
    # no device_get occurs until after this trace is closed.
    trace_dir = args.out.parent / f"{args.run_id}.transfer_trace"
    if os.path.lexists(trace_dir):
        raise RuntimeError(f"unique transfer trace path exists: {trace_dir}")
    with jax.profiler.trace(str(trace_dir)):
        with jax.profiler.TraceAnnotation(RANGE_NAME):
            _run_iterations(candidate_executable, device_arguments, 100, jax)
    h2d_bytes, d2h_bytes, transfer_files = count_transfer_bytes(trace_dir)
    no_hidden_transfer = h2d_bytes == 0 and d2h_bytes == 0

    # Compile has completed and one correctness/warm pass has run for both arms.
    # Every pair times 100 dispatches per arm, and order alternates mechanically.
    pair_records = []
    ratios = []
    for pair_index in range(args.alternating_pairs):
        order = (
            ("reference", reference_executable),
            ("pallas", candidate_executable),
        )
        if pair_index % 2:
            order = tuple(reversed(order))
        seconds = {}
        for name, executable in order:
            seconds[name] = _run_iterations(
                executable, device_arguments, args.warm_iterations, jax
            )
        ratio = seconds["reference"] / seconds["pallas"]
        ratios.append(float(ratio))
        pair_records.append(
            {
                "pair": pair_index + 1,
                "order": [name for name, _ in order],
                "reference_seconds": seconds["reference"],
                "pallas_seconds": seconds["pallas"],
                "speedup": ratio,
            }
        )
    median_speedup = float(st.median(ratios))
    lower95 = _bootstrap_median_lower95(ratios)
    verdict = classify(
        native=True,
        correctness_ok=(
            correctness_ok and within_envelope and no_hidden_transfer
        ),
        speedup=median_speedup,
        lower95=lower95,
    )
    return {
        "schema": "wrf_gpu2.v025.m0.pallas_sm120_viability.v1",
        "status": "OK",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "backend": "native_mosaic_gpu",
        "operator": "advance_w_vertical_implicit_thomas",
        "state_source_is_real": True,
        "savepoint": {
            "manifest": str(args.savepoint),
            "manifest_sha256": _sha256_file(args.savepoint),
            "payload": manifest["payload_path"],
            "payload_sha256": manifest["payload_sha256"],
            "shape": list(arrays["rhs"].shape),
        },
        "operator_selection": {
            "contract_preferred": "advance_w vertical implicit/Thomas solve",
            "substitution_used": False,
            "entrypoint": "gpuwrf.dynamics.tridiag_solve.thomas_solve_scan",
            "wrf_source": "dyn_em/module_small_step_em.F:1533-1550",
        },
        "authorization": authorization,
        "timing": {
            "parent_process_launch_monotonic_ns":
                authorization["parent_launch_monotonic_ns"],
            "child_executable_ready_monotonic_ns": executable_ready_ns,
            "compile_separated": True,
            "reference": reference_compile,
            "pallas": candidate_compile,
        },
        "correctness": {
            "cpu_reference_pass": correctness_ok,
            "native_output_finite": bool(
                candidate_vs_xla["forward"]["all_finite"]
                and candidate_vs_xla["solved"]["all_finite"]
            ),
            "within_wrf_fp32_vs_fp64_envelope": within_envelope,
            "no_hidden_transfer_in_timed_loop": no_hidden_transfer,
            "same_dtype_tolerance": same_dtype_tolerance,
            "fp32_vs_fp64_p99_maximum": envelope,
            "pallas_vs_production_xla": candidate_vs_xla,
            "pallas_vs_independent_numpy_oracle": candidate_vs_numpy,
            "production_xla_vs_independent_numpy_oracle": xla_vs_numpy,
        },
        "transfer_audit": {
            "range_name": RANGE_NAME,
            "iterations": 100,
            "host_to_device_bytes_post_init": h2d_bytes,
            "device_to_host_bytes_post_init": d2h_bytes,
            "trace_dir": str(trace_dir),
            "trace_transfer_event_files": transfer_files,
        },
        "performance": {
            "warm_iterations_per_arm": args.warm_iterations,
            "alternating_pairs": args.alternating_pairs,
            "paired_warm_median_speedup": median_speedup,
            "bootstrap_lower_95": lower95,
            "pair_records": pair_records,
            "raw_artifacts": [str(trace_dir)],
        },
        "environment": {
            "host": platform.node(),
            "jax": jax.__version__,
            "python": sys.version.split()[0],
            "platforms": platforms,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--interpret", action="store_true", help="CPU/interpreter (Phase A)")
    mode.add_argument("--native", action="store_true", help="native Mosaic GPU (Phase B only)")
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--columns", type=int, default=256,
                    help="interpreter is slow; full production is %d columns" % PROD_COLUMNS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--handoff", type=Path)
    # ``--session-label`` is the canonical lock/receipt authority; ``--window``
    # is the stage identity inside it.  They are two namespaces and the stage
    # identity is never checked against the lock.
    ap.add_argument("--launch-mode", default="held")
    ap.add_argument("--session-label")
    ap.add_argument("--window")
    ap.add_argument("--stage")
    ap.add_argument("--run-id")
    ap.add_argument(
        "--savepoint",
        type=Path,
        default=DEFAULT_SAVEPOINT_MANIFEST,
        help="CPU-prepared real FAST savepoint manifest (native mode)",
    )
    ap.add_argument("--warm-iterations", type=int, default=100)
    ap.add_argument("--alternating-pairs", type=int, default=5)
    args = ap.parse_args()

    if args.native:
        # Importing this module is safe.  The handoff is consumed and the
        # canonical lock rechecked before the first JAX/gpuwrf import.
        missing = [
            name
            for name in (
                "handoff",
                "session_label",
                "window",
                "stage",
                "run_id",
                "out",
            )
            if getattr(args, name) is None
        ]
        if missing:
            refusal = {
                "schema": "wrf_gpu2.v025.m0.pallas_native_refusal.v1",
                "status": "REFUSED_PRE_DEVICE_IMPORT",
                "missing": missing,
                "handoff_consumed": False,
                "jax_imported": "jax" in sys.modules,
                "gpuwrf_imported": "gpuwrf" in sys.modules,
                "device_touched": False,
            }
            print(json.dumps(refusal, indent=2, sort_keys=True))
            return 2
        import m0_exact_boundary_child as exact_child

        try:
            authorization = exact_child.consume_authorization_handoff(
                args.handoff,
                expected_launch_mode=args.launch_mode,
                expected_session_label=args.session_label,
                expected_stage_identity=args.window,
                expected_stage=args.stage,
                expected_run_id=args.run_id,
            )
        except Exception as exc:  # noqa: BLE001 - refusal stays pre-import
            refusal = {
                "schema": "wrf_gpu2.v025.m0.pallas_native_refusal.v1",
                "status": "REFUSED_PRE_DEVICE_IMPORT",
                "reason": str(exc),
                "launch_mode": args.launch_mode,
                "session_label": args.session_label,
                "window": args.window,
                "handoff_consumed": False,
                "jax_imported": "jax" in sys.modules,
                "gpuwrf_imported": "gpuwrf" in sys.modules,
                "device_touched": False,
            }
            print(json.dumps(refusal, indent=2, sort_keys=True))
            return 2
        try:
            payload = run_native(args, authorization)
            _atomic_result(args.out, payload)
        except Exception as exc:  # noqa: BLE001 - persist exact native boundary
            failure = {
                "schema": "wrf_gpu2.v025.m0.pallas_sm120_viability.v1",
                "status": "FAILED",
                "verdict": "PALLAS_BLOCKED",
                "backend": "native_mosaic_gpu",
                "operator": "advance_w_vertical_implicit_thomas",
                "state_source_is_real": True,
                "authorization": authorization,
                "toolchain_boundary": f"{type(exc).__name__}: {exc}",
                "typed_ffi_fallback_smoke": {
                    "ran": False,
                    "reason": (
                        "not run automatically inside the same failed window; "
                        "a manager-approved fallback command is still required"
                    ),
                },
            }
            try:
                _atomic_result(args.out, failure)
            except Exception:
                pass
            print(json.dumps(failure, indent=2, sort_keys=True))
            return 1
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    import cpu_guard  # noqa: F401  MUST precede JAX in interpreter mode
    import jax
    import jax.numpy as jnp

    from gpuwrf._x64_config import configure_jax_x64

    configure_jax_x64()

    nz = PROD_NZ
    ncols = args.columns
    ny, nx = 1, ncols
    a, alpha, gamma, rhs = make_state(nz=nz, ny=ny, nx=nx)
    dtype = np.float64 if args.dtype == "float64" else np.float32

    # Oracle 1: independent NumPy transcription of WRF's own sweep.
    fwd_np, out_np = thomas_numpy(a, alpha, gamma, rhs)

    # Oracle 2: the production XLA reference, imported read-only.
    ref = production_reference()
    ja, jal, jg, jr = (jnp.asarray(x, dtype) for x in (a, alpha, gamma, rhs))
    t0 = time.perf_counter()
    fwd_ref, out_ref = jax.block_until_ready(ref(ja, jal, jg, jr))
    ref_seconds = time.perf_counter() - t0

    # Candidate: the Pallas kernel, in the interpreter.
    run = build_pallas_kernel(nz, interpret=True)
    t0 = time.perf_counter()
    out_pl, fwd_pl = jax.block_until_ready(jax.jit(run)(ja, jal, jg, jr))
    pallas_seconds = time.perf_counter() - t0

    vs_numpy = error_metrics(out_pl, out_np)
    vs_xla = error_metrics(out_pl, out_ref)
    xla_vs_numpy = error_metrics(out_ref, out_np)

    # The primary gate is Pallas vs the production XLA reference AT THE SAME
    # dtype: that is the "did the kernel compute the right thing" question, and
    # it must hold bit-tight. Comparing an fp32 kernel against an fp64 oracle
    # measures fp32 rounding, not kernel correctness, and would fail a perfectly
    # correct kernel.
    same_dtype_tol = 1e-12 if dtype is np.float64 else 1e-6
    correctness_ok = bool(
        vs_xla["max_rel"] < same_dtype_tol and vs_xla["all_finite"]
    )
    # The fp32-vs-fp64 delta is a separate quantity: §11 requires it to sit
    # inside the envelope taken from the cancellation map. That map does not
    # exist yet (Phase A5), so the envelope check is recorded as unresolved
    # rather than silently assumed to pass.
    fp32_vs_fp64 = vs_numpy if dtype is not np.float64 else None
    tol = same_dtype_tol

    verdict = classify(native=False, correctness_ok=correctness_ok, speedup=None, lower95=None)
    obj = {
        "schema": "wrf_gpu2.v025.m0.pallas_sm120_viability.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "backend": "interpreter",
        "backend_note": (
            "§11: interpreter-only and deprecated-backend results are DIAGNOSTIC, "
            "never viability evidence. This file cannot carry a PALLAS_* verdict."
        ),
        "operator": "advance_w_vertical_implicit_thomas",
        "operator_selection": {
            "contract_preferred": "advance_w vertical implicit/Thomas solve",
            "substitution_used": False,
            "isolatable_without_production_change": True,
            "entrypoint": "gpuwrf.dynamics.tridiag_solve.thomas_solve_scan",
            "wrf_source": "dyn_em/module_small_step_em.F:1533-1550",
        },
        "state_source_is_real": False,
        "state_source_note": (
            "synthetic well-conditioned coefficients at production vertical extent; "
            "a real FAST-v025 savepoint is required before any verdict"
        ),
        "shape": {
            "nz": nz,
            "columns_tested": ncols,
            "production_columns": PROD_COLUMNS,
            "dtype": args.dtype,
        },
        "correctness": {
            "cpu_reference_pass": bool(correctness_ok),
            "native_output_finite": None,
            "within_wrf_fp32_vs_fp64_envelope": None,
            "no_hidden_transfer_in_timed_loop": None,
            "tolerance": tol,
            "primary_gate": "pallas vs production XLA at the same dtype",
            "fp32_vs_fp64_envelope_delta": fp32_vs_fp64,
            "fp32_vs_fp64_envelope_source": (
                None if fp32_vs_fp64 is None else
                "UNRESOLVED: requires cancellation_map.json (Phase A5)"
            ),
            "pallas_vs_independent_numpy_oracle": vs_numpy,
            "pallas_vs_production_xla": vs_xla,
            "production_xla_vs_numpy_oracle": xla_vs_numpy,
        },
        "interpreter_timings_not_performance_evidence": {
            "xla_reference_seconds": ref_seconds,
            "pallas_interpreter_seconds": pallas_seconds,
            "note": "interpreter wallclock is meaningless as a speed signal; recorded for completeness only",
        },
        "environment": {
            "host": platform.node(),
            "jax": jax.__version__,
            "python": sys.version.split()[0],
        },
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps(obj, indent=2, sort_keys=True))
    return 0 if correctness_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
