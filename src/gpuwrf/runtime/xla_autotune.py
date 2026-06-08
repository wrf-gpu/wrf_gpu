"""Persistent XLA GPU autotuning cache + parallel-compile wiring (v0.13 #2/#4).

WHY
---
XLA AOT-compiles the whole jitted timestep graph before the first run, and a
large chunk of that wall-clock for the GPU backend is **autotuning**: XLA runs
candidate GEMM / convolution kernels at compile time to pick the fastest, which
is heavier in fp64 (fp64 is forced operationally; fp32 detonates the acoustic
solver, so the fp64-inherent autotune cost stays). The v0.13 speed roadmap
(#2) calls for *persisting* those autotune results to disk so even a new-but-
similar graph reuses the tuned kernels instead of re-running the autotuner.

This is the autotune-cache analogue of :mod:`gpuwrf.runtime.compile_cache`
(which persists the *compiled executable*). The two are complementary:

* compile_cache  -> warm hit  = identical HLO + backend + flags (whole-program
  executable served from disk).
* this module     -> warm hit = same fusion's tuned kernel choice reused even
  when the surrounding program differs, so partial / first-of-a-family compiles
  still skip the autotuner.

NUMERICS
--------
**Numerically inert.** Autotuning only selects *which* kernel implementation
computes a GEMM/conv; persisting that choice changes nothing about the math.
The autotune *level* (``--xla_gpu_autotune_level``) is left at the XLA default
(4 = on+init+reinit+check) for production so the picked kernels are still the
fastest correct ones; only the *caching* of the result is added here.

WHY THIS MODULE IS OFF BY DEFAULT (the v0.12.0 regression + its fix)
-------------------------------------------------------------------
An earlier version of this module injected ``--xla_gpu_*`` autotune-cache flags
into ``XLA_FLAGS`` **at package import** whenever a GPU was merely *detected*.
On the production GPU box the bundled jaxlib/XLA build did **not recognise**
those particular flag names: XLA printed its flag-help text and **fatally
aborted** the whole process before the first compile. That broke the default
GPU path (the v0.12.0 gate aborted in ~3 s) and the merge was reverted
(969d435).

This re-landed version is HARD-SAFE by construction:

1. **Explicit opt-in (default OFF).** The autotune cache does nothing unless the
   operator explicitly sets ``GPUWRF_XLA_AUTOTUNE_CACHE`` to a truthy value
   (``1``/``true``/``on``/``yes``/``force``). With the var unset or falsey this
   module is a pure no-op and injects nothing -- so the default GPU path is
   **byte-for-byte unchanged** from a build without this module. Detecting a GPU
   is NOT sufficient to activate it.
2. **Flag validation against the actual build (no blind inject).** Even when
   opted in, every candidate ``--xla_gpu_*`` flag is first PROBED in an isolated
   child process that tries to initialise the GPU backend with only that flag
   set. If the child aborts / errors / the flag is unknown, the flag is dropped.
   The probe runs in a SUBPROCESS so an XLA fatal-abort kills only the child,
   never this process. Only flags the build provably accepts are injected.
3. **No-op + log on anything unsupported.** A rejected/unprobeable flag is
   recorded in :data:`AUTOTUNE_STATUS` and simply skipped; this module never
   raises and never aborts at import.

PLATFORM GUARD
--------------
These are GPU-plugin flags (``--xla_gpu_*``). They are valid only when a CUDA
backend is loaded; on a CPU-only jaxlib build XLA logs ``unknown flag`` and (in
some builds) can abort init. We therefore inject them ONLY when (a) the operator
opted in AND (b) a CUDA backend is the likely target, detected WITHOUT
importing/initialising JAX in THIS process (importing jax here would lock the
flags in before we set them) AND (c) the per-flag subprocess probe confirms the
build accepts the flag.

Best-effort: any failure leaves ``XLA_FLAGS`` untouched and never raises at
import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

__all__ = [
    "AUTOTUNE_STATUS",
    "PARALLEL_COMPILE_STATUS",
    "configure_autotune_cache",
    "configure_parallel_compile",
    "resolve_autotune_cache_dir",
    "resolve_parallel_compile",
    "autotune_env_help",
    "parallel_compile_env_help",
    "probe_flag_supported",
]

_DISABLE_VALUES = {"0", "false", "off", "no"}
_FORCE_VALUES = {"1", "true", "on", "yes", "force"}

# Per-flag probe timeout (seconds). Initialising a GPU backend in a cold child
# is fast; cap it so a wedged probe can never stall import.
_PROBE_TIMEOUT_S = 30.0

# Record of what was configured, for audit / proof scripts.
AUTOTUNE_STATUS: dict[str, object] = {
    "enabled": False,
    "dir": None,
    "mode": None,
    "parallelism": None,
    "injected_flags": None,
    "source": None,
    "reason": None,
    "opted_in": False,
    "probed": None,
    "rejected_flags": None,
    "error": None,
}


def _reset_status() -> None:
    """Reset the module-global status so a re-call reflects the CURRENT env."""
    AUTOTUNE_STATUS.update(
        {
            "enabled": False,
            "dir": None,
            "mode": None,
            "parallelism": None,
            "injected_flags": None,
            "source": None,
            "reason": None,
            "opted_in": False,
            "probed": None,
            "rejected_flags": None,
            "error": None,
        }
    )


# Record of the standalone parallel-compile knob, for audit / proof scripts. This
# is INDEPENDENT of the autotune cache above: a deployment can enable N-way
# parallel XLA compilation without opting into the persistent autotune cache.
PARALLEL_COMPILE_STATUS: dict[str, object] = {
    "enabled": False,
    "parallelism": None,
    "injected_flags": None,
    "source": None,
    "reason": None,
    "opted_in": False,
    "probed": None,
    "rejected_flags": None,
    "error": None,
}


def _reset_parallel_status() -> None:
    PARALLEL_COMPILE_STATUS.update(
        {
            "enabled": False,
            "parallelism": None,
            "injected_flags": None,
            "source": None,
            "reason": None,
            "opted_in": False,
            "probed": None,
            "rejected_flags": None,
            "error": None,
        }
    )


def _opt_in() -> bool:
    """True only if the operator EXPLICITLY enabled the autotune cache.

    Default is OFF: a merely-detected GPU does NOT activate this module. This is
    the gate that makes the default GPU path byte-unchanged (regression fix d).
    """
    return os.environ.get("GPUWRF_XLA_AUTOTUNE_CACHE", "").strip().lower() in _FORCE_VALUES


def resolve_autotune_cache_dir() -> Path | None:
    """Return the autotune-cache directory, or ``None`` if disabled / not opted in.

    Mirrors :func:`gpuwrf.runtime.compile_cache.resolve_cache_dir` precedence so
    the two caches sit side by side under one project cache root. Pure apart
    from recording ``AUTOTUNE_STATUS['source']``.
    """
    flag = os.environ.get("GPUWRF_XLA_AUTOTUNE_CACHE", "").strip().lower()
    if flag in _DISABLE_VALUES:
        AUTOTUNE_STATUS["source"] = "disabled-by-GPUWRF_XLA_AUTOTUNE_CACHE"
        return None
    if flag not in _FORCE_VALUES:
        # Not opted in (default). Pure no-op; the JIT compile cache still applies.
        AUTOTUNE_STATUS["source"] = "not-opted-in"
        return None

    explicit = os.environ.get("GPUWRF_XLA_AUTOTUNE_CACHE_DIR", "").strip()
    if explicit:
        AUTOTUNE_STATUS["source"] = "env:GPUWRF_XLA_AUTOTUNE_CACHE_DIR"
        return Path(explicit).expanduser()

    root = os.environ.get("GPUWRF_CACHE", "").strip()
    if root:
        AUTOTUNE_STATUS["source"] = "env:GPUWRF_CACHE"
        return Path(root).expanduser() / "autotune"

    # Sit beside the JIT cache under the same per-user default root.
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else (Path.home() / ".cache")
    AUTOTUNE_STATUS["source"] = "default"
    return base / "gpuwrf" / "autotune"


def _gpu_is_target() -> tuple[bool, str]:
    """Decide whether a GPU backend is the target, WITHOUT importing jax.

    Returns ``(is_gpu, reason)``. Importing jax here would freeze ``XLA_FLAGS``
    before we add ours, so detection is purely environmental. NOTE: this is the
    *platform* gate only; activation ALSO requires the explicit opt-in
    (:func:`_opt_in`) -- a detected GPU alone never injects anything.
    """
    # Explicit platform pin takes precedence: a CPU/TPU pin means the
    # --xla_gpu_* flags are unknown to the active backend, and on a CPU-only
    # jaxlib build XLA can fatally abort on an unknown flag. So never inject when
    # the operator pinned cpu/tpu, even when opted in.
    platforms = (
        os.environ.get("JAX_PLATFORMS", "")
        or os.environ.get("JAX_PLATFORM_NAME", "")
    ).strip().lower()
    if platforms:
        first = platforms.split(",")[0].strip()
        if first in ("cpu", "tpu"):
            return False, f"platform-pinned:{first}"
        if first in ("cuda", "gpu", "rocm"):
            return True, f"platform-pinned:{first}"

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        cvd = cvd.strip()
        if cvd == "" or cvd == "-1":
            return False, "CUDA_VISIBLE_DEVICES-empty"
        return True, "CUDA_VISIBLE_DEVICES-set"

    # Fall back to probing for an nvidia device node (cheap, no jax import).
    for node in ("/dev/nvidia0", "/dev/nvidiactl"):
        if os.path.exists(node):
            return True, "nvidia-device-node"
    return False, "no-gpu-detected"


def _parse_existing_flags(xla_flags: str) -> set[str]:
    """Return the set of bare flag names already present in ``XLA_FLAGS``."""
    names: set[str] = set()
    for tok in xla_flags.split():
        if tok.startswith("--"):
            names.add(tok[2:].split("=", 1)[0])
    return names


# Child program: set XLA_FLAGS to ONLY the candidate flag, then force the GPU
# backend to initialise (which is what parses --xla_gpu_* flags). Exit 0 iff the
# build accepted the flag. Any unknown-flag abort makes the child exit non-zero
# (or be killed by the abort), so the parent treats it as unsupported. This
# isolates the fatal-abort failure mode to the child process.
_PROBE_CHILD = r"""
import os, sys
# The parent passes the single candidate flag via _GPUWRF_PROBE_FLAG; set it as
# the ONLY XLA flag so a parse error is unambiguously about this flag.
os.environ["XLA_FLAGS"] = os.environ.get("_GPUWRF_PROBE_FLAG", "")
# Force the GPU/CUDA backend: that is the code path that parses --xla_gpu_*.
os.environ["JAX_PLATFORMS"] = "cuda"
try:
    import jax
    # Initialise the backend; this triggers XLA flag parsing for the GPU plugin.
    _ = jax.devices()
except SystemExit:
    raise
except BaseException as exc:  # noqa: BLE001 - any failure => unsupported
    sys.stderr.write("PROBE_FAIL:%s:%s\n" % (type(exc).__name__, exc))
    sys.exit(3)
sys.exit(0)
"""


def probe_flag_supported(flag: str, timeout_s: float = _PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """Probe whether the installed XLA/jaxlib build accepts ``flag``.

    Runs a short child process that sets ``XLA_FLAGS`` to ONLY ``flag`` and then
    initialises the CUDA backend (the code path that parses ``--xla_gpu_*``
    flags). Returns ``(ok, detail)``. ``ok`` is True iff the child exited 0
    (flag accepted + backend initialised). The probe is fully isolated: an XLA
    fatal-abort kills only the child, never the caller, and the function never
    raises.

    ``flag`` is the full ``--name=value`` token.
    """
    env = dict(os.environ)
    env["_GPUWRF_PROBE_FLAG"] = flag
    # Make sure the child is not itself short-circuited to CPU by an inherited
    # pin; the child sets JAX_PLATFORMS=cuda explicitly, but clear conflicting
    # vars so the pin actually takes.
    env.pop("JAX_PLATFORM_NAME", None)
    # Do not let the child inherit a CPU pin from the parent env.
    if env.get("JAX_PLATFORMS", "").strip().lower().startswith(("cpu", "tpu")):
        env["JAX_PLATFORMS"] = "cuda"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_CHILD],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, "probe-timeout"
    except OSError as exc:
        return False, f"probe-spawn-failed:{exc}"

    if proc.returncode == 0:
        return True, "accepted"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    detail = tail[-1] if tail else f"rc={proc.returncode}"
    return False, f"rejected:rc={proc.returncode}:{detail[:200]}"


def configure_autotune_cache() -> dict[str, object]:
    """Inject build-validated persistent-autotune-cache + parallel-compile flags.

    Must be called BEFORE the first JAX/XLA compile (i.e. at package import,
    before any jitted fn is built). Idempotent, best-effort, never raises.

    HARD-SAFE by construction (see the module docstring): does NOTHING unless the
    operator explicitly opts in via ``GPUWRF_XLA_AUTOTUNE_CACHE`` truthy; even
    then, every candidate ``--xla_gpu_*`` flag is validated against the installed
    build in an isolated subprocess and dropped if the build rejects it (so XLA
    can never abort this process on an unknown flag).

    Returns :data:`AUTOTUNE_STATUS`.
    """
    _reset_status()

    AUTOTUNE_STATUS["opted_in"] = _opt_in()

    cache_dir = resolve_autotune_cache_dir()
    if cache_dir is None:
        # Disabled or (default) not opted in. Pure no-op: XLA_FLAGS untouched.
        AUTOTUNE_STATUS["reason"] = AUTOTUNE_STATUS.get("source") or "disabled"
        return AUTOTUNE_STATUS

    is_gpu, reason = _gpu_is_target()
    AUTOTUNE_STATUS["reason"] = reason
    if not is_gpu:
        # CPU/TPU run: these --xla_gpu_* flags are unknown to the backend and
        # would log noise (or abort). Skip silently; the JIT compile cache
        # (compile_cache.py) still applies on CPU.
        return AUTOTUNE_STATUS

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        AUTOTUNE_STATUS["error"] = f"mkdir failed: {exc}"
        return AUTOTUNE_STATUS

    existing_raw = os.environ.get("XLA_FLAGS", "")
    existing = _parse_existing_flags(existing_raw)

    # Mode: "update" reads existing results AND writes newly-tuned ones back, so
    # the cache fills incrementally and is reused across runs.
    mode = os.environ.get("GPUWRF_XLA_AUTOTUNE_CACHE_MODE", "update").strip() or "update"

    candidate: list[tuple[str, str]] = [
        ("xla_gpu_per_fusion_autotune_cache_dir", str(cache_dir)),
        ("xla_gpu_experimental_autotune_cache_mode", mode),
    ]

    results_file = os.environ.get("GPUWRF_XLA_AUTOTUNE_RESULTS_FILE", "").strip()
    if results_file:
        candidate.append(("xla_gpu_dump_autotune_results_to", results_file))
        if os.path.exists(results_file):
            candidate.append(("xla_gpu_load_autotune_results_from", results_file))

    par = os.environ.get("GPUWRF_XLA_COMPILE_PARALLELISM", "").strip()
    parallelism: int | None = None
    if par:
        try:
            parallelism = int(par)
            candidate.append(("xla_gpu_force_compilation_parallelism", str(parallelism)))
        except ValueError:
            AUTOTUNE_STATUS["error"] = f"bad GPUWRF_XLA_COMPILE_PARALLELISM={par!r}"

    # Operator may have opted out of the (slow) subprocess validation, e.g. once
    # they have confirmed the build accepts the flags and want a faster import.
    # Default is to validate (safe). =0/false skips validation and trusts the
    # candidate flags (operator's explicit responsibility).
    validate = os.environ.get("GPUWRF_XLA_AUTOTUNE_PROBE", "1").strip().lower() not in _DISABLE_VALUES

    injected: list[str] = []
    probed: dict[str, str] = {}
    rejected: list[str] = []
    for name, value in candidate:
        if name in existing:
            # Never clobber an operator-set flag.
            continue
        token = f"--{name}={value}"
        if validate:
            ok, detail = probe_flag_supported(token)
            probed[name] = detail
            if not ok:
                # Build rejects this flag: drop it, log in status, continue.
                rejected.append(token)
                continue
        injected.append(token)

    if injected:
        new_flags = (existing_raw + " " + " ".join(injected)).strip()
        os.environ["XLA_FLAGS"] = new_flags

    AUTOTUNE_STATUS["enabled"] = bool(injected)
    AUTOTUNE_STATUS["dir"] = str(cache_dir)
    AUTOTUNE_STATUS["mode"] = mode
    AUTOTUNE_STATUS["parallelism"] = parallelism
    AUTOTUNE_STATUS["injected_flags"] = injected
    AUTOTUNE_STATUS["probed"] = probed or None
    AUTOTUNE_STATUS["rejected_flags"] = rejected or None
    return AUTOTUNE_STATUS


def resolve_parallel_compile() -> int | None:
    """Resolve the requested XLA compile-parallelism, or ``None`` if not opted in.

    STANDALONE knob (default OFF), INDEPENDENT of the autotune cache. Precedence:

    1. ``GPUWRF_XLA_PARALLEL_COMPILE`` -- the primary opt-in.
       * falsey (``0``/``false``/``off``/``no``) -> disabled (``None``).
       * a positive integer -> that many parallel compile threads.
       * truthy (``1``/``true``/``on``/``yes``/``force``) without a number ->
         a default thread count taken from ``GPUWRF_XLA_COMPILE_PARALLELISM`` if
         set, else ``os.cpu_count()`` capped at 8 (a sane default that does not
         oversubscribe the box reserved for nightly WRF).
    2. ``GPUWRF_XLA_COMPILE_PARALLELISM`` alone (legacy var, also consumed by the
       autotune-cache path) is honoured as an opt-in when ``GPUWRF_XLA_PARALLEL_COMPILE``
       is UNSET, so a deployment that only sets the parallelism count still gets
       parallel compile. (If both are set, ``GPUWRF_XLA_PARALLEL_COMPILE`` wins.)

    Records the chosen ``source`` in :data:`PARALLEL_COMPILE_STATUS`. Returns the
    integer parallelism (>=1) or ``None`` when disabled / not opted in / invalid.
    """
    raw = os.environ.get("GPUWRF_XLA_PARALLEL_COMPILE", "").strip().lower()
    legacy = os.environ.get("GPUWRF_XLA_COMPILE_PARALLELISM", "").strip()

    if raw in _DISABLE_VALUES:
        PARALLEL_COMPILE_STATUS["source"] = "disabled-by-GPUWRF_XLA_PARALLEL_COMPILE"
        return None

    if raw == "":
        # Primary var unset: fall back to the legacy count-only var as an opt-in.
        if not legacy:
            PARALLEL_COMPILE_STATUS["source"] = "not-opted-in"
            return None
        try:
            n = int(legacy)
        except ValueError:
            PARALLEL_COMPILE_STATUS["error"] = (
                f"bad GPUWRF_XLA_COMPILE_PARALLELISM={legacy!r}"
            )
            PARALLEL_COMPILE_STATUS["source"] = "invalid"
            return None
        if n < 1:
            PARALLEL_COMPILE_STATUS["source"] = "disabled-by-GPUWRF_XLA_COMPILE_PARALLELISM<1"
            return None
        PARALLEL_COMPILE_STATUS["source"] = "env:GPUWRF_XLA_COMPILE_PARALLELISM"
        return n

    # Primary var present and truthy/numeric.
    # A bare integer in GPUWRF_XLA_PARALLEL_COMPILE is both an opt-in AND the count.
    try:
        n = int(raw)
        if n < 1:
            PARALLEL_COMPILE_STATUS["source"] = "disabled-by-GPUWRF_XLA_PARALLEL_COMPILE<1"
            return None
        PARALLEL_COMPILE_STATUS["source"] = "env:GPUWRF_XLA_PARALLEL_COMPILE(count)"
        return n
    except ValueError:
        pass

    if raw not in _FORCE_VALUES:
        PARALLEL_COMPILE_STATUS["error"] = (
            f"bad GPUWRF_XLA_PARALLEL_COMPILE={raw!r}"
        )
        PARALLEL_COMPILE_STATUS["source"] = "invalid"
        return None

    # Truthy keyword: derive the count from the legacy var or a capped cpu_count.
    if legacy:
        try:
            n = int(legacy)
            if n >= 1:
                PARALLEL_COMPILE_STATUS["source"] = "env:GPUWRF_XLA_COMPILE_PARALLELISM"
                return n
        except ValueError:
            pass  # fall through to the cpu_count default
    n = min(os.cpu_count() or 1, 8)
    PARALLEL_COMPILE_STATUS["source"] = "default:cpu_count(<=8)"
    return n


def configure_parallel_compile() -> dict[str, object]:
    """Inject the build-validated ``--xla_gpu_force_compilation_parallelism`` flag.

    STANDALONE, default-OFF, GPU-only counterpart to the autotune cache. Must be
    called BEFORE the first JAX/XLA compile. Idempotent, best-effort, never raises.

    HARD-SAFE by the same construction as :func:`configure_autotune_cache`:

    1. Does NOTHING unless the operator opts in via ``GPUWRF_XLA_PARALLEL_COMPILE``
       (or the legacy ``GPUWRF_XLA_COMPILE_PARALLELISM`` count). A merely-detected
       GPU never activates it -> default GPU path byte-unchanged.
    2. Respects the platform pin (never injects ``--xla_gpu_*`` on a cpu/tpu pin,
       where a CPU jaxlib could fatally abort on an unknown flag).
    3. The single candidate flag is PROBED in an isolated subprocess (the SAME
       :func:`probe_flag_supported` used by the autotune path) and dropped if the
       build rejects it -- so an unsupported flag can never abort this process
       (this is the explicit guard against re-introducing the v0.12.0 GPU-abort).
    4. Never clobbers an operator-set ``--xla_gpu_force_compilation_parallelism``
       already in ``XLA_FLAGS``.

    NUMERICALLY INERT: compile parallelism only changes how many threads XLA uses
    to compile; it changes no floating-point op and no executable bytes.

    Returns :data:`PARALLEL_COMPILE_STATUS`.
    """
    _reset_parallel_status()

    parallelism = resolve_parallel_compile()
    PARALLEL_COMPILE_STATUS["opted_in"] = parallelism is not None
    if parallelism is None:
        # Disabled / not opted in. Pure no-op: XLA_FLAGS untouched.
        PARALLEL_COMPILE_STATUS["reason"] = (
            PARALLEL_COMPILE_STATUS.get("source") or "disabled"
        )
        return PARALLEL_COMPILE_STATUS

    is_gpu, reason = _gpu_is_target()
    PARALLEL_COMPILE_STATUS["reason"] = reason
    if not is_gpu:
        # CPU/TPU run: --xla_gpu_* is unknown to the backend (and can abort a
        # CPU jaxlib). Skip silently; parallel compile is a GPU-compile speedup.
        PARALLEL_COMPILE_STATUS["parallelism"] = parallelism
        return PARALLEL_COMPILE_STATUS

    existing_raw = os.environ.get("XLA_FLAGS", "")
    existing = _parse_existing_flags(existing_raw)

    name = "xla_gpu_force_compilation_parallelism"
    if name in existing:
        # Operator already set it: respect their value, inject nothing.
        PARALLEL_COMPILE_STATUS["parallelism"] = parallelism
        PARALLEL_COMPILE_STATUS["reason"] = f"{reason};operator-preset"
        return PARALLEL_COMPILE_STATUS

    token = f"--{name}={parallelism}"

    # Same opt-out as the autotune path: default validate (safe). =0/false trusts
    # the flag and skips the (slow) subprocess probe (operator's responsibility).
    validate = (
        os.environ.get("GPUWRF_XLA_AUTOTUNE_PROBE", "1").strip().lower()
        not in _DISABLE_VALUES
    )

    probed: dict[str, str] = {}
    if validate:
        ok, detail = probe_flag_supported(token)
        probed[name] = detail
        if not ok:
            PARALLEL_COMPILE_STATUS["parallelism"] = parallelism
            PARALLEL_COMPILE_STATUS["probed"] = probed
            PARALLEL_COMPILE_STATUS["rejected_flags"] = [token]
            return PARALLEL_COMPILE_STATUS

    new_flags = (existing_raw + " " + token).strip()
    os.environ["XLA_FLAGS"] = new_flags

    PARALLEL_COMPILE_STATUS["enabled"] = True
    PARALLEL_COMPILE_STATUS["parallelism"] = parallelism
    PARALLEL_COMPILE_STATUS["injected_flags"] = [token]
    PARALLEL_COMPILE_STATUS["probed"] = probed or None
    return PARALLEL_COMPILE_STATUS


def autotune_env_help() -> str:
    """One-line human summary of the env vars for ``--help`` / docs / logs."""
    return (
        "Persistent XLA autotune cache env vars (GPU only, OFF by default): "
        "GPUWRF_XLA_AUTOTUNE_CACHE=1 opts in (unset/0 = no-op, default GPU path "
        "unchanged); "
        "GPUWRF_XLA_AUTOTUNE_CACHE_DIR sets the dir (default $GPUWRF_CACHE/autotune "
        "or $HOME/.cache/gpuwrf/autotune); "
        "GPUWRF_XLA_AUTOTUNE_CACHE_MODE=update|read; "
        "GPUWRF_XLA_AUTOTUNE_RESULTS_FILE adds dump/load of a single results proto; "
        "GPUWRF_XLA_COMPILE_PARALLELISM=N enables N-way parallel compile; "
        "GPUWRF_XLA_AUTOTUNE_PROBE=0 skips the per-flag build-validation subprocess "
        "(default 1 = validate each --xla_gpu_* flag in an isolated child before "
        "injecting, so an unknown flag can never abort the main process)."
    )


def parallel_compile_env_help() -> str:
    """One-line human summary of the standalone parallel-compile env vars."""
    return (
        "Parallel XLA compile env vars (GPU only, OFF by default, INDEPENDENT of "
        "the autotune cache): GPUWRF_XLA_PARALLEL_COMPILE=1 opts in (default thread "
        "count = GPUWRF_XLA_COMPILE_PARALLELISM if set else min(cpu_count,8)); "
        "GPUWRF_XLA_PARALLEL_COMPILE=N opts in with N threads directly; "
        "GPUWRF_XLA_PARALLEL_COMPILE=0 disables; GPUWRF_XLA_COMPILE_PARALLELISM=N "
        "alone (without GPUWRF_XLA_PARALLEL_COMPILE) also opts in with N threads. "
        "The single --xla_gpu_force_compilation_parallelism flag is validated in an "
        "isolated subprocess (GPUWRF_XLA_AUTOTUNE_PROBE=0 to skip) and dropped if "
        "the build rejects it, so it can never abort the main process. Numerically "
        "inert (only changes compile-thread count, not the executable)."
    )
