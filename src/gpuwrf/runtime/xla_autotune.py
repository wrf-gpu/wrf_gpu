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

WHEN THE SEPARATE AUTOTUNE CACHE ACTUALLY HELPS (vs the executable cache)
------------------------------------------------------------------------
The standard JAX persistent executable cache (``compile_cache``) ALREADY embeds
the autotuner's results inside each cached executable. So for an *exact-same*
program -- recompiling the identical HLO on the identical backend -- the
executable cache is a full warm hit and this module adds NOTHING: the autotuner
never reruns because the whole executable is served from disk.

This separate per-fusion autotune cache earns its keep only when the executable
cache MISSES but the work is internally similar -- i.e. a NEW-but-related HLO
that shares fusions (GEMM/conv shapes) with something tuned before. Concretely
for gpuwrf:

* a first-of-a-family nest compile (e.g. d03 1km) whose fusions overlap an
  already-tuned d02 program -- the executable differs so the executable cache
  misses, but the shared fusions are served from the autotune cache instead of
  re-running the (fp64-heavy) autotuner;
* the same grid with a changed static flag/namelist (different whole-program
  HLO, overlapping kernels);
* incrementally warming a brand-new version-keyed dir, where many programs share
  GEMM/conv shapes.

It does NOT help a pure cold box with zero prior autotune entries (nothing to
reuse) and does NOT replace the executable cache. B1 turns it on by default
WHENEVER the executable cache is on so these partial-overlap cases stop paying
the autotuner twice; it is a strict superset of value with no downside beyond a
small extra cache dir.

NUMERICS
--------
**Numerically inert.** Autotuning only selects *which* kernel implementation
computes a GEMM/conv; persisting that choice changes nothing about the math.
The autotune *level* (``--xla_gpu_autotune_level``) is left at the XLA default
(4 = on+init+reinit+check) for production so the picked kernels are still the
fastest correct ones; only the *caching* of the result is added here.

DEFAULT-ON-WITH-COMPILE-CACHE (B1) + the v0.12.0 regression history
-------------------------------------------------------------------
An earlier version of this module injected ``--xla_gpu_*`` autotune-cache flags
into ``XLA_FLAGS`` **at package import** whenever a GPU was merely *detected*,
WITHOUT validating them. On the production GPU box the bundled jaxlib/XLA build
did **not recognise** those particular flag names: XLA printed its flag-help
text and **fatally aborted** the whole process before the first compile. That
broke the default GPU path (the v0.12.0 gate aborted in ~3 s) and the merge was
reverted (969d435). The fix that re-landed it was the per-flag SUBPROCESS PROBE
(point 2 below) -- NOT the opt-in gate. The opt-in gate was belt-and-braces.

B1 keeps the probe (the actual safety mechanism) and now activates the cache BY
DEFAULT when the executable compile cache is on, so a fresh user / paid run gets
the autotune reuse without setting any env var. Tri-state on
``GPUWRF_XLA_AUTOTUNE_CACHE``:

* truthy (``1``/``true``/``on``/``yes``/``force``) -> ON (explicit opt-in).
* falsey (``0``/``false``/``off``/``no``)          -> OFF (explicit opt-out, kept).
* UNSET -> ON when the compile cache is on (the import hook passes
  ``default_on=True``); a *direct* call to :func:`configure_autotune_cache` with
  no argument keeps the historical OFF default for callers that bypass the hook.

This is HARD-SAFE by construction REGARDLESS of the default:

1. **Opt-out always wins + platform guard.** ``GPUWRF_XLA_AUTOTUNE_CACHE=0`` is a
   pure no-op, and the flags are NEVER injected on a CPU/TPU pin (where a CPU
   jaxlib could fatally abort on an unknown ``--xla_gpu_*`` flag).
2. **Flag validation against the actual build (no blind inject).** Every
   candidate ``--xla_gpu_*`` flag is first PROBED in an isolated child process
   that tries to initialise the GPU backend with only that flag set. If the
   child aborts / errors / the flag is unknown, the flag is dropped. The probe
   runs in a SUBPROCESS so an XLA fatal-abort kills only the child, never this
   process. Only flags the build provably accepts are injected. This is what
   makes default-on safe: an unknown flag can never abort the main process.
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
    # B1: "explicit-opt-in" vs "default-on-with-compile-cache" vs None (off).
    "activation": None,
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
            "activation": None,
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
    # B2: "explicit-opt-in" vs "default-on-with-compile-cache" vs None (off).
    "activation": None,
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
            "activation": None,
            "probed": None,
            "rejected_flags": None,
            "error": None,
        }
    )


def _opt_in(default_on: bool = False) -> bool:
    """Whether the autotune cache should activate.

    Tri-state on ``GPUWRF_XLA_AUTOTUNE_CACHE``:

    * truthy (``1``/``true``/``on``/``yes``/``force``) -> ON (explicit opt-in).
    * falsey (``0``/``false``/``off``/``no``)          -> OFF (explicit opt-out).
    * UNSET -> ``default_on``.

    B1 change: the import hook (compile cache) now calls this with
    ``default_on=True`` AFTER the persistent compile cache is configured on, so
    the autotune cache rides ON by default WHENEVER the executable cache is on
    (preserving the ``=0`` opt-out). A *direct* call with no argument keeps the
    historical default OFF, so unit tests / callers that hit this module without
    the compile-cache hook stay byte-unchanged unless they explicitly opt in.

    The v0.12.0 abort guard is UNCHANGED and still holds regardless of the
    default: even when this returns True, every ``--xla_gpu_*`` flag is still
    validated against the build in an isolated subprocess before injection, so an
    unknown flag can never abort this process.
    """
    raw = os.environ.get("GPUWRF_XLA_AUTOTUNE_CACHE", "").strip().lower()
    if raw in _FORCE_VALUES:
        return True
    if raw in _DISABLE_VALUES:
        return False
    return default_on


def resolve_autotune_cache_dir(default_on: bool = False) -> Path | None:
    """Return the autotune-cache directory, or ``None`` if disabled / not opted in.

    Mirrors :func:`gpuwrf.runtime.compile_cache.resolve_cache_dir` precedence so
    the two caches sit side by side under one project cache root. Pure apart
    from recording ``AUTOTUNE_STATUS['source']``. ``default_on`` controls the
    UNSET case (see :func:`_opt_in`)."""
    flag = os.environ.get("GPUWRF_XLA_AUTOTUNE_CACHE", "").strip().lower()
    if flag in _DISABLE_VALUES:
        AUTOTUNE_STATUS["source"] = "disabled-by-GPUWRF_XLA_AUTOTUNE_CACHE"
        return None
    if flag not in _FORCE_VALUES and not default_on:
        # Unset AND not default-on (a direct call). Pure no-op; the JIT compile
        # cache still applies.
        AUTOTUNE_STATUS["source"] = "not-opted-in"
        return None
    # Activation reason for audits: explicit opt-in vs default-on-with-compile-cache.
    AUTOTUNE_STATUS["activation"] = (
        "explicit-opt-in" if flag in _FORCE_VALUES else "default-on-with-compile-cache"
    )

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


def configure_autotune_cache(default_on: bool = False) -> dict[str, object]:
    """Inject build-validated persistent-autotune-cache + parallel-compile flags.

    Must be called BEFORE the first JAX/XLA compile (i.e. at package import,
    before any jitted fn is built). Idempotent, best-effort, never raises.

    ``default_on`` (B1): when the persistent executable compile cache is
    configured ON, the import hook calls this with ``default_on=True`` so the
    autotune cache rides ON by default too (the executable cache already carries
    autotune results for the SAME graph; this separate cache adds reuse across
    DIFFERENT-but-similar graphs -- e.g. a first-of-a-family nest compile). The
    ``GPUWRF_XLA_AUTOTUNE_CACHE=0`` opt-out is always honoured. A direct call
    (``default_on`` left False) keeps the historical OFF-unless-opted-in default.

    HARD-SAFE by construction (see the module docstring) REGARDLESS of the
    default: every candidate ``--xla_gpu_*`` flag is validated against the
    installed build in an isolated subprocess and dropped if the build rejects it
    (so XLA can never abort this process on an unknown flag), and the flags are
    never injected on a CPU/TPU pin. So default-on cannot reintroduce the v0.12.0
    GPU abort.

    Returns :data:`AUTOTUNE_STATUS`.
    """
    _reset_status()

    AUTOTUNE_STATUS["opted_in"] = _opt_in(default_on=default_on)

    cache_dir = resolve_autotune_cache_dir(default_on=default_on)
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


def _default_parallelism() -> int:
    """The default N-way compile thread count when parallel compile is on.

    ``os.cpu_count()`` capped at 8 -- enough to overlap the many independent
    fusion compiles of a cold nest compile (the ~40 min B200 case) without
    oversubscribing a box that may be running nightly CPU-WRF in parallel. XLA's
    ``--xla_gpu_force_compilation_parallelism`` parallelises HOST-side LLVM
    codegen across these threads; it changes the compile-thread count only, never
    the emitted executable (numerically inert).
    """
    return min(os.cpu_count() or 1, 8)


def resolve_parallel_compile(default_on: bool = False) -> int | None:
    """Resolve the requested XLA compile-parallelism, or ``None`` if disabled.

    Tier-2 compile-speed knob. Precedence:

    1. ``GPUWRF_XLA_PARALLEL_COMPILE`` -- the primary control.
       * falsey (``0``/``false``/``off``/``no``) -> disabled (``None``); the
         explicit opt-OUT, always honoured regardless of ``default_on``.
       * a positive integer -> that many parallel compile threads.
       * truthy (``1``/``true``/``on``/``yes``/``force``) without a number ->
         a default thread count taken from ``GPUWRF_XLA_COMPILE_PARALLELISM`` if
         set, else ``os.cpu_count()`` capped at 8 (a sane default that does not
         oversubscribe the box reserved for nightly WRF).
    2. ``GPUWRF_XLA_COMPILE_PARALLELISM`` alone (legacy var, also consumed by the
       autotune-cache path) is honoured as an opt-in when ``GPUWRF_XLA_PARALLEL_COMPILE``
       is UNSET, so a deployment that only sets the parallelism count still gets
       parallel compile. (If both are set, ``GPUWRF_XLA_PARALLEL_COMPILE`` wins.)
    3. ``default_on`` (B2): when BOTH vars are UNSET and the persistent compile
       cache is on, the import hook calls this with ``default_on=True`` so N-way
       parallel compile rides ON by default at :func:`_default_parallelism`
       threads. The ``=0`` opt-out always wins. This is what slashes the cold
       nest-compile wall (many independent fusions compiled in parallel) on a
       fresh machine without the operator setting any env var. STILL HARD-SAFE:
       the single ``--xla_gpu_*`` flag is subprocess-probed before injection and
       never injected on a CPU/TPU pin, so default-on cannot reintroduce the
       v0.12.0 GPU abort.

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
            if default_on:
                # B2 default-on: both vars unset + compile cache on -> default N.
                n = _default_parallelism()
                PARALLEL_COMPILE_STATUS["source"] = "default-on-with-compile-cache"
                PARALLEL_COMPILE_STATUS["activation"] = "default-on-with-compile-cache"
                return n
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
        PARALLEL_COMPILE_STATUS["activation"] = "explicit-opt-in"
        return n

    # Primary var present and truthy/numeric.
    # A bare integer in GPUWRF_XLA_PARALLEL_COMPILE is both an opt-in AND the count.
    try:
        n = int(raw)
        if n < 1:
            PARALLEL_COMPILE_STATUS["source"] = "disabled-by-GPUWRF_XLA_PARALLEL_COMPILE<1"
            return None
        PARALLEL_COMPILE_STATUS["source"] = "env:GPUWRF_XLA_PARALLEL_COMPILE(count)"
        PARALLEL_COMPILE_STATUS["activation"] = "explicit-opt-in"
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
    PARALLEL_COMPILE_STATUS["activation"] = "explicit-opt-in"
    if legacy:
        try:
            n = int(legacy)
            if n >= 1:
                PARALLEL_COMPILE_STATUS["source"] = "env:GPUWRF_XLA_COMPILE_PARALLELISM"
                return n
        except ValueError:
            pass  # fall through to the cpu_count default
    n = _default_parallelism()
    PARALLEL_COMPILE_STATUS["source"] = "default:cpu_count(<=8)"
    return n


def configure_parallel_compile(default_on: bool = False) -> dict[str, object]:
    """Inject the build-validated ``--xla_gpu_force_compilation_parallelism`` flag.

    GPU-only Tier-2 compile-speed knob. Must be called BEFORE the first JAX/XLA
    compile. Idempotent, best-effort, never raises.

    ``default_on`` (B2): when the persistent executable compile cache is
    configured ON, the import hook calls this with ``default_on=True`` so N-way
    parallel XLA compile rides ON by default (at :func:`_default_parallelism`
    threads) -- this is what slashes the cold nest-compile wall (the ~40 min B200
    case) by overlapping the many independent fusion compiles. The
    ``GPUWRF_XLA_PARALLEL_COMPILE=0`` opt-out is always honoured. A *direct* call
    (``default_on`` left False) keeps the historical OFF-unless-opted-in default,
    so unit tests / callers that bypass the compile-cache hook stay byte-unchanged
    unless they opt in.

    HARD-SAFE by the same construction as :func:`configure_autotune_cache`
    REGARDLESS of the default:

    1. The ``GPUWRF_XLA_PARALLEL_COMPILE=0`` opt-out always wins; only an UNSET
       var defers to ``default_on``.
    2. Respects the platform pin (never injects ``--xla_gpu_*`` on a cpu/tpu pin,
       where a CPU jaxlib could fatally abort on an unknown flag).
    3. The single candidate flag is PROBED in an isolated subprocess (the SAME
       :func:`probe_flag_supported` used by the autotune path) and dropped if the
       build rejects it -- so an unsupported flag can never abort this process
       (this is the explicit guard against re-introducing the v0.12.0 GPU-abort;
       the bundled jaxlib aborts on an UNKNOWN ``--xla_gpu_*`` flag, so default-on
       MUST keep this probe). This is precisely why default-on is safe.
    4. Never clobbers an operator-set ``--xla_gpu_force_compilation_parallelism``
       already in ``XLA_FLAGS``.

    NUMERICALLY INERT: compile parallelism only changes how many threads XLA uses
    to compile; it changes no floating-point op and no executable bytes.

    Returns :data:`PARALLEL_COMPILE_STATUS`.
    """
    _reset_parallel_status()

    parallelism = resolve_parallel_compile(default_on=default_on)
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
        "Persistent XLA autotune cache env vars (GPU only; ON by default when the "
        "persistent compile cache is on -- B1): "
        "GPUWRF_XLA_AUTOTUNE_CACHE=0 opts OUT (1/on forces on; unset = on-with-"
        "compile-cache); "
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
        "Parallel XLA compile env vars (GPU only; ON by default when the persistent "
        "compile cache is on -- B2; INDEPENDENT of the autotune cache): "
        "GPUWRF_XLA_PARALLEL_COMPILE=0 opts OUT (default thread count = "
        "GPUWRF_XLA_COMPILE_PARALLELISM if set else min(cpu_count,8)); "
        "GPUWRF_XLA_PARALLEL_COMPILE=N forces N threads directly; "
        "GPUWRF_XLA_PARALLEL_COMPILE=1 forces on; GPUWRF_XLA_COMPILE_PARALLELISM=N "
        "alone (without GPUWRF_XLA_PARALLEL_COMPILE) also opts in with N threads; "
        "unset = on-with-compile-cache at min(cpu_count,8). "
        "The single --xla_gpu_force_compilation_parallelism flag is validated in an "
        "isolated subprocess (GPUWRF_XLA_AUTOTUNE_PROBE=0 to skip) and dropped if "
        "the build rejects it, so it can never abort the main process. Numerically "
        "inert (only changes compile-thread count, not the executable)."
    )
