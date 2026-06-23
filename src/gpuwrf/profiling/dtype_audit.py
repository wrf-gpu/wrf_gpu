"""Per-field dtype + XLA convert-counter audit for the precision program (v0.20).

This is the S1 "free wins + harness" instrument from the fp32 sprint plan
(``proofs/v020/fp32_analysis/FINAL_FP32_SPRINT_PLAN.md`` S1): a precision-invariant
auditor that (a) reports the dtype of every leaf flowing through the hot path,
(b) counts ``convert-element-type`` ops in a jitted program's HLO (split into
f32->f64 / f64->f32 / other so a *silent* promotion is visible), and (c) provides
a dtype-STABILITY check that FAILS when a traced function changes a float leaf's
precision between input and output (the f32->f64 / f64->f32 leak the later fp32
work must never re-introduce).

Nothing here changes numerics: it traces/lowers only (``jax.eval_shape`` /
``jax.jit(...).lower(...)``) and never mutates state, so it is safe to call on the
bit-identical ``fp64_default`` path.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# Per-leaf dtype reporting                                                     #
# --------------------------------------------------------------------------- #
def _leaf_paths_and_values(tree: Any) -> list[tuple[str, Any]]:
    """Return (readable_path, leaf) for every array leaf in ``tree``.

    Uses ``jax.tree_util.tree_flatten_with_path`` so a leaf's logical name
    (e.g. ``state.p_total``) is preserved in the report instead of a bare index.
    """

    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(tree)
    out: list[tuple[str, Any]] = []
    for key_path, leaf in leaves_with_path:
        out.append((jax.tree_util.keystr(key_path), leaf))
    return out


def field_dtypes(tree: Any) -> dict[str, str]:
    """Map every array leaf of ``tree`` to its dtype string (per-field report)."""

    report: dict[str, str] = {}
    for path, leaf in _leaf_paths_and_values(tree):
        dtype = getattr(leaf, "dtype", None)
        if dtype is None:
            continue
        report[path] = str(dtype)
    return report


def dtype_histogram(tree: Any) -> dict[str, int]:
    """Count leaves per dtype (a compact precision fingerprint of a carry)."""

    hist: dict[str, int] = {}
    for dtype in field_dtypes(tree).values():
        hist[dtype] = hist.get(dtype, 0) + 1
    return dict(sorted(hist.items()))


def named_dtypes(obj: Any, *, prefix: str = "") -> dict[str, str]:
    """Per-field dtype report using LOGICAL names (``state.p_total``, ...).

    State / OperationalCarry register as plain (index-keyed) pytrees, so the
    generic tree-path report shows integer indices. This reporter walks
    ``__slots__`` / dataclass fields / containers by name so the hot-path dtype
    audit is human-readable. Falls back to recursing arbitrary containers.
    """

    out: dict[str, str] = {}

    def visit(name: str, value: Any) -> None:
        if value is None:
            return
        dtype = getattr(value, "dtype", None)
        if dtype is not None and hasattr(value, "shape"):
            out[name] = str(dtype)
            return
        slots = getattr(type(value), "__slots__", None)
        if slots:
            for slot in slots:
                visit(f"{name}.{slot}" if name else slot, getattr(value, slot, None))
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for f in dataclasses.fields(value):
                visit(f"{name}.{f.name}" if name else f.name, getattr(value, f.name, None))
            return
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(f"{name}[{i}]", item)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{name}.{key}" if name else str(key), item)
            return

    visit(prefix, obj)
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# XLA convert-element-type counter (from lowered HLO text)                     #
# --------------------------------------------------------------------------- #
# StableHLO:  %0 = "stablehlo.convert"(%a) : (tensor<...xf32>) -> tensor<...xf64>
# classic HLO: %x = f64[...]{...} convert(f32[...]{...} %y)
_STABLEHLO_CONVERT = re.compile(
    r"stablehlo\.convert.*?tensor<[^>]*?x?(f16|bf16|f32|f64|f8[^>]*)>\s*\)\s*->\s*tensor<[^>]*?x?(f16|bf16|f32|f64|f8[^>]*)>"
)
_CLASSIC_CONVERT = re.compile(
    r"(f16|bf16|f32|f64)\[[^\]]*\][^=]*=\s*(f16|bf16|f32|f64)\[[^\]]*\][^c]*convert\("
)
# Generic fallback: any line with "convert" and two distinct float element types.
_FLOAT_TOKENS = ("f64", "f32", "bf16", "f16")


@dataclass(frozen=True)
class ConvertCounts:
    """Element-type convert tally from one lowered HLO module."""

    total: int
    f32_to_f64: int
    f64_to_f32: int
    other: int
    by_transition: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "f32_to_f64": self.f32_to_f64,
            "f64_to_f32": self.f64_to_f32,
            "other": self.other,
            "by_transition": dict(self.by_transition),
        }


def count_converts(hlo_text: str) -> ConvertCounts:
    """Count element-type converts in lowered HLO/StableHLO text, by transition.

    Robust to both the StableHLO textual form (``jax...lower().as_text()``) and the
    classic optimized-HLO form (``...compile().as_text()``). A line that mentions
    ``convert`` but exposes only one float type (e.g. integer index conversions) is
    tallied under ``other``.
    """

    by_transition: dict[str, int] = {}
    total = 0
    for line in hlo_text.splitlines():
        if "convert" not in line:
            continue
        src = dst = None
        m = _STABLEHLO_CONVERT.search(line)
        if m:
            src, dst = m.group(1), m.group(2)
        else:
            m = _CLASSIC_CONVERT.search(line)
            if m:
                dst, src = m.group(1), m.group(2)  # classic writes dst first
        if src is None or dst is None:
            # Generic: pull the two right-most float tokens on the line if present.
            found = [t for t in re.findall(r"f64|f32|bf16|f16", line)]
            if len(found) >= 2 and "convert" in line:
                # heuristic: last two distinct tokens are (in,out) or (out,in);
                # only count when they differ, transition direction unknown.
                a, b = found[-1], found[-2]
                if a != b:
                    src, dst = b, a
        if src is None or dst is None:
            continue
        total += 1
        key = f"{src}->{dst}"
        by_transition[key] = by_transition.get(key, 0) + 1
    f32_to_f64 = by_transition.get("f32->f64", 0)
    f64_to_f32 = by_transition.get("f64->f32", 0)
    other = total - f32_to_f64 - f64_to_f32
    return ConvertCounts(
        total=total,
        f32_to_f64=f32_to_f64,
        f64_to_f32=f64_to_f32,
        other=other,
        by_transition=dict(sorted(by_transition.items())),
    )


def count_converts_for(fn: Callable[..., Any], *args, optimized: bool = False, **kwargs) -> ConvertCounts:
    """Lower ``fn`` and count its element-type converts (HLO trace; no exec)."""

    lowered = jax.jit(fn).lower(*args, **kwargs)
    text = lowered.compile().as_text() if optimized else lowered.as_text()
    return count_converts(text)


# --------------------------------------------------------------------------- #
# dtype-stability check (catches silent f32<->f64 promotion across a function) #
# --------------------------------------------------------------------------- #
class DtypePromotionError(AssertionError):
    """Raised when a traced function changes a float leaf's precision."""


def _is_float(dtype: Any) -> bool:
    return jnp.issubdtype(jnp.dtype(dtype), jnp.floating)


def diff_float_dtypes(before: Any, after: Any) -> dict[str, tuple[str, str]]:
    """Return leaves whose FLOAT precision changed between two pytrees.

    Both trees must share structure (e.g. a scan carry in vs out). Integer/bool
    leaves are ignored (only float precision drift is a silent-promotion bug).
    """

    before_map = {p: l for p, l in _leaf_paths_and_values(before)}
    after_map = {p: l for p, l in _leaf_paths_and_values(after)}
    changed: dict[str, tuple[str, str]] = {}
    for path, b_leaf in before_map.items():
        a_leaf = after_map.get(path)
        if a_leaf is None:
            continue
        b_dt, a_dt = getattr(b_leaf, "dtype", None), getattr(a_leaf, "dtype", None)
        if b_dt is None or a_dt is None:
            continue
        if not (_is_float(b_dt) or _is_float(a_dt)):
            continue
        if jnp.dtype(b_dt) != jnp.dtype(a_dt):
            changed[path] = (str(b_dt), str(a_dt))
    return changed


def assert_dtype_stable(
    fn: Callable[[Any], Any],
    example_input: Any,
    *,
    label: str = "fn",
) -> dict[str, str]:
    """Trace ``fn(example_input)`` and FAIL on any float-precision change.

    Uses ``jax.eval_shape`` so nothing executes (cheap + side-effect free). The
    returned mapping is the per-leaf dtype of the OUTPUT (the per-field dtype
    report through the hot path). Raises :class:`DtypePromotionError` listing every
    leaf that silently promoted f32<->f64.
    """

    out = jax.eval_shape(fn, example_input)
    changed = diff_float_dtypes(example_input, out)
    if changed:
        lines = [f"  {path}: {b} -> {a}" for path, (b, a) in sorted(changed.items())]
        raise DtypePromotionError(
            f"[{label}] silent float-precision change on {len(changed)} leaf(es):\n"
            + "\n".join(lines)
        )
    return field_dtypes(out)


__all__ = [
    "ConvertCounts",
    "DtypePromotionError",
    "assert_dtype_stable",
    "count_converts",
    "count_converts_for",
    "diff_float_dtypes",
    "dtype_histogram",
    "field_dtypes",
    "named_dtypes",
]
