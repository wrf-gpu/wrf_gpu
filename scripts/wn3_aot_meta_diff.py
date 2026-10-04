"""Diff the static (treedef node_data) content of two AOT .meta files, leaf by leaf.

Usage (CPU, PYTHONPATH=<tree>/src): meta_diff.py A.meta B.meta
Descends into dataclasses, NamedTuples, dicts, tuples, __slots__ objects and
_StaticHolder.value; arrays compared by content. Prints every differing path.
"""
from __future__ import annotations

import dataclasses
import pickle
import sys

import numpy as np


def _arr(x):
    return isinstance(x, np.ndarray) or (hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "__array__"))


def walk(x, y, path, out, seen):
    if id(x) in seen:
        return
    seen.add(id(x))
    if x is None or y is None:
        if (x is None) != (y is None):
            out.append((path, "None-vs-value"))
        return
    if type(x) is not type(y):
        out.append((path, f"type {type(x).__name__} vs {type(y).__name__}"))
        return
    if _arr(x):
        a, b = np.asarray(x), np.asarray(y)
        if a.shape != b.shape or a.dtype != b.dtype:
            out.append((path, f"shape/dtype {a.shape}{a.dtype} vs {b.shape}{b.dtype}"))
        elif not np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"):
            out.append((path, f"array {a.shape} {a.dtype} ndiff={int(np.count_nonzero(a != b))}"))
        return
    if isinstance(x, (str, bytes, int, float, bool, complex)):
        if x != y:
            out.append((path, f"{x!r:.100} vs {y!r:.100}"))
        return
    if type(x).__name__ == "_StaticHolder":
        walk(x.value, y.value, path + ".value", out, seen)
        return
    if dataclasses.is_dataclass(x):
        for f in dataclasses.fields(x):
            walk(getattr(x, f.name), getattr(y, f.name), f"{path}.{f.name}", out, seen)
        return
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        for k in x._fields:
            walk(getattr(x, k), getattr(y, k), f"{path}.{k}", out, seen)
        return
    if isinstance(x, (tuple, list)):
        if len(x) != len(y):
            out.append((path, f"len {len(x)} vs {len(y)}"))
            return
        for i, (p, q) in enumerate(zip(x, y)):
            walk(p, q, f"{path}[{i}]", out, seen)
        return
    if isinstance(x, dict):
        for k in sorted(set(x) | set(y), key=repr):
            walk(x.get(k), y.get(k), f"{path}[{k!r}]", out, seen)
        return
    slots = [s for c in type(x).__mro__ for s in getattr(c, "__slots__", ()) if s != "__weakref__"]
    if slots:
        for s in slots:
            walk(getattr(x, s, None), getattr(y, s, None), f"{path}.{s}", out, seen)
        return
    if hasattr(x, "__dict__"):
        for k in sorted(set(vars(x)) | set(vars(y))):
            walk(vars(x).get(k), vars(y).get(k), f"{path}.{k}", out, seen)
        return
    if x != y:
        out.append((path, f"{type(x).__name__}: {x!r:.80} vs {y!r:.80}"))


def walk_tree(ta, tb, path, out):
    na, nb = ta.node_data(), tb.node_data()
    if (na is None) != (nb is None):
        out.append((path, "node None mismatch"))
        return
    if na is not None:
        name = getattr(na[0], "__name__", str(na[0]))
        walk(na[1], nb[1], f"{path}:{name}", out, set())
    ca, cb = ta.children(), tb.children()
    if len(ca) != len(cb):
        out.append((path, f"children {len(ca)} vs {len(cb)}"))
        return
    for i, (p, q) in enumerate(zip(ca, cb)):
        walk_tree(p, q, f"{path}/{i}", out)


def main():
    a = pickle.load(open(sys.argv[1], "rb"))
    b = pickle.load(open(sys.argv[2], "rb"))
    out = []
    for k in ("in_tree", "out_tree"):
        walk_tree(getattr(a, k), getattr(b, k), k, out)
    for k in ("in_avals", "kept_var_idx", "key_schema", "trace_metadata", "fingerprint"):
        x, y = getattr(a, k, None), getattr(b, k, None)
        if repr(x) != repr(y):
            out.append((k, f"{repr(x)[:150]} vs {repr(y)[:150]}"))
    for row in out:
        print(*row, sep="  |  ")
    print(f"{len(out)} differing paths")


if __name__ == "__main__":
    main()
