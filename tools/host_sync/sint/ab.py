"""SINT-kernel GPU A/B on real PROD d02 + WN3 d03 force-down operands.

1. full coupled force-down, GPUWRF_SINT_KERNEL=0 vs 1: every output leaf byte-compared;
2. per field: kernel (div rn / full) vs unfused XLA SINT+sides vs NumPy source-literal;
3. interleaved force-down wall (10 pairs) per case.
"""
import argparse
import functools
import hashlib
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--source", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--cpu-dry", action="store_true", help="harness logic check only (interpret, no claims)")
args = ap.parse_args()
sys.path[:0] = [str(args.source / "src"), str(args.source / "scripts")]

import jax
import jax.numpy as jnp

DRY = bool(args.cpu_dry)
if not DRY:
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
    assert jax.devices()[0].platform == "gpu", jax.devices()

from gpuwrf.nesting.boundary_construction import (
    _compiled_sides2d, _compiled_sides3d, _compiled_sint, build_child_boundary_package,
    build_nest_force_weights, couple_state_for_forcedown,
)
from gpuwrf.nesting import sint_kernel as _sk
from gpuwrf.nesting.sint_kernel import sint_sides as _sint_sides

if DRY:

    _sk.sint_kernel_enabled = lambda: os.environ.get("GPUWRF_SINT_KERNEL", "1") == "1"
    import gpuwrf.nesting.boundary_construction as _bc
    _bc.sint_kernel_enabled = _sk.sint_kernel_enabled
    _bc.sint_sides = functools.partial(_sint_sides, interpret=True)
sint_sides = functools.partial(_sint_sides, interpret=True) if DRY else _sint_sides
from gpuwrf.runtime.domain_tree import DomainTree
from gpuwrf.runtime.operational_mode import _initial_carry_for_run
from v0234_wrf_sint_source_oracle import sint_full, wrf_sides

SPECIES = ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr")
try:  # B36 stack: live children carry the seven scalar records
    from gpuwrf.nesting.boundary_construction import initialize_child_scalar_boundaries as _seed
except ImportError:
    _seed = None
cases = []
with Path("<USER_HOME>/wrf_gpu2_lanes/integrate/diagnose/common_inputs.pkl").open("rb") as fh:
    h, b, _, _, dts, _ = pickle.load(fh)
edge = DomainTree.from_domains(h, b).edges["d01"][0]
cases.append(dict(
    name="PROD-d02",
    parent=_initial_carry_for_run(b["d01"].state, b["d01"].namelist).state,
    child=_initial_carry_for_run(b["d02"].state, b["d02"].namelist).state,
    pm=b["d01"].namelist.metrics, cm=b["d02"].namelist.metrics, weights=edge.weights,
    start=(edge.spec.i_parent_start, edge.spec.j_parent_start),
))
rain = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/W3/ni_probe/rain01")
payload = []
for n in (2, 3):
    with (rain / f"d0{n}_first_exceed.pkl").open("rb") as fh:
        payload.append(pickle.load(fh))
cases.append(dict(
    name="WN3-d03",
    parent=_initial_carry_for_run(payload[0]["before"].state, payload[0]["namelist"]).state,
    child=_initial_carry_for_run(payload[1]["before"].state, payload[1]["namelist"]).state,
    pm=payload[0]["namelist"].metrics, cm=payload[1]["namelist"].metrics,
    weights=build_nest_force_weights(parent_grid_ratio=3, i_parent_start=92, j_parent_start=36,
                                     parent_grid=payload[0]["grid"], child_grid=payload[1]["grid"]),
    start=(92, 36),
))


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    same_shape = a.shape == b.shape and a.dtype == b.dtype
    out = dict(shape=list(a.shape), dtype=str(a.dtype), bytes_equal=bool(same_shape and a.tobytes() == b.tobytes()))
    if same_shape:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        out.update(value_equal=bool(np.array_equal(a, b)), n_diff=int(np.count_nonzero(a != b)),
                   max_abs=float(np.nanmax(diff)) if diff.size else 0.0,
                   max_ref=float(np.nanmax(np.abs(b))) if b.size else 0.0)
    return out


VARIANTS = {  # name: (GPUWRF_SINT_KERNEL, GPUWRF_FORCEDOWN_SINGLE_JIT); "main" = released path
    "main": ("0", "0"), "kernel": ("1", "0"), "single_jit": ("0", "1"), "kernel_single_jit": ("1", "1"),
}


def force(case, variant):
    os.environ["GPUWRF_SINT_KERNEL"], os.environ["GPUWRF_FORCEDOWN_SINGLE_JIT"] = VARIANTS[variant]
    child = case["child"] if _seed is None else _seed(case["child"])
    w = int(child.u_bdy.shape[2])
    out = build_child_boundary_package(
        child, case["parent"], case["weights"], bdy_width=w,
        parent_metrics=case["pm"], child_metrics=case["cm"], coupled_forcedown=True,
        parent_grid_ratio=3, _compiled_producers=True,
    )
    jax.block_until_ready(out)
    return out


report = dict(b36_species_seeded=_seed is not None, platform=jax.devices()[0].platform, device=str(jax.devices()[0]),
              affinity=sorted(os.sched_getaffinity(0)), source=str(args.source), cases=[])
for case in cases:
    row = dict(name=case["name"], variants={}, fields=[], timing={})
    ref = force(case, "main")
    rflat = jax.tree_util.tree_flatten_with_path(ref)[0]
    for variant in ("kernel", "single_jit", "kernel_single_jit"):
        new = force(case, variant)
        nflat = jax.tree_util.tree_flatten_with_path(new)[0]
        assert [p for p, _ in rflat] == [p for p, _ in nflat]
        leaves = []
        for (path, x), (_, y) in zip(rflat, nflat):
            r = compare(y, x)
            r["leaf"] = jax.tree_util.keystr(path)
            r["same_object"] = x is y
            leaves.append(r)
        row["variants"][variant] = dict(
            all_leaves_bytes_equal=all(r["bytes_equal"] for r in leaves),
            changed_leaves=[r for r in leaves if not r["bytes_equal"]],
            untouched_same_object=sum(r["same_object"] for r in leaves),
        )

    child = case["child"]
    w, side_len = int(child.u_bdy.shape[2]), int(max(child.u_bdy.shape[-1], child.v_bdy.shape[-1]))
    pf = couple_state_for_forcedown(case["parent"], case["pm"])
    p = case["parent"]
    c1 = np.asarray(case["pm"].c1h, np.float32)[:, None, None]
    c2 = np.asarray(case["pm"].c2h, np.float32)[:, None, None]
    mass32 = c1 * np.asarray(p.mu_total, np.float32)[None] + c2
    fields = [("theta", pf["theta"], case["weights"].mass, {}),
              ("qv", pf["qv"], case["weights"].mass, {}),
              ("w", pf["w"], case["weights"].mass, {}),
              ("p", p.p_perturbation, case["weights"].mass, {}),
              ("ph", pf["ph"], case["weights"].mass, {}),
              ("u", pf["u"], case["weights"].u, {"xstag": True}),
              ("v", pf["v"], case["weights"].v, {"ystag": True}),
              ("mu", p.mu_perturbation, case["weights"].mass, {})]
    fields += [(f"{n}_real4", jnp.asarray(np.asarray(getattr(p, n), np.float32) * mass32), case["weights"].mass, {})
               for n in SPECIES]
    fields += [("theta_f32", jnp.asarray(pf["theta"], jnp.float32), case["weights"].mass, {})]
    for name, field, plan, stag in fields:
        sides = _compiled_sides2d if field.ndim == 2 else _compiled_sides3d
        xla = sides(_compiled_sint(field, plan, parent_grid_ratio=3, **stag), w, side_len)
        frow = dict(field=name, dtype=str(field.dtype), active=int(np.count_nonzero(np.asarray(field))))
        for div in (("rn",) if DRY else ("rn", "full")):
            got = sint_sides(field, plan, parent_grid_ratio=3, width=w, side_len=side_len, _div=div, **stag)
            frow[f"kernel_{div}_vs_xla"] = compare(got, xla)
            if field.dtype == jnp.float32 and field.ndim == 3:
                lit = sint_full(np.asarray(field), ratio=3, i_parent_start=case["start"][0],
                                j_parent_start=case["start"][1], child_ny=int(plan.child_ny),
                                child_nx=int(plan.child_nx), xstag=bool(stag.get("xstag")),
                                ystag=bool(stag.get("ystag")))
                frow[f"kernel_{div}_vs_numpy_literal"] = compare(got, wrf_sides(lit, width=w, side_len=side_len))
                if div == "rn":
                    frow["xla_vs_numpy_literal"] = compare(xla, wrf_sides(lit, width=w, side_len=side_len))
        row["fields"].append(frow)

    samples = {v: [] for v in VARIANTS}
    order = list(VARIANTS)
    for i in range(1 if DRY else 10):
        for variant in (order if i % 2 == 0 else order[::-1]):
            t0 = time.perf_counter()
            force(case, variant)
            samples[variant].append(time.perf_counter() - t0)
    row["timing"] = {k: dict(median_s=statistics.median(v), samples=v) for k, v in samples.items()}
    report["cases"].append(row)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(name=row["name"],
                          variants={k: (v["all_leaves_bytes_equal"], [(r["leaf"], r.get("n_diff"), r.get("max_abs")) for r in v["changed_leaves"]])
                                    for k, v in row["variants"].items()},
                          timing={k: v["median_s"] for k, v in row["timing"].items()},
                          fields=[(f["field"], f["dtype"], f["kernel_rn_vs_xla"]["bytes_equal"],
                                   f.get("kernel_full_vs_xla", {}).get("bytes_equal"),
                                   f.get("kernel_rn_vs_numpy_literal", {}).get("bytes_equal"),
                                   f.get("xla_vs_numpy_literal", {}).get("bytes_equal")) for f in row["fields"]])),
          flush=True)
report["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
args.out.write_text(json.dumps(report, indent=2) + "\n")
