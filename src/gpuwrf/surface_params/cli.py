"""`gpuwrf surface-params` — Track P controls and the EKI scaffold (experimental, opt-in).

Backend-dark registration. Subcommands:
  basis   load a case once (CPU is fine) and write the float64 table-value basis of every control per domain
  expand  multipliers (uniform or per vegetation category) -> a validated control file for `gpuwrf run
          --surface-controls` (NumPy only)
  eki     init / update an ensemble of log-multiplier parameters (host fp64) and write one control file per member
  osse    offline Noah-MP twin experiment: EKI recovers a known roughness truth (CPU, minutes)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BASIS_SCHEMA = "gpuwrf-surface-basis-v1"


def register(subparsers):
    top = subparsers.add_parser("surface-params", help="EXPERIMENTAL Track P: traced Noah-MP surface parameters + EKI")
    actions = top.add_subparsers(dest="sp_action", required=True)
    basis = actions.add_parser("basis", help="write the per-domain table-value basis of every surface control")
    basis.add_argument("--input-dir", type=Path, required=True)
    basis.add_argument("--max-dom", type=int, default=None)
    basis.add_argument("--output", type=Path, required=True)
    basis.set_defaults(func=command)
    expand = actions.add_parser("expand", help="multipliers -> validated control file (no model import)")
    expand.add_argument("--basis", type=Path, required=True)
    expand.add_argument("--output", type=Path, required=True)
    expand.add_argument("--set", action="append", default=[], metavar="[DOM:]NAME=FACTOR",
                        help="uniform multiplier, e.g. z0mvt=1.2 or d02:albsat=0.9 (repeatable)")
    expand.add_argument("--class", dest="classes", action="append", default=[], metavar="[DOM:]NAME:CLASS=FACTOR",
                        help="1-based category: IVGTYP for vegetation, ISLTYP for soil; e.g. z0mvt:5=1.5 or d03:z0bare:16=10 (EXPERIMENTAL, not WRF-faithful; repeatable)")
    expand.set_defaults(func=command)
    eki = actions.add_parser("eki", help="ensemble Kalman inversion scaffold (host fp64)")
    steps = eki.add_subparsers(dest="eki_action", required=True)
    init = steps.add_parser("init", help="draw the prior ensemble and write member control files")
    init.add_argument("--spec", type=Path, required=True, help="JSON parameter spec (see docs/surface_params.md)")
    init.add_argument("--basis", type=Path, required=True)
    init.add_argument("--members", type=int, required=True)
    init.add_argument("--seed", type=int, required=True)
    init.add_argument("--output-dir", type=Path, required=True)
    init.set_defaults(func=command)
    upd = steps.add_parser("update", help="one (tempered) TEKI step from member predictions and observations")
    upd.add_argument("--state", type=Path, required=True)
    upd.add_argument("--predictions", type=Path, required=True, help=".npy (members x obs)")
    upd.add_argument("--obs", type=Path, required=True, help=".npz with y and sigma (same obs order)")
    upd.add_argument("--dt", type=float, default=1.0)
    upd.add_argument("--output-dir", type=Path, required=True)
    upd.set_defaults(func=command)
    col = steps.add_parser("collect", help="member wrfouts + station observations -> predictions/obs matrices")
    col.add_argument("--member-dir", action="append", required=True, type=Path,
                     help="one member's output directory, in ensemble order (repeat)")
    col.add_argument("--domain", default="d01")
    col.add_argument("--stations", type=Path, required=True)
    col.add_argument("--obs", type=Path, required=True)
    col.add_argument("--obs-format", choices=["csv", "alisios"], default="csv")
    col.add_argument("--variables", default="t2,wind")
    col.add_argument("--sigma", action="append", default=[], metavar="VAR=VALUE",
                     help="observation error (representation + instrument) per variable, e.g. t2=1.0 wind=1.5")
    col.add_argument("--wind-frame", choices=["grid", "earth"], default="earth")
    col.add_argument("--alisios-issue-days", default=None, help="YYYYMMDD,... (required for ALISIOS observations)")
    col.add_argument("--alisios-valid-days", default=None, help="YYYYMMDD,... days the observation file covers")
    col.add_argument("--guard-receipt-dir", type=Path, default=None)
    col.add_argument("--output-dir", type=Path, required=True)
    col.set_defaults(func=command)
    osse = actions.add_parser("osse", help="offline Noah-MP twin experiment (CPU)")
    osse.add_argument("--members", type=int, default=16)
    osse.add_argument("--iterations", type=int, default=4)
    osse.add_argument("--output", type=Path, default=None)
    osse.set_defaults(func=command)


def command(args) -> int:
    try:
        if args.sp_action == "basis":
            return _basis(args)
        if args.sp_action == "expand":
            return _expand(args)
        if args.sp_action == "eki":
            return {"init": _eki_init, "update": _eki_update, "collect": _eki_collect}[args.eki_action](args)
        return _osse(args)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"gpuwrf surface-params: error: {exc}", file=sys.stderr)
        return 2


def _new(path: Path) -> Path:
    if Path(path).exists():
        raise FileExistsError(f"{path} exists; use a new destination")
    return Path(path)


# ----------------------------------------------------------------------------- basis
def _basis(args) -> int:
    import tempfile

    out = _new(args.output)
    from gpuwrf.assimilation.letkf.member_driver import (
        ProtectedInputError, refuse_protected_inputs, refuse_protected_opens)
    from gpuwrf.integration import nested_pipeline as pipeline

    # fail closed BEFORE any read: the case dir (directory aliases followed cycle-safely, namelist-named files) and
    # every runtime root the load reads through the runtime's own resolvers (WRF tables, asset/compile caches, env
    # paths) -- the da-cycle preflight, one implementation; the open-layer backstop covers anything not found statically
    refuse_protected_inputs("surface-params basis", [args.input_dir], runtime=True)
    max_dom = args.max_dom
    if max_dom is None:
        from gpuwrf.cli import _namelist_max_dom
        max_dom = _namelist_max_dom(Path(args.input_dir) / "namelist.input")
    names = pipeline.domain_names_for(int(max_dom))
    with tempfile.TemporaryDirectory() as tmp, refuse_protected_opens() as refused:
        cfg = pipeline.NestedPipelineConfig(Path(args.input_dir), Path(tmp) / "out", Path(tmp) / "proof", hours=1,
                                            max_dom=int(max_dom))
        _h, bundles, _m, _rs, _dt, _c = pipeline._load_domains(cfg, names)
    if refused:  # a reader swallowed the refusal: still no basis from a partly protected read
        raise ProtectedInputError(f"surface-params basis: refused ALISIOS opens during the load: {sorted(set(refused))[:3]}")
    manifest = write_basis({name: bundles[name].namelist for name in names}, out, input_dir=args.input_dir,
                           land_states={name: carry.noahmp_land for name, carry in _c.items()})
    print(json.dumps({"basis": str(out), "domains": {d: sorted(c for c, v in e["controls"].items() if v["active"])
                                                     for d, e in manifest["domains"].items()}}))
    return 0


def write_basis(namelists: dict, out: Path, *, input_dir=None, land_states=None) -> dict:
    """Float64 table basis + consumer dtype/activity per control for every Noah-MP domain."""
    from gpuwrf.surface_params import controls as sc

    arrays, manifest = {}, {"schema": BASIS_SCHEMA, "input_dir": None if input_dir is None else
                            str(Path(input_dir).resolve()), "domains": {}}
    for name, nl in namelists.items():
        if land_states is not None and name in land_states:
            from gpuwrf.surface_params.lsmparams import validation_namelist
            nl = validation_namelist(nl, land_states[name])
        noah = bool(getattr(nl, "use_noahmp", False))
        if not noah and sc._topo_reason(nl) is not None:
            continue
        entry = {"controls": {}, "grid_shape": list(sc.grid_shape(nl))}
        if noah:
            arrays[f"{name}/_ivgtyp"] = np.asarray(nl.noahmp_static.ivgtyp, np.int32)
            arrays[f"{name}/_isltyp"] = np.asarray(nl.noahmp_static.isltyp, np.int32)
            from gpuwrf.surface_params.lsmparams import active_columns
            arrays[f"{name}/_w2_active"] = active_columns(nl)
            entry["soil_class_count"] = len(nl.noahmp_static.parameters.bexp) - 1
            from gpuwrf.physics.noahmp.noahmp_driver import _urban_soil_on
            entry["urban_category"] = int(nl.noahmp_static.parameters.isurban) if _urban_soil_on() else None
            for key in ("taul", "taus"):
                arrays[f"{name}/_{key}"] = np.asarray(getattr(nl.noahmp_rad_params, key), np.float64)
        # Traced controls of a Noah-MP domain (inactive ones listed with the reason); host controls only when active.
        listed = (sorted({**sc.CONTROLS, **sc.LSM_CONTROLS}) if noah else []) + [c for c in sorted(sc.HOST_CONTROLS) if sc._topo_reason(nl) is None]
        for control in listed:
            if control in sc.UNAVAILABLE_SURFACE_CONTROLS:
                entry["controls"][control] = {"active": False,
                    "reason": "not available in v0.4.4: failed the GPU qualification; see KNOWN_ISSUES"}
                continue
            reason = sc._inactive_reason(control, nl)
            if reason:
                entry["controls"][control] = {"active": False, "reason": reason}
                continue
            base = sc.table_base(control, nl)
            dtype = sc.target_dtype(control, nl)
            prior = sc.expand_controls(nl, {control: 1.0}, device=False)[control]
            if not np.array_equal(base.astype(dtype), prior):
                raise ValueError(f"{name}/{control}: table basis does not reproduce the bound value")
            arrays[f"{name}/{control}"] = base
            entry["controls"][control] = {"active": True, "dtype": str(dtype), "shape": list(base.shape),
                                          "prior_digest": sc.controls_digest({control: prior})}
        manifest["domains"][name] = entry
    if not manifest["domains"]:
        raise ValueError("no controllable domain in this case (Noah-MP sf_surface_physics=4, or topo_wind 1/2 with "
                         "YSU/Shin-Hong for ctopo)")
    arrays["manifest"] = np.frombuffer(json.dumps(manifest, sort_keys=True).encode(), dtype=np.uint8)
    with open(out, "wb") as handle:
        np.savez(handle, **arrays)
    return manifest


def load_basis(path):
    data = np.load(Path(path), allow_pickle=False)
    manifest = json.loads(bytes(data["manifest"]).decode())
    if manifest.get("schema") != BASIS_SCHEMA:
        raise ValueError(f"{path}: not a surface-params basis file")
    return data, manifest


def expand_from_basis(basis, manifest, multipliers: dict[str, dict[str, object]]) -> dict[str, dict[str, np.ndarray]]:
    """{domain: {control: scalar | {VEGTYP: factor} | field}} -> validated fields (NumPy, table-edit exact).

    Host controls (ctopo) take a scalar/field factor (or [(None, factor), ...]) of the excess form."""
    from gpuwrf.surface_params.controls import ALL_CONTROLS, HOST_CONTROLS, LSM_CONTROLS, _LEADING, _TRANSMITTANCE, excess_scale
    from gpuwrf.surface_params.controls import refuse_unqualified_controls
    refuse_unqualified_controls({name for controls in multipliers.values() for name in controls})

    out: dict[str, dict[str, np.ndarray]] = {}
    for domain, controls in multipliers.items():
        entry = manifest["domains"].get(domain)
        if entry is None:
            raise ValueError(f"basis has no controllable domain {domain}")
        out[domain] = {}
        for name, mult in controls.items():
            info = entry["controls"].get(name)
            if info is None or not info["active"]:
                raise ValueError(f"{domain}/{name}: not an active control ({(info or {}).get('reason', 'unknown')})")
            base = basis[f"{domain}/{name}"]
            spec = ALL_CONTROLS[name]
            if name in HOST_CONTROLS:
                if isinstance(mult, list):  # member_multipliers terms: only domain-wide factors apply
                    if any(classes is not None for classes, _ in mult):
                        raise ValueError(f"{domain}/{name}: per-vegetation-class factors do not apply")
                    mult = float(np.prod([factor for _, factor in mult]))
                if isinstance(mult, dict):
                    raise ValueError(f"{domain}/{name}: per-vegetation-class factors do not apply")
                value = excess_scale(base, np.broadcast_to(np.asarray(mult, np.float64), base.shape))
                value = value.astype(np.dtype(info["dtype"]))
                if not np.all(np.isfinite(value)) or value.min() < spec.lower or value.max() > spec.upper:
                    raise ValueError(f"{domain}/{name}: values outside [{spec.lower}, {spec.upper}] {spec.units}")
                out[domain][name] = value
                continue
            axis = "_isltyp" if spec.axis == "soil" else "_ivgtyp"
            vt = basis[f"{domain}/{axis}"]
            if isinstance(mult, dict):  # {VEGTYP: factor}
                mult = [((int(cat),), float(factor)) for cat, factor in mult.items()]
            if isinstance(mult, list):  # [(classes | None, factor), ...] multiplied together
                m2 = np.ones(vt.shape)
                for classes, factor in mult:
                    if spec.axis == "soil" and classes is not None and any(
                            not 1 <= c <= entry["soil_class_count"] for c in classes):
                        raise ValueError(f"{domain}/{name}: soil class must be 1-based (slot 0 is unused)")
                    m2 = m2 * np.where(np.ones(vt.shape, bool) if classes is None else np.isin(vt, classes),
                                       float(factor), 1.0)
                field = np.broadcast_to(m2, base.shape)
            else:
                field = np.broadcast_to(np.asarray(mult, np.float64), base.shape)
            value = (base * field).astype(np.dtype(info["dtype"]))
            active = basis[f"{domain}/_w2_active"] if name in LSM_CONTROLS else None
            if active is not None:
                value = np.where(active, value, base.astype(value.dtype))
            if name == "smcmax" and entry.get("urban_category") is not None:
                vegetation = basis[f"{domain}/_ivgtyp"]
                urban = (vegetation == entry["urban_category"]) | ((vegetation >= 51) & (vegetation <= 61))
                value = np.where(urban, np.asarray(.45, value.dtype), value)
            if value.shape != tuple(_LEADING[spec.axis]) + tuple(entry["grid_shape"]):
                raise ValueError(f"{domain}/{name}: shape {value.shape}")
            checked = value if active is None else value[np.broadcast_to(active, value.shape)]
            if not np.all(np.isfinite(value)) or checked.size and (checked.min() < spec.lower or checked.max() > spec.upper):
                raise ValueError(f"{domain}/{name}: values outside [{spec.lower}, {spec.upper}] {spec.units}")
            if name in _TRANSMITTANCE and np.any(value + basis[f"{domain}/_{_TRANSMITTANCE[name]}"] > 1.0):
                raise ValueError(f"{domain}/{name}: reflectance + transmittance > 1")
            out[domain][name] = value
    return out


def _parse_multipliers(sets, classes, default_domain="d01"):
    mult: dict[str, dict[str, object]] = {}

    def split(text):
        if ":" in text.split("=")[0] and text.split(":")[0].startswith("d") and text.split(":")[0][1:].isdigit():
            dom, rest = text.split(":", 1)
            return dom, rest
        return default_domain, text

    for item in sets:
        dom, rest = split(item)
        name, value = rest.split("=")
        mult.setdefault(dom, {})[name] = float(value)
    for item in classes:
        dom, rest = split(item)
        left, value = rest.split("=")
        name, cat = left.split(":")
        current = mult.setdefault(dom, {}).setdefault(name, {})
        if not isinstance(current, dict):
            raise ValueError(f"{dom}/{name}: both uniform and per-class multipliers given")
        current[int(cat)] = float(value)
    if not mult:
        raise ValueError("no multipliers given (--set/--class)")
    return mult


def _expand(args) -> int:
    from gpuwrf.surface_params.controls import save_controls

    out = _new(args.output)
    basis, manifest = load_basis(args.basis)
    mult = _parse_multipliers(args.set, args.classes)
    fields = expand_from_basis(basis, manifest, mult)
    saved = save_controls(out, fields, provenance={"basis": str(Path(args.basis).resolve()),
                                                   "multipliers": json.loads(json.dumps(mult, default=str))})
    print(json.dumps({"controls": str(out), "domains": saved["domains"]}, default=str))
    return 0


# ----------------------------------------------------------------------------- EKI
def _spec_problem(spec_path, obs=None):
    spec = json.loads(Path(spec_path).read_text())
    params = spec["parameters"]
    from gpuwrf.surface_params.controls import refuse_unqualified_controls
    refuse_unqualified_controls({q["control"] for q in params})
    from gpuwrf.surface_params import eki
    p = len(params)
    prior_sd = np.array([float(q.get("prior_sd_log", 0.5)) for q in params])
    lower = np.log([float(q.get("bounds", [0.2, 5.0])[0]) for q in params])
    upper = np.log([float(q.get("bounds", [0.2, 5.0])[1]) for q in params])
    y = np.zeros(1) if obs is None else obs["y"]
    sigma = np.ones(1) if obs is None else obs["sigma"]
    problem = eki.EKIProblem(names=tuple(q["name"] for q in params), prior_mean=np.zeros(p),
                             prior_cov=np.diag(prior_sd ** 2), obs=np.asarray(y, np.float64),
                             obs_cov=np.asarray(sigma, np.float64) ** 2, lower=lower, upper=upper,
                             tikhonov=bool(spec.get("tikhonov", True)))
    return spec, problem


def member_multipliers(spec, theta_row):
    """theta (p,) -> {domain: {control: [(classes | None, factor), ...]}} (terms multiply where they overlap)."""
    from gpuwrf.surface_params.controls import refuse_unqualified_controls
    refuse_unqualified_controls({q["control"] for q in spec["parameters"]})
    mult: dict[str, dict[str, list]] = {}
    for q, value in zip(spec["parameters"], np.exp(np.asarray(theta_row, np.float64)), strict=True):
        classes = None if q.get("classes", "all") == "all" else tuple(int(c) for c in q["classes"])
        for dom in q.get("domains", ["d01"]):
            mult.setdefault(dom, {}).setdefault(q["control"], []).append((classes, float(value)))
    return mult


def _write_members(spec, theta, basis_path, out_dir: Path, iteration: int):
    from gpuwrf.surface_params.controls import save_controls, refuse_unqualified_controls
    refuse_unqualified_controls({q["control"] for q in spec["parameters"]})
    basis, manifest = load_basis(basis_path)
    paths = []
    for j, row in enumerate(theta):
        mult = member_multipliers(spec, row)
        path = out_dir / f"member_{j:03d}.controls.npz"
        save_controls(path, expand_from_basis(basis, manifest, mult),
                      provenance={"iteration": iteration, "member": j, "theta": list(map(float, row))})
        paths.append(str(path))
    return paths


def _save_state(out_dir: Path, spec_path, basis_path, state, iteration, paths):
    record = {"spec": str(Path(spec_path).resolve()), "basis": str(Path(basis_path).resolve()),
              "iteration": iteration, "theta": state.theta.tolist(), "history": state.history,
              "member_controls": paths}
    (out_dir / "eki_state.json").write_text(json.dumps(record, indent=1))
    return record


def _eki_init(args) -> int:
    from gpuwrf.surface_params import eki
    out_dir = Path(args.output_dir)
    spec, problem = _spec_problem(args.spec)
    out_dir.mkdir(parents=True, exist_ok=False)
    state = eki.initial_ensemble(problem, args.members, seed=args.seed)
    paths = _write_members(spec, state.theta, args.basis, out_dir, 0)
    _save_state(out_dir, args.spec, args.basis, state, 0, paths)
    print(json.dumps({"state": str(out_dir / "eki_state.json"), "members": len(paths)}))
    return 0


def _eki_update(args) -> int:
    from gpuwrf.surface_params import eki
    prev = json.loads(Path(args.state).read_text())
    obs = np.load(args.obs)
    spec, problem = _spec_problem(prev["spec"], obs=obs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    state = eki.EKIState(theta=np.asarray(prev["theta"], np.float64), iteration=int(prev["iteration"]),
                         history=list(prev["history"]))
    new = eki.update(problem, state, np.load(args.predictions), dt=args.dt, seed=1000 + state.iteration)
    paths = _write_members(spec, new.theta, prev["basis"], out_dir, new.iteration)
    _save_state(out_dir, prev["spec"], prev["basis"], new, new.iteration, paths)
    print(json.dumps({"state": str(out_dir / "eki_state.json"), "misfit": new.history[-1]["normalised_misfit"],
                      "theta_mean": new.theta.mean(0).tolist()}))
    return 0


def _eki_collect(args) -> int:
    from gpuwrf.surface_params.stations import collect

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    members = [sorted(Path(d).glob(f"wrfout_{args.domain}_*")) for d in args.member_dir]
    if any(not m for m in members):
        raise ValueError("a member directory has no wrfout files for the domain")
    sigma = {k: float(v) for k, v in (item.split("=") for item in args.sigma)}
    guard = None
    if args.alisios_issue_days:
        guard = {"issue_days": args.alisios_issue_days.split(","),
                 "valid_days": (args.alisios_valid_days or args.alisios_issue_days).split(","),
                 "receipt_dir": str(args.guard_receipt_dir or out_dir / "guard")}
    pred, y, sig, keys = collect(members, args.stations, args.obs, variables=tuple(args.variables.split(",")),
                                 sigma=sigma, obs_format=args.obs_format, guard=guard, wind_frame=args.wind_frame)
    np.save(out_dir / "predictions.npy", pred)
    np.savez(out_dir / "obs.npz", y=y, sigma=sig)
    (out_dir / "keys.json").write_text(json.dumps(keys))
    if guard is not None and guard.get("receipt") is not None:
        (out_dir / "guard_receipt.json").write_text(json.dumps(guard["receipt"], default=str, indent=1))
    print(json.dumps({"members": int(pred.shape[0]), "observations": int(pred.shape[1]),
                      "output": str(out_dir)}))
    return 0


def _osse(args) -> int:
    from gpuwrf.surface_params.osse import TwinConfig, run_twin
    result = run_twin(TwinConfig(members=args.members, iterations=args.iterations))
    text = json.dumps(result, indent=1)
    if args.output is not None:
        _new(args.output).write_text(text)
    print(text)
    return 0


def main_args(argv) -> int:
    """Run ``gpuwrf surface-params <argv>`` without the full CLI (tests, scripts)."""
    import argparse

    parser = argparse.ArgumentParser(prog="gpuwrf")
    register(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["surface-params", *argv])
    return int(args.func(args))
