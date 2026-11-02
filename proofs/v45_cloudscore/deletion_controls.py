"""Literal scorer mutations must fail existing tests by assertion, never error.

All mutated modules live in memory. Immutable snapshot source is never edited.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gpuwrf.optimize import cloud, continuous
import gpuwrf.optimize as optimize_package


def root_controls(args, root, controls):
    """Changed admission guards under the real ROOT conftest, without editing source."""
    child = r'''
import importlib,json,pathlib,sys,types
spec=json.loads(sys.argv[1])
sys.path.insert(0,str(pathlib.Path(spec['root'])/'scripts/v025'))
import cpu_guard  # before gpuwrf/JAX, exactly as ROOT conftest requires
import gpuwrf.optimize as package
if spec['mode']=='mutant':
 reference=importlib.import_module('gpuwrf.optimize.'+spec['target'])
 source=pathlib.Path(reference.__file__).read_text()
 assert source.count(spec['old'])==1
 module=types.ModuleType('gpuwrf.optimize._correlation_mutant')
 module.__package__='gpuwrf.optimize';module.__file__=reference.__file__
 sys.modules[module.__name__]=module
 exec(compile(source.replace(spec['old'],spec['new'],1),module.__file__,'exec'),module.__dict__)
 sys.modules['gpuwrf.optimize.'+spec['target']]=module;setattr(package,spec['target'],module)
import pytest
raise SystemExit(pytest.main([spec['selector'],'-q','--basetemp',spec['basetemp'],'--junitxml',spec['xml']]))
'''
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    results, baselines = [], set()
    for control in controls:
        label, old, new, name, _ = control[:5]
        name = args.selector or name
        target = control[5] if len(control) > 5 else "cloud"
        modes = ["mutant"] if name in baselines else ["baseline", "mutant"]
        for mode in modes:
            stem = scratch/(label+"_"+mode)
            spec = {"mode": mode, "target": target, "old": old, "new": new, "root": str(root),
                    "selector": str(root/args.test_file)+"::"+name,
                    "basetemp": str(scratch.parent/"pytest"), "xml": str(stem.with_suffix('.xml'))}
            with stem.with_suffix('.log').open('w') as log:
                process = subprocess.run([sys.executable, '-c', child, json.dumps(spec)], cwd=root,
                                         stdout=log, stderr=subprocess.STDOUT, timeout=60)
            suites = ET.parse(stem.with_suffix('.xml')).getroot().findall('.//testsuite')
            counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
            passed = counts == {'tests':1,'failures':0,'errors':0,'skipped':0} and process.returncode == 0
            killed = counts == {'tests':1,'failures':1,'errors':0,'skipped':0} and process.returncode == 1
            status = 'PASS' if passed else ('ASSERT' if killed else 'ERROR')
            results.append({'control':label,'mode':mode,'status':status,'counts':counts,'rc':process.returncode,
                            'xml':str(stem.with_suffix('.xml'))})
        baselines.add(name)
    payload = {'root_conftest':True,'baseline_reuse':'one exact-source baseline per selector', 'controls':results,
               'pass':all(r['status']==('PASS' if r['mode']=='baseline' else 'ASSERT') for r in results)}
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload))
    return 0 if payload['pass'] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--only", help="comma-separated changed controls; unlisted accepted controls are carried")
    parser.add_argument("--root-pytest", action="store_true", help="run selected zero-argument tests under ROOT pytest")
    parser.add_argument("--test-file", default="tests/test_optimize_cloudscore.py", help="ROOT test module for a composed guard check")
    parser.add_argument("--selector", help="ROOT function/parameter selector for a composed guard check")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("cloud_controls_tests", root/"tests/test_optimize_cloudscore.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    controls = [
        ("finite", "if not np.isfinite(out).all():", "if False:",
         "test_nonfinite_column_rejected_even_one_cell", {"bad": np.nan, "field": "QCLOUD"}),
        ("night", "valid = (obs != 255) & day & (area[None, :] > 0)",
         "valid = (obs != 255) & (area[None, :] > 0)", "test_invalid_night_uncovered_and_pixel_region_counts", {}),
        ("invalid255", "valid = (obs != 255) & day & (area[None, :] > 0)",
         "valid = day & (area[None, :] > 0)", "test_invalid_night_uncovered_and_pixel_region_counts", {}),
        ("common90", "if day[j].any() and coverage < config[\"min_common_area\"]:", "if False:",
         "test_exact_match_no_nearest_and_common_area_floor", {}),
        ("inclusive_native", "columns >= config[\"threshold_g_m2\"]", "columns > config[\"threshold_g_m2\"]",
         "test_native_cwp_threshold_inclusive", {}),
        ("inclusive_pixel", "yes, pred = obs == 1, values >= pixel_threshold", "yes, pred = obs == 1, values > pixel_threshold",
         "test_invalid_night_uncovered_and_pixel_region_counts", {}),
        ("dry_density", "rho = p/(287.*temperature*(1+vapor/(287./461.6)))",
         "rho = p*(1+vapor)/(287.*temperature*(1+vapor/(287./461.6)))", "test_dry_density_not_moist_density", {}),
        ("cwp_heat_capacity", "temperature = theta*(p/100000.)**R_D_OVER_CP",
         "temperature = theta*(p/100000.)**(287./1004.)", "test_cwp_density_uses_original_wrf_heat_capacity", {}),
        ("protected", 'refuse_protected_inputs("cloud truth (stage ALISIOS facade)", paths)', "pass",
         "test_protected_truth_refused_before_payload_open", {}),
        ("frozen_day", "if time.tz is None or time.value % 10**9", "if time.value % 10**9",
         "test_frozen_day_and_area_contract_refusals", {}),
        ("fss_mask", "matrix@(ap*av*active_area)", "matrix@(ap*active_area)",
         "test_fss_excludes_invalid_neighbours_and_undefined_centroid", {}),
        ("ensemble_footprint", "if footprint is not None and current_footprint != footprint:", "if False:",
         "test_shared_footprint_identity_required_even_with_same_pixel_support", {}),
        ("discrete_footprint", "if any(f != footprints[0] for f in footprints):", "if False:",
         "test_discrete_cloud_study_load_and_real_fetch_wiring", {}),
        ("exact_daytime_support", 'if any(truth["daylight"][s].any() and slots[s] not in records for s in eligible):',
         "if False:", "test_exact_match_no_nearest_and_common_area_floor", {}),
        ("correlated_sigma", "np.full(observations.shape, sigma)", "np.full(observations.shape, sigma/np.sqrt(2))",
         "test_superobservations_area_weight_both_sides_and_keep_sigma", {}),
        ("superobs_truth_area", "weights=weights*obs", "weights=obs",
         "test_superobservation_known_area_weights_and_region_partition", {}),
        ("legacy_total_cap", "if count > LEGACY_MAX_OBSERVATIONS and not available:", "if False:",
         "test_continuous_cloud_load_and_driver_wiring", {"observations": 1100}),
        ("native_correlation", 'if operator == "native":', "if False:",
         "test_memory_capability_does_not_admit_native_correlation", {}),
        ("direct_correlation", '    require_correlation_contract(config, operator)\n', "",
         "test_memory_capability_does_not_admit_native_correlation", {}),
        ("pre_payload_correlation", '        require_correlation_contract(None, operator)  # before truth/model payloads or forecasts\n', "",
         "test_memory_capability_does_not_admit_native_correlation", {}, "continuous"),
        ("pre_prefetch_correlation", '        require_correlation_contract(cloud_scorer.get("cloud") or {}, operator)  # before any member prefetch\n', "",
         "test_memory_capability_does_not_admit_native_correlation", {}, "continuous"),
    ]
    if args.only:
        selected = set(args.only.split(','))
        controls = [c for c in controls if c[0] in selected]
        if {c[0] for c in controls} != selected:
            raise ValueError("unknown selected deletion control")
    if args.root_pytest:
        if not (root/args.test_file).resolve().is_relative_to((root/"tests").resolve()):
            raise ValueError("ROOT deletion test must be inside the immutable tests tree")
        return root_controls(args, root, controls)
    results = []
    Path(args.scratch).mkdir(parents=True, exist_ok=True)
    for control in controls:
        label, old, new, name, kwargs = control[:5]
        target = control[5] if len(control) > 5 else "cloud"
        reference = cloud if target == "cloud" else continuous
        source = Path(reference.__file__).read_text()
        if source.count(old) != 1:
            raise ValueError(f"{label}: literal mutation must hit exactly once")
        module = types.ModuleType("gpuwrf.optimize._cloud_"+label)
        module.__package__ = "gpuwrf.optimize"
        module.__file__ = reference.__file__
        sys.modules[module.__name__] = module
        exec(compile(source.replace(old, new, 1), str(reference.__file__), "exec"), module.__dict__)
        for mode, implementation in (("baseline", reference), ("mutant", module)):
            suite.cloud = implementation if target == "cloud" else cloud
            function = getattr(suite, name)
            with tempfile.TemporaryDirectory(dir=args.scratch) as directory, pytest.MonkeyPatch.context() as patch:
                patch.setitem(sys.modules, "gpuwrf.optimize."+target, implementation)
                patch.setattr(optimize_package, target, implementation)
                values = dict(kwargs)
                if "tmp_path" in function.__code__.co_varnames[:function.__code__.co_argcount]:
                    values["tmp_path"] = Path(directory)
                if "monkeypatch" in function.__code__.co_varnames[:function.__code__.co_argcount]:
                    values["monkeypatch"] = patch
                try:
                    function(**values)
                except (AssertionError, pytest.fail.Exception) as exc:
                    status, detail = "ASSERT", type(exc).__name__
                except BaseException as exc:
                    status, detail = "ERROR", type(exc).__name__
                else:
                    status, detail = "PASS", ""
            results.append({"control": label, "mode": mode, "status": status, "detail": detail})
        del sys.modules[module.__name__]
    payload = {"controls": results, "pass": all(r["status"] == ("PASS" if r["mode"] == "baseline" else "ASSERT") for r in results)}
    with open(args.output, "x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps(payload))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
