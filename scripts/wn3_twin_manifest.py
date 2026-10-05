"""Build the GPU twin manifest for alisios R32-01 (compare_gpu_cpu_R32_01_v2.py schema). CPU only, read-only inputs.

Usage: twin_manifest.py ISSUE ARM_OUT THROUGH_TAU SNAPSHOT OUT_JSON [--version-tag TAG]
  ISSUE e.g. 20260227_18z; ARM_OUT = wn3 forecast out dir (receipt.json, arm_env.txt, wrfout/);
  THROUGH_TAU 24 or 72 (smaller allowed only with --partial, for tests); SNAPSHOT = immutable source tree.
Records the frozen twin thresholds (path + sha256) the verdict must use.
Refuses (no manifest, no diff written) unless the MEASURED process attests SNAPSHOT: receipt commit + src tree equal the
snapshot's HEAD (clean), and every input file the measured CLI read (--input-dir of its argv) has the server file's sha256.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

CASES = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen")
THRESHOLDS = Path("<DATA_ROOT>/alisios/state/manager/cross/TO_WRF_GPU2_WN3_TWIN_THRESHOLDS_v1.json")
INPUTS = ("wrfinput_d01", "wrfinput_d02", "wrfinput_d03", "wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02", "wrflowinp_d03")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path)}


def git(snapshot: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(snapshot), *args], capture_output=True, text=True, check=True).stdout.strip()


def attest_source(receipt: dict, launcher: bool, snapshot: Path) -> dict:
    """The measured process's own source record must name the snapshot (commit, src tree, clean); never inferred."""
    commit, tree = git(snapshot, "rev-parse", "HEAD"), git(snapshot, "rev-parse", "HEAD:src/gpuwrf")
    if git(snapshot, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(f"snapshot {snapshot} has uncommitted changes: source not attestable")
    measured = (receipt.get("source") or {}) if launcher else receipt
    head, src_tree = measured.get("git_head"), measured.get("src_tree")
    if head != commit or src_tree != tree:
        raise SystemExit(f"measured receipt source commit {head} / src tree {src_tree} != snapshot {commit} / {tree}")
    if launcher:
        if measured.get("dirty") is not False:
            raise SystemExit("launcher receipt source is dirty or unrecorded")
        where = Path(measured.get("gpuwrf_path") or "/nonexistent").resolve()
        if where != (snapshot / "src" / "gpuwrf").resolve():
            raise SystemExit(f"launcher receipt imported gpuwrf from {where}, not the snapshot {snapshot}")
    else:
        script = Path((receipt.get("argv") or ["/nonexistent"])[0]).resolve()
        if not script.is_relative_to(snapshot.resolve()):
            raise SystemExit(f"measured harness script {script} is not inside the snapshot {snapshot}")
    return {"git_head": commit, "src_tree": tree, "clean": True,
            "evidence": "receipt.source" if launcher else "receipt.git_head/src_tree/argv[0]"}


def option(argv: list[str], name: str) -> str | None:
    for i, tok in enumerate(argv):
        if tok == name and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return None


def resolved_options(proofs: dict) -> tuple[dict, str]:
    """Executed per-domain options of a launcher (product CLI) run, from its run proof: every scalar control of the resolved
    OperationalNamelist when the product records it (metadata.domains[d].namelist_resolved, v0.3.1+), else the 14-control
    summary of v0.3.0 proofs (metadata.domains[d].namelist) — the source is attested next to the options."""
    doms = proofs.get("metadata", {}).get("domains", {}) or {}
    if doms and all(isinstance(v, dict) and v.get("namelist_resolved") for v in doms.values()):
        return ({d: v["namelist_resolved"] for d, v in doms.items()},
                "run proof metadata.domains[*].namelist_resolved (every scalar control of the resolved OperationalNamelist)")
    return ({d: v.get("namelist") for d, v in doms.items()},
            "run proof metadata.domains[*].namelist (v0.3.0 proof: 14-control summary per domain)")


def attest_inputs(argv: list[str], case_dir: Path, namelist: str | None) -> dict:
    """The files the measured CLI read (its --input-dir, its namelist) must be byte-identical to the server case files."""
    input_dir = option(argv or [], "--input-dir")
    if not input_dir:
        raise SystemExit("measured receipt argv has no --input-dir: input path not attestable")
    read = {n: Path(input_dir) / n for n in INPUTS}
    read["namelist.input"] = Path(option(argv, "--namelist") or namelist or Path(input_dir) / "namelist.input")
    out, bad = {}, []
    for name, path in read.items():
        server = case_dir / name
        got = sha256(path) if path.is_file() else None
        want = sha256(server)
        out[name] = {"read_path": str(path), "read_realpath": str(path.resolve()), "sha256": got, "server_sha256": want}
        if got != want:
            bad.append(name)
    if bad:
        raise SystemExit(f"measured CLI input path {input_dir}: {bad} missing or not byte-identical to server case {case_dir}")
    return {"input_dir": input_dir, "files": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("issue")
    ap.add_argument("arm_out", type=Path)
    ap.add_argument("through", type=int)
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--version-tag", default="wrf_gpu2 v0.25-dev")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    if a.through not in (24, 72) and not a.partial:
        raise SystemExit("THROUGH must be 24 or 72 (alisios GPU scope)")
    case_dir = CASES / f"wg_{a.issue}_a1" / "run" / "run"
    receipt = json.loads((a.arm_out / "receipt.json").read_text())
    launcher = receipt.get("schema") == "gpuwrf-parallel-case-v1"  # scripts/run_parallel_cases.sh case dir
    proofs_path = a.arm_out / "wrfout" / "proofs" / "nested_pipeline_run.json"
    proofs = json.loads(proofs_path.read_text()) if proofs_path.exists() else {}
    if launcher:
        assert receipt.get("rc") == 0 and str(proofs.get("device", "")).startswith("cuda"), "launcher case must be rc0 on GPU"
    else:
        assert receipt.get("rc") == 0 and receipt.get("device", {}).get("platform") == "gpu", "receipt must be rc0 on GPU"

    # Attest BEFORE writing anything: measured source == snapshot, measured CLI inputs == server case files.
    source = attest_source(receipt, launcher, a.snapshot)
    argv = receipt.get("command") if launcher else receipt.get("cli_argv")
    inputs = attest_inputs(argv, case_dir, proofs.get("namelist_path") if launcher else None)
    input_sha = {n: inputs["files"][n]["sha256"] for n in INPUTS}

    frames, start = [], None
    for nest in ("d01", "d02", "d03"):
        files = sorted((a.arm_out / "wrfout").glob(f"wrfout_{nest}_*"))[: a.through + 1]
        if len(files) != a.through + 1:
            raise SystemExit(f"{nest}: {len(files)} frames < {a.through + 1}")
        for tau, f in enumerate(files):
            utc = datetime.strptime(f.name[11:], "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
            start = start or utc
            frames.append({"nest": nest, "tau_h": tau, "path": str(f), "sha256": sha256(f), "bytes": f.stat().st_size,
                           "native_utc": utc.isoformat(), "actual_age_h": round((utc - start).total_seconds() / 3600.0, 6)})

    # Namelist diff: the GPU reads the CPU namelist.input itself; GPU-side choices are env flags + resolved options.
    nml_cpu = case_dir / "namelist.input"
    proof_nml = {d: v.get("namelist") for d, v in proofs.get("metadata", {}).get("domains", {}).items()}
    executed, executed_src = resolved_options(proofs) if launcher else (receipt.get("runtime_namelist"), "harness runtime_namelist")
    diff = {"cpu_namelist": ref(nml_cpu), "gpu_reads_same_file": True,  # attested above (sha256, actual CLI path)
            "gpu_namelist_read": inputs["files"]["namelist.input"], "measured_input_dir": inputs["input_dir"],
            "gpu_resolved_options": executed, "gpu_resolved_options_source": executed_src,
            "gpu_env_flags": receipt.get("env") if launcher else (a.arm_out / "arm_env.txt").read_text().splitlines(),
            "cli_argv": receipt.get("command") if launcher else receipt.get("cli_argv")}
    diff_path = a.out_json.with_name(a.out_json.stem + "_namelist_diff.json")
    diff_path.write_text(json.dumps(diff, indent=1) + "\n")

    if launcher:  # product run proof (nested_pipeline metadata: effective per-domain controls + static load)
        terrain = {d: {"topo_shading": (n or {}).get("topo_shading"), "slope_rad": (n or {}).get("slope_rad"),
                       "statics_loaded": (n or {}).get("radiation_static_loaded")} for d, n in proof_nml.items()}
        evidence = str(proofs_path)
    else:
        terrain, evidence = receipt.get("terrain_radiation") or {}, str(a.arm_out / "receipt.json")
    rad = {d: {"topo_shading": bool(terrain.get(d, {}).get("topo_shading") == 1),
               "slope_rad": bool(terrain.get(d, {}).get("slope_rad") == 1),
               "statics_loaded": bool(terrain.get(d, {}).get("statics_loaded")),
               "evidence_ref": evidence} for d in ("d01", "d02", "d03")}

    commit = source["git_head"]
    tarball = subprocess.run(["git", "-C", str(a.snapshot), "archive", "HEAD", "src"], capture_output=True, check=True).stdout
    g0 = [ref(a.arm_out / "receipt.json"), ref(diff_path)] + ([ref(proofs_path)] if launcher else [ref(a.arm_out / "arm_env.txt")])
    census = a.arm_out / "wrfout" / "census.json"
    if census.exists():
        g0.append(ref(census))
    if launcher:  # hourly d01 publication times (s after case start) -> per-hour wall, first frame = pre-step
        pub = [o["t_published_s"] for o in receipt.get("outputs", []) if o["d"] == "d01"]
        wallclock = {"process_wall_s": receipt.get("wall_s"),
                     "segment_rate_s_per_sim_h": [round(b - x, 3) for x, b in zip(pub, pub[1:])],
                     "pre_step_s": pub[0] if pub else None, "compile_s": None,
                     "hardware": f"1x {proofs.get('device')} via scripts/run_parallel_cases.sh, "
                                 f"{receipt.get('concurrent_at_start', 0) + 1}+ cases concurrent"}
    else:
        seg = receipt["derived"]["segments"]
        wallclock = {"process_wall_s": receipt.get("t_end"), "segment_rate_s_per_sim_h": receipt["derived"]["segment_rate_s_per_fc_h"],
                     "pre_step_s": seg[0]["t_start"] if seg else None, "compile_s": sum(s.get("compile_s_in_call") or 0 for s in seg),
                     "hardware": f"1x {receipt.get('device', {}).get('kind')}, host cpus {receipt.get('cpu_affinity')}"}
    manifest = {
        "schema": "wrf_gpu2.wn3.twin_gpu_manifest.v1", "issue": a.issue, "attempt": 1,
        "input_sha256": input_sha, "gpu_version_tag": f"{a.version_tag} {commit[:12]}",
        "gpu_code_sha256": hashlib.sha256(tarball).hexdigest(), "gpu_commit": commit, "gpu_source_attestation": source,
        "gpu_input_attestation": inputs,
        "namelist_diff_ref": ref(diff_path), "g0_refs": g0, "effective_radiation_features": rad,
        "output_through_tau_h": a.through, "frames": frames,
        "twin_thresholds_ref": ref(THRESHOLDS),
        "wallclock": wallclock,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    a.out_json.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"issue": a.issue, "frames": len(frames), "through": a.through, "commit": commit[:12],
                      "radiation": {d: (v["topo_shading"], v["slope_rad"]) for d, v in rad.items()}}))


if __name__ == "__main__":
    main()
