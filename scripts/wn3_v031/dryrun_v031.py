"""(A)+(B) dry-run evidence, CPU only: the lane/wn3 launcher (warm/cold admission per required executable + one-compile host
headroom) on a SCRATCH cache whose sidecar is written by record_peaks() from the REAL FINAL-b proofs (cold sizing 0227:
VmHWM 14,761,780 kB / 3,928 MiB; warm R3 0614: VmHWM 6,248,212 kB / 4,972 MiB, its real AOT blobs + .meta in W9/cache).
Six WN3 72 h cases, --max-parallel 8, idle VRAM of the FINAL-b session-2 start (30.27 GB free); MemAvailable injected:
43 GB (alisios) and 65.68 GB (our idle). Arms: warm (artifacts present), artifact deleted (simulated by recording a missing
alias path), v0.3.0 legacy sidecar. Writes result.json next to this file."""
import contextlib, importlib.util, io, json, os, sys
from pathlib import Path

D = Path(__file__).resolve().parent
LANE = Path("<USER_HOME>/src/wrf_gpu2_wt/wn3")
W9 = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/W9")
CASES = [f"<USER_HOME>/wrf_gpu2_lanes/wn3/cases/{c}" for c in ("20260227_18z_a1", "20260502_18z_a1", "20260614_18z_a1",
                                                             "20260220_18z_a1", "20260608_18z_a1", "20260120_18z_a1")]
CLI = ["--max-dom", "3", "--emit-initial-history", "--hours", "72"]
idle = json.loads((W9 / "finalb_meta/admission_r3_s2.json").read_text())
C = D / "cache"
os.environ.update({"GPUWRF_JAX_CACHE_DIR": str(C / "jax"), "JAX_COMPILATION_CACHE_DIR": str(C / "jax"),
                   "GPUWRF_XLA_AUTOTUNE_CACHE_DIR": str(C / "autotune"), "CUDA_CACHE_PATH": str(C / "cuda"),
                   "TRITON_CACHE_DIR": str(C / "triton"), "GPUWRF_WRF_ROOT": "<DATA_ROOT>/wrf_gpu2/v025/tenerife_b4/wrf_root",
                   "JAX_PLATFORMS": "cpu", "GPUWRF_GPU_LOCK_HELD": "1", "PYTHONPATH": str(LANE / "src")})
sys.path.insert(0, str(LANE / "src"))
spec = importlib.util.spec_from_file_location("pc_lane", LANE / "scripts/parallel_cases.py")
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
MEM = {"alisios_43GB": 43_000_000, "idle_65.7GB": int(idle["host_capacity_kb"]) + 8_000_000}


def dry(mem_kb):
    pc.mem_available_kb = lambda: mem_kb
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = pc.main(["--dry-run", "--max-parallel", "8", "--out-root", str(D / "out"), *CASES, "--", *CLI],
                     gpu=lambda: (int(idle["vram_total_bytes"]), int(idle["vram_free_bytes"])))
    r = json.loads(buf.getvalue()); c0 = r["cases"][0]
    return {"rc": rc, "would_start_now": r["would_start_now"], "host_capacity_kb": r["host_capacity_kb"], "need_bytes": c0["need_bytes"],
            "host_need_kb": c0["host_need_kb"], "host_need_cold_kb": c0["host_need_cold_kb"], "host_need_source": c0["host_need_source"],
            "aot_cache_warm": c0["aot_cache_warm"], "plan_path": c0.get("plan_path")}


probe = dry(MEM["alisios_43GB"])  # no sidecar yet -> solo sizing run; also yields the plan path
side = Path(probe["plan_path"]).with_suffix(".measured.json")
assert str(side).startswith(str(D)) and not side.exists(), side
sizing = {"plan_path": probe["plan_path"]}
cold = pc.cache_state(W9 / "finalb_cold/20260227_18z_a1")
warm = pc.cache_state(W9 / "finalb_r3/20260614_18z_a1")
assert cold["warm"] is False and warm["warm"] is True, (cold, warm)
out = {"no_sidecar": probe, "states": {"cold": cold, "warm": warm}}
pc.record_peaks(sizing, 14_761_780, 3928, cold)
out["after_cold_sizing"] = {k: dry(v) for k, v in MEM.items()}
pc.record_peaks(sizing, 6_248_212, 4972, warm)
out["after_warm_run"] = {k: dry(v) for k, v in MEM.items()}
gone = {**warm, "executables": {**warm["executables"], "fused/d02": [str(D / "gone.xlaexec"), str(D / "gone.meta")]}}
pc.record_peaks(sizing, 6_248_212, 4972, gone)
out["warm_artifact_deleted"] = {k: dry(v) for k, v in MEM.items()}
side.write_text(json.dumps({"vmhwm_kb_max": 14_761_780, "vram_peak_mib_max": 4972}) + "\n")
out["v030_legacy_sidecar"] = {k: dry(v) for k, v in MEM.items()}
side.unlink()
(D / "result.json").write_text(json.dumps(out, indent=1) + "\n")
for k, v in out.items():
    if k != "states":
        print(k, json.dumps({m: (x["would_start_now"], round(x["host_need_kb"] / 1e6, 2), round(x["host_need_cold_kb"] / 1e6, 2))
                             for m, x in v.items()} if "rc" not in v else (v["would_start_now"], v["host_need_source"])))
