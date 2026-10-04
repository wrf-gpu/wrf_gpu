"""WRF-side runner for the v0234 d03 step-1 dry-dycore/nest suboperator ladder.

Sprint: 2026-07-18-v0234-dycore-suboperator-kimi. CPU-only. All runs confined
to physical cores 13-15 (logical 13-15,29-31) with kernel-enforced affinity
(systemd transient service, CPUAffinity + seccomp sched_setaffinity denial),
zero overlap with ALISIOS production cores 0-11 proven per run by a two-sample
admission and a 1 Hz watchdog.

Subcommands (execution order; build/run require the 0:3 ADMIT):
  build-tree       copy the frozen isolated WRF tree, apply the ladder patch,
                   configure and compile (incremental not possible across
                   absolute-path CMake caches -> fresh build dir)
  prepare-runs     create control + 6 member run directories
  admit-run --member ID
                   two-sample admission -> systemd launch -> 1 Hz watchdog ->
                   execution receipt -> raw archive (sha256 tar.zst)
  verify-control   output-neutrality (wrfout byte-identity vs retained truth)
                   + reproducibility (SP1/SP4 steps 1-2 bitwise vs baseline)
  analyze          reassemble L1-L5 + SP1-SP4 for control and members, compute
                   rung/span envelopes and band splits -> ladder-analysis.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi"
CONTINUATION_SPRINT = (
    REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-continuation"
)
PATCH_PATH = SPRINT / "wrf-ladder-instrumentation.patch"
LADDER_ENVELOPES_OUTPUT = CONTINUATION_SPRINT / "ladder-envelopes.json"

KIMI_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi")
FROZEN_TREE = KIMI_ROOT / "wrf_iso"
BASE_RUN = KIMI_ROOT / "run/momsp_arm"
BASE_DUMPS = KIMI_ROOT / "momsp_dumps"
BASE_CACHE = KIMI_ROOT / "compare/wrf_global_cache"
MEMBERS_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_conditioning_ensemble_gpt/members")

WORK = Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi")
LADDER_TREE = WORK / "wrf_iso_ladder"
RUNS = WORK / "runs"

WRF_ENV = Path("<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build")
CPUSET = "13-15,29-31"
SYSTEMD_CPU_AFFINITY = "13 14 15 29 30 31"
SYSTEMD_AFFINITY_SYSCALL_FILTER = "~sched_setaffinity"
EXPECTED_PHYSICAL = {13, 14, 15}
PRODUCTION_LOGICAL = set(range(12)) | set(range(16, 28))
MPI_RANKS = 12
DISK_FLOOR_GIB = 40.0
MEM_FLOOR_GIB = 32.0

MEMBERS = ("control", "mask-a-minus", "mask-a-plus", "mask-b-minus",
           "mask-b-plus", "mask-c-minus", "mask-c-plus")

FROZEN_HASHES = {
    "namelist": "9ce4336dd4b878685871370a2bdf048ce27c4bb52b655b5a33089008026caaf3",
    "wrfbdy_d01": "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec",
    "wrfinput_d01": "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756",
    "wrfinput_d02": "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964",
    "wrfinput_d03": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}
TRUTH_WRFOUT = {
    "wrfout_d01_2025-03-01_00:00:00": "f91ef2336179e816a37ceb0a749396072b7e41a64fd2846a8ba9e808650df96e",
    "wrfout_d02_2025-03-01_00:00:00": "9904a25719941df667c152bf5f2e0a1729cbe645b170da428f006d484b5f21fc",
    "wrfout_d03_2025-03-01_00:00:00": "37db046dad2540340ce8847e76c7f32b1808d6012f1b14c84e24431280677202",
    "wrfout_d03_2025-03-01_00:20:00": "0a1157771f8b00f2c2c4fb66ec1cb63e4534cf3c305981ca81e1cfaad0d8d7f1",
}
# member wrfinput_d03 hashes are read from each member's audited
# perturbation-manifest.json at prepare time (independently re-verified there).

LADDER_TAGS = {
    "l1_rk1_tend": ("ru_tend", "rv_tend", "u_save", "v_save"),
    "l2_rk1_fin": ("u", "v"),
    "l3_rk2_fin": ("u", "v"),
    "l4_rk3_fin": ("u", "v"),
    "l5_prebdry": ("u", "v"),
}
CHAIN_TAGS = {
    "sp1_entry": ("u", "v"),
    "sp2_pbl": ("rublten", "rvblten"),
    "sp3_tendf": ("ru_tendf", "rv_tendf"),
    "sp4_exit": ("u", "v"),
    **LADDER_TAGS,
}
ANALYSIS_TAGS = {
    **CHAIN_TAGS,
    # Frozen plan section 5 defines the L1 state metric on the RU/RV pair.
    # u_save/v_save are retained auxiliary evidence, not extra samples in the
    # combined primary RMSE.
    "l1_rk1_tend": ("ru_tend", "rv_tend"),
}
# dump files written per rank per step by the instrumentation (incl. sp2 mut, sp5)
DUMP_FILES_PER_STEP = 2 + 3 + 2 + 2 + 2 + 4 + 2 + 2 + 2 + 2  # 23
STEPS = (1, 2)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value, self_key: str | None = None) -> str:
    if self_key is not None and isinstance(value, dict):
        value = {k: v for k, v in value.items() if k != self_key}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_self_hashed(path: Path, payload: dict) -> None:
    out = dict(payload)
    out["proof_sha256"] = canonical_digest(out, "proof_sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def run_logged(cmd: list[str], log: Path | None = None, **kwargs) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, **kwargs)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as handle:
            handle.write(f"$ {' '.join(str(c) for c in cmd)}\nrc={proc.returncode}\n".encode())
    return proc


# ---------------------------------------------------------------- resources

def _cpu_set_of(pid: int) -> set[int]:
    out = subprocess.run(["taskset", "-pc", str(pid)], text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    m = re.search(r":\s*([0-9,\-]+)\s*$", out)
    cpus: set[int] = set()
    for part in (m.group(1) if m else "").split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        elif part.strip():
            cpus.add(int(part))
    return cpus


def _physical_cores(cpus: set[int]) -> set[int]:
    # logical 0-15 -> physical 0-15; logical 16-31 -> physical (n-16)
    return {c if c < 16 else c - 16 for c in cpus}


def production_processes() -> list[dict]:
    """Real ALISIOS production processes, identified by executable link.

    pgrep -f on the command line self-matches any monitoring shell whose
    cmdline contains the pattern text; filter on /proc/<pid>/exe instead.
    """
    pids = subprocess.run(
        ["pgrep", "-f", "install_gen2_dmpar/run/wrf.exe|prterun.*install_gen2_dmpar"],
        text=True, stdout=subprocess.PIPE).stdout.split()
    rows = []
    for pid_text in pids:
        pid = int(pid_text)
        if pid == os.getpid():
            continue
        try:
            exe = os.readlink(f"/proc/{pid}/exe")
        except (FileNotFoundError, PermissionError):
            continue
        is_rank = "install_gen2_dmpar" in exe
        is_launcher = exe.rsplit("/", 1)[-1] in {"prte", "prterun"}
        if not (is_rank or is_launcher):
            continue
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
            if is_launcher and b"install_gen2_dmpar" not in cmdline:
                continue
            cpus = _cpu_set_of(pid)
            stat = Path(f"/proc/{pid}/stat").read_text().split()
            rows.append({
                "pid": pid,
                "exe": exe,
                "start_ticks": int(stat[21]),
                "logical_cpus": sorted(cpus),
                "physical_cores": sorted(_physical_cores(cpus)),
            })
        except (FileNotFoundError, PermissionError):
            continue
    return rows


def capacity_snapshot() -> dict:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            mem["mem_available_gib"] = int(line.split()[1]) / 2**20
    out = subprocess.run(["df", "--output=avail", "-BG", "<DATA_ROOT>"],
                         text=True, stdout=subprocess.PIPE).stdout.splitlines()[-1].strip()
    mem["mnt_data_free_gib"] = float(out.rstrip("G"))
    return mem


def resource_snapshot() -> dict:
    prods = production_processes()
    confined = all(set(p["physical_cores"]) <= set(range(12)) for p in prods)
    return {
        "captured_at_utc": utc_now(),
        "production_processes": prods,
        "production_confined_to_physical_0_11": confined,
        "production_identity": hashlib.sha256(
            json.dumps([(p["exe"], p["start_ticks"]) for p in sorted(prods, key=lambda r: r["pid"])],
                       sort_keys=True).encode()).hexdigest(),
        **capacity_snapshot(),
    }


def admission(member: str) -> dict:
    first = resource_snapshot()
    time.sleep(2.0)
    second = resource_snapshot()
    stable = (first["production_confined_to_physical_0_11"]
              and second["production_confined_to_physical_0_11"]
              and first["production_identity"] == second["production_identity"]
              and first["mem_available_gib"] >= MEM_FLOOR_GIB
              and second["mem_available_gib"] >= MEM_FLOOR_GIB
              and first["mnt_data_free_gib"] >= DISK_FLOOR_GIB
              and second["mnt_data_free_gib"] >= DISK_FLOOR_GIB)
    payload = {
        "schema": "gpuwrf.v0234.dycore-suboperator-kimi.wrf-ladder-admission.v1",
        "member": member,
        "requested_logical_cpuset": CPUSET,
        "expected_physical_cores": sorted(EXPECTED_PHYSICAL),
        "first": first,
        "second": second,
        "verdict": "ADMITTED_ISOLATED_CPU_SET" if stable else "DENIED_RESOURCE_BOUNDARY",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_self_hashed(RUNS / member / f"admissions/{stamp}-admitted.json", payload)
    if not stable:
        raise RuntimeError(f"admission denied for {member}")
    return payload


class Watchdog:
    """1 Hz sampling of production + member affinities during a run."""

    def __init__(self, member: str, unit: str, baseline_identity: str):
        self.member = member
        self.unit = unit
        self.baseline_identity = baseline_identity
        self.samples: list[dict] = []
        self.violations: list[str] = []

    def _unit_process_rows(self) -> list[dict]:
        """Enumerate every process in the transient unit's cgroup.

        The inherited monitor used a ``pgrep -f`` expression ending in
        ``wrf.exe`` even though the admitted executable is launched through
        its installed ``.../bin/wrf`` path.  It therefore never observed a
        member rank.  The systemd cgroup is the kernel-owned membership
        authority and covers the launcher plus every MPI child without
        relying on a command-line spelling.
        """
        control_group = subprocess.run(
            ["systemctl", "--user", "show", self.unit,
             "--property=ControlGroup", "--value"],
            text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
        if not control_group.startswith("/") or ".." in control_group.split("/"):
            return []
        cgroup_root = Path("/sys/fs/cgroup") / control_group.lstrip("/")
        if not cgroup_root.is_dir():
            return []
        pids: set[int] = set()
        for procs_file in cgroup_root.rglob("cgroup.procs"):
            try:
                pids.update(int(value) for value in procs_file.read_text().split())
            except (FileNotFoundError, PermissionError, ValueError):
                continue
        rows = []
        for pid in sorted(pids):
            try:
                exe = os.readlink(f"/proc/{pid}/exe")
                cpus = _cpu_set_of(pid)
                stat = Path(f"/proc/{pid}/stat").read_text().split()
            except (FileNotFoundError, PermissionError):
                continue
            rows.append({
                "pid": pid,
                "exe": exe,
                "start_ticks": int(stat[21]),
                "logical_cpus": sorted(cpus),
                "physical_cores": sorted(_physical_cores(cpus)),
            })
        return rows

    def sample(self) -> None:
        snap = resource_snapshot()
        unit_state = subprocess.run(
            ["systemctl", "--user", "show", self.unit,
             "-p", "ActiveState,CPUAffinity,ControlGroup,MainPID"],
            text=True, stdout=subprocess.PIPE).stdout
        unit_rows = self._unit_process_rows()
        member_rows = [row for row in unit_rows if "wrf_iso_ladder" in row["exe"]]
        overlap = set()
        for row in unit_rows:
            if not set(row["physical_cores"]) <= EXPECTED_PHYSICAL:
                overlap.add(row["pid"])
        row = {
            "captured_at_utc": snap["captured_at_utc"],
            "production_identity": snap["production_identity"],
            "production_confined": snap["production_confined_to_physical_0_11"],
            "member_rank_count": len(member_rows),
            "member_rows": member_rows,
            "unit_process_count": len(unit_rows),
            "unit_rows": unit_rows,
            "unit_state": unit_state.strip().replace("\n", ";"),
            "mem_available_gib": snap["mem_available_gib"],
            "mnt_data_free_gib": snap["mnt_data_free_gib"],
        }
        self.samples.append(row)
        if not row["production_confined"]:
            self.violations.append("production-confinement")
        if row["production_identity"] != self.baseline_identity:
            self.violations.append("production-identity-drift")
        if overlap:
            self.violations.append(f"member-core-violation:{sorted(overlap)}")
        if row["mem_available_gib"] < MEM_FLOOR_GIB or row["mnt_data_free_gib"] < DISK_FLOOR_GIB:
            self.violations.append("capacity")

    def finish(self) -> dict:
        max_member_rank_count = max(
            (s["member_rank_count"] for s in self.samples), default=0,
        )
        if max_member_rank_count != MPI_RANKS:
            self.violations.append(
                f"member-rank-observation:{max_member_rank_count}!={MPI_RANKS}"
            )
        return {
            "member": self.member,
            "unit": self.unit,
            "baseline_production_identity": self.baseline_identity,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "violations": self.violations,
            "all_samples_safe": not self.violations,
            "max_member_rank_count": max_member_rank_count,
            "max_unit_process_count": max(
                (s["unit_process_count"] for s in self.samples), default=0,
            ),
        }


# ---------------------------------------------------------------- build

def build_tree() -> None:
    if LADDER_TREE.exists():
        raise FileExistsError(f"refusing to overwrite {LADDER_TREE}")
    WORK.mkdir(parents=True, exist_ok=True)
    log = WORK / "build-ladder.log"
    run_logged(["cp", "-a", str(FROZEN_TREE), str(LADDER_TREE)], log)
    run_logged(["patch", "-p1", "-d", str(LADDER_TREE), "-i", str(PATCH_PATH),
                "--no-backup-if-mismatch", "-r", str(WORK / "patch.rej")], log)
    env = dict(
        PATH=f"{WRF_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        LD_LIBRARY_PATH=str(WRF_ENV / "lib"),
        CONDA_PREFIX=str(WRF_ENV),
        CC="mpicc", CXX="mpicxx", FC="mpifort", F77="mpifort",
        FFLAGS=f"-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem {WRF_ENV}/include -I{WRF_ENV}/include",
        CFLAGS=f"-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem {WRF_ENV}/include",
        CXXFLAGS=f"-fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem {WRF_ENV}/include",
        LDFLAGS=f"-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined -Wl,-rpath,{WRF_ENV}/lib -Wl,-rpath-link,{WRF_ENV}/lib -L{WRF_ENV}/lib",
        HOME="<USER_HOME>",
    )
    configure = [
        "./configure_new", "-p", "GNU", "-x", "-d", "_build_ladder",
        "-i", str(LADDER_TREE / "install_ladder"), "--",
        "-DWRF_CORE=ARW", "-DWRF_CASE=EM_REAL", "-DWRF_NESTING=BASIC",
        f"-DMPI_ROOT={WRF_ENV}", f"-DnetCDF_ROOT={WRF_ENV}",
        f"-DnetCDF-Fortran_ROOT={WRF_ENV}", f"-DHDF5_ROOT={WRF_ENV}",
        f"-DJasper_ROOT={WRF_ENV}", f"-DZLIB_ROOT={WRF_ENV}",
        "-DUSE_MPI=ON", "-DUSE_OPENMP=OFF", "-DBUILD_EXTERNALS=OFF",
    ]
    with log.open("ab") as handle:
        proc = subprocess.run(configure, cwd=LADDER_TREE, env=env,
                              stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError("configure failed; see build-ladder.log")
        proc = subprocess.run(["./compile_new", "_build_ladder", "-j", "6"],
                              cwd=LADDER_TREE, env=env,
                              stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError("compile failed; see build-ladder.log")
    binary = LADDER_TREE / "install_ladder/bin/wrf"
    if not binary.is_file():
        raise RuntimeError(f"missing binary {binary}")
    receipt = {
        "schema": "gpuwrf.v0234.dycore-suboperator-kimi.wrf-ladder-build.v1",
        "tree": str(LADDER_TREE),
        "frozen_source_tree": str(FROZEN_TREE),
        "patch": str(PATCH_PATH),
        "patch_sha256": sha256_file(PATCH_PATH),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "built_at_utc": utc_now(),
    }
    write_self_hashed(WORK / "build-receipt.json", receipt)
    print(json.dumps({"binary_sha256": receipt["binary_sha256"]}))


# ---------------------------------------------------------------- run dirs

def prepare_runs() -> None:
    template = BASE_RUN
    for member in MEMBERS:
        dest = RUNS / member / "run"
        if dest.exists():
            # rerunnable: verify the existing dir instead of failing
            pass
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(template, dest, symlinks=True,
                            ignore=shutil.ignore_patterns(
                                "wrfout*", "rsl.*", "momsp_dumps", "*.log",
                                "restart*", "wrfrst*"))
        if member != "control":
            src_input = MEMBERS_ROOT / member / "run" / "wrfinput_d03"
            manifest = json.loads((MEMBERS_ROOT / member / "perturbation-manifest.json").read_text())
            expected = manifest["member_wrfinput_d03"]["sha256"]
            observed = sha256_file(src_input)
            if observed != expected:
                raise RuntimeError(f"{member} wrfinput_d03 hash drift: {observed} != {expected}")
            target = dest / "wrfinput_d03"
            if target.exists() or target.is_symlink():
                target.chmod(0o644)
                target.unlink()
            shutil.copy2(src_input, target)
            target.chmod(0o444)
            member_d03_hash = expected
        else:
            member_d03_hash = FROZEN_HASHES["wrfinput_d03"]
        checks = {
            "namelist.input": FROZEN_HASHES["namelist"],
            "wrfbdy_d01": FROZEN_HASHES["wrfbdy_d01"],
            "wrfinput_d01": FROZEN_HASHES["wrfinput_d01"],
            "wrfinput_d02": FROZEN_HASHES["wrfinput_d02"],
            "wrfinput_d03": member_d03_hash,
        }
        for name, expected in checks.items():
            observed = sha256_file(dest / name)
            if observed != expected:
                raise RuntimeError(f"{member}/{name}: {observed} != {expected}")
        print(json.dumps({"member": member, "run_dir": str(dest), "inputs_verified": True}))


# ---------------------------------------------------------------- run

def _launch(member: str) -> tuple[subprocess.Popen, str, Path]:
    run_dir = RUNS / member / "run"
    dumps = RUNS / member / "momsp_dumps"
    dumps.mkdir(parents=True, exist_ok=True)
    unit = f"v0234-ladder-{member}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ').lower()}"
    env = {
        "HOME": "<USER_HOME>", "USER": "user", "LOGNAME": "user",
        "PATH": f"{WRF_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": str(WRF_ENV / "lib"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "WRFGPU2_MOM_SP": "1", "WRFGPU2_MOM_SP_GRID": "3",
        "WRFGPU2_MOM_SP_SMIN": str(min(STEPS)), "WRFGPU2_MOM_SP_SMAX": str(max(STEPS)),
        "WRFGPU2_MOM_SP_LADDER": "1",
        "WRFGPU2_MOM_SP_ROOT": str(dumps),
    }
    command = [
        "/usr/bin/taskset", "-c", CPUSET,
        "/usr/bin/systemd-run", "--user", "--wait", "--pipe", "--collect", "--quiet",
        f"--unit={unit}",
        f"--working-directory={run_dir}",
        f"--property=CPUAffinity={SYSTEMD_CPU_AFFINITY}",
        f"--property=SystemCallFilter={SYSTEMD_AFFINITY_SYSCALL_FILTER}",
        "--property=SystemCallErrorNumber=EPERM",
        "--property=NoNewPrivileges=yes",
        "--property=KillMode=control-group",
    ]
    command.extend(f"--setenv={k}={v}" for k, v in env.items())
    command += [
        "/usr/bin/nice", "-n", "10", "/usr/bin/ionice", "-c", "3",
        str(WRF_ENV / "bin/prterun"),
        "--map-by", ":oversubscribe:hwtcpus", "--bind-to", "none",
        "-np", str(MPI_RANKS),
        str(LADDER_TREE / "install_ladder/bin/wrf"),
    ]
    log = (RUNS / member / "launch.log").open("wb")
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    return proc, unit, dumps


def admit_run(member: str) -> None:
    adm = admission(member)
    proc, unit, dumps = _launch(member)
    watchdog = Watchdog(member, unit, adm["second"]["production_identity"])
    watchdog.sample()
    while proc.poll() is None:
        time.sleep(1.0)
        watchdog.sample()
        if watchdog.violations:
            subprocess.run(
                ["systemctl", "--user", "stop", unit],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            proc.wait()
            break
    rc = proc.returncode
    time.sleep(0.2)
    watchdog.sample()
    monitor = watchdog.finish()
    write_self_hashed(RUNS / member / "resource-monitor.json", {
        "schema": "gpuwrf.v0234.dycore-suboperator-kimi.wrf-ladder-monitor.v1",
        **monitor,
    })
    frames = {}
    for name, expected in TRUTH_WRFOUT.items():
        frame = RUNS / member / "run" / name
        if frame.is_file():
            frames[name] = {"sha256": sha256_file(frame), "bytes": frame.stat().st_size}
    dump_files = sorted(dumps.glob("rank*/*.f64"))
    inventory = [{
        "path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p),
    } for p in dump_files] + [{
        "path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p),
    } for p in sorted(dumps.glob("rank*/meta.txt"))]
    n_ranks = len(list(dumps.glob("rank*")))
    expected_count = n_ranks * DUMP_FILES_PER_STEP * len(STEPS) + n_ranks
    success_marker = any(
        b"SUCCESS COMPLETE WRF" in path.read_bytes()
        for path in sorted((RUNS / member / "run").glob("rsl.error.*"))
    )
    receipt = {
        "schema": "gpuwrf.v0234.dycore-suboperator-kimi.wrf-ladder-execution.v1",
        "member": member,
        "systemd_unit": unit,
        "returncode": rc,
        "success_complete_wrf": rc == 0 and success_marker,
        "binary": str(LADDER_TREE / "install_ladder/bin/wrf"),
        "binary_sha256": sha256_file(LADDER_TREE / "install_ladder/bin/wrf"),
        "wrfout_frames": frames,
        "dump_file_count": len(inventory),
        "expected_dump_file_count_at_steps_1_2": expected_count,
        "monitor_safe": monitor["all_samples_safe"],
        "monitor_violations": monitor["violations"],
        "finished_at_utc": utc_now(),
    }
    write_self_hashed(RUNS / member / "execution-receipt.json", receipt)
    archive = RUNS / member / "evidence" / "momsp-ladder-dumps.tar.zst"
    archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "--sort=name", "-I", "zstd -3 -T4", "-cf", str(archive),
                    "-C", str(dumps.parent), dumps.name], check=True)
    subprocess.run(["zstd", "-q", "-t", str(archive)], check=True)
    write_self_hashed(RUNS / member / "archive-receipt.json", {
        "schema": "gpuwrf.v0234.dycore-suboperator-kimi.wrf-ladder-archive.v1",
        "member": member,
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "zstd_test_passed": True,
        "file_count": len(inventory),
        "inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
    })
    print(json.dumps({"member": member, "rc": rc, "dumps": len(inventory),
                      "monitor_safe": monitor["all_samples_safe"]}))
    if not (
        rc == 0
        and success_marker
        and monitor["all_samples_safe"]
        and len(inventory) == expected_count
    ):
        raise SystemExit(1)


# ---------------------------------------------------------------- control gates

def verify_control() -> None:
    sys.path.insert(0, str(REPO))
    from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble

    result: dict[str, object] = {"schema": "gpuwrf.v0234.dycore-suboperator-kimi.control-gates.v1"}
    frames_ok = True
    frames = {}
    for name, expected in TRUTH_WRFOUT.items():
        frame = RUNS / "control/run" / name
        observed = sha256_file(frame) if frame.is_file() else None
        frames[name] = {"expected": expected, "observed": observed,
                        "byte_identical": observed == expected}
        frames_ok &= observed == expected
    result["wrfout_byte_identity"] = frames
    ranks = reassemble.load_ranks(RUNS / "control/momsp_dumps")
    bitwise = {}
    all_ok = frames_ok
    for step in STEPS:
        for tag in ("sp1_entry", "sp4_exit"):
            for field in ("u", "v"):
                key = f"{tag}__{field}"
                arr = reassemble.reassemble3d(key, step, ranks)
                cache = np.load(BASE_CACHE / f"step{step:06d}_{key}.npy")
                same = bool(np.array_equal(arr, cache))
                bitwise[f"step{step}/{key}"] = same
                all_ok &= same
    result["sp1_sp4_bitwise_vs_baseline"] = bitwise
    result["passed"] = bool(all_ok)
    write_self_hashed(RUNS / "control/control-gates.json", result)
    print(json.dumps({"passed": result["passed"]}))
    if not result["passed"]:
        raise SystemExit(1)


# ---------------------------------------------------------------- analysis

def analyze() -> None:
    sys.path.insert(0, str(REPO))
    from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble

    control_ranks = reassemble.load_ranks(RUNS / "control/momsp_dumps")
    member_ranks = {m: reassemble.load_ranks(RUNS / m / "momsp_dumps")
                    for m in MEMBERS if m != "control"}

    # control arrays, cached once
    control_arrays: dict[tuple[int, str, str], np.ndarray] = {}
    for step in STEPS:
        for tag, fields in ANALYSIS_TAGS.items():
            for field in fields:
                control_arrays[(step, tag, field)] = reassemble.reassemble3d(
                    f"{tag}__{field}", step, control_ranks)

    # The pre-instrumentation baseline contains SP1-SP4 but cannot contain the
    # newly introduced L1-L5 dumps.  Authenticate every comparable chain rung;
    # ladder output neutrality is instead bound by the already-green wrfout and
    # SP1/SP4 control gates.
    baseline_ranks = reassemble.load_ranks(BASE_DUMPS)
    control_bitwise = {}
    for tag, fields in ANALYSIS_TAGS.items():
        if tag in LADDER_TAGS:
            continue
        for field in fields:
            key = f"{tag}__{field}"
            if tag in ("sp1_entry", "sp4_exit"):
                ref = np.load(BASE_CACHE / f"step000001_{key}.npy")
            else:
                ref = reassemble.reassemble3d(key, 1, baseline_ranks)
            control_bitwise[key] = bool(np.array_equal(control_arrays[(1, tag, field)], ref))

    def combined(diffs: list[np.ndarray]) -> float:
        sse = sum(float(np.sum(np.square(d, dtype=np.float64), dtype=np.float64)) for d in diffs)
        n = sum(d.size for d in diffs)
        return float(np.sqrt(sse / n))

    envelopes: dict[str, dict] = {}
    for step in STEPS:
        step_key = f"step{step}"
        envelopes[step_key] = {}
        for tag, fields in ANALYSIS_TAGS.items():
            rows = {}
            for member, ranks in member_ranks.items():
                diffs = [
                    reassemble.reassemble3d(f"{tag}__{field}", step, ranks)
                    - control_arrays[(step, tag, field)]
                    for field in fields
                ]
                rows[member] = combined(diffs)
            envelopes[step_key][tag] = {
                "member_rmse": rows,
                "wrf_envelope_min": min(rows.values()),
                "wrf_envelope_max": max(rows.values()),
            }
    payload = {
        "schema": "gpuwrf.v0234.dycore-suboperator-kimi.ladder-envelopes.v1",
        "control_bitwise_equal_to_baseline_step1": control_bitwise,
        "ladder_baseline_status": {
            "available": False,
            "reason": "L1-L5 did not exist in the pre-instrumentation baseline",
            "output_neutrality_authority": str(RUNS / "control/control-gates.json"),
        },
        "l1_auxiliary_fields": {
            "fields": ["u_save", "v_save"],
            "captured": True,
            "included_in_primary_l1_rmse": False,
        },
        "envelopes": envelopes,
        "note": "GPU rung values are merged by the GPU-arm analysis step; "
                "span update-difference envelopes are computed in the same merge.",
    }
    write_self_hashed(LADDER_ENVELOPES_OUTPUT, payload)
    print(json.dumps({"wrote": str(LADDER_ENVELOPES_OUTPUT),
                      "control_bitwise_all_available_rungs": all(control_bitwise.values())}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-tree")
    sub.add_parser("prepare-runs")
    run_p = sub.add_parser("admit-run")
    run_p.add_argument("--member", required=True, choices=MEMBERS)
    sub.add_parser("verify-control")
    sub.add_parser("analyze")
    args = parser.parse_args()
    if args.cmd == "build-tree":
        build_tree()
    elif args.cmd == "prepare-runs":
        prepare_runs()
    elif args.cmd == "admit-run":
        admit_run(args.member)
    elif args.cmd == "verify-control":
        verify_control()
    elif args.cmd == "analyze":
        analyze()


if __name__ == "__main__":
    main()
