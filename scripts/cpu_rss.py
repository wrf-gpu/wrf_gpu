#!/usr/bin/env python3
"""cpu_rss.py — CPU-WRF PROD peak RSS for the R1 denominator (lane bench).

Stages D5's pristine CPU case (inputs + the exact dmpar wrf.exe D5 ran) in a writable dir, runs
`mpirun -np N ./wrf.exe` pinned + nic'd, samples the sum of VmRSS (and PSS) of every wrf rank every 2 s,
and reports peak total / per-rank / wall. Everything is written under the staged dir (never <DATA_ROOT>),
which is deleted at the end unless --keep.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, signal, subprocess, sys, threading, time
from pathlib import Path

D5 = Path("<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/"
          "alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case")
WRF_EXE = Path("<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar/bin/wrf")
MPI_DIR = Path("<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin")
LIB_DIR = Path("<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/lib")
ROOT = Path("<USER_HOME>/wrf_gpu2_lanes/bench/cpu_rss")
CPUS = "12,13,28,29"
QUIET = Path("/tmp/wrf_gpu2_quiet")
INPUTS = ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01")
NPROC = {4: (2, 2), 2: (2, 1), 12: (4, 3)}   # ranks -> (nproc_x, nproc_y)


def _set_namelist(src: Path, dst: Path, ranks: int, hours: int):
    text = src.read_text()
    nx, ny = NPROC[ranks]
    # time control: start 2026-07-26_00 (unchanged) -> end = start + hours
    text = re.sub(r"run_days\s*=\s*\d+", "run_days                            = 0", text, count=1)
    text = re.sub(r"run_hours\s*=\s*\d+", f"run_hours                           = {hours}", text, count=1)
    text = re.sub(r"end_month\s*=\s*\d+,\s*\d+", "end_month                         = 7, 7", text, count=1)
    text = re.sub(r"end_day\s*=\s*\d+,\s*\d+", "end_day                           = 26, 26", text, count=1)
    text = re.sub(r"end_hour\s*=\s*\d+,\s*\d+", f"end_hour                          = {hours}, {hours}", text, count=1)
    text = re.sub(r"nproc_x\s*=\s*\d+", f"nproc_x                             = {nx}", text, count=1)
    text = re.sub(r"nproc_y\s*=\s*\d+", f"nproc_y                             = {ny}", text, count=1)
    dst.write_text(text)


def _rss_pss(pids):
    rss = pss = 0
    for pid in pids:
        try:
            for ln in open(f"/proc/{pid}/status"):
                if ln.startswith("VmRSS:"):
                    rss += int(ln.split()[1]); break
        except OSError:
            pass
        try:
            for ln in open(f"/proc/{pid}/smaps_rollup"):
                if ln.startswith("Pss:"):
                    pss += int(ln.split()[1]); break
        except OSError:
            pass
    return rss, pss


def _my_ranks(root: int):
    """PIDs of wrf ranks under MY prterun (never other lanes' wrf.exe)."""
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        kids = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()
        for c in kids:
            c = int(c)
            try:
                comm = open(f"/proc/{c}/comm").read().strip()
            except OSError:
                comm = ""
            if comm.startswith("wrf"):
                out.append(c)
            else:
                stack.append(c)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, required=True)
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--stage-only", action="store_true", help="stage + print, do not run")
    a = ap.parse_args(argv)
    if QUIET.exists():
        print(f"cpu_rss: refuse -- /tmp/wrf_gpu2_quiet is set: {QUIET.read_text().strip()!r}", file=sys.stderr)
        return 2
    stage = ROOT / a.tag
    if stage.exists():
        print(f"cpu_rss: stage {stage} already exists; remove it or pick another --tag", file=sys.stderr)
        return 3
    stage.mkdir(parents=True)
    try:
        # assets + wrf.exe: symlink by PATH only (no resolve() -- resolving a CIFS-backed symlink can hang)
        for f in D5.iterdir():
            if f.name in INPUTS or f.name.startswith(("wrfout", "rsl.", "namelist")):
                continue
            os.symlink(str(f), stage / f.name)
        # the 3 inputs are read repeatedly over the run -> COPY them locally (real files, no CIFS read stalls)
        for name in INPUTS:
            src = (D5 / name).resolve()
            t = time.time()
            shutil.copyfile(src, stage / name)
            print(f"cpu_rss: copied {name} {Path(src).stat().st_size/1e6:.1f} MB in {time.time()-t:.1f}s",
                  file=sys.stderr)
        _set_namelist(D5 / "namelist.input", stage / "namelist.input", a.ranks, a.hours)
        (stage / "namelist.input").chmod(0o644)
        if a.stage_only:
            print(f"cpu_rss: staged {stage} (smoke test mode)")
            print((stage / "namelist.input").read_text().split("&domains")[1][:200])
            nx, ny = NPROC[a.ranks]
            print(f"nproc_x={nx} nproc_y={ny}; entries: {len(list(stage.iterdir()))}")
            return 0

        env = dict(os.environ, LD_LIBRARY_PATH=str(LIB_DIR), OMP_NUM_THREADS="1")
        cmd = ["taskset", "-c", CPUS, "nice", "-n", "10", str(MPI_DIR / "mpirun"),
               "--bind-to", "none", "-np", str(a.ranks), "./wrf.exe"]
        samples, stop = [], threading.Event()

        t0 = time.time()
        logf = open(stage / "stdout.log", "w")
        proc = subprocess.Popen(cmd, cwd=stage, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True)

        def _kill(sig, frame):  # never orphan the MPI ranks on the GPU/CPU after releasing the lock
            for s_ in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), s_)
                except OSError:
                    pass
                if s_ is signal.SIGTERM:
                    time.sleep(5)
            sys.exit(130)
        signal.signal(signal.SIGTERM, _kill); signal.signal(signal.SIGINT, _kill)

        def sample2():
            log = open(stage / "rss_samples.log", "w")
            log.write(f"start cmd={' '.join(cmd)}\n"); log.flush()
            while not stop.is_set():
                pids = _my_ranks(proc.pid)
                rss, pss = _rss_pss(pids)
                samples.append((time.time(), len(pids), rss, pss))
                log.write(f"{time.strftime('%H:%M:%S', time.gmtime())} ranks={len(pids)} "
                          f"sum_rss_kb={rss} sum_pss_kb={pss}\n"); log.flush()
                stop.wait(2)
            log.close()

        th = threading.Thread(target=sample2, daemon=True); th.start()
        rc = proc.wait()
        wall = time.time() - t0
        stop.set(); th.join(timeout=10); logf.close()
        try:  # make sure no rank survives past the measured window
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            pass

        peak_rss = max((s[2] for s in samples if s[1] > 0), default=0)
        peak_pss = max((s[3] for s in samples if s[1] > 0), default=0)
        peak_ranks = max((s[1] for s in samples), default=0)
        done = "SUCCESS COMPLETE WRF" in (stage / "stdout.log").read_text(errors="ignore")
        # also check rsl.error for success
        rsl_ok = any("SUCCESS COMPLETE WRF" in (stage / f"rsl.error.{i:04d}").read_text(errors="ignore")
                     for i in range(a.ranks)) if (stage / "rsl.error.0000").exists() else done
        res = {"tag": a.tag, "ranks": a.ranks, "hours": a.hours, "rc": rc, "wall_s": round(wall, 1),
               "wrf_success": bool(done or rsl_ok),
               "peak_total_rss_mib": round(peak_rss / 1024, 1), "peak_total_pss_mib": round(peak_pss / 1024, 1),
               "peak_ranks_seen": peak_ranks,
               "per_rank_rss_mib": round(peak_rss / 1024 / max(peak_ranks, 1), 1),
               "n_samples": len(samples), "wrf_exe": str(WRF_EXE),
               "namelist_sha256": subprocess.run(["sha256sum", str(stage / "namelist.input")],
                                                 capture_output=True, text=True).stdout.split()[0]}
        (stage / "rss_result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return 0 if rc == 0 else 1
    finally:
        if a.keep:
            print(f"cpu_rss: kept stage {stage}", file=sys.stderr)
        else:
            shutil.rmtree(stage, ignore_errors=True)
            print(f"cpu_rss: removed stage {stage}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
