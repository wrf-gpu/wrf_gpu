"""scripts/run_parallel_cases.sh helper: admission, scheduling, receipts, C-auto plan lookup, rolling compression (CPU)."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GIB = 1024**3


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pc = _load("parallel_cases")

FAKE_CLI = r'''
import json, os, signal, sys, time, pathlib
a = sys.argv
out = pathlib.Path(a[a.index("--output-dir") + 1]); out.mkdir(parents=True)
name = pathlib.Path(a[a.index("--input-dir") + 1]).name
if os.environ.get("FAKE_KILL_PARENT_" + name):
    time.sleep(0.5)
    os.kill(os.getppid(), signal.SIGTERM)
time.sleep(float(os.environ.get("FAKE_SLEEP_" + name, os.environ.get("FAKE_SLEEP", "1.5"))))
(out / "wrfout_d01_2026-01-01_00:00:00").write_bytes(b"x")
flag = os.environ.get("FAKE_PLAN_FLAG")
if flag:
    pathlib.Path(flag).write_text("plan")
print(json.dumps({"effective_hours": 2, "all_domains_finite": True, "all_outputs_present": True}))
sys.exit(int(os.environ.get("FAKE_RC", "0")))
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI)
    cases = []
    for name in ("c1", "c2", "c3"):
        (tmp_path / "cases" / name).mkdir(parents=True)
        cases.append(str(tmp_path / "cases" / name))
    monkeypatch.setenv("GPUWRF_GPU_LOCK_HELD", "1")
    monkeypatch.setattr(pc, "vram_by_pid", lambda: {})
    for key in pc.EXPLICIT_ALLOCATOR_ENV:
        monkeypatch.delenv(key, raising=False)
    base = ["--out-root", str(tmp_path / "out"), "--gpu-log-interval", "0", "--min-mem-available-gb", "0",
            "--host-gb-per-case", "0.001", "--cli-json", json.dumps([sys.executable, str(cli)])]
    return tmp_path, cases, base


def test_case_names_take_the_shortest_unique_tail():
    assert pc.case_names([Path("/a/x"), Path("/b/y")]) == ["x", "y"]
    assert pc.case_names([Path("/w/wg_0227_a1/run/run"), Path("/w/wg_0502_a1/run/run")]) == \
        ["wg_0227_a1_run_run", "wg_0502_a1_run_run"]
    with pytest.raises(ValueError):
        pc.case_names([Path("/a/x"), Path("/a/x")])


def test_fits_vram_host_parallel_and_exclusive_sizing_runs():
    def case(need, host=1):
        c = pc.Case(Path("/x"), "x", Path("/o"))
        c.sizing, c.host_kb = {"need_bytes": need}, host
        return c
    four = [case(4 * GIB) for _ in range(3)]
    assert pc.fits(four[:1], four[1], 9 * GIB, 10, 8)
    assert not pc.fits(four[:2], four[2], 9 * GIB, 10, 8)          # VRAM
    assert not pc.fits(four[:1], four[1], 9 * GIB, 1, 8)           # host
    assert not pc.fits(four[:1], four[1], 9 * GIB, 10, 1)          # --max-parallel
    assert pc.fits([], case(None), 0, 0, 1)                         # sizing run alone, whatever the budget
    assert not pc.fits(four[:1], case(None), 99 * GIB, 99, 8)       # ... but only alone
    assert not pc.fits([case(None)], four[0], 99 * GIB, 99, 8)       # and nothing next to it


def test_vram_admission_limits_concurrency_and_writes_receipts(fake, monkeypatch):
    tmp, cases, base = fake
    sizer = lambda ds, a, e: [{"source": "c-auto-plan", "need_bytes": 4 * GIB} for _ in ds]  # noqa: E731
    seen = []
    vram = lambda: {p: 700 for p in seen}  # noqa: E731

    def capture(*args, **kwargs):  # every child pid reports 700 MiB
        proc = real_popen(*args, **kwargs)
        seen.append(proc.pid)
        return proc
    real_popen = pc.subprocess.Popen
    monkeypatch.setattr(pc.subprocess, "Popen", capture)
    rc = pc.main([*base, "--vram-sample-interval", "0.5", *cases, "--", "--hours", "2"], sizer=sizer, gpu=lambda: (32 * GIB, 9 * GIB), vram=vram)
    assert rc == 0
    run = json.loads((tmp / "out" / "parallel_run.json").read_text())
    assert run["peak_concurrency"] == 2
    assert [e["start"] for e in run["events"]] == ["c1", "c2", "c3"]
    assert run["events"][1]["t_s"] < 1.0 <= run["events"][2]["t_s"]  # c3 waits until c1/c2 (1.5 s each) end
    assert run["case_hours"] == 6 and run["wall_s_per_case_hour"] > 0
    assert {v["vram_peak_mib"] for v in run["cases"].values()} == {700}
    for name in ("c1", "c2", "c3"):
        rec = json.loads((tmp / "out" / name / "receipt.json").read_text())
        assert rec["rc"] == 0 and rec["outputs"][0]["file"].startswith("wrfout_d01") and rec["outputs"][0]["d"] == "d01"
        assert 1.0 <= rec["outputs"][0]["t_published_s"] <= rec["wall_s"] + 1.0  # fake CLI publishes after 1.5 s
        assert rec["command"][-4:] == ["--output-dir", str(tmp / "out" / name / "wrfout"), "--hours", "2"]
        assert rec["cli_summary"]["all_domains_finite"] is True
        assert "GPUWRF_GPU_LOCK_HELD=1" in rec["env"] and not any("TOKEN" in e for e in rec["env"])
        assert set(rec["source"]) == {"gpuwrf_path", "git_head", "src_tree", "dirty"} and rec["source"]["gpuwrf_path"]


def test_case_without_plan_runs_alone_then_the_rest_in_parallel(fake, monkeypatch):
    tmp, cases, base = fake
    flag = tmp / "plan_recorded"
    monkeypatch.setenv("FAKE_PLAN_FLAG", str(flag))
    plan = tmp / "cache" / "k.json"

    def sizer(ds, a, e):
        rec = {"plan_path": str(plan)}
        return [rec | ({"source": "c-auto-plan", "need_bytes": GIB} if flag.exists() else {"source": "sizing-run", "need_bytes": None})
                for _ in ds]
    rc = pc.main([*base, *cases], sizer=sizer, gpu=lambda: (32 * GIB, 30 * GIB))
    assert rc == 0
    run = json.loads((tmp / "out" / "parallel_run.json").read_text())
    assert [e["start"] for e in run["events"]] == ["c1", "c2", "c3"]
    assert [e["running"] for e in run["events"]] == [1, 1, 2]
    assert json.loads((tmp / "out" / "c1" / "receipt.json").read_text())["sizing"]["source"] == "sizing-run"
    assert json.loads((tmp / "out" / "c3" / "receipt.json").read_text())["sizing"]["source"] == "c-auto-plan"
    assert json.loads(plan.with_suffix(".measured.json").read_text())["vmhwm_kb_max"] > 0  # host peak recorded


def test_measured_fb_admits_cases_without_plan_and_floors_a_thin_plan(tmp_path):
    side = tmp_path / "k.measured.json"
    side.write_text(json.dumps({"vmhwm_kb_max": 10_000_000, "vram_peak_mib_max": 6000}))
    no_plan = pc.apply_vram_floor({"plan_path": str(tmp_path / "k.json"), "source": "sizing-run", "need_bytes": None})
    assert no_plan["source"] == "measured-fb" and no_plan["need_bytes"] == int(6000 * 1.1 * 2**20)
    assert no_plan["child_env"] == {"GPUWRF_MIN_FREE_VRAM_GIB": f"{6000 * 1.1 / 1024:.2f}"}
    thin = pc.apply_vram_floor({"plan_path": str(tmp_path / "k.json"), "source": "c-auto-plan", "need_bytes": GIB})
    assert thin["source"] == "c-auto-plan" and thin["need_bytes"] == int(6000 * 1.1 * 2**20)
    assert pc.host_need_kb({"plan_path": str(tmp_path / "k.json")}, 16.0) == (11_000_000, "1.1 x measured VmHWM")
    assert pc.apply_vram_floor({"plan_path": str(tmp_path / "other.json"), "need_bytes": None})["need_bytes"] is None


def test_case_larger_than_the_gpu_is_not_admitted(fake):
    tmp, cases, base = fake
    rc = pc.main([*base, cases[0]], sizer=lambda ds, a, e: [{"need_bytes": 40 * GIB}], gpu=lambda: (32 * GIB, 30 * GIB))
    assert rc == 1
    assert json.loads((tmp / "out" / "c1" / "receipt.json").read_text())["rc"] == 75


def test_refusals(fake, monkeypatch):
    tmp, cases, base = fake
    sizer, gpu = (lambda ds, a, e: [{"need_bytes": GIB} for _ in ds]), (lambda: (32 * GIB, 30 * GIB))
    (tmp / "out" / "c2").mkdir(parents=True)
    assert pc.main([*base, *cases], sizer=sizer, gpu=gpu) == 2           # existing output dir
    assert not (tmp / "out" / "c1").exists()
    shutil.rmtree(tmp / "out" / "c2")
    monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.5")
    assert pc.main([*base, *cases], sizer=sizer, gpu=gpu) == 2           # explicit fraction overrides sizing
    monkeypatch.delenv("XLA_CLIENT_MEM_FRACTION")
    monkeypatch.delenv("GPUWRF_GPU_LOCK_HELD")
    dev_lock = tmp / "dev.lock"
    dev_lock.touch()
    monkeypatch.setattr(pc, "DEV_LOCK_FILE", dev_lock)
    assert pc.main([*base, *cases], sizer=sizer, gpu=gpu) == 2           # dev lock infra present, lock not held
    with pytest.raises(SystemExit):
        pc.main([*base, *cases, "--", "--output-dir", "x"], sizer=sizer, gpu=gpu)
    with pytest.raises(SystemExit):
        pc.main([*base, "--compress", *cases, "--", "--checkpoint-dir", "x"], sizer=sizer, gpu=gpu)


def test_public_machine_runs_without_the_dev_lock(fake, monkeypatch):
    tmp, cases, base = fake
    monkeypatch.delenv("GPUWRF_GPU_LOCK_HELD")
    monkeypatch.delenv("GPUWRF_REQUIRE_GPU_LOCK", raising=False)
    monkeypatch.setattr(pc, "DEV_LOCK_FILE", tmp / "absent.lock")
    rc = pc.main([*base, cases[0]], sizer=lambda ds, a, e: [{"need_bytes": GIB}], gpu=lambda: (32 * GIB, 30 * GIB))
    assert rc == 0
    lock = json.loads((tmp / "out" / "parallel_run.json").read_text())["gpu_lock"]
    assert lock["required"] is False and lock["held"] is False
    monkeypatch.setenv("GPUWRF_REQUIRE_GPU_LOCK", "1")
    shutil.rmtree(tmp / "out")
    assert pc.main([*base, cases[0]], sizer=lambda ds, a, e: [{"need_bytes": GIB}], gpu=lambda: (32 * GIB, 30 * GIB)) == 2


def test_pool_gib_pins_every_case(fake):
    tmp, cases, base = fake
    rc = pc.main([*base, "--pool-gib", "3", *cases[:1]], sizer=None, gpu=lambda: (32 * GIB, 30 * GIB))
    assert rc == 0
    sizing = json.loads((tmp / "out" / "c1" / "receipt.json").read_text())["sizing"]
    assert sizing["need_bytes"] == 4 * GIB
    assert sizing["child_env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert float(sizing["child_env"]["XLA_CLIENT_MEM_FRACTION"]) == pytest.approx(3 / 32, abs=1e-6)
    assert sizing["child_env"]["GPUWRF_MIN_FREE_VRAM_GIB"] == "1.0"  # pool reserved before the case preflight


def _synthetic_case(path: Path, nx: int = 30) -> Path:
    netCDF4 = pytest.importorskip("netCDF4")
    path.mkdir(parents=True)
    (path / "namelist.input").write_text(" &time_control\n start_year = 2026, 2026,\n run_hours = 3,\n /\n"
                                         " &domains\n max_dom = 2,\n /\n")
    for d in ("d01", "d02"):
        with netCDF4.Dataset(path / f"wrfinput_{d}", "w") as ds:
            ds.createDimension("west_east", nx)
            ds.createDimension("south_north", 20)
            ds.DX = ds.DY = 3000.0
    return path


def test_product_sizing_finds_the_plan_the_case_cli_records(tmp_path):
    """Keyed in a child with the case environment, as the case CLI keys it; dates do not split plans, geometry does."""
    from gpuwrf.runtime import gpu_allocator as ga
    env = {**os.environ, "GPUWRF_JAX_CACHE_DIR": str(tmp_path / "cache"), "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(REPO / "src")}
    a = _synthetic_case(tmp_path / "a")
    size = lambda d: pc.product_sizing([d], ["--max-dom", "2"], env)[0]  # noqa: E731
    first = size(a)
    assert first["source"] == "sizing-run" and first["need_bytes"] is None
    programs = {d: {"v": {"argument_size_in_bytes": GIB, "output_size_in_bytes": GIB // 2, "temp_size_in_bytes": GIB,
                          "alias_size_in_bytes": GIB // 2, "generated_code_size_in_bytes": 1000}} for d in ("d01", "d02")}
    plan = ga.write_plan(Path(first["plan_path"]), first["plan_key"], 2, programs, live_peak_bytes=GIB)
    again = size(a)
    assert again["source"] == "c-auto-plan"
    assert again["need_bytes"] == plan["budget_bytes"] + ga.outside_headroom(plan) and again["headroom_bytes"] >= GIB
    later = _synthetic_case(tmp_path / "b")
    text = (later / "namelist.input").read_text().replace("start_year = 2026", "start_year = 2027")
    (later / "namelist.input").write_text(text)
    other = _synthetic_case(tmp_path / "c", nx=31)
    both = pc.product_sizing([later, other], ["--max-dom", "2"], env)
    assert both[0]["plan_key"] == first["plan_key"] and both[0]["source"] == "c-auto-plan"
    assert both[1]["source"] == "sizing-run"
    assert size(a)["plan_key"] != pc.product_sizing([a], ["--max-dom", "2"], env | {"GPUWRF_SOME_FLAG": "1"})[0]["plan_key"]
    with pytest.raises(ValueError):
        pc.product_sizing([a], ["--max-dom", "1"], env)


def test_rolling_compression_is_lossless_and_waits_for_finished_frames(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    if shutil.which("nccopy") is None:
        pytest.skip("nccopy not installed")
    cw = _load("compress_wrfout")
    wrfout = tmp_path / "case" / "wrfout"
    wrfout.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for h in range(2):
        with netCDF4.Dataset(wrfout / f"wrfout_d01_2026-01-01_0{h}:00:00", "w") as ds:
            ds.createDimension("x", 64)
            v = ds.createVariable("T", "f4", ("x",))
            v[:] = np.where(np.arange(64) == 3, np.nan, rng.standard_normal(64)).astype("f4")
            v.units = "K"
    assert [p.name[-8:] for p in cw.finished(tmp_path, set())] == ["00:00:00"]  # newest frame may still be written
    (tmp_path / ".cases_done").write_text("x")
    todo = cw.finished(tmp_path, set())
    assert len(todo) == 2
    before = netCDF4.Dataset(todo[0])["T"][:].filled(np.nan).copy()
    rec = cw.compress_one(str(todo[0]))
    assert rec["value_identical"] and rec["mismatches"] == []
    np.testing.assert_array_equal(netCDF4.Dataset(todo[0])["T"][:].filled(np.nan), before)


# --- review-s2small Item 18 C1: exact allowlist for the case options (the product argparse expands abbreviations) -----

ABBREVIATION_PROBES = [["--input-d", "x"], ["--input-d=x"], ["--output-d", "x"], ["--output-d=x"],
                       ["--checkpoint-d", "x"], ["--checkpoint-d=x"], ["--resume-check", "x"], ["--resume-check=x"]]


def _run_parser():
    from gpuwrf import cli
    return cli.build_parser()._subparsers._group_actions[0].choices["run"]


@pytest.mark.parametrize("probe", ABBREVIATION_PROBES)
def test_abbreviations_are_refused_although_the_product_parser_expands_them(probe):
    from gpuwrf import cli
    base = ["run", "--input-dir", "in", "--output-dir", "out"]
    assert vars(cli.build_parser().parse_args([*base, *probe])) != vars(cli.build_parser().parse_args(base))  # real bypass
    with pytest.raises(ValueError, match="not an allowed exact case option"):
        pc.check_cli_args(probe)
    with pytest.raises(SystemExit):
        pc.parse(["--out-root", "o", "case", "--", *probe])


def test_allowlist_matches_the_product_parser_and_classifies_every_option():
    run = _run_parser()
    arity = {}
    for action in run._actions:
        for opt in action.option_strings:
            arity[opt] = 0 if action.nargs == 0 else 1
    assert set(pc.CLI_ALLOWED) | set(pc.CLI_REFUSED) == set(arity)  # a new CLI option forces a decision here
    assert not set(pc.CLI_ALLOWED) & set(pc.CLI_REFUSED)
    assert {k: arity[k] for k in pc.CLI_ALLOWED} == pc.CLI_ALLOWED


def test_case_option_parser_accepts_exact_forms_and_refuses_the_rest():
    ok = ["--domains-from-namelist", "--hours", "24", "--emit-initial-history", "--max-dom=3", "--no-aot-prefetch"]
    assert pc.check_cli_args(ok) == ok
    for bad in (["--hours"], ["--hours", "--emit-initial-history"], ["24"], ["--feedback=1"], ["--hours="],
                ["--namelist", "x"], ["--scratch-dir=x"], ["--checkpoint-dir", "x"], ["--dry-run"], ["--hour", "3"]):
        with pytest.raises(ValueError):
            pc.check_cli_args(bad)


# --- review-s2small Item 18 C2: no started case or helper survives a launcher failure ------------------------------

def _alive(proc) -> bool:
    return proc.poll() is None


def test_failed_second_launch_tears_down_the_first_case(fake, monkeypatch):
    tmp, cases, base = fake
    monkeypatch.setattr(pc, "TEARDOWN_GRACE_S", 3.0)
    monkeypatch.setenv("FAKE_SLEEP", "30")
    real, started = pc.subprocess.Popen, []

    def popen(*args, **kwargs):
        if started:
            raise OSError("injected spawn failure")
        started.append(real(*args, **kwargs))
        return started[-1]
    monkeypatch.setattr(pc.subprocess, "Popen", popen)
    t0 = time.time()
    with pytest.raises(OSError, match="injected spawn failure"):
        pc.main([*base, *cases[:2]], sizer=lambda ds, a, e: [{"need_bytes": GIB} for _ in ds], gpu=lambda: (32 * GIB, 30 * GIB))
    assert not _alive(started[0]) and time.time() - t0 < 20
    r1 = json.loads((tmp / "out" / "c1" / "receipt.json").read_text())
    r2 = json.loads((tmp / "out" / "c2" / "receipt.json").read_text())
    assert r1["rc"] != 0 and r1["note"].startswith("stopped: launcher error: OSError")
    assert r2["note"].startswith("launch failed: OSError")
    assert json.loads((tmp / "out" / "parallel_run.json").read_text())["launcher_error"].startswith("OSError")


def test_failed_receipt_write_tears_down_the_running_case(fake, monkeypatch):
    tmp, cases, base = fake
    monkeypatch.setattr(pc, "TEARDOWN_GRACE_S", 3.0)
    monkeypatch.setenv("FAKE_SLEEP_c1", "0.5")
    monkeypatch.setenv("FAKE_SLEEP_c2", "30")
    real_write, real_popen, started, calls = pc.write_receipt, pc.subprocess.Popen, [], []

    def write(c, cli):
        calls.append(c.name)
        if len(calls) == 1:
            raise OSError("injected receipt failure")
        return real_write(c, cli)

    def popen(*args, **kwargs):
        started.append(real_popen(*args, **kwargs))
        return started[-1]
    monkeypatch.setattr(pc, "write_receipt", write)
    monkeypatch.setattr(pc.subprocess, "Popen", popen)
    with pytest.raises(OSError, match="injected receipt failure"):
        pc.main([*base, *cases[:2]], sizer=lambda ds, a, e: [{"need_bytes": GIB} for _ in ds], gpu=lambda: (32 * GIB, 30 * GIB))
    assert len(started) == 2 and not any(_alive(p) for p in started)
    assert json.loads((tmp / "out" / "c2" / "receipt.json").read_text())["note"].startswith("stopped: launcher error")


def test_sigterm_stops_every_case_and_restores_the_handler(fake, monkeypatch):
    tmp, cases, base = fake
    monkeypatch.setenv("FAKE_SLEEP", "30")
    monkeypatch.setenv("FAKE_KILL_PARENT_c1", "1")  # c1 signals the launcher (this process) after 0.5 s
    before = signal.getsignal(signal.SIGTERM)
    t0 = time.time()
    rc = pc.main([*base, *cases[:2]], sizer=lambda ds, a, e: [{"need_bytes": GIB} for _ in ds], gpu=lambda: (32 * GIB, 30 * GIB))
    assert rc == 1 and time.time() - t0 < 20
    run = json.loads((tmp / "out" / "parallel_run.json").read_text())
    assert run["stop_reason"] == f"signal {int(signal.SIGTERM)}"
    assert all(v["rc"] is not None and v["rc"] < 0 for v in run["cases"].values())
    assert signal.getsignal(signal.SIGTERM) == before


def test_host_watchdog_stops_every_case(fake, monkeypatch):
    tmp, cases, base = fake
    monkeypatch.setenv("FAKE_SLEEP", "30")
    t0 = time.time()
    rc = pc.main([*base, "--min-mem-available-gb", "1e9", *cases[:2]],
                 sizer=lambda ds, a, e: [{"need_bytes": GIB} for _ in ds], gpu=lambda: (32 * GIB, 30 * GIB))
    assert rc == 1 and time.time() - t0 < 20
    run = json.loads((tmp / "out" / "parallel_run.json").read_text())
    assert run["stop_reason"].startswith("host watchdog") and all(v["rc"] < 0 for v in run["cases"].values())


def test_older_product_without_lock_policy_still_requires_the_lock(fake, monkeypatch):
    import gpuwrf.runtime.gpu_preflight as gp
    tmp, cases, base = fake
    monkeypatch.delenv("GPUWRF_GPU_LOCK_HELD")
    monkeypatch.setattr(pc, "DEV_LOCK_FILE", tmp / "absent.lock")
    monkeypatch.delattr(gp, "gpu_lock_required")
    assert pc.main([*base, cases[0]], sizer=lambda ds, a, e: [{"need_bytes": GIB}], gpu=lambda: (32 * GIB, 30 * GIB)) == 2


# --- review-s2small/review-writer Item 18 C3: helper DESCENDANTS (compressor pool workers, dmon) stop with the launcher -----

HELPER_WITH_WORKER = r'''
import json, os, subprocess, sys, time, pathlib
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pathlib.Path(sys.argv[1]).write_text(json.dumps({"parent": os.getpid(), "child": child.pid}))
time.sleep(60)
'''


def _pid_alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1].split()[0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


def _intercept_helper(monkeypatch, tmp, marker: str, record: Path, fail_second_case: bool):
    helper = tmp / "helper_with_worker.py"
    helper.write_text(HELPER_WITH_WORKER)
    real, cases_started, helpers = pc.subprocess.Popen, [], []

    def popen(args, **kwargs):
        if any(marker in str(x) for x in args):
            helpers.append(real([sys.executable, str(helper), str(record)], **kwargs))
            deadline = time.time() + 10
            while not record.exists() and time.time() < deadline:
                time.sleep(0.02)
            return helpers[-1]
        if "--input-dir" in args:
            if fail_second_case and cases_started:
                raise OSError("injected second-case spawn failure")
            cases_started.append(real(args, **kwargs))
            return cases_started[-1]
        return real(args, **kwargs)
    monkeypatch.setattr(pc.subprocess, "Popen", popen)
    return cases_started, helpers


def _all_dead(pids, timeout_s=5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline and any(_pid_alive(p) for p in pids):
        time.sleep(0.05)
    return not any(_pid_alive(p) for p in pids)


@pytest.mark.parametrize("flags,marker", [(["--compress"], "compress_wrfout.py"), (["--gpu-log-interval", "1"], "nvidia-smi")])
def test_launch_failure_stops_helper_groups_including_workers(fake, monkeypatch, flags, marker):
    tmp, cases, base = fake
    monkeypatch.setattr(pc, "TEARDOWN_GRACE_S", 0.5)
    monkeypatch.setattr(pc, "HELPER_GRACE_S", 0.5)
    monkeypatch.setenv("FAKE_SLEEP", "30")
    record = tmp / "helper.json"
    cases_started, helpers = _intercept_helper(monkeypatch, tmp, marker, record, fail_second_case=True)
    try:
        with pytest.raises(OSError, match="second-case spawn failure"):
            pc.main([*base, *flags, *cases[:2]], sizer=lambda ds, a, e: [{"need_bytes": GIB} for _ in ds],
                    gpu=lambda: (32 * GIB, 30 * GIB))
        own = json.loads(record.read_text())
        assert _all_dead([cases_started[0].pid, own["parent"], own["child"]]), "helper worker survived"
    finally:
        if record.exists():
            for pid in json.loads(record.read_text()).values():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)


def test_sigterm_stops_compressor_workers_too(fake, monkeypatch):
    tmp, cases, base = fake
    monkeypatch.setattr(pc, "HELPER_GRACE_S", 0.5)
    monkeypatch.setenv("FAKE_SLEEP", "30")
    monkeypatch.setenv("FAKE_KILL_PARENT_c1", "1")
    record = tmp / "helper.json"
    _intercept_helper(monkeypatch, tmp, "compress_wrfout.py", record, fail_second_case=False)
    try:
        assert pc.main([*base, "--compress", *cases[:2]], sizer=lambda ds, a, e: [{"need_bytes": GIB} for _ in ds],
                       gpu=lambda: (32 * GIB, 30 * GIB)) == 1
        own = json.loads(record.read_text())
        assert _all_dead([own["parent"], own["child"]]), "compressor worker survived SIGTERM stop"
    finally:
        if record.exists():
            for pid in json.loads(record.read_text()).values():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
