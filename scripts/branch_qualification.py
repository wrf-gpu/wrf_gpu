"""Checkpoint-branch GPU qualification driver (v0.5.0 item 0.1): G1 null resume, G2 null branch, G3 edited series.

    python scripts/branch_qualification.py prep|gpu|verify|cleanup PLAN.json RUN_ROOT

The launcher (lane gpu_arm.sh) runs this file from a detached snapshot of the lane commit and sets
V50_DEADLINE_EPOCH (window end of this share), V50_COMMIT (the pinned snapshot commit) and, for the CPU stub dry run
only, V50_DRY_RUN=1 + V50_STUB_PYTHONPATH + V50_E41_TOOL. ``gpu`` runs inside scripts/with_gpu_lock.sh; every other
phase is CPU-only (JAX_PLATFORMS=cpu).

PLAN.json (schema gpuwrf.branch_qualification.plan.v1): case_src + case_sha256 (every input incl. namelist.input),
neutral/variant surface-control files, hours, branch_step (root steps), cli_args, emit_initial_history, caps_s per
arm, order, cpu_reserve_s, kill_grace_s, floors_gib {data, root}, projection_gib {data, root}, own_cap_gib,
source_fingerprint, lease, run_parent, dry_run_parent, evidence_dir, nightly (timer/status/static window).

Arms (fresh process each; only C may compile): C continuous 0->H with checkpoints every branch_step; G2
--branch-from C@branch_step, empty manifest; G3 branch_series --mode inprocess (all variants); G3p first variant
--mode process; R1 0->seam with checkpoints; R2 ordinary --resume-checkpoint R1@branch_step --extend-run -> H.

Admission before EVERY arm (refuse, never truncate): QUIET flag; DATA free - projection >= floor and ROOT >= floor;
own run-root bytes + projection <= own cap; the arm's whole budget (cap incl. teardown grace) + CPU reserve ends
before min(window deadline, lease expiry); it does not overlap the ALISIOS Nightly protected window (18:50-06:00
Atlantic/Canary local, DST-correct; the timer is a 10-min poller, its next elapse is recorded only); nightly_gate(): the
latest issue (scheduled 18z floor) is finished under the ALISIOS rule of 2026-11-11T03:39Z (exact verdict DONE|DEGRADED,
tz-aware done_utc <= now, steps.gpu object DONE, no busy step) and no recent issue is busy/unreadable. While an arm runs, the
disk floors and the deadline are re-checked every 5 s (continuous); TERM/INT/HUP tear the arm's session down.
The GPU lock holder file must carry this process's token at every arm start (no foreign holder); queued lock waiters
(the next lane of a shared window) are recorded, not refused.

verify: frame inventory from the namelist alone (compare_branch_streams.py inventory; seam from the generation),
digests (per-file SHA-256) of every stream, comparisons with exit code AND parsed verdict, platform gpu in every
measured process (acct receipts + run payload device), compile counters (cache misses, cache-disabled and fresh
compiles, new cache entries) for every arm after C, mandatory E41 receipt on the fresh cache. Missing evidence of
any kind is FAIL; PASS needs every gate. cleanup: journaled deletion of own RAW by fixed names (sha256 per frame from
the digests), evidence copied to evidence_dir.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

SNAP = Path(__file__).resolve().parents[1]
PY = sys.executable
GIB = 1 << 30
SCHEMA = "gpuwrf.branch_qualification.plan.v1"
RUN_NAME = re.compile(r"gpuarm_\d{8}T\d{6}Z(_dry)?")
ARMS = ("C", "G2", "G3", "G3p", "R1", "R2")
ACCT_ARMS = ("C", "G2", "R1", "R2")
NIGHTLY_BUSY = ("RUNNING", "STARTED", "PENDING", "WAITING", "READY", "QUEUED")


def _busy(value):
    """Refusal side is case-insensitive ('running' is busy); acceptance stays an exact match."""
    return str(value).strip().upper() in NIGHTLY_BUSY


QUIET = Path("/tmp/wrf_gpu2_quiet")
_spec = importlib.util.spec_from_file_location("compare_branch_streams", SNAP / "scripts" / "compare_branch_streams.py")
cmpmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmpmod)


def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log(msg):
    print(f"[branch_qualification {utc()}] {msg}", file=sys.stderr, flush=True)


def dry() -> bool:
    return os.environ.get("V50_DRY_RUN") == "1"


def free(path) -> int:
    return shutil.disk_usage(path).free


# ----------------------------------------------------------------------------------------------------------- plan
def load_plan(path) -> dict:
    plan = load(path)
    if plan.get("schema") != SCHEMA:
        raise SystemExit(f"plan schema must be {SCHEMA}")
    need = {"name", "case_src", "case_sha256", "neutral_controls", "variants", "hours", "branch_step", "cli_args",
            "emit_initial_history", "caps_s", "order", "cpu_reserve_s", "kill_grace_s", "floors_gib", "projection_gib",
            "own_cap_gib", "source_fingerprint", "lease", "run_parent", "dry_run_parent", "evidence_dir", "nightly",
            "model", "compile_allowed_arms", "case_in_place", "e41_decisions"}
    missing = sorted(need - set(plan))
    if missing:
        raise SystemExit(f"plan misses {missing}")
    model = plan["model"]
    if model.get("kind") not in ("snapshot", "bundle_consumer"):
        raise SystemExit("model.kind must be snapshot or bundle_consumer")
    if model["kind"] == "bundle_consumer" and not {"bundle", "layer", "env"} <= set(model):
        raise SystemExit("bundle_consumer model needs bundle, layer and env (the exact env the layer was built with)")
    if not set(plan["compile_allowed_arms"]) <= set(ARMS):
        raise SystemExit("compile_allowed_arms must name arms")
    if dry() and model["kind"] == "bundle_consumer":
        # The CPU dry run never touches the sealed bundle or the real layer: a lane sandbox stands in for both
        # (sandbox/bundle/src -> read-only bundle src, sandbox/bundle/tools/bundle_consumer.sh = stand-in).
        sandbox = os.environ.get("V50_DRY_SANDBOX")
        if not sandbox:
            raise SystemExit("dry run of a bundle_consumer plan needs V50_DRY_SANDBOX")
        real = model["bundle"]
        plan["model"] = dict(model, bundle=f"{sandbox}/bundle", layer=f"{sandbox}/layer",
                             env={k: v.replace(real, f"{sandbox}/bundle") for k, v in model["env"].items()})
    if sorted(plan["order"]) != sorted(ARMS) or set(plan["caps_s"]) != set(ARMS):
        raise SystemExit(f"plan order/caps must cover exactly {ARMS}")
    if "namelist.input" not in plan["case_sha256"]:
        raise SystemExit("case_sha256 must pin namelist.input")
    nl = cmpmod._namelist_values(Path(plan["case_src"], "namelist.input").read_text())
    dt = float(nl["time_step"][0])
    total, seam = plan["hours"] * 3600.0, plan["branch_step"] * dt
    if total % dt or not 0 < seam < total:
        raise SystemExit(f"hours x 3600 must be a multiple of the root dt {dt} and the seam inside the run (E225)")
    if seam % 60:
        raise SystemExit("the seam must fall on a whole minute (R1 runs --minutes to the seam)")
    plan["_root_dt"], plan["_seam_minutes"] = dt, int(seam // 60)
    return plan


def expected_programs(namelist_text: str) -> list[str]:
    """The step programs a nested run must load, from the namelist alone: the root advance ``d01`` plus one fused
    cascade ``fused/dNN`` per direct child of d01 (its whole subtree runs inside it)."""
    nl = cmpmod._namelist_values(namelist_text)
    n = int(nl.get("max_dom", ["1"])[0])
    parent = [int(v) for v in nl.get("parent_id", ["1"] * n)][:n]
    return ["d01"] + [f"fused/d{c:02d}" for c in range(2, n + 1) if parent[c - 1] == 1]


def checked_root(plan, root) -> Path:
    root = Path(root)
    parent = Path(plan["dry_run_parent"] if dry() else plan["run_parent"])
    if root.parent != parent or not RUN_NAME.fullmatch(root.name) or root.name.endswith("_dry") != dry() or root.is_symlink():
        raise SystemExit(f"refused run root {root}")
    return root


# ------------------------------------------------------------------------------------------------------ admission
def lease_end(plan):
    lease = Path(plan["lease"])
    if dry():
        return None
    if not lease.is_file():
        raise RuntimeError(f"lease missing: {lease}")
    found = re.findall(r"^expires-utc:\s*(\S+)", lease.read_text(errors="replace"), re.M)
    if len(found) != 1:
        raise RuntimeError("lease has no single expires-utc line")
    return datetime.strptime(found[0].replace("Z", "")[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def hard_end(plan) -> float:
    ends = [float(os.environ["V50_DEADLINE_EPOCH"])]
    lease = lease_end(plan)
    if lease is not None:
        ends.append(lease)
    return min(ends)


def own_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(p.lstat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def nightly_window(cfg, now):
    """Protected spans of the ALISIOS Nightly in Atlantic/Canary LOCAL time (DST-correct): ALISIOS nightly9.require_time
    admits heavy starts from 19:00 local until 06:00 local minus its reserve, so every local night
    [protected_local[0], next day protected_local[1]] is protected (default 18:50-06:00 local; WET = UTC, WEST = UTC+1).
    The timer is a 10-min poller; its next elapse is recorded for information only. Returns (spans, info)."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(cfg.get("local_tz", "Atlantic/Canary"))
    lo, hi = cfg.get("protected_local", ["18:50", "06:00"])
    today = datetime.fromtimestamp(now, tz).date()
    spans = []
    for shift in (-1, 0, 1):
        d = today + timedelta(days=shift)
        a = datetime(d.year, d.month, d.day, int(lo[:2]), int(lo[3:]), tzinfo=tz)
        e = d + timedelta(days=1)
        b = datetime(e.year, e.month, e.day, int(hi[:2]), int(hi[3:]), tzinfo=tz)
        spans.append((a.timestamp(), b.timestamp()))
    info = {"local_tz": str(tz), "protected_local": [lo, hi]}
    try:
        out = subprocess.run(["systemctl", "--user", "show", cfg["timer"], "-p", "NextElapseUSecRealtime", "--value",
                              "--timestamp=unix"], capture_output=True, text=True, timeout=10)
        m = re.match(r"@(\d+)", out.stdout.strip())
        info["timer_next_poll_utc"] = utc(int(m.group(1))) if m else None
    except (OSError, subprocess.SubprocessError, KeyError):
        info["timer_next_poll_utc"] = None
    return spans, info


ISSUE_DIR = re.compile(r"[0-9]{8}_(12|18)z")


def scheduled_issue(now: float) -> str:
    """The 18z issue whose Nightly must already have run: the last 18z fallback start (~01:45Z) before now, minus a day
    (port of v50-evert evert_tools nightly, critic NR2 PASS)."""
    t = datetime.fromtimestamp(now, timezone.utc)
    fallback = t.replace(hour=1, minute=45, second=0, microsecond=0)
    last = fallback if t >= fallback else fallback - timedelta(days=1)
    return (last - timedelta(days=1)).strftime("%Y%m%d") + "_18z"


def _read_status(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("STATUS.json is not an object")
    return value


def nightly_gate(status_root, now: float) -> dict:
    """ALISIOS rule 2026-11-11T03:39Z ('Nightly fertig') with the critic hardening (port of v50-evert 7face710):
    latest issue = newest [0-9]{8}_(12|18)z dir not older than the scheduled 18z issue (a SKIPPED 12z is the only issue
    passed over, never an 18z); its STATUS.json must exist, be an object with issue == dir, verdict exactly DONE or
    DEGRADED, a tz-aware ISO done_utc <= now, steps.gpu an object with status exactly DONE, and no step in a busy state.
    Any recent issue (4 days) with a busy verdict/step or an unreadable STATUS refuses. Everything else fails closed."""
    root = Path(status_root)
    sched = scheduled_issue(now)
    doc = {"scheduled_issue": sched, "problems": []}
    why = doc["problems"].append
    try:
        dirs = sorted(p.name for p in root.iterdir() if p.is_dir() and ISSUE_DIR.fullmatch(p.name))
    except OSError as exc:
        why(f"Nightly status root unreadable: {exc}")
        doc["ok"] = False
        return doc
    t = datetime.fromtimestamp(now, timezone.utc)
    recent = {(t - timedelta(days=k)).strftime("%Y%m%d") for k in range(0, 4)}
    busy = []
    for name in dirs:
        path = root / name / "STATUS.json"
        if name[:8] not in recent or not path.exists():
            continue
        try:
            st = _read_status(path)
        except (OSError, ValueError, RecursionError):
            busy.append(f"{name}:unreadable")
            continue
        if _busy(st.get("verdict")):
            busy.append(f"{name}:verdict={st.get('verdict')}")
        steps = st.get("steps") if isinstance(st.get("steps"), dict) else {}
        busy += [f"{name}:{k}={v.get('status')}" for k, v in steps.items() if isinstance(v, dict) and _busy(v.get("status"))]
    doc["busy"] = busy
    if busy:
        why(f"Nightly work active or unreadable: {busy}")
    candidates = sorted(set([n for n in dirs if n >= sched] + [sched]), reverse=True)
    chosen = None
    for name in candidates:  # newest first; only a SKIPPED 12z is passed over, the first 18z always decides
        if name.endswith("_12z"):
            try:
                st = _read_status(root / name / "STATUS.json")
            except (OSError, ValueError, RecursionError):
                chosen = name  # decided below: missing/unreadable refuses
                break
            if st.get("verdict") == "SKIPPED" and st.get("issue") == name:
                continue  # a 12z that did not run ('18z-primaer')
        chosen = name
        break
    doc["latest_issue"] = chosen
    try:
        st = _read_status(root / chosen / "STATUS.json")
    except (OSError, ValueError, RecursionError) as exc:
        why(f"Nightly {chosen} STATUS missing/unreadable: {exc}")
        doc["ok"] = False
        return doc
    steps = st.get("steps") if isinstance(st.get("steps"), dict) else {}
    gpu = steps.get("gpu") if isinstance(steps.get("gpu"), dict) else {}
    done = st.get("done_utc")
    try:
        parsed = datetime.fromisoformat(done.replace("Z", "+00:00")) if isinstance(done, str) and done else None
        done_ok = parsed is not None and parsed.tzinfo is not None and parsed.timestamp() <= now
    except ValueError:
        done_ok = False
    doc.update(verdict=st.get("verdict"), done_utc=done, gpu_status=gpu.get("status"))
    if st.get("issue") != chosen:
        why(f"Nightly STATUS issue {st.get('issue')!r} != {chosen}")
    if st.get("verdict") not in ("DONE", "DEGRADED"):
        why(f"Nightly {chosen} not finished (verdict {st.get('verdict')!r}; need DONE or DEGRADED)")
    if not done_ok:
        why(f"Nightly {chosen} done_utc missing/invalid/naive/future ({done!r})")
    if gpu.get("status") != "DONE":
        why(f"Nightly {chosen} steps.gpu.status {gpu.get('status')!r} != DONE (steps.gpu must be an object)")
    running = [k for k, v in steps.items() if isinstance(v, dict) and _busy(v.get("status"))]
    if running:
        why(f"Nightly {chosen} steps busy: {running}")
    doc["ok"] = not doc["problems"]
    return doc


def admit(plan, root: Path, budget_s: float, *, own_holder_token: str | None = None) -> dict:
    """Refuse (RuntimeError) an arm whose whole budget is not admissible now. Writes ADMISSION.jsonl."""
    now = time.time()
    rec = {"utc": utc(now), "budget_s": budget_s}
    try:
        if QUIET.exists():
            raise RuntimeError(f"QUIET active ({QUIET})")
        fl, pr = plan["floors_gib"], plan["projection_gib"]
        rec["free_gib"] = {"data": free(plan["data_mount"]) / GIB, "root": free("/") / GIB}
        # projection = peak own bytes of the whole run; only the part not yet written is still to come
        rec["own_gib"] = own_bytes(root) / GIB
        growth = max(0.0, pr["data"] - rec["own_gib"])
        rec["remaining_growth_gib"] = growth
        if dry():
            rec["free_gib"]["data_check"] = "dry run: no DATA writes"
        elif free(plan["data_mount"]) - growth * GIB < fl["data"] * GIB:
            raise RuntimeError(f"DATA projected floor {fl['data']} GiB: free {rec['free_gib']['data']:.1f} - growth {growth:.1f}")
        if free("/") - pr["root"] * GIB < fl["root"] * GIB:
            raise RuntimeError(f"ROOT floor {fl['root']} GiB: free {rec['free_gib']['root']:.1f} - {pr['root']}")
        if rec["own_gib"] + growth > plan["own_cap_gib"]:
            raise RuntimeError(f"own byte cap {plan['own_cap_gib']} GiB: own {rec['own_gib']:.1f} + growth {growth:.1f}")
        end = hard_end(plan)
        rec["hard_end_utc"] = utc(end)
        if now + budget_s + plan["cpu_reserve_s"] > end:
            raise RuntimeError(f"budget {budget_s:.0f} s + CPU reserve {plan['cpu_reserve_s']} s would cross the hard end {utc(end)}")
        cfg = plan["nightly"]
        if cfg and not dry():
            spans, info = nightly_window(cfg, now)
            rec["nightly_window"] = {"spans": [[utc(x), utc(y)] for x, y in spans], **info}
            for lo, hi in spans:
                if now < hi and now + budget_s > lo:
                    raise RuntimeError(f"budget would overlap the Nightly window {utc(lo)}..{utc(hi)}")
            gate_doc = nightly_gate(cfg["status_root"], now)
            rec["nightly_gate"] = gate_doc
            if not gate_doc["ok"]:
                raise RuntimeError(f"Nightly not finished (ALISIOS rule 2026-11-11): {gate_doc['problems']}")
        if not dry():
            rows = subprocess.run(["lslocks", "-n", "-o", "PID,COMMAND,MODE,PATH"], capture_output=True, text=True,
                                  check=True).stdout.splitlines()
            waiters = [r.strip() for r in rows if "/tmp/wrf_gpu2_gpu.lock" in r and ("WRITE*" in r or "READ*" in r)]
            rec["lock_waiters"] = waiters  # recorded only: the next lane of a shared window queues on the lock
            if own_holder_token is not None:
                holder = Path(os.environ.get("GPUWRF_GPU_LOCK_HOLDER_FILE", "/nonexistent")).read_text(errors="replace")
                if own_holder_token not in holder:
                    raise RuntimeError("GPU lock holder file does not carry this process's token")
        rec["ok"] = True
        return rec
    except Exception as exc:
        rec.update(ok=False, refused=str(exc))
        raise RuntimeError(str(exc)) from None
    finally:
        if root.exists():
            with (root / "ADMISSION.jsonl").open("a") as stream:
                stream.write(json.dumps(rec, default=str) + "\n")


# ----------------------------------------------------------------------------------------------------------- prep
def snapshot_ok(plan) -> dict:
    out = {"snap": str(SNAP)}
    git = lambda *a: subprocess.run(["git", "-C", str(SNAP), *a], capture_output=True, text=True)  # noqa: E731
    out["head"] = git("rev-parse", "HEAD").stdout.strip()
    out["src_clean"] = subprocess.run(["git", "-C", str(SNAP), "diff", "--quiet", "HEAD", "--", "src", "scripts", "data"]).returncode == 0
    out["untracked"] = [u for u in git("status", "--porcelain", "--", "src", "scripts", "data").stdout.split("\n")
                        if u and u != "?? data/wrf_pristine"]
    out["wrf_pristine"] = os.path.realpath(SNAP / "data" / "wrf_pristine") == "<USER_HOME>/src/wrf_pristine"
    src = model_src(plan)
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "JAX_PLATFORMS": "cpu",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(src)}
    fp = subprocess.run([PY, "-B", "-c", "from gpuwrf.runtime.aot_cheap_key import source_fingerprint_hash as f; "
                         "import gpuwrf; print(gpuwrf.__file__); print(f())"], env=env, capture_output=True, text=True, cwd="/")
    out["gpuwrf_file"], out["fingerprint"] = (fp.stdout.split() + ["", ""])[:2]
    out["model_src"] = str(src)
    model = plan["model"]
    if model["kind"] == "bundle_consumer":
        out["bundle_consumer"] = str(Path(model["bundle"]) / "tools" / "bundle_consumer.sh")
        out["layer_exists"] = (Path(model["layer"]) / "upper" / "jax").is_dir()
    out["ok"] = (out["head"] == os.environ.get("V50_COMMIT") and out["src_clean"] and not out["untracked"]
                 and out["wrf_pristine"] and out["fingerprint"] == plan["source_fingerprint"]
                 and out["gpuwrf_file"] == str(src / "gpuwrf/__init__.py")
                 and (model["kind"] == "snapshot" or (Path(out["bundle_consumer"]).is_file() and out["layer_exists"])))
    return out


def prep(plan, root):
    root = Path(root)
    checked_root(plan, root)
    if root.exists():
        raise SystemExit(f"{root} exists: fresh run root per attempt (E42)")
    pre = {"utc": utc(), "dry": dry(), "plan": plan["name"], "snapshot": snapshot_ok(plan)}
    problems = [] if pre["snapshot"]["ok"] else ["snapshot identity"]
    try:
        pre["admission"] = admit(plan, root, plan["caps_s"][plan["order"][0]])
    except RuntimeError as exc:
        problems.append(f"admission: {exc}")
    if problems:  # refuse before writing anything to the run filesystem
        out = Path(plan["evidence_dir"]) / root.name
        out.mkdir(parents=True, exist_ok=True)
        save(out / "RESULT.json", {"schema": "gpuwrf.branch_qualification.result.v1", "verdict": "NOSTART",
                                   "reasons": problems, "preflight": pre, "utc": utc()})
        raise SystemExit(f"NOSTART: {problems} (receipt {out / 'RESULT.json'})")
    root.mkdir(parents=True)
    cases = {}
    if plan["case_in_place"]:  # a staged, read-only case copy used where it is (the private layer was built from it)
        case_dir = Path(plan["case_src"])
        extra = sorted(p.name for p in case_dir.iterdir() if p.name not in plan["case_sha256"])
        if extra:
            cases["_extra_files"] = {"expected": [], "copied": extra}
        for name, digest in sorted(plan["case_sha256"].items()):
            cases[name] = {"expected": digest, "copied": sha(case_dir / name)}
    else:
        case_dir = root / "case"
        case_dir.mkdir()
        for name, digest in sorted(plan["case_sha256"].items()):
            dst = case_dir / name
            shutil.copyfile(Path(plan["case_src"]) / name, dst)
            dst.chmod(0o444)
            cases[name] = {"expected": digest, "copied": sha(dst)}
    pre["case"], pre["case_dir"] = cases, str(case_dir)
    plan["_case_dir"] = str(case_dir)
    save(root / "PREFLIGHT.json", pre)
    bad = sorted(n for n, v in cases.items() if v["expected"] != v["copied"])
    if bad:
        save(root / "RESULT.json", {"schema": "gpuwrf.branch_qualification.result.v1", "verdict": "NOSTART",
                                    "reasons": [f"case sha256 mismatch {bad}"], "preflight": pre, "utc": utc()})
        raise SystemExit(f"NOSTART: case sha256 mismatch {bad}")
    save(root / "PLAN.json", plan)
    log(f"prep ok: {plan['name']} commit {pre['snapshot']['head'][:10]} fingerprint {pre['snapshot']['fingerprint'][:8]}")


# ------------------------------------------------------------------------------------------------------------ gpu
def model_src(plan) -> Path:
    model = plan["model"]
    return SNAP / "src" if model["kind"] == "snapshot" else Path(model["bundle"]) / "src" / "src"


def cache_dir(plan, root: Path) -> Path:
    """Where this run's executables land: the fresh private cache, or the private layer's upper jax dir."""
    model = plan["model"]
    return root / "cache" if model["kind"] == "snapshot" else Path(model["layer"]) / "upper"


def child_env(plan, root: Path) -> dict:
    model = plan["model"]
    stub = (os.environ["V50_STUB_PYTHONPATH"] + ":") if dry() else ""
    env = {"HOME": os.environ["HOME"], "PATH": "<USER_HOME>/miniconda3/bin:/usr/local/cuda/bin:/usr/bin:/bin",
           "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": stub + str(model_src(plan)), "JAX_PLATFORMS": "cpu" if dry() else "cuda",
           "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}", "TMPDIR": str(root / "_tmp"), "XDG_CACHE_HOME": str(root / "_xdg")}
    if model["kind"] == "snapshot":
        env.update({"GPUWRF_JAX_CACHE_DIR": str(root / "cache/jax"), "JAX_COMPILATION_CACHE_DIR": str(root / "cache/jax"),
                    "GPUWRF_XLA_AUTOTUNE_CACHE_DIR": str(root / "cache/autotune"), "TRITON_CACHE_DIR": str(root / "cache/triton"),
                    "CUDA_CACHE_PATH": str(root / "cache/cuda")})
    else:  # exactly the env the private layer was built with (cache/autotune paths inside the consumer view of B)
        env.update(model["env"])
    for name in ("GPUWRF_GPU_LOCK_FD", "GPUWRF_GPU_LOCK_HELD", "GPUWRF_GPU_LOCK_FILE", "GPUWRF_GPU_LOCK_HOLDER_FILE",
                 "GPUWRF_GPU_LOCK_LABEL", "GPUWRF_GPU_LOCK_TOKEN"):
        if name not in os.environ:
            raise SystemExit(f"GPU lock variable absent: {name} (run under scripts/with_gpu_lock.sh)")
        env[name] = os.environ[name]
    if dry():
        env.update({k: v for k, v in os.environ.items() if k.startswith("V50_STUB_")})
        env["V50_STUB_CACHE"] = str(cache_dir(plan, root) / "jax")  # the stub writes fake executables only there
    for key in ("TMPDIR", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def consumer_prefix(plan) -> list[str]:
    """bundle_consumer view of the sealed bundle with the private layer (dry run: the sandbox stand-in)."""
    model = plan["model"]
    if model["kind"] == "snapshot":
        return []
    return [str(Path(model["bundle"]) / "tools" / "bundle_consumer.sh"), "--bundle", model["bundle"], "--layer",
            model["layer"], "--"]


CONSUMER_LINE = re.compile(r"^BUNDLE_CONSUMER rc=(-?\d+) compiled_new=(\d+) .*?receipt=(\S+)", re.M)


def consumer_evidence(log_text: str) -> dict | None:
    """The last BUNDLE_CONSUMER line and whether the sealed bundle checked BUNDLE_OK afterwards."""
    lines = CONSUMER_LINE.findall(log_text)
    if not lines:
        return None
    rc, compiled_new, receipt = lines[-1]
    after = re.findall(r"^P check after: (\S+)", log_text, re.M)
    return {"rc": int(rc), "compiled_new": int(compiled_new), "receipt": receipt,
            "bundle_check_after": after[-1] if after else None}


def env_digest(env: dict) -> str:
    """sha256 of the child environment with the lock token masked (no secret in receipts, E245)."""
    shown = {k: ("<masked>" if k == "GPUWRF_GPU_LOCK_TOKEN" else v) for k, v in sorted(env.items())}
    return hashlib.sha256(json.dumps(shown, sort_keys=True).encode()).hexdigest()


def executables(cache: Path) -> list[str]:
    jax_dir = cache / "jax"
    entries = sorted(p.name for p in jax_dir.glob("*-cache")) if jax_dir.is_dir() else []
    aot = sorted(str(p.relative_to(cache)) for p in (jax_dir / "aot").rglob("*.xlaexec")) if (jax_dir / "aot").is_dir() else []
    return entries + aot


def generation_at(store: Path, step: int) -> Path:
    found = [m.parent for m in sorted(store.glob("generation-*/manifest.json"))
             if int(load(m).get("own_steps", {}).get(load(m)["domains"][0], -1)) == step]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one generation at root step {step} in {store}, found {found}")
    return found[0]


CURRENT: dict = {"proc": None, "root": None, "arms": None, "arm": None}


def session_members(sid: int) -> list[int]:
    """Live (non-zombie) processes whose session id is ``sid`` (descendants survive their leader's exit)."""
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2:].split()
        if int(fields[3]) == sid and fields[0] != "Z":
            out.append(int(entry.name))
    return out


def _signal_session(sid: int, sig) -> None:
    try:
        os.killpg(sid, sig)
    except ProcessLookupError:
        pass
    for pid in session_members(sid):  # members that left the leader's process group
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _kill_session(proc, grace):
    """TERM the whole session (leader + every descendant), wait, then KILL what is left."""
    if proc is None:
        return
    sid = proc.pid
    if proc.poll() is None or session_members(sid):
        _signal_session(sid, signal.SIGTERM)
        deadline = time.time() + grace
        while time.time() < deadline and (proc.poll() is None or session_members(sid)):
            time.sleep(0.2)
        if proc.poll() is None or session_members(sid):
            _signal_session(sid, signal.SIGKILL)
        proc.wait()


def _terminate(signum, _frame):
    """TERM/INT/HUP from the launcher or the lock wrapper: kill the arm's session, journal, mark STOPPED, exit."""
    _kill_session(CURRENT["proc"], 30)
    if CURRENT["root"] is not None:
        CURRENT["arms"].append({"arm": CURRENT["arm"], "status": "STOPPED_SIGNAL", "signal": signum, "utc_end": utc()})
        save(CURRENT["root"] / "ARMS.json", CURRENT["arms"])
    log(f"stopped by signal {signum}")
    raise SystemExit(128 + signum)


def run_arm(command, env, fds, cwd, log_path, budget_s, grace_s, plan, root):
    """Run one arm in its own session; enforce its budget and the disk floors continuously (5 s)."""
    t0 = time.time()
    status, rc = None, None
    with open(log_path, "ab") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, pass_fds=fds,
                                start_new_session=True, cwd=str(cwd))
        CURRENT["proc"] = proc
        while True:
            try:
                rc = proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.time() - t0 > budget_s - grace_s:
                status = "TIMEOUT"
            elif (not dry() and free(plan["data_mount"]) < plan["floors_gib"]["data"] * GIB) or \
                    free("/") < plan["floors_gib"]["root"] * GIB:
                status = "STOPPED_DISK"
            elif own_bytes(root) > plan["own_cap_gib"] * GIB:
                status = "STOPPED_OWN_CAP"
            if status:
                _kill_session(proc, grace_s)
                rc = proc.returncode
                break
        leftover = session_members(proc.pid)
        if leftover:  # the leader exited but descendants of its session live on: tear them down, the arm FAILs
            _kill_session(proc, grace_s)
        CURRENT["proc"] = None
    row = {"rc": rc, "status": status or ("OK" if rc == 0 and not leftover else "FAILED"),
           "wall_s": round(time.time() - t0, 1)}
    if leftover:
        row["leftover_pids"] = leftover
    return row


def gpu(plan, root):
    root = checked_root(plan, root)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _terminate)
    env = child_env(plan, root)
    fds = (int(env["GPUWRF_GPU_LOCK_FD"]),) if os.path.exists(f"/proc/self/fd/{env['GPUWRF_GPU_LOCK_FD']}") else ()
    for sub in ("acct", "logs"):
        (root / sub).mkdir(exist_ok=True)
    case = plan["case_src"] if plan["case_in_place"] else str(root / "case")
    neutral = str(Path(plan["neutral_controls"]).resolve())
    prefix = consumer_prefix(plan)
    acct = [PY, "-B", str(SNAP / "scripts" / "jax_acct_run.py")]
    model = ["run", "--input-dir", case, "--surface-controls", neutral, *plan["cli_args"]]
    hours = str(plan["hours"])
    empty = root / "empty_manifest.json"
    empty.write_text('{"schema_version": 1}\n')
    arms: list = []
    save(root / "ARMS.json", arms)
    CURRENT.update(root=root, arms=arms)

    def series(name, variants):
        spec = {"schema": "gpuwrf.branch_series.v1", "input_dir": case,
                "branch_from": str(generation_at(root / "C_ckpt", plan["branch_step"])), "out_root": str(root / name),
                "end": {"hours": plan["hours"]}, "parent_surface_controls": neutral, "cli_args": plan["cli_args"],
                "variants": [{"name": v["name"], "surface_controls": str(Path(v["controls"]).resolve())} for v in variants]}
        save(root / f"series_{name}.json", spec)
        return str(root / f"series_{name}.json")

    def argv(arm):
        ck = lambda d: ["--checkpoint-dir", str(root / d), "--checkpoint-interval-steps", str(plan["branch_step"])]  # noqa: E731
        if arm == "C":
            return acct + ["--receipt", str(root / "acct/C.json"), "--", *model, "--hours", hours,
                           "--output-dir", str(root / "C"), "--scratch-dir", str(root / "C_scratch"), *ck("C_ckpt")]
        if arm == "G2":
            return acct + ["--receipt", str(root / "acct/G2.json"), "--", *model, "--hours", hours,
                           "--output-dir", str(root / "G2"), "--scratch-dir", str(root / "G2_scratch"),
                           "--branch-from", str(generation_at(root / "C_ckpt", plan["branch_step"])),
                           "--branch-manifest", str(empty)]
        if arm == "G3":
            return [PY, "-B", str(SNAP / "scripts/branch_series.py"), series("G3", plan["variants"]), "--mode", "inprocess"]
        if arm == "G3p":
            return [PY, "-B", str(SNAP / "scripts/branch_series.py"), series("G3p", plan["variants"][:1]), "--mode", "process"]
        if arm == "R1":
            return acct + ["--receipt", str(root / "acct/R1.json"), "--", *model, "--minutes", str(plan["_seam_minutes"]),
                           "--output-dir", str(root / "R"), "--scratch-dir", str(root / "R_scratch"), *ck("R_ckpt")]
        if arm == "R2":  # ordinary resume without --checkpoint-dir (CPU-proven form): store = generation parent
            return acct + ["--receipt", str(root / "acct/R2.json"), "--", *model, "--hours", hours, "--extend-run",
                           "--output-dir", str(root / "R"), "--scratch-dir", str(root / "R_scratch"),
                           "--resume-checkpoint", str(generation_at(root / "R_ckpt", plan["branch_step"]))]
        raise KeyError(arm)

    stop = None
    token = os.environ.get("GPUWRF_GPU_LOCK_TOKEN")
    for arm in plan["order"]:
        CURRENT["arm"] = arm
        cap = float(plan["caps_s"][arm])
        row = {"arm": arm, "cap_s": cap, "utc_start": utc()}
        if stop:
            row.update(status="NOT_RUN", reason=stop)
        else:
            try:
                row["admission"] = admit(plan, root, cap, own_holder_token=None if dry() else token)
                command = argv(arm)
            except RuntimeError as exc:
                stop = f"{arm} refused: {exc}"
                row.update(status="REFUSED", reason=str(exc))
                command = None
            if command is not None:
                command = prefix + command
                before = executables(cache_dir(plan, root))
                row.update(argv=command, env_sha256=env_digest(env), executables_before=len(before))
                row.update(run_arm(command, env, fds, SNAP, root / "logs" / f"{arm}.log", cap,
                                   float(plan["kill_grace_s"]), plan, root))
                row["new_executables"] = sorted(set(executables(cache_dir(plan, root))) - set(before))
                if row["status"] != "OK":
                    stop = f"arm {arm} {row['status']} rc={row['rc']}"
        row["utc_end"] = utc()
        arms.append(row)
        save(root / "ARMS.json", arms)
        log(f"{arm}: {row['status']} rc={row.get('rc')} wall={row.get('wall_s')} new_exec={len(row.get('new_executables', []))}")
    save(root / "GPU_DONE.json", {"utc": utc(), "stop": stop, "arms": [[a["arm"], a["status"]] for a in arms]})
    return 0 if stop is None else 1


# --------------------------------------------------------------------------------------------------------- verify
def _run_cmp(argv, out: Path, env) -> dict:
    proc = subprocess.run(argv, env=env, check=False, capture_output=True, text=True)
    verdict = load(out).get("verdict") if out.exists() else None
    ok = proc.returncode == 0 and verdict == "PASS"
    # exit code and parsed verdict must agree (E247 class: never trust one alone)
    return {"rc": proc.returncode, "parsed": verdict, "verdict": "PASS" if ok else "FAIL", "file": str(out),
            "stderr_tail": proc.stderr[-800:]}


def stream_dirs(plan, root):
    dirs = {"C": root / "C", "R": root / "R", "G2": root / "G2"}
    for v in plan["variants"]:
        dirs[f"G3_{v['name']}"] = root / "G3" / v["name"] / "wrfout"
    dirs[f"G3p_{plan['variants'][0]['name']}"] = root / "G3p" / plan["variants"][0]["name"] / "wrfout"
    return dirs


def collect(plan, root) -> dict:
    """Gather every evidence item (no verdict here). Missing items stay missing (None)."""
    cpu = {"PATH": "<USER_HOME>/miniconda3/bin:/usr/local/cuda/bin:/usr/bin:/bin", "HOME": os.environ["HOME"],
           "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(SNAP / "src"), "LC_ALL": "C.UTF-8"}
    tool = str(SNAP / "scripts/compare_branch_streams.py")
    ev: dict = {"gpu_done": load(root / "GPU_DONE.json") if (root / "GPU_DONE.json").exists() else None,
                "arms": {a["arm"]: a for a in load(root / "ARMS.json")} if (root / "ARMS.json").exists() else {},
                "acct": {}, "series": {}, "payload_device": {}, "payload_walls": {}, "payload_aot": {},
                "stray_files": [], "cmp": {}, "consumer": {}}
    if plan["model"]["kind"] == "bundle_consumer":
        for arm in ARMS:
            logf = root / "logs" / f"{arm}.log"
            ev["consumer"][arm] = consumer_evidence(logf.read_text(errors="replace")) if logf.exists() else None
    for arm in ACCT_ARMS:
        p = root / "acct" / f"{arm}.json"
        ev["acct"][arm] = load(p) if p.exists() else None
    for name in ("G3", "G3p"):
        p = root / name / "series_receipt.json"
        ev["series"][name] = load(p) if p.exists() else None
    dirs = stream_dirs(plan, root)
    (root / "digests").mkdir(exist_ok=True)
    (root / "cmp").mkdir(exist_ok=True)
    for name, path in dirs.items():
        ev["stray_files"] += [f"{name}/{n}" for n in cmpmod.stray(path)]  # anything that is not a well-formed frame
        proof = path / "proofs" / "nested_pipeline_run.json"
        if proof.exists():
            payload = load(proof)
            ev["payload_device"][name] = payload.get("device")
            domains = ((payload.get("metadata") or {}).get("nested_aot") or {}).get("domains") or {}
            ev["payload_aot"][name] = {d: {k: v.get(k) for k in ("loaded", "source", "error")} for d, v in domains.items()}
            ev["payload_walls"][name] = {k: payload.get(k) for k in ("wall_clock_total_s", "wall_clock_forecast_only_s",
                                                                     "root_steps", "hours")}
        else:
            ev["payload_device"][name] = None
        if any(path.glob("wrfout_d0*")):
            subprocess.run([PY, "-B", tool, "digest", str(path), str(root / "digests" / f"{name}.json")], env=cpu, check=False)
    # independent expected inventories (namelist only; seam from the generation itself)
    nl = str(Path(plan["case_src"] if plan["case_in_place"] else root / "case") / "namelist.input")
    try:
        ev["expected_programs"] = expected_programs(Path(nl).read_text())
    except (OSError, ValueError, KeyError) as exc:
        ev["expected_programs"], ev["expected_programs_error"] = None, str(exc)
    ev["inventory"] = {}
    try:
        gen_c = generation_at(root / "C_ckpt", plan["branch_step"])
        ev["seam"] = cmpmod.seam_time(gen_c)
    except Exception as exc:
        gen_c, ev["seam"] = None, None
        ev["seam_error"] = str(exc)
    try:
        gen_r = generation_at(root / "R_ckpt", plan["branch_step"])
    except Exception:
        gen_r = None
    whole = ["--whole-stream"] + (["--include-initial"] if plan["emit_initial_history"] else [])
    for key, extra in (("whole", whole), ("branch", ["--branch-generation", str(gen_c)] if gen_c else None)):
        out = root / "cmp" / f"inventory_{key}.json"
        if extra is not None and subprocess.run([PY, "-B", tool, "inventory", str(out), "--namelist", nl, "--hours",
                                                 str(plan["hours"]), *extra], env=cpu, check=False).returncode == 0:
            ev["inventory"][key] = load(out)["frames"]
        else:
            ev["inventory"][key] = None
    d = root / "digests"
    first = plan["variants"][0]["name"]

    def cmp(name, ref, test, mode, *extra, inv="branch", gen=gen_c):
        inv_file = root / "cmp" / f"inventory_{inv}.json"
        if not (Path(ref).exists() and Path(test).exists() and gen is not None and ev["inventory"].get(inv)):
            ev["cmp"][name] = {"verdict": "MISSING_EVIDENCE"}
            return
        out = root / "cmp" / f"{name}.json"
        ev["cmp"][name] = _run_cmp([PY, "-B", tool, "--ref", str(ref), "--test", str(test), "--out", str(out), "--mode",
                                    mode, "--branch-generation", str(gen), "--expect-frames", str(inv_file), *extra], out, cpu)

    cmp("C_inventory", d / "C.json", d / "C.json", "exact", "--whole-stream", inv="whole")
    cmp("G1_null_resume", d / "C.json", d / "R.json", "exact", "--whole-stream", inv="whole", gen=gen_r)
    cmp("G2_null_branch", d / "C.json", d / "G2.json", "exact")
    for v in plan["variants"]:  # raw files: the verdict lists max |diff| per differing variable
        cmp(f"G3_{v['name']}", dirs["G2"], dirs[f"G3_{v['name']}"], "edited", "--expect-differ", *v["expect_differ"])
    cmp("G3x_inprocess_vs_process", d / f"G3_{first}.json", d / f"G3p_{first}.json", "exact")
    # E41 on the cache this run used (fresh private cache or the private layer) before any cleanup
    gate = root / "E41" / "wave_gate.json"
    cache = cache_dir(plan, root)
    if executables(cache):
        tool41 = os.environ.get("V50_E41_TOOL") if dry() else None
        proof = root / "C" / "proofs" / "nested_pipeline_run.json"
        argv = [PY, "-B", tool41 or str(SNAP / "scripts/wave_cubin_gate.py"), str(cache / "jax"), str(root / "E41"),
                plan.get("e41_glob", "jit__advance_chunk*,jit_fused_jit*")] + (["--run-proof", str(proof)] if proof.exists() else [])
        t0 = time.time()
        limit = max(60.0, min(float(plan.get("e41_cap_s", 600)), float(os.environ["V50_DEADLINE_EPOCH"]) - time.time() - 60))
        e41_env = dict(cpu, **({k: v for k, v in os.environ.items() if k.startswith("V50_STUB_")} if dry() else {}))
        try:
            with open(root / "logs" / "E41.log", "ab") as stream:
                rc = subprocess.run(argv, env=e41_env, check=False, timeout=limit, stdout=stream, stderr=subprocess.STDOUT).returncode
        except subprocess.TimeoutExpired:
            rc = 124
        ev["e41"] = {"rc": rc, "wall_s": round(time.time() - t0, 1), "argv": argv,
                     "receipt": load(gate) if gate.exists() else None}
    else:
        ev["e41"] = {"rc": None, "receipt": None, "reason": "no executables in the cache"}
    ev["e41"]["assessment"] = e41_assess(root / "E41", ev["e41"]["rc"], plan["e41_decisions"])
    return ev


FLAG_NAMES_KEY = "fusions"
# wave_cubin_gate NO_KERNEL class (kernel-less eager JAX-cache entries, optimized HLO only buffer moves); re-checked here
NO_KERNEL_OPS = frozenset({"parameter", "copy", "bitcast", "tuple", "get-tuple-element", "constant"})


def _decision_ok(decision) -> tuple[bool, str]:
    """A decision is usable only when its file (and every file it builds on, ``also``) exists with the pinned sha256."""
    if not decision:
        return False, "no decision in the plan"
    for item in [decision] + list(decision.get("also", [])):
        path = Path(item["file"])
        if not path.is_file():
            return False, f"decision file missing: {path}"
        if sha(path) != item["sha256"]:
            return False, f"decision file changed: {path}"
    return True, ""


def e41_assess(out_dir: Path, rc, decisions: dict) -> dict:
    """E41 verdict from the gate receipt: rc 0 + PASS + full coverage, or a non-PASS that a pinned manager decision
    permits exactly (Option 3: INCOMPLETE only from UNKNOWN_AOT_METADATA; structural WARN: only the named kInput
    fusions within HLO/SASS bounds measured from that entry's cubin scan, calls 0, stack 0, <= 1 flagged entry per
    entry family). Anything else, a missing receipt, rc not in {0, 2}, or incomplete coverage is FAIL."""
    out_dir = Path(out_dir)
    gate_path = out_dir / "wave_gate.json"
    res = {"verdict": "FAIL", "reasons": [], "rc": rc, "disclosed": []}
    why = res["reasons"].append
    if not gate_path.is_file():
        why("no wave_gate.json receipt")
        return res
    gate = load(gate_path)
    cov = gate.get("coverage") or {}
    if rc not in (0, 2):
        why(f"gate rc {rc} (only 0, or 2 for a classified non-PASS, is accepted)")
    if cov.get("status") != "PASS" or not cov.get("n_discovered") or cov.get("n_scanned") != cov.get("n_discovered") \
            or len(gate.get("entries") or []) != cov.get("n_scanned"):
        why(f"coverage incomplete: {cov}")
    if gate.get("hard_stack_gate") != "PASS":
        why("STACK > 256 B")
    entries = gate.get("entries") or []
    for e in entries:
        for elf in e.get("elfs") or []:
            if elf.get("nonleaf_over1k") or elf.get("callgraph_unknown") or elf.get("stack_over256"):
                why(f"{e['entry']}: non-leaf/callgraph/stack flags")
    # NO_KERNEL entries (no ELF, buffer-move HLO only): scanned, never measured; they need no decision but must be
    # well-formed, and a gate with nothing but NO_KERNEL entries measured no kernel at all (FAIL).
    no_kernel = [e for e in entries if e.get("kernel_class") == "NO_KERNEL"]
    for e in no_kernel:
        ops = e.get("no_kernel_hlo_ops") or {}
        if (e.get("elfs") or e.get("source_kind") != "jax_cache" or e.get("kinput_transpose_status") != "NO_KERNEL_IDENTITY_HLO"
                or not ops or set(ops) - NO_KERNEL_OPS or e.get("kinput_transpose_over500")):
            why(f"{e.get('entry')}: invalid NO_KERNEL record")
    if gate.get("no_measured_kernels") or (entries and len(no_kernel) == len(entries)):
        why("no measured kernels: every entry is NO_KERNEL")
    res["no_kernel"] = len(no_kernel)
    verdict = gate.get("e41_gate")
    if verdict == "PASS":
        if rc != 0:
            why(f"gate PASS but rc {rc}")
        if not res["reasons"]:
            res["verdict"] = "PASS"
        return res
    unknown = [e for e in entries if e not in no_kernel and not str(e.get("kinput_transpose_status", "")).startswith("MEASURED")]
    flagged = [e for e in entries if e.get("kinput_transpose_over500")]
    if unknown:
        bad = [e["entry"] for e in unknown if e.get("kinput_transpose_status") != "UNKNOWN_AOT_METADATA"]
        ok, msg = _decision_ok(decisions.get("option3_incomplete"))
        rich_elfs = {x.get("sha256") for e in entries if str(e.get("kinput_transpose_status", "")).startswith("MEASURED")
                     for x in e.get("elfs") or []}
        unmatched = [e["entry"] for e in unknown if not e.get("elfs")
                     or not {x.get("sha256") for x in e["elfs"]} <= rich_elfs]
        if bad:
            why(f"incomplete for a reason other than UNKNOWN_AOT_METADATA: {bad}")
        elif unmatched:  # Option 3 (v0.5 scope): every compact blob's native ELF matches a scanned rich entry
            why(f"compact AOT entries whose native ELF matches no scanned rich entry: {unmatched}")
        elif not ok:
            why(f"INCOMPLETE (UNKNOWN_AOT_METADATA) without a usable decision: {msg}")
        else:
            res["disclosed"].append({"decision": decisions["option3_incomplete"]["file"], "entries": [e["entry"] for e in unknown],
                                     "residual": "compact-AOT kInput <= 500 not proven"})
    if flagged:
        dec = decisions.get("structural_warn")
        ok, msg = _decision_ok(dec)
        if not ok:
            why(f"kInput > 500 flags without a usable decision: {msg}")
        else:
            bounds = dec[FLAG_NAMES_KEY]
            families = {}
            for e in flagged:
                family = e["entry"].split("-")[0] if e.get("source_kind") == "jax_cache" else f"aot:{e.get('program')}"
                families[family] = families.get(family, 0) + 1
                names = {n for n, _ in e["kinput_transpose_over500"]}
                if names - set(bounds):
                    why(f"{e['entry']}: kInput flags outside the decision: {sorted(names - set(bounds))}")
                    continue
                label = hashlib.sha256(str(e["path"]).encode()).hexdigest()[:12]
                rows = []
                for f in sorted(out_dir.glob(f"cubin_scan_{label}_e*.json")):
                    rows += load(f)
                for name, hlo in e["kinput_transpose_over500"]:
                    b = bounds[name]
                    row = [r for r in rows if r.get("kernel") == name.replace(".", "_")]
                    if len(row) != 1:
                        why(f"{e['entry']}: SASS of {name} not measurable from its cubin scan")
                        continue
                    r = row[0]
                    if not (hlo <= b["hlo"] and 0 < r["sass"] <= b["sass"] and r["calls"] == 0 and r["stack"] == 0):
                        why(f"{e['entry']}: {name} HLO {hlo}/{b['hlo']} SASS {r['sass']}/{b['sass']} calls {r['calls']} "
                            f"stack {r['stack']} outside the decision")
                    else:
                        res["disclosed"].append({"decision": dec["file"], "entry": e["entry"], "fusion": name, "hlo": hlo,
                                                 "sass": r["sass"], "elf": [x.get("sha256") for x in e.get("elfs") or []]})
            if any(n > 1 for n in families.values()):
                why(f"more than one flagged entry per family: {families}")
    if verdict not in ("INCOMPLETE", "FAIL"):
        why(f"unknown gate verdict {verdict!r}")
    if not res["reasons"]:
        res["verdict"] = "WARN_DISCLOSED" if flagged else "PASS_DISCLOSED"
    return res


WARM_SOURCES = ("aot_blob", "aot_memory_cache")
STREAM_ARM = {"C": "C", "R": "R2", "G2": "G2"}


def evaluate(ev: dict, plan: dict, is_dry: bool) -> tuple[str, list[str], dict]:
    """PASS only with every gate green and every evidence item present. Returns (verdict, reasons, gates)."""
    reasons, gates = [], {}

    def gate(name, ok, why):
        gates[name] = "PASS" if ok else "FAIL"
        if not ok:
            reasons.append(f"{name}: {why}")

    allowed = set(plan.get("compile_allowed_arms", []))
    names = [v["name"] for v in plan["variants"]]
    gate("gpu_done", ev.get("gpu_done") is not None, "GPU_DONE.json missing")
    arms = ev.get("arms") or {}
    gate("arms", all(arms.get(a, {}).get("status") == "OK" for a in ARMS),
         {a: arms.get(a, {}).get("status", "MISSING") for a in ARMS})
    want_platform = "cpu" if is_dry else "gpu"
    acct = ev.get("acct") or {}
    series = ev.get("series") or {}
    g3p_rows = (series.get("G3p") or {}).get("variants") or []
    plat = {a: (acct.get(a) or {}).get("platform") for a in ACCT_ARMS}
    plat.update({f"G3p:{r.get('name')}": r.get("platform") for r in g3p_rows})
    dev = ev.get("payload_device") or {}
    dev_ok = bool(dev) and all(isinstance(v, str) and v and (v.startswith("STUB") if is_dry else
                                                             (v != "none" and "cpu" not in v.lower())) for v in dev.values())
    gate("platform", bool(g3p_rows) and all(v == want_platform for v in plat.values()) and dev_ok,
         {"measured_process_platform": plat, "device": dev})
    keys = ("cache_requests", "cache_hits", "cache_misses", "cache_disabled", "backend_compile_events", "fresh_compiles")
    counters, complete = {}, True
    for a in ACCT_ARMS:
        runs = (acct.get(a) or {}).get("runs") or []
        if not runs or any(k not in runs[0] for k in keys):
            complete = False
            continue
        counters[a] = {k: sum(r[k] for r in runs) for k in keys}
    g3_rows = {v.get("name"): v for v in (series.get("G3") or {}).get("variants", [])}
    for arm, rows, want in (("G3", list(g3_rows.values()), names), ("G3p", g3p_rows, names[:1])):
        if [r.get("name") for r in rows] != want or any(r.get("status") != "OK" or any(k not in (r.get("jax_counters") or {})
                                                                                      for k in keys) for r in rows):
            complete = False
            continue
        for r in rows:
            counters[f"{arm}:{r['name']}"] = {k: r["jax_counters"][k] for k in keys}
    gate("receipts", complete, "acct/series receipts or their counters missing (G3p child counters included)")
    must_warm = [a for a in ARMS if a not in allowed]
    later = {k: v for k, v in counters.items() if k.split(":")[0] not in allowed}
    new_exec = {a: len(arms.get(a, {}).get("new_executables", [])) for a in must_warm}
    gate("compile", complete and all(arms.get(a, {}).get("new_executables") is not None for a in must_warm)
         and all(v["cache_misses"] == 0 and v["fresh_compiles"] == 0 and v["cache_disabled"] == 0 for v in later.values())
         and all(n == 0 for n in new_exec.values()),
         {"counters": later, "new_executables": new_exec, "compile_allowed (E41-bound)": sorted(allowed)})
    aot = ev.get("payload_aot") or {}
    expected = set(ev.get("expected_programs") or [])
    warm_streams = {s: STREAM_ARM.get(s, s.split("_")[0]) for s in (ev.get("payload_device") or {})}
    cold = {} if expected else {"expected_programs": "missing (namelist-derived program set)"}
    for stream, arm in warm_streams.items():
        if arm in allowed:
            continue
        programs = aot.get(stream) or {}
        bad = {d: v for d, v in programs.items() if not (v.get("loaded") and v.get("error") is None
                                                           and v.get("source") in WARM_SOURCES)}
        missing, extra = sorted(expected - set(programs)), sorted(set(programs) - expected)
        if not programs or bad or missing or extra:  # the complete expected set, each program warm
            cold[stream] = {"bad": bad, "missing": missing, "extra": extra} if programs else "no per-program AOT evidence"
    gate("aot_warm", not cold, cold)
    if plan["model"]["kind"] == "bundle_consumer":
        cons = ev.get("consumer") or {}
        bad = {a: c for a, c in ((a, cons.get(a)) for a in ARMS)
               if not c or c["rc"] != 0 or not str(c.get("bundle_check_after") or "").startswith("BUNDLE_OK")
               or (a not in allowed and c["compiled_new"] != 0)}
        gate("bundle_consumer", not bad, bad)
    gate("inventory", bool((ev.get("inventory") or {}).get("whole")) and bool((ev.get("inventory") or {}).get("branch")),
         "expected frame inventory missing")
    gate("no_stray_files", not ev.get("stray_files"), ev.get("stray_files"))
    expected_cmp = ["C_inventory", "G1_null_resume", "G2_null_branch", *[f"G3_{n}" for n in names], "G3x_inprocess_vs_process"]
    for name in expected_cmp:
        gate(f"cmp:{name}", (ev.get("cmp") or {}).get(name, {}).get("verdict") == "PASS",
             (ev.get("cmp") or {}).get(name, {"verdict": "MISSING_EVIDENCE"}))
    assessment = (ev.get("e41") or {}).get("assessment") or {"verdict": "FAIL", "reasons": ["no E41 assessment"]}
    gate("e41", assessment["verdict"] in ("PASS", "PASS_DISCLOSED", "WARN_DISCLOSED"), assessment.get("reasons"))
    if reasons:
        return "FAIL", reasons, gates
    if is_dry:
        return "DRYRUN_PASS", [], gates
    return ("PASS" if assessment["verdict"] == "PASS" else "PASS_E41_DISCLOSED"), [], gates


def verify(plan, root):
    root = checked_root(plan, root)
    ev = collect(plan, root)
    verdict, reasons, gates = evaluate(ev, plan, dry())
    result = {"schema": "gpuwrf.branch_qualification.result.v1", "verdict": verdict, "reasons": reasons, "gates": gates,
              "dry_run": dry(), "utc": utc(), "plan": plan, "commit": os.environ.get("V50_COMMIT"),
              "seam": ev.get("seam"), "arms": list(ev["arms"].values()), "acct": ev["acct"], "series": ev["series"],
              "payload_walls": ev["payload_walls"], "payload_device": ev["payload_device"], "cmp": ev["cmp"],
              "e41": ev["e41"], "bundle_consumer": "N/A: snapshot model with a fresh private cache (no bundle)",
              "preflight": load(root / "PREFLIGHT.json") if (root / "PREFLIGHT.json").exists() else None,
              "disk_after_gib": {"data": free(plan["data_mount"]) / GIB, "root": free("/") / GIB}}
    save(root / "RESULT.json", result)
    log(f"VERDICT {verdict}" + ("" if not reasons else ": " + "; ".join(reasons)[:1500]))
    return 0 if verdict in ("PASS", "PASS_E41_DISCLOSED", "DRYRUN_PASS") else 1


# -------------------------------------------------------------------------------------------------------- cleanup
BULK = ("C", "R", "G2", "G3", "G3p", "C_ckpt", "R_ckpt", "C_scratch", "G2_scratch", "R_scratch", "case", "_tmp", "_xdg")


def cleanup(plan, root):
    """Journaled deletion of own RAW by fixed names under the checked root; keep JSON evidence and logs."""
    root = checked_root(plan, root)
    if not (root / "RESULT.json").exists():
        raise SystemExit(f"cleanup refused: no RESULT.json in {root} (verify did not finish); nothing deleted")
    result = load(root / "RESULT.json")
    e41_done = (result.get("e41") or {}).get("receipt") is not None
    keep = root / "keep_proofs"
    keep.mkdir(exist_ok=True)
    for proof in root.glob("**/proofs/nested_pipeline_run.json"):
        rel = proof.relative_to(root)
        if rel.parts[0] in BULK:
            shutil.copyfile(proof, keep / "__".join(rel.parts))
    for receipt in root.glob("G3*/series_receipt.json"):
        shutil.copyfile(receipt, keep / f"{receipt.parent.name}_series_receipt.json")
    hashes = {}
    for digest in (root / "digests").glob("*.json"):
        frames = load(digest).get("frames", {})
        src = Path(load(digest).get("source", ""))
        hashes.update({str(src / name): row.get("file_sha") for name, row in frames.items()})
    names = list(BULK) + (["cache"] if e41_done else [])
    journal = root / "RAW_DELETE_JOURNAL.jsonl"
    removed = []
    with journal.open("a") as stream:
        stream.write(json.dumps({"utc": utc(), "event": "START", "root": str(root), "names": names}) + "\n")
        for name in names:
            target = root / name
            if target.is_symlink():
                raise SystemExit(f"refusing symlink {target}")
            if not target.is_dir():
                continue
            for path in sorted(p for p in target.rglob("*") if p.is_file() or p.is_symlink()):
                # every RAW file is hashed before it goes: the digest's frame sha, else hashed here (failure paths too)
                if path.is_symlink():
                    digest, how = "symlink:" + os.readlink(path), "symlink"
                elif hashes.get(str(path)):
                    digest, how = hashes[str(path)], "stream_digest"
                else:
                    digest, how = sha(path), "hashed_at_delete"
                stream.write(json.dumps({"path": str(path), "bytes": path.lstat().st_size, "sha256": digest,
                                         "hash_source": how}) + "\n")
            shutil.rmtree(target)
            removed.append(name)
        stream.write(json.dumps({"utc": utc(), "event": "END", "removed": removed,
                                 "cache_kept_for_e41": not e41_done}) + "\n")
    result["cleanup"] = {"utc": utc(), "removed": removed, "journal": str(journal), "cache_kept_for_e41": not e41_done,
                         "kept": sorted(p.name for p in root.iterdir())}
    save(root / "RESULT.json", result)
    dst = Path(plan["evidence_dir"]) / root.name
    dst.mkdir(parents=True, exist_ok=True)
    for p in root.iterdir():
        if p.is_file():
            shutil.copyfile(p, dst / p.name)
        elif p.name in ("acct", "cmp", "digests", "logs", "keep_proofs", "E41"):
            shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
    log(f"cleanup: removed {removed}; journal {journal}; evidence copied to {dst}")


if __name__ == "__main__":
    phase, plan_path, root_arg = sys.argv[1:4]
    plan_value = load_plan(plan_path)
    raise SystemExit({"prep": prep, "gpu": gpu, "verify": verify, "cleanup": cleanup}[phase](plan_value, root_arg) or 0)
