#!/usr/bin/env python3
"""Run one consumer of a sealed compile bundle P inside a private copy-on-write view of P; P is never written.

usage: bundle_consumer.py [--no-verify] [--grace S] --bundle P --layer DIR -- CMD [ARGS...]
  (or bundle_consumer.sh, same arguments). Run it INSIDE with_gpu_lock.sh:
    with_gpu_lock.sh --label L -- bundle_consumer.sh --bundle P --layer DIR -- run_case.sh ...
The command sees P at the SAME absolute path (JAX/AOT keys bind cache and source paths): reads come from P, every write
lands in DIR/upper (unprivileged bubblewrap overlayfs). Same PID namespace and inherited fds, so the with_gpu_lock fd 9 /
holder-token preflight is unchanged. DIR may be reused by later runs of one consumer (a private warm layer); one run per
DIR at a time.

Safety:
- Every existing component of DIR (DIR, upper, work, runs, .consumer.lock) is validated before anything is created:
  opened without following symlinks and refused if it lies inside or aliases P. That is judged by the filesystem
  location of the open file (fdinfo mnt_id + mountinfo root), so bind mounts count. Also refused: a DIR that
  contains P or a mount point, an upper file that is a hard link into P, and a missing DIR whose nearest existing
  ancestor lies in P. Only then are missing components created, through the validated handles. bwrap receives the
  opened directories (/proc/self/fd), never a path that could be swapped.
- This process is a child subreaper. On exit, TERM/INT/HUP or an internal error it terminates the whole consumer tree:
  TERM, then SIGKILL after --grace seconds, repeated and reaped until no descendant is left (no time bound, so no race
  under load). Only then are the layer flock and the inherited GPU-lock fd released. The consumer tree also holds both, so neither is free while any part of it lives.
  A SIGKILL of this process cannot run that teardown; the tree then keeps running and keeps holding both locks.

Classification: after the run, P's own checker (P/tools/compile_bundle.py check) runs on the consumer's view, i.e. P
with the whole layer on top, mounted read-only. Its report (new / aliases / changed / missing) classifies every layer
entry; the entries this run wrote (content hash changed or new) give the per-run counts. Last stdout line:
  BUNDLE_CONSUMER rc=.. compiled_new=N aliases_new=A runtime_new=M other_new=K layer_compiled=T
  - compiled_new: entries this run wrote that the checker calls new or changed (programs/tunes outside the bundle),
    plus toolchain cache files (autotune/ triton/ cuda/ of P) with new content.
  - aliases_new: entries this run wrote that the checker accepts as sealed cheap-key aliases (no compile).
  - layer_compiled: new/changed entries of the whole layer (programs a reused layer can serve).
  Receipts: DIR/runs/<utc>_<pid>.json and the checker report <utc>_<pid>.view.json. Unless --no-verify, P itself is
  re-checked with the real checker as well (must stay BUNDLE_OK).
Exit status: the command's; 125 = refused before it ran; 128+N after signal N; 1 if the P check or the view
classification failed.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time

REFUSED = 125
CHILDREN = ("upper", "work", "runs")
TOOLCHAIN = ("autotune/", "triton/", "cuda/")  # compile caches of P outside the sealed JAX cache


class Refused(Exception):
    pass


class Terminated(Exception):
    def __init__(self, signum):
        super().__init__(signum)
        self.signum = signum


def payload_options(readonly_dirs, cwd):
    """Validate host payload mounts before creating a layer or acquiring its lock."""
    options = []
    for directory in readonly_dirs:
        if (not os.path.isabs(directory) or os.path.realpath(directory) != os.path.normpath(directory)
                or not os.path.isdir(directory)):
            raise Refused('read-only mount must be an existing canonical absolute directory')
        options += ['--ro-bind', directory, directory]
    if cwd:
        if (not os.path.isabs(cwd) or os.path.realpath(cwd) != os.path.normpath(cwd)
                or not os.path.isdir(cwd)):
            raise Refused('payload cwd must be an existing canonical absolute directory')
        options += ['--chdir', cwd]
    return options


def _unescape(field: str) -> str:  # mountinfo octal escapes (\040 = space, ...)
    out, i = [], 0
    while i < len(field):
        if field[i] == "\\" and field[i + 1:i + 4].isdigit():
            out.append(chr(int(field[i + 1:i + 4], 8)))
            i += 4
        else:
            out.append(field[i])
            i += 1
    return "".join(out)


def mounts(text: str | None = None) -> dict:
    """mount id -> (major:minor, root inside its filesystem, mount point) from /proc/self/mountinfo."""
    text = open("/proc/self/mountinfo").read() if text is None else text
    table = {}
    for line in text.splitlines():
        f = line.split()
        table[f[0]] = (f[2], _unescape(f[3]), _unescape(f[4]))
    return table


def location(fd: int, table: dict) -> tuple[str, str]:
    """(filesystem, path inside that filesystem) of an open file: bind mounts resolve to the same location."""
    mnt_id = next(line.split()[1] for line in open(f"/proc/self/fdinfo/{fd}") if line.startswith("mnt_id:"))
    dev, root, point = table[mnt_id]
    path = os.readlink(f"/proc/self/fd/{fd}")
    if not path.startswith("/") or path.endswith(" (deleted)"):
        raise Refused(f"cannot locate fd {fd} ({path})")
    rel = os.path.relpath(path, point)
    if rel == ".." or rel.startswith("../"):
        raise Refused(f"fd {fd} path {path} is outside its mount {point}")
    return dev, os.path.normpath(os.path.join(root, rel))


def within(inner: tuple[str, str], outer: tuple[str, str]) -> bool:
    return inner[0] == outer[0] and (inner[1] == outer[1] or inner[1].startswith(outer[1].rstrip("/") + "/"))


def open_layer(bundle: str, layer: str, table: dict | None = None) -> dict:
    """Validate P and DIR, then open DIR's components without following symlinks.

    Phase 1 judges everything that already exists and creates nothing. Phase 2 creates only the missing components,
    through the no-follow directory handle. Anything inside or aliasing P is refused in phase 1.
    """
    if not (os.path.isabs(bundle) and os.path.isabs(layer)):
        raise Refused("--bundle and --layer must be absolute paths")
    if os.path.realpath(bundle) != os.path.normpath(bundle) or not os.path.isdir(bundle):
        raise Refused(f"bundle {bundle} must be an existing directory given by its real path")
    if os.path.realpath(layer) != os.path.normpath(layer):
        raise Refused(f"layer {layer} must not contain symbolic links")
    bundle, layer = os.path.normpath(bundle), os.path.normpath(layer)
    table = mounts() if table is None else table
    fds = {"P": os.open(bundle, os.O_RDONLY | os.O_DIRECTORY)}
    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        where = {"P": location(fds["P"], table)}

        def judge(name: str) -> None:
            where[name] = location(fds[name], table)
            if within(where[name], where["P"]):
                raise Refused(f"layer component {name} lies inside or aliases the bundle {bundle}")

        # phase 1: nothing is created
        for _dev, _root, point in table.values():
            if point == layer or point.startswith(layer + "/"):
                raise Refused(f"mount point {point} inside the layer")
        if os.path.lexists(layer):
            fds["L"] = os.open(layer, dir_flags)
            judge("L")
            if within(where["P"], where["L"]):
                raise Refused(f"layer {layer} contains the bundle {bundle}")
            for name in (*CHILDREN, "lock"):
                entry = ".consumer.lock" if name == "lock" else name
                try:
                    st = os.stat(entry, dir_fd=fds["L"], follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if name == "lock":
                    if not stat.S_ISREG(st.st_mode):
                        raise Refused("layer lock is not a regular file")
                    fds[name] = os.open(entry, os.O_RDWR | os.O_NOFOLLOW, dir_fd=fds["L"])
                else:
                    if not stat.S_ISDIR(st.st_mode):
                        raise Refused(f"layer component {name} is not a directory")
                    fds[name] = os.open(entry, dir_flags, dir_fd=fds["L"])
                judge(name)
            if "upper" in fds:
                sealed_inodes = set()
                for d, _dirs, files in os.walk(bundle):
                    for f in files:
                        s = os.lstat(os.path.join(d, f))
                        sealed_inodes.add((s.st_dev, s.st_ino))
                for d, _dirs, files in os.walk(f"/proc/self/fd/{fds['upper']}"):
                    for f in files:
                        s = os.lstat(os.path.join(d, f))
                        if (s.st_dev, s.st_ino) in sealed_inodes:
                            raise Refused(f"layer file {os.path.join(d, f)} is a hard link into the bundle")
        else:
            existing = layer
            while not os.path.lexists(existing):
                existing = os.path.dirname(existing)
            fds["ancestor"] = os.open(existing, dir_flags)
            judge("ancestor")
            os.close(fds.pop("ancestor"))
        # phase 2: create what is missing, through the validated handles
        if "L" not in fds:
            os.makedirs(layer, 0o755, exist_ok=True)
            fds["L"] = os.open(layer, dir_flags)
            judge("L")
        for name in CHILDREN:
            if name not in fds:
                os.mkdir(name, 0o755, dir_fd=fds["L"])
                fds[name] = os.open(name, dir_flags, dir_fd=fds["L"])
                judge(name)
        if "lock" not in fds:
            fds["lock"] = os.open(".consumer.lock", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644,
                                  dir_fd=fds["L"])
            judge("lock")
        if where["upper"][0] != where["work"][0]:
            raise Refused("upper and work must be on one filesystem")
        return fds
    except BaseException as exc:
        for fd in fds.values():
            os.close(fd)
        if isinstance(exc, OSError):
            raise Refused(f"layer component cannot be opened safely: {exc}") from exc
        raise


def content(root_fd: int) -> dict:
    """rel -> content identity of every entry below an opened directory (never follows symlinks)."""
    root = f"/proc/self/fd/{root_fd}"
    out = {}
    for d, _dirs, files in os.walk(root):
        for f in files:
            path = os.path.join(d, f)
            rel = os.path.relpath(path, root)
            s = os.lstat(path)
            if stat.S_ISREG(s.st_mode):
                h = hashlib.sha256()
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                out[rel] = "sha256:" + h.hexdigest()
            elif stat.S_ISLNK(s.st_mode):
                out[rel] = "symlink:" + os.readlink(path)
            elif stat.S_ISCHR(s.st_mode) and s.st_rdev == 0:
                out[rel] = "whiteout"  # overlayfs deletion marker: the file is absent in the consumer's view
            else:
                out[rel] = f"special:{s.st_mode:o}"
    return out


def view_check(bundle: str, manifest_path: str, upper_fd: int, runs_fd: int, name: str, env: dict) -> dict:
    """The bundle's own checker (P/tools/compile_bundle.py check) on the consumer's view: P with the layer on top,
    mounted read-only. Its report (new / aliases / changed / missing, cache-relative) is the classification."""
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=runs_fd)
    os.close(fd)
    out = f"/proc/self/fd/{runs_fd}/{name}"
    r = subprocess.run(["bwrap", "--dev-bind", "/", "/", "--die-with-parent", "--overlay-src", bundle,
                        "--overlay-src", f"/proc/self/fd/{upper_fd}", "--ro-overlay", bundle, "--", sys.executable,
                        "-I", os.path.join(bundle, "tools", "compile_bundle.py"), "check", "--cache",
                        os.path.join(bundle, "jax"), "--manifest", manifest_path, "--out", out],
                       capture_output=True, text=True, close_fds=False,
                       env={**env, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    with open(out) as fh:
        report = json.load(fh)  # an unreadable report is a failure of this tool, never a silent zero
    report["summary"] = (r.stdout.strip().splitlines() or [""])[-1]
    return report


def classify(after: dict, report: dict, bundle: str) -> dict:
    """rel -> class for every upper entry, from the real checker's report on the merged view."""
    verdict = {}
    for key, cls in (("new", "compiled"), ("changed", "changed_sealed"), ("aliases", "alias"),
                     ("missing", "removed_in_view")):
        for crel in report.get(key, []):
            verdict["jax/" + crel] = cls
    out = {}
    for rel, ident in after.items():
        if rel in verdict:
            out[rel] = verdict[rel]
        elif rel.startswith("jax/") or "/__pycache__/" in f"/{rel}" or rel.startswith("tmp/"):
            out[rel] = "runtime"  # access stamps, locks, prefetch hints, C-auto plans, copies of sealed bytes
        elif rel.startswith(TOOLCHAIN):
            lower = os.path.join(bundle, rel)
            same = False
            if ident.startswith("sha256:") and os.path.isfile(lower) and not os.path.islink(lower):
                h = hashlib.sha256()
                with open(lower, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                same = "sha256:" + h.hexdigest() == ident
            out[rel] = "runtime" if same else "compiled"
        else:
            out[rel] = "other"
    return out


def descendants(root: int) -> list[int]:
    children = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as fh:
                    ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(entry))
    out, todo = [], [root]
    while todo:
        for pid in children.get(todo.pop(), ()):
            out.append(pid)
            todo.append(pid)
    return out


def reap() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _state(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return "gone"


def teardown(grace: float, report_every: float = 30.0) -> dict:
    """TERM every descendant, SIGKILL after ``grace`` seconds, and reap until NO descendant is left.

    There is no time bound, so no race under load. SIGKILL cannot be caught or ignored. A process in uninterruptible
    sleep dies when its system call returns, and every orphan is reparented to this subreaper. The loop therefore
    ends exactly when the whole consumer tree is gone, and only then may the caller release the locks. A process
    that never leaves uninterruptible sleep keeps this process, and so the locks, waiting; it is reported on stderr
    every ``report_every`` seconds.
    """
    me, t0 = os.getpid(), time.monotonic()
    rec = {"termed": 0, "killed": 0}
    for pid in descendants(me):
        try:
            os.kill(pid, signal.SIGTERM)
            rec["termed"] += 1
        except ProcessLookupError:
            pass
    end = t0 + grace
    while descendants(me) and time.monotonic() < end:
        reap()
        time.sleep(0.05)
    report = time.monotonic() + report_every
    while pids := descendants(me):
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                rec["killed"] += 1
            except ProcessLookupError:
                pass
        reap()
        if time.monotonic() >= report:
            states = ", ".join(f"{pid}:{_state(pid)}" for pid in pids[:20])
            print(f"BUNDLE_CONSUMER: teardown still waiting for {len(pids)} descendant(s) ({states}) after "
                  f"{time.monotonic() - t0:.0f} s; the layer and GPU locks stay held until they are gone", file=sys.stderr)
            report += report_every
        time.sleep(0.05)
    reap()
    rec["survivors"] = descendants(me)  # empty by construction; kept in the receipt as the check
    rec["seconds"] = round(time.monotonic() - t0, 3)
    return rec


KINDS = ("compiled", "changed_sealed", "alias", "removed_in_view", "runtime", "other")


def release_gpu_lock_fd(env: dict) -> None:
    """Close this process's copy of the with_gpu_lock fd (only when it really is that lock file)."""
    fd, lock = env.get("GPUWRF_GPU_LOCK_FD", ""), env.get("GPUWRF_GPU_LOCK_FILE", "")
    if fd.isdigit() and lock:
        try:
            if os.path.realpath(f"/proc/self/fd/{fd}") == os.path.realpath(lock):
                os.close(int(fd))
        except OSError:
            pass


def write_receipt(runs_fd: int, name: str, data: dict) -> str:
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=runs_fd)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=1)
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bundle_consumer", add_help=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--no-verify", action="store_true", help="skip the final real-checker run on P")
    ap.add_argument("--grace", type=float, default=15.0, help="seconds between TERM and KILL at teardown")
    ap.add_argument("--readonly-dir", action="append", default=[],
                    help="Read-only payload mount in this consumer's existing namespace (no nested bwrap).")
    ap.add_argument("--cwd", help="Existing absolute payload working directory.")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    env = dict(os.environ)  # the consumer's environment, captured before anything here can touch os.environ
    try:
        if not cmd:
            raise Refused("no command")
        if shutil.which("bwrap") is None:
            raise Refused("bubblewrap (bwrap) not installed")
        bundle, layer = a.bundle, a.layer
        payload_mounts = payload_options(a.readonly_dir, a.cwd)
        manifest_path = os.path.join(bundle, "sealed", "BUNDLE_MANIFEST.json")
        for need in (manifest_path, os.path.join(bundle, "tools", "compile_bundle.py")):
            if not os.path.isfile(need):
                raise Refused(f"missing {need}")
        fds = open_layer(bundle, layer)
        bundle, layer = os.path.normpath(bundle), os.path.normpath(layer)
    except Refused as exc:
        print(f"BUNDLE_CONSUMER_REFUSED: {exc}", file=sys.stderr)
        return REFUSED
    import fcntl
    try:
        fcntl.flock(fds["lock"], fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"BUNDLE_CONSUMER_REFUSED: layer {layer} is in use by another run", file=sys.stderr)
        return REFUSED
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
    before = content(fds["upper"])
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER: orphans of the consumer stay our descendants

    def on_signal(signum, _frame):
        raise Terminated(signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    for name in ("lock", "upper", "work"):  # the tree holds the layer lock too; bwrap opens the validated dirs
        os.set_inheritable(fds[name], True)
    argv_bwrap = ["bwrap", "--dev-bind", "/", "/", "--die-with-parent", "--overlay-src", bundle,
                  "--overlay", f"/proc/self/fd/{fds['upper']}", f"/proc/self/fd/{fds['work']}", bundle,
                  *payload_mounts, "--", *cmd]
    rc, signum, error, td = None, None, None, None
    try:
        proc = subprocess.Popen(argv_bwrap, env=env, close_fds=False,
                                preexec_fn=lambda: libc.prctl(1, signal.SIGKILL, 0, 0, 0))  # PR_SET_PDEATHSIG
        rc = proc.wait()
    except Terminated as exc:
        signum = exc.signum
    except BaseException as exc:  # noqa: BLE001 - any failure still tears the tree down before the locks go
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)  # teardown itself is not interruptible
        td = teardown(a.grace)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_DFL)  # the tree is gone: from here a signal may end this process at once
    release_gpu_lock_fd(env)  # bookkeeping below needs no GPU; teardown returned, so the whole tree is gone
    after = content(fds["upper"])
    written = {rel for rel, ident in after.items() if before.get(rel) != ident}
    view, classes, view_error = None, {}, None
    if signum:
        view_error = "skipped (terminated)"
    else:
        try:
            os.set_inheritable(fds["runs"], True)
            view = view_check(bundle, manifest_path, fds["upper"], fds["runs"], f"{stamp}.view.json", env)
            classes = classify(after, view, bundle)
        except Exception as exc:  # noqa: BLE001 - reported; the run's status then fails
            view_error = f"FAILED ({type(exc).__name__}: {exc})"
    by = {k: sorted(r for r in written if classes.get(r) == k) for k in KINDS}
    compiled_new = by["compiled"] + by["changed_sealed"]
    layer_compiled = sorted(r for r, c in classes.items() if c in ("compiled", "changed_sealed"))
    check = "skipped (--no-verify)" if a.no_verify else ("skipped (terminated)" if signum else None)
    if check is None:
        r = subprocess.run([sys.executable, "-I", os.path.join(bundle, "tools", "compile_bundle.py"), "check",
                            "--cache", os.path.join(bundle, "jax"), "--manifest", manifest_path],
                           capture_output=True, text=True,
                           env={**env, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0", "PYTHONDONTWRITEBYTECODE": "1"},
                           preexec_fn=lambda: libc.prctl(1, signal.SIGKILL, 0, 0, 0))
        check = (r.stdout.strip().splitlines() or [r.stderr.strip()])[-1]
    status = 128 + signum if signum else (1 if error else rc)
    receipt = {"bundle": bundle, "layer": layer, "cmd": cmd, "rc": status, "command_rc": rc, "signal": signum,
               "error": error, "teardown": td, "bundle_check_after": check,
               "view_check": view["summary"] if view else view_error, "written_by_this_run": by,
               "layer_compiled": layer_compiled}
    name = write_receipt(fds["runs"], f"{stamp}.json", receipt)
    for rel in compiled_new + by["other"] + by["removed_in_view"]:
        print(f"  wrote: {rel} ({classes[rel]})")
    print(f"view check (P + layer): {view['summary'] if view else view_error}")
    print(f"P check after: {check}")
    if view_error:
        counts = f"compiled_new={view_error.split()[0]} aliases_new=- runtime_new=- other_new=- layer_compiled=-"
    else:
        counts = (f"compiled_new={len(compiled_new)} aliases_new={len(by['alias'])} runtime_new={len(by['runtime'])} "
                  f"other_new={len(by['other']) + len(by['removed_in_view'])} layer_compiled={len(layer_compiled)}")
    print(f"BUNDLE_CONSUMER rc={status} {counts} receipt={layer}/runs/{name}")
    if view_error and not signum:
        print(f"BUNDLE_CONSUMER: view classification {view_error}", file=sys.stderr)
        status = status or 1
    if check.startswith("skipped") or check.startswith("BUNDLE_OK missing=0 changed=0 new=0 "):
        return status
    print(f"BUNDLE_CONSUMER: P CHECK FAILED ({check})", file=sys.stderr)
    return status or 1


if __name__ == "__main__":
    sys.exit(main())
