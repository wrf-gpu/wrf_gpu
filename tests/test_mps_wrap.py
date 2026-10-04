"""SI41: scripts/mps_wrap.sh stops its private MPS daemon on every exit path (CPU-only: fake MPS binaries)."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

WRAP = Path(__file__).resolve().parents[1] / "scripts" / "mps_wrap.sh"
FAKE_CONTROL = r"""#!/usr/bin/env bash
P=${CUDA_MPS_PIPE_DIRECTORY:?}
if [[ ${1:-} == -d ]]; then
  [[ -n ${FAKE_FAIL_START:-} ]] && exit 1
  ( exec -a nvidia-cuda-mps-control sleep 600 ) &
  echo $! > "$P/fake.pid"; exit 0
fi
read -r cmd
case $cmd in
  quit) echo "fake quit"; [[ -n ${FAKE_STUBBORN:-} ]] || kill "$(cat "$P/fake.pid")" 2>/dev/null ;;
  get_server_list) echo "fake-server-list" ;;
esac
"""


@pytest.fixture
def env(tmp_path):
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    (fakebin / "nvidia-cuda-mps-control").write_text(FAKE_CONTROL)
    (fakebin / "nvidia-smi").write_text("#!/usr/bin/env bash\necho fake-nvidia-smi\n")
    for f in fakebin.iterdir():
        f.chmod(0o755)
    e = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_MPS_", "FAKE_"))}
    e.update(PATH=f"{fakebin}:{e['PATH']}", GPUWRF_GPU_LOCK_HELD="1", MPS_WRAP_QUIT_WAIT_S="3")
    return e


def _daemon_alive(mps_dir: Path) -> bool:
    pid_file = mps_dir / "pipe" / "fake.pid"
    if not pid_file.exists():
        return False
    try:
        os.kill(int(pid_file.read_text()), 0)
    except ProcessLookupError:
        return False
    return True


def _run(env, mps_dir, *cmd, **extra):
    return subprocess.run(["bash", str(WRAP), str(mps_dir), "--", *cmd], env={**env, **extra},
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("cmd,rc", [(["true"], 0), (["bash", "-c", "exit 7"], 7)])
def test_daemon_stopped_and_rc_propagated(env, tmp_path, cmd, rc):
    d = tmp_path / "mps"
    assert _run(env, d, *cmd).returncode == rc
    assert not _daemon_alive(d)
    log = (d / "mps_wrap.log").read_text()
    assert "daemon up" in log and "fake quit" in log and "daemon down" in log


def test_command_inherits_private_pipe_dir(env, tmp_path):
    d = tmp_path / "mps"
    out = _run(env, d, "bash", "-c", 'echo "$CUDA_MPS_PIPE_DIRECTORY"')
    assert out.returncode == 0 and out.stdout.strip() == str(d / "pipe")


def test_refuses_outside_gpu_lock(env, tmp_path):
    d = tmp_path / "mps"
    assert _run(env, d, "true", GPUWRF_GPU_LOCK_HELD="").returncode == 3
    assert not (d / "pipe" / "fake.pid").exists()


def test_start_failure_exits_4(env, tmp_path):
    assert _run(env, tmp_path / "mps", "true", FAKE_FAIL_START="1").returncode == 4


def test_stubborn_daemon_is_terminated(env, tmp_path):
    d = tmp_path / "mps"
    assert _run(env, d, "true", FAKE_STUBBORN="1").returncode == 0
    time.sleep(0.3)
    assert not _daemon_alive(d) and "still up after quit" in (d / "mps_wrap.log").read_text()


def test_term_is_forwarded_and_daemon_stopped(env, tmp_path):
    d = tmp_path / "mps"
    proc = subprocess.Popen(["bash", str(WRAP), str(d), "--", "sleep", "30"], env=env)
    time.sleep(1.5)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=60) == 128 + signal.SIGTERM
    assert not _daemon_alive(d)
    assert "signal TERM -> child" in (d / "mps_wrap.log").read_text()


def test_timeout_group_kill_stops_daemon(env, tmp_path):
    d = tmp_path / "mps"
    out = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30", "2", "bash", str(WRAP), str(d), "--",
                          "sleep", "30"], env=env, timeout=60)
    assert out.returncode == 124
    time.sleep(0.3)
    assert not _daemon_alive(d)
