import os
def test_env_after_restart_tests():
    leaked = {k: os.environ[k] for k in ("GPUWRF_SCRATCH", "GPUWRF_TMPDIR") if k in os.environ}
    print("LEAKED", leaked)
    assert not leaked, leaked
