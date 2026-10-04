"""Use the existing real PROD harness against this worktree, with no model substitutes."""
import importlib.util
import os
from pathlib import Path
import sys

if os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
    raise SystemExit("PROD arm requires scripts/with_gpu_lock.sh --label census")

root = Path(__file__).resolve().parents[3]
harness_path = root / ".agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools/s0_nested_harness.py"
spec = importlib.util.spec_from_file_location("census_prod_harness", harness_path)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
# The historical harness prepends MAIN. Restore this lane before importing gpuwrf.
sys.path.insert(0, str(root / "src"))
harness.REPO = root
raise SystemExit(harness.main())
