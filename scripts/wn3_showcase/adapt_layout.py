"""Post-step for a run_parallel_cases.sh arm dir: write release-docs' layout next to the launcher receipts (CPU).
start.json (start_unix, cases, hours), analysis.json (per_case vram_peak_mib/wall/rc + arm wall), dmon.log -> gpu_dmon.log."""
import json, sys
from datetime import datetime
from pathlib import Path
arm = Path(sys.argv[1]); run = json.loads((arm / "parallel_run.json").read_text())
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()  # noqa: E731
hours = None
if "--hours" in run["cli_args"]:
    hours = int(run["cli_args"][run["cli_args"].index("--hours") + 1])
(arm / "start.json").write_text(json.dumps({"start_unix": ts(run["start_utc"]), "cases": " ".join(run["cases"]), "hours": hours,
                                            "source": "run_parallel_cases.sh", "cpus": run["cpu_affinity"]}) + "\n")
(arm / "analysis.json").write_text(json.dumps({"arm": str(arm), "n_cases": len(run["cases"]), "hours": hours, "arm_wall_s": run["wall_s"],
    "wall_s_per_case_hour": run.get("wall_s_per_case_hour"), "per_case": run["cases"]}, indent=1) + "\n")
if (arm / "gpu_dmon.log").exists() and not (arm / "dmon.log").exists():
    (arm / "dmon.log").symlink_to("gpu_dmon.log")
print(arm, "adapted")
