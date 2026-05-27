#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(".")
PAPER = ROOT / "publication/draft/paper.md"
BIB = ROOT / "publication/draft/references.bib"

TEXT_FILES = [
    PAPER,
    BIB,
    ROOT / "publication/draft/tables/performance_summary.md",
    ROOT / "publication/draft/tables/skill_regression_summary.md",
    ROOT / "publication/draft/honesty_audit.md",
    ROOT / ".agent/sprints/2026-05-27-publication-revision-pass/revision_decisions.md",
    ROOT / "scripts/m7_publication_audit.sh",
]

PROOF_OBJECTS = [
    ".agent/decisions/MILESTONE-M6-CLOSEOUT.md",
    ".agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md",
    ".agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json",
    ".agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json",
    ".agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json",
    ".agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json",
    ".agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json",
    ".agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json",
    ".agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json",
    ".agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json",
    ".agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json",
    ".agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json",
    ".agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json",
    ".agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json",
    ".agent/sprints/2026-05-27-m7-skill-fix-algorithmic/invariant_preservation.json",
    ".agent/sprints/2026-05-27-publication-revision-pass/revision_decisions.md",
    "publication/draft/honesty_audit.md",
]

errors = []

paper_text = PAPER.read_text(encoding="utf-8")
main_text = paper_text.split("## References", 1)[0]
words = re.findall(r"[A-Za-z0-9_./+-]+", main_text)
word_count = len(words)
if not (6000 <= word_count <= 12000):
    errors.append(f"paper word count {word_count} outside [6000, 12000]")

non_ascii = {}
for path in TEXT_FILES:
    if not path.exists():
        errors.append(f"missing text file: {path}")
        continue
    bad_lines = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if any(ord(ch) > 127 for ch in line):
            bad_lines.append(lineno)
    if bad_lines:
        non_ascii[str(path)] = bad_lines
if non_ascii:
    errors.append(f"non-ascii characters found: {non_ascii}")

try:
    import bibtexparser
except Exception as exc:
    errors.append(f"bibtexparser import failed: {type(exc).__name__}: {exc}")
    entries = []
else:
    with BIB.open(encoding="utf-8") as handle:
        bib_db = bibtexparser.load(handle)
    entries = bib_db.entries
    if not entries:
        errors.append("bibtexparser parsed zero entries")

bib_keys = {entry.get("ID") for entry in entries if entry.get("ID")}
cite_keys = set()
for match in re.findall(r"\\cite\{([^}]+)\}", paper_text):
    cite_keys.update(key.strip() for key in match.split(",") if key.strip())
missing_citations = sorted(cite_keys - bib_keys)
if missing_citations:
    errors.append(f"missing BibTeX entries for citations: {missing_citations}")
uncited_entries = sorted(bib_keys - cite_keys)

missing_proofs = [path for path in PROOF_OBJECTS if not (ROOT / path).exists()]
if missing_proofs:
    errors.append(f"missing proof objects: {missing_proofs}")

agentos = subprocess.run(
    [sys.executable, "scripts/validate_agentos.py"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
try:
    agentos_json = json.loads(agentos.stdout)
except json.JSONDecodeError:
    agentos_json = {"ok": False, "raw_stdout": agentos.stdout, "stderr": agentos.stderr}
if agentos.returncode != 0 or not agentos_json.get("ok"):
    errors.append(f"validate_agentos failed: returncode={agentos.returncode} stdout={agentos.stdout!r} stderr={agentos.stderr!r}")

summary = {
    "ok": not errors,
    "paper_word_count": word_count,
    "bib_entries": len(entries),
    "cited_keys": len(cite_keys),
    "missing_citations": missing_citations,
    "uncited_entries": uncited_entries,
    "proof_objects_checked": len(PROOF_OBJECTS),
    "validate_agentos": agentos_json,
    "errors": errors,
}
print(json.dumps(summary, indent=2, sort_keys=True))
if errors:
    sys.exit(1)
PY
