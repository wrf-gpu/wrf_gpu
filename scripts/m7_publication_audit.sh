#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(".")
PAPER = ROOT / "paper" / "paper.md"
BIB = ROOT / "paper" / "references.bib"

TEXT_FILES = [
    ROOT / "README.md",
    ROOT / "INSTALL.md",
    ROOT / "CONTRIBUTING.md",
    PAPER,
    BIB,
    ROOT / "paper" / "honesty_audit.md",
    ROOT / "tables" / "performance_evolution.md",
    ROOT / "tables" / "skill_evolution.md",
    ROOT / "tables" / "m7_gates.md",
    ROOT / "scripts" / "m7_publication_audit.sh",
]

PROOF_OBJECTS = [
    ROOT / "proofs" / "2026-05-26-m7-gpu-profile-prep__wall_clock.json",
    ROOT / "proofs" / "2026-05-26-m7-gpu-profile-prep__nsys_summary.json",
    ROOT / "proofs" / "2026-05-27-m7-profiler-window-fix__reproducibility_v2.json",
    ROOT / "proofs" / "2026-05-27-m7-profiler-window-fix__d2h_audit_v2.json",
    ROOT / "proofs" / "2026-05-27-m7-daily-pipeline-integration__pipeline_run_20260521.json",
    ROOT / "proofs" / "2026-05-27-m7-daily-pipeline-integration__speedup_vs_cpu_24h.json",
    ROOT / "proofs" / "2026-05-27-m7-honest-speedup-skill-diff__honest_speedup_table.json",
    ROOT / "proofs" / "2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json",
    ROOT / "proofs" / "2026-05-27-m7-restart-continuity__restart_continuity.json",
    ROOT / "proofs" / "2026-05-27-m7-1km-memory-audit__step_feasibility.json",
    ROOT / "proofs" / "2026-05-27-m7-skill-fix-algorithmic__pipeline_run_20260521.json",
    ROOT / "proofs" / "2026-05-27-m7-skill-fix-algorithmic__post_fix_speedup.json",
    ROOT / "proofs" / "2026-05-27-m7-skill-fix-algorithmic__post_fix_skill_diff.json",
    ROOT / "proofs" / "2026-05-27-m7-skill-fix-algorithmic__invariant_preservation.json",
    ROOT / "proofs" / "2026-05-27-m7-skill-fix-iter2__pipeline_run_20260521.json",
    ROOT / "proofs" / "2026-05-27-m7-skill-fix-iter2__post_iter2_speedup.json",
    ROOT / "proofs" / "2026-05-27-m7-skill-fix-iter2__post_iter2_skill_diff.json",
    ROOT / "proofs" / "2026-05-27-m7-skill-fix-iter2__invariant_preservation_iter2.json",
]

forbidden_patterns = [
    re.escape("/home/" + "enric"),
    re.escape("/mnt/data/" + "canairy"),
    re.escape("enric" + ".r.g@"),
    re.escape("@" + "gmail"),
]
FORBIDDEN = re.compile("|".join(forbidden_patterns))
AUDITED_EXTENSIONS = {".md", ".py", ".sh", ".toml", ".yml", ".json", ".cff"}

errors: list[str] = []
for path in TEXT_FILES:
    if not path.exists():
        errors.append(f"missing text file: {path}")

missing_proofs = [str(path) for path in PROOF_OBJECTS if not path.exists()]
if missing_proofs:
    errors.append(f"missing proof objects: {missing_proofs}")

if PAPER.exists():
    paper_text = PAPER.read_text(encoding="utf-8")
    main_text = paper_text.split("## References", 1)[0]
    word_count = len(re.findall(r"[A-Za-z0-9_./+-]+", main_text))
else:
    paper_text = ""
    word_count = 0
    errors.append(f"missing paper: {PAPER}")

if BIB.exists():
    bib_text = BIB.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))
else:
    bib_keys = set()
    errors.append(f"missing bibliography: {BIB}")

cite_keys: set[str] = set()
for match in re.findall(r"\\cite\{([^}]+)\}", paper_text):
    cite_keys.update(key.strip() for key in match.split(",") if key.strip())
missing_citations = sorted(cite_keys - bib_keys)
if missing_citations:
    errors.append(f"missing BibTeX entries for citations: {missing_citations}")

scrub_hits: list[str] = []
for path in ROOT.rglob("*"):
    if ".git" in path.parts or not path.is_file() or path.suffix not in AUDITED_EXTENSIONS:
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    if FORBIDDEN.search(text):
        scrub_hits.append(str(path))
if scrub_hits:
    errors.append(f"forbidden personal path/email patterns remain: {scrub_hits}")

summary = {
    "ok": not errors,
    "paper_word_count": word_count,
    "bib_entries": len(bib_keys),
    "cited_keys": len(cite_keys),
    "missing_citations": missing_citations,
    "proof_objects_checked": len(PROOF_OBJECTS),
    "errors": errors,
}
print(json.dumps(summary, indent=2, sort_keys=True))
if errors:
    sys.exit(1)
PY
