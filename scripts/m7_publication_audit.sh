#!/usr/bin/env bash
set -euo pipefail

# v0.1.0 publication audit (CPU-only; no GPU/CPU model reruns).
# Checks: paper word count, ASCII-cleanliness of text assets, BibTeX/cited-key
# integrity, existence of the current v0.1.0 proof objects + generated tables,
# presence of the binding proof contract (publish/VERIFICATION.md + verify_all.sh),
# and AgentOS structural validation. Heavy GPU/CPU reruns are separate and live
# under scripts/verify_all.sh (VERIFY_RUN_GPU=1) — NOT invoked here.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(".")
PAPER = ROOT / "publish/paper/paper.md"
BIB = ROOT / "publish/paper/references.bib"

# Text assets whose ASCII-cleanliness matters at assembly.
TEXT_FILES = [
    PAPER,
    BIB,
    ROOT / "publish/paper/honesty_audit.md",
    ROOT / "publish/paper/missing_elements.md",
    ROOT / "scripts/m7_publication_audit.sh",
]

# The 9 current v0.1.0 evidence tables + the two process ledgers.
CURRENT_TABLES = [
    "publish/tables/performance_current.md",
    "publish/tables/v010_d02_validation.md",
    "publish/tables/v010_d03_status.md",
    "publish/tables/idealized_gate_summary.md",
    "publish/tables/optimization_refutations.md",
    "publish/tables/wind_persistence_skill.md",
    "publish/tables/systems_invariants.md",
    "publish/tables/v010_claim_boundary.md",
    "publish/tables/comparators.md",
    "publish/tables/ai_process_ledger.md",
    "publish/tables/effort_accounting.md",
]

# The binding proof contract that gates the v0.1.0 tag.
PROOF_CONTRACT = [
    "publish/VERIFICATION.md",
    "scripts/verify_all.sh",
]

# Current v0.1.0 proof objects the paper cites as load-bearing.
PROOF_OBJECTS = [
    "proofs/f7/DYCORE_STATUS.md",
    "proofs/f7n/skamarock_bubble_diagnostics.json",
    "proofs/f7n/straka_density_current_diagnostics.json",
    "proofs/v010_validation/v010_d02_result.json",
    "proofs/v010_validation/d03_summary_run24h_hfxfix4.json",
    "proofs/v010_validation/d03_validation_run24h_hfxfix4.json",
    "proofs/v010_validation/sfclay_hfx_oracle_parity.json",
    "proofs/v010_validation/speedup_vs_cpu_24h.json",
    "proofs/perf/roofline_costonly.json",
    "proofs/perf/speedup_denominator.md",
    "proofs/m19/verdict_result.json",
]

errors = []
warnings = []

if not PAPER.exists():
    print(json.dumps({"ok": False, "errors": [f"missing paper: {PAPER}"]}, indent=2))
    sys.exit(1)

paper_text = PAPER.read_text(encoding="utf-8")
main_text = paper_text.split("## References", 1)[0]
words = re.findall(r"[A-Za-z0-9_./+-]+", main_text)
word_count = len(words)
if not (6000 <= word_count <= 13000):
    errors.append(f"paper word count {word_count} outside [6000, 13000]")

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
    # Non-ASCII (unit symbols, em dashes) is allowed in the markdown paper but
    # surfaced as a warning so a LaTeX/plain-text export can decide.
    warnings.append(f"non-ascii characters present (informational): {sorted(non_ascii)}")

# BibTeX / citation integrity.
entries = []
try:
    import bibtexparser
    with BIB.open(encoding="utf-8") as handle:
        bib_db = bibtexparser.load(handle)
    entries = bib_db.entries
    if not entries:
        errors.append("bibtexparser parsed zero entries")
except Exception as exc:  # noqa: BLE001
    # Fall back to a regex scan of @type{key, lines so the audit still runs.
    warnings.append(f"bibtexparser unavailable ({type(exc).__name__}); using regex bib scan")
    bib_text = BIB.read_text(encoding="utf-8")
    entries = [{"ID": m} for m in re.findall(r"^@[A-Za-z]+\{([^,]+),", bib_text, re.MULTILINE)]

bib_keys = {entry.get("ID") for entry in entries if entry.get("ID")}
cite_keys = set()
for match in re.findall(r"\\cite\{([^}]+)\}", paper_text):
    cite_keys.update(key.strip() for key in match.split(",") if key.strip())
missing_citations = sorted(cite_keys - bib_keys)
if missing_citations:
    errors.append(f"orphan citation keys (cited but not in references.bib): {missing_citations}")
uncited_entries = sorted(bib_keys - cite_keys)

# Current evidence tables + proof contract + proof objects must all exist.
missing_tables = [p for p in CURRENT_TABLES if not (ROOT / p).exists()]
if missing_tables:
    errors.append(f"missing current evidence tables: {missing_tables}")

missing_contract = [p for p in PROOF_CONTRACT if not (ROOT / p).exists()]
if missing_contract:
    errors.append(f"missing proof-contract files: {missing_contract}")

missing_proofs = [p for p in PROOF_OBJECTS if not (ROOT / p).exists()]
if missing_proofs:
    errors.append(f"missing proof objects: {missing_proofs}")

# Cross-check: VERIFICATION.md should reference verify_all.sh as the runner.
verification_text = ""
vpath = ROOT / "publish/VERIFICATION.md"
if vpath.exists():
    verification_text = vpath.read_text(encoding="utf-8")
    if "verify_all.sh" not in verification_text:
        warnings.append("publish/VERIFICATION.md does not mention scripts/verify_all.sh")

# Guard against stale path resurfacing in the paper.
for stale in ("publication/draft", "22.26x", "22.26×"):
    if stale in paper_text and "retract" not in paper_text.lower():
        warnings.append(f"stale token {stale!r} present in paper without a retraction context")

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
    errors.append(
        f"validate_agentos failed: returncode={agentos.returncode} "
        f"stdout={agentos.stdout!r} stderr={agentos.stderr!r}"
    )

summary = {
    "ok": not errors,
    "paper": str(PAPER),
    "paper_word_count": word_count,
    "bib_entries": len(entries),
    "cited_keys": len(cite_keys),
    "orphan_citation_keys": missing_citations,
    "uncited_bib_entries": uncited_entries,
    "current_tables_checked": len(CURRENT_TABLES),
    "proof_contract_checked": PROOF_CONTRACT,
    "proof_objects_checked": len(PROOF_OBJECTS),
    "validate_agentos": agentos_json.get("ok", False),
    "warnings": warnings,
    "errors": errors,
}
print(json.dumps(summary, indent=2, sort_keys=True))
if errors:
    sys.exit(1)
PY
