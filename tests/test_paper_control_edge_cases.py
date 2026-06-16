"""Tester edge-case tests for Sprint #6 paper-control-opus.

These tests are written from the perspective of the sprint tester. They lock
in the Sprint #4 PUBLISHABLE_AS_IS preconditions that Sprint #5 was supposed
to satisfy, and add adversarial checks the worker may have missed:

- Title must not contain "Canary" (any case).
- Option-2 novelty wording from `novelty_bounds.md` must appear verbatim in
  paper.md, with whitespace normalization.
- The Canary skill regression must appear in Abstract, Results, Limitations,
  and Discussion.
- Rejected/diagnostic-only headline numbers (156.82x and 50.20x) must never
  appear as the *current* result; the only current result is 22.26x.
- Every \\cite{...} key in paper.md resolves in references.bib (defensive
  re-implementation of the audit check, so a paper-only edit can be caught
  in isolation).
- BibTeX has no duplicate IDs.
- References.bib and paper.md are pure ASCII (no smart quotes / Unicode that
  would break LaTeX rendering).
- Every `.agent/...` proof object referenced literally in paper.md exists on
  disk.
- Every quantitative claim row in honesty_audit.md cites a proof file that
  exists on disk.
- The publication-audit JSON contract is satisfied (ok=true, word count in
  [6000, 12000], no missing citations).

The tests intentionally only read project files; they do not import GPU
modules, so they run on CPU-only CI and inside `taskset -c 0-3`.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "publication" / "draft" / "paper.md"
BIB = ROOT / "publication" / "draft" / "references.bib"
HONESTY = ROOT / "publication" / "draft" / "honesty_audit.md"
NOVELTY = (
    ROOT
    / ".agent"
    / "sprints"
    / "2026-05-28-gpu-wrf-history-research"
    / "novelty_bounds.md"
)
AUDIT = ROOT / "scripts" / "m7_publication_audit.sh"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Fixtures: lazy-cached file reads.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paper_text() -> str:
    return _read(PAPER)


@pytest.fixture(scope="module")
def paper_lines(paper_text: str) -> list[str]:
    return paper_text.splitlines()


@pytest.fixture(scope="module")
def bib_text() -> str:
    return _read(BIB)


@pytest.fixture(scope="module")
def novelty_text() -> str:
    return _read(NOVELTY)


@pytest.fixture(scope="module")
def section_index(paper_lines: list[str]) -> dict[str, tuple[int, int]]:
    """Map of section heading -> (start_line_idx, end_line_idx_exclusive)."""
    headings: list[tuple[int, str]] = []
    for i, line in enumerate(paper_lines):
        if line.startswith("## "):
            headings.append((i, line[3:].strip()))
    out: dict[str, tuple[int, int]] = {}
    for idx, (start, name) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(paper_lines)
        out[name] = (start, end)
    return out


def _section_text(
    paper_lines: list[str], section_index: dict[str, tuple[int, int]], name: str
) -> str:
    for key, (start, end) in section_index.items():
        if key == name or key.startswith(name):
            return "\n".join(paper_lines[start:end])
    raise AssertionError(f"section not found: {name!r}; have {list(section_index)}")


# ---------------------------------------------------------------------------
# AC1: precondition compliance.
# ---------------------------------------------------------------------------


def test_paper_title_does_not_contain_canary(paper_lines: list[str]) -> None:
    # The first non-empty line must be the H1 title.
    title = next(line for line in paper_lines if line.strip())
    assert title.startswith("# "), f"first non-empty line is not H1: {title!r}"
    # AC1: title must not contain Canary (any case).
    assert "canary" not in title.lower(), (
        f"paper title contains 'Canary' which violates AC1 of the sprint contract: {title!r}"
    )


def test_paper_title_is_stable_string(paper_lines: list[str]) -> None:
    title = next(line for line in paper_lines if line.strip())
    expected = "# wrf_gpu: An Open-Source JAX-Native WRF v4 Port with Whole-State GPU Residency"
    assert title == expected, f"title drift detected: {title!r} != {expected!r}"


def test_option_2_novelty_wording_verbatim(
    paper_text: str, novelty_text: str
) -> None:
    """The Option-2 paragraph from novelty_bounds.md must appear verbatim."""
    # Locate the Option-2 block in novelty_bounds.md and extract the quoted text.
    block = re.search(
        r"\*\*Option 2[^*]*\*\*[^\"]*\"([^\"]+)\"",
        novelty_text,
        re.DOTALL,
    )
    assert block is not None, "could not locate Option 2 quote in novelty_bounds.md"
    quote = block.group(1).strip()
    assert len(quote) > 200, f"Option-2 quote too short to be the real text: {quote!r}"
    # Tolerate whitespace differences but require byte-equal content.
    assert _normalize_ws(quote) in _normalize_ws(paper_text), (
        "Option-2 novelty wording is not present verbatim in paper.md "
        "(AC1 of paper-control contract). Quote was:\n" + quote
    )


def test_skill_regression_in_required_sections(
    paper_lines: list[str], section_index: dict[str, tuple[int, int]]
) -> None:
    """Sprint #4 binding precondition: skill regression in 4 specific sections."""
    # The skill regression numbers (T2/U10/V10 percentages) must be present in
    # each of Abstract, 7. Results / 7.3, Limitations, and 9. Discussion.

    def section_has_skill_regression(text: str) -> bool:
        text_l = text.lower()
        # Either the percentage band OR a clear regression statement.
        percent_band = bool(
            re.search(r"\+\s*1\d\d\s*%", text)
            or re.search(r"\+\s*[2-3]\d\d\s*%", text)
        )
        regression_phrase = (
            "materially less skilful" in text_l
            or "materially less skilful" in text_l
            or "materially less skillful" in text_l
            or "materially worse station rmse" in text_l
            or "skill gap" in text_l
            or "skill regression" in text_l
            or "skill-skill" in text_l
        )
        return percent_band or regression_phrase

    abstract = _section_text(paper_lines, section_index, "Abstract")
    results = _section_text(paper_lines, section_index, "7. Results")
    discussion = _section_text(paper_lines, section_index, "9. Discussion")
    limitations = _section_text(paper_lines, section_index, "11. Limitations")

    assert section_has_skill_regression(abstract), (
        "Abstract is missing the Canary skill regression statement"
    )
    assert section_has_skill_regression(results), (
        "Results is missing the Canary skill regression statement"
    )
    assert section_has_skill_regression(discussion), (
        "Discussion is missing the Canary skill regression statement"
    )
    assert section_has_skill_regression(limitations), (
        "Limitations is missing the Canary skill regression statement"
    )


# ---------------------------------------------------------------------------
# Anti-overclaim guardrails: Option-1 and rejected numbers.
# ---------------------------------------------------------------------------


def test_no_unqualified_first_gpu_wrf_claim(paper_text: str) -> None:
    """The paper must never say 'first GPU WRF' / 'first GPU-enabled WRF' as an
    unqualified positive claim. Every occurrence must be inside an explicit
    denial ('do not claim', 'does not claim')."""
    # Find every sentence containing the dangerous phrasing.
    sentences = re.split(r"(?<=[.!?])\s+", paper_text)
    bad: list[str] = []
    for s in sentences:
        s_l = s.lower()
        if (
            "first gpu-enabled wrf" in s_l
            or "first gpu wrf" in s_l
            or "the first full gpu" in s_l
            or "first commercial gpu wrf" in s_l
        ):
            if not (
                "do not claim" in s_l
                or "does not claim" in s_l
                or "not claim" in s_l
            ):
                bad.append(s.strip())
    assert not bad, (
        "Found 'first GPU-enabled WRF'-style claims without an explicit denial:\n"
        + "\n--\n".join(bad)
    )


def test_no_first_full_open_source_gpu_wrf_outside_denial(paper_text: str) -> None:
    """Same guardrail for the more academic 'first full open-source' phrasing,
    which is Option-1 (banned)."""
    # Look at the entire paper for "first full open-source GPU".
    # If present, it must be wrapped in the strict novelty-bound discussion,
    # not used as a headline claim. The strict bound IS allowed in line 19
    # because it is quoted as a defined term, not a claim.
    pattern = re.compile(r"first full open[- ]source[^.]*GPU", re.IGNORECASE)
    matches = list(pattern.finditer(paper_text))
    for m in matches:
        start = max(0, m.start() - 200)
        end = min(len(paper_text), m.end() + 200)
        window = paper_text[start:end].lower()
        # Must be either (a) inside quotes (strict bound definition), or
        # (b) wrapped in a denial.
        assert (
            '"' in window
            or "strict novelty bound" in window
            or "do not claim" in window
            or "does not claim" in window
            or "not claim" in window
        ), f"Option-1 style 'first full open-source GPU' claim is not bounded:\n{paper_text[start:end]}"


def test_rejected_speedup_numbers_are_marked_as_rejected(paper_text: str) -> None:
    """156.82x must appear ONLY as a rejected/corrected reference, never as the
    current operational result. 50.20x must appear only as diagnostic
    history/warning, never as the current claim."""
    # 156.82x rule: every occurrence must be in the same sentence as
    # 'overclaim', 'rejected', 'correction', 'corrected from', 'original',
    # 'before the fix' or similar.
    rejected_markers = (
        "overclaim",
        "rejected",
        "correction",
        "corrected",
        "original",
        "self-correction",
        "before the fix",
        "the original",
        "from 156.82",
    )
    sentences = re.split(r"(?<=[.!?])\s+", paper_text)
    for s in sentences:
        if "156.82" in s:
            s_l = s.lower()
            assert any(marker in s_l for marker in rejected_markers), (
                "156.82x appears without rejection context:\n" + s.strip()
            )

    # 50.20x: must be in a sentence that calls it 'pre-fix', 'diagnostic',
    # 'warning', 'less valid', or similar.
    pre_fix_markers = (
        "pre-fix",
        "diagnostic",
        "warning",
        "less valid",
        "diagnostic history",
    )
    for s in sentences:
        if "50.20" in s:
            s_l = s.lower()
            assert any(marker in s_l for marker in pre_fix_markers), (
                "50.20x appears without pre-fix/diagnostic context:\n" + s.strip()
            )


def test_only_current_speedup_is_22_26x(paper_text: str) -> None:
    """The only number described as the *current* speedup must be 22.26x."""
    sentences = re.split(r"(?<=[.!?])\s+", paper_text)
    current_speedup_sentences = [
        s
        for s in sentences
        if re.search(r"current\s+(?:speedup|result|iteration|.*speedup)", s.lower())
        and "x" in s.lower()
        and re.search(r"\d+\.\d+\s*x", s)
    ]
    # At least one sentence should explicitly tie 22.26x to "current".
    has_current_22 = any("22.26" in s for s in current_speedup_sentences)
    assert has_current_22, (
        "No sentence ties 22.26x to the current result; current_speedup_sentences="
        + repr(current_speedup_sentences)
    )


# ---------------------------------------------------------------------------
# AC3: citation audit (defensive reimplementation).
# ---------------------------------------------------------------------------


def _bib_keys(bib_text: str) -> set[str]:
    return set(re.findall(r"@\w+\{\s*([A-Za-z0-9_:\-]+)\s*,", bib_text))


def _cite_keys(paper_text: str) -> set[str]:
    found: set[str] = set()
    for match in re.findall(r"\\cite\{([^}]+)\}", paper_text):
        for key in match.split(","):
            key = key.strip()
            if key:
                found.add(key)
    return found


def test_all_cite_keys_resolve_in_bib(paper_text: str, bib_text: str) -> None:
    bibs = _bib_keys(bib_text)
    cites = _cite_keys(paper_text)
    missing = sorted(cites - bibs)
    assert not missing, f"unresolved \\cite keys: {missing}"


def test_bib_has_no_duplicate_ids(bib_text: str) -> None:
    ids = re.findall(r"@\w+\{\s*([A-Za-z0-9_:\-]+)\s*,", bib_text)
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"duplicate BibTeX IDs: {duplicates}"


def test_no_placeholder_citations(paper_text: str, bib_text: str) -> None:
    """Catch obvious placeholder/fake citations."""
    bad_keys = []
    placeholder_patterns = re.compile(
        r"^(todo|fixme|placeholder|xxx|tbd|fake|example)",
        re.IGNORECASE,
    )
    for key in _cite_keys(paper_text):
        if placeholder_patterns.match(key):
            bad_keys.append(key)
    for key in _bib_keys(bib_text):
        if placeholder_patterns.match(key):
            bad_keys.append(key)
    assert not bad_keys, f"placeholder-looking keys present: {sorted(set(bad_keys))}"


def test_bib_entries_have_minimum_fields(bib_text: str) -> None:
    """Every @article/@inproceedings/@book must carry title and year.

    @misc / @software are allowed to omit year, but must still have title."""
    # Parse entries by manual splitting on @ at column 0.
    entries = re.split(r"\n@", bib_text)
    missing = []
    for raw in entries:
        if not raw.strip().startswith("@") and not raw.strip().startswith("article"):
            # First entry doesn't lead with @ because we split on it; re-add.
            chunk = "@" + raw if not raw.strip().startswith("@") else raw
        else:
            chunk = raw
        m = re.match(r"@?(\w+)\s*\{\s*([A-Za-z0-9_:\-]+)\s*,", chunk)
        if not m:
            continue
        entry_type, key = m.group(1).lower(), m.group(2)
        if entry_type in {"comment", "preamble", "string"}:
            continue
        if "title" not in chunk.lower():
            missing.append(f"{key}: missing title")
        # Year required unless entry type is one of these.
        flexible = {"misc", "software", "online", "unpublished"}
        if entry_type not in flexible and not re.search(
            r"\byear\s*=", chunk, re.IGNORECASE
        ):
            missing.append(f"{key}: missing year")
    assert not missing, "BibTeX entries missing required fields: " + "; ".join(missing)


# ---------------------------------------------------------------------------
# AC2: honesty audit / proof object existence.
# ---------------------------------------------------------------------------


def test_referenced_proof_objects_exist(paper_text: str) -> None:
    """Every literal `.agent/...` reference in backticks must point to a
    real file or directory under the repo."""
    pattern = re.compile(r"`(\.agent/[^`]+)`")
    referenced = sorted({m.group(1) for m in pattern.finditer(paper_text)})
    missing: list[str] = []
    for ref in referenced:
        # Allow trailing slash; allow directory references like `.agent/sprints/`.
        candidate = ROOT / ref
        if not candidate.exists():
            # Strip trailing slash and try again as directory.
            if candidate.with_suffix("").exists():
                continue
            missing.append(ref)
    assert not missing, "missing referenced proof objects:\n  " + "\n  ".join(missing)


def test_honesty_audit_rows_point_to_existing_files() -> None:
    """Each row in honesty_audit.md that names a `.agent/...` or `publication/...`
    file path must resolve to an existing file."""
    text = _read(HONESTY)
    refs = re.findall(r"`(\.agent/[^`]+|publication/[^`]+)`", text)
    missing = []
    for ref in sorted(set(refs)):
        if not (ROOT / ref).exists():
            missing.append(ref)
    assert not missing, "honesty_audit.md cites missing proof objects: " + str(missing)


def test_honesty_audit_covers_skill_table_rows() -> None:
    """All nine RMSE entries in the Results 7.3 table must have a backing
    honesty-audit row."""
    paper_text = _read(PAPER)
    honesty_text = _read(HONESTY)
    rmse_rows = re.findall(
        r"(2026-05-(?:09|21|25))\s*\|\s*(T2|U10|V10).*?\+(\d+)\s*%",
        paper_text,
    )
    assert len(rmse_rows) == 9, f"expected 9 RMSE rows, got {rmse_rows}"
    for date, var, pct in rmse_rows:
        row_pattern = re.compile(
            rf"{re.escape(date)}.*{re.escape(var)}.*\+{pct}\s*%",
            re.DOTALL,
        )
        assert row_pattern.search(honesty_text), (
            f"honesty_audit.md lacks a row for {date}/{var} +{pct}%"
        )


# ---------------------------------------------------------------------------
# Sanity / hygiene.
# ---------------------------------------------------------------------------


def test_paper_and_bib_are_ascii() -> None:
    """The audit script enforces ASCII; we redundantly assert here so a
    paper-only edit that introduces smart quotes is caught even if the
    user is not running the shell audit."""
    for label, path in (("paper.md", PAPER), ("references.bib", BIB)):
        text = _read(path)
        bad_lines = []
        for i, line in enumerate(text.splitlines(), 1):
            if any(ord(ch) > 127 for ch in line):
                bad_lines.append((i, line))
        assert not bad_lines, f"{label} contains non-ASCII chars at lines {[l for l, _ in bad_lines]}"


def test_word_count_inside_audit_band(paper_text: str) -> None:
    main_text = paper_text.split("## References", 1)[0]
    words = re.findall(r"[A-Za-z0-9_./+-]+", main_text)
    n = len(words)
    assert 6000 <= n <= 12000, f"paper word count {n} outside [6000, 12000]"


def test_required_sections_present(section_index: dict[str, tuple[int, int]]) -> None:
    required = [
        "Abstract",
        "1. Introduction",
        "2. Background and Related Work",
        "3. The Code: Architecture",
        "4. The Code: Physics",
        "5. Methodology: Multi-Agent Engineering",
        "6. Validation Strategy",
        "7. Results",
        "8. Canary Case Study",
        "9. Discussion",
        "10. Open Source Release Plan",
        "11. Limitations",
        "12. Reproducibility",
        "13. Author Contributions and AI Use Disclosure",
        "14. Acknowledgements",
        "References",
    ]
    actual = list(section_index)
    for name in required:
        assert any(a == name or a.startswith(name) for a in actual), (
            f"missing section: {name!r}; have {actual}"
        )


def test_limitations_section_has_seven_items(
    paper_lines: list[str], section_index: dict[str, tuple[int, int]]
) -> None:
    text = _section_text(paper_lines, section_index, "11. Limitations")
    ids = sorted(set(re.findall(r"\*\*L(\d)\.\s", text)))
    assert ids == ["1", "2", "3", "4", "5", "6", "7"], (
        f"limitations section expected to enumerate L1..L7; found {ids}"
    )


def test_author_disclosure_present(paper_text: str) -> None:
    # The AI use disclosure is a load-bearing publication-ethics statement.
    text_l = paper_text.lower()
    assert "ai system" in text_l, "missing AI system disclosure"
    assert "user r.g." in text_l, "missing human corresponding-author identification"
    assert (
        "ai use disclosure" in text_l
        or "ai-use disclosure" in text_l
        or "ai use" in text_l
    ), "missing 'AI use disclosure' section reference"


# ---------------------------------------------------------------------------
# AC7: audit script self-test.
# ---------------------------------------------------------------------------


def _run_audit_json() -> dict:
    """Run scripts/m7_publication_audit.sh and return the JSON the inner
    Python block prints to stdout."""
    proc = subprocess.run(
        ["bash", str(AUDIT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    # The script prints one JSON document.
    raw = proc.stdout
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        pytest.fail(
            f"audit produced non-JSON output:\nstdout={raw!r}\nstderr={proc.stderr!r}"
        )
    payload["_returncode"] = proc.returncode
    return payload


def test_publication_audit_returns_ok_true() -> None:
    payload = _run_audit_json()
    assert payload["_returncode"] == 0, (
        f"audit script exit != 0; payload={json.dumps(payload, indent=2)}"
    )
    assert payload["ok"] is True, (
        f"audit script reported ok=false; payload={json.dumps(payload, indent=2)}"
    )
    assert payload["errors"] == [], f"audit errors: {payload['errors']}"
    assert not payload["missing_citations"], (
        f"missing citations: {payload['missing_citations']}"
    )
    assert payload["validate_agentos"]["ok"] is True, (
        f"validate_agentos.py failed: {payload['validate_agentos']}"
    )


def test_audit_recorded_uncited_set_is_tracked() -> None:
    """The 9 known uncited entries are tracked in the sprint contract as a
    decision item (CITED-or-TRIM). This test pins the current set so a
    silent additions to the bib show up as a test failure rather than as
    drift in the published draft."""
    payload = _run_audit_json()
    known_uncited = {
        "anthropic2024effective",
        "anthropic2026claude",
        "fredj2023adios2wrf",
        "huang2013thermal",
        "jakobs2024wsm7",
        "milroy2018ensemble",
        "roberts2008scale",
        "schmidt2025senior",
        "wernli2008sal",
    }
    actual = set(payload["uncited_entries"])
    # Drift in either direction is informative.
    extra = actual - known_uncited
    gone = known_uncited - actual
    assert not (extra or gone), (
        f"uncited-entries set drifted; +{sorted(extra)} -{sorted(gone)}"
    )
