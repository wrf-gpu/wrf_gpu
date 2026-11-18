#!/usr/bin/env python3
"""Static-site generator for the wrf_gpu User's Guide.

Pure Python standard library (no deps). Reads HTML content fragments from
``content/<id>.html``, wraps each in ``template.html`` with a generated sidebar,
breadcrumb and prev/next nav, and emits the built site plus a client-side
``search-index.json`` into the output directory and its ``_assets`` directory
(default: repo ``docs/``). The root copy preserves the historical public
artifact; the browser loads the asset copy.

Usage:
    python3 tools/docsite/build.py            # build into ../../docs
    python3 tools/docsite/build.py --out DIR  # build elsewhere
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

VERSION = "0.5.1"

# ---- site structure: ordered groups -> pages -------------------------------
# Each page: (id, nav_title, <title>, meta-description)
SITE = [
    ("Introduction", [
        ("index",        "Overview",                    "Overview",
         "wrf_gpu is a GPU-native, WRF-compatible regional weather model — an AI-written JAX rewrite validated against WRF as an oracle."),
        ("concepts",     "How wrf_gpu Works",           "How wrf_gpu Works",
         "Architecture and design of wrf_gpu: GPU-native JAX dynamical core, native initialization, and validation against WRF as an oracle."),
    ]),
    ("Getting started", [
        ("installation", "Installation & Requirements", "Installation & Requirements",
         "System requirements and installation of wrf_gpu: GPU, CUDA 13, JAX, Python, and the runtime table prerequisites."),
        ("quickstart",   "Quickstart",                  "Quickstart: Your First Forecast",
         "Run your first standalone GPU forecast with the bundled Switzerland 3 km case and read the WRF-compatible wrfout."),
        ("ai-quickstart","Quickstart with an AI Agent", "Quickstart with an AI Agent",
         "Clone the repo, open an AI coding agent, and say 'here is my input, get it running' — the repo ships an operator skill the agent auto-loads."),
        ("running",      "Running the Model",           "Running the Model",
         "Run a forecast with gpuwrf run: the bundled case, your own case, nests, checkpoints and restart, GPU memory."),
        ("tools",        "Tools & Commands",            "Tools and Commands",
         "Every gpuwrf command: region set-up, case checks, weather products, station verification, ensembles, export, cycling and several GPUs."),
        ("input-data",   "Input Data & Initialization", "Input Data & Initialization",
         "Prepared wrfinput/wrfbdy inputs, optional wrflowinp SST forcing, namelist controls and verified restart."),
    ]),
    ("Configuration", [
        ("namelist",     "Namelist Compatibility",      "Namelist Compatibility",
         "How your WRF namelist is checked, the status of each model feature (nudging, DFI, urban, lake, ocean, wind farms, tracers, diagnostics) and what is not supported."),
        ("physics",      "Physics Options",             "Physics Options",
         "Every WRF physics option code (microphysics, cumulus, PBL, surface layer, land, radiation) and Noah-MP option, with its v0.4.0 status."),
        ("environment",  "Environment Variables",       "Environment Variables",
         "Reference for the GPUWRF_* environment variables: paths, output, performance and precision modes, caching, nesting, and debugging."),
        ("output",       "Output Files",                "Output Files",
         "The WRF-compatible wrfout history file, the focused training subset, full output, colon-free naming, and restart files."),
    ]),
    ("The model", [
        ("dynamics",     "Dynamical Core",              "Dynamical Core",
         "The nonhydrostatic split-explicit ARW dynamical core: RK3, WRF-derived acoustic substeps, default fp32 with DOUBLE islands, diffusion and boundaries."),
        ("nesting",      "Nesting",                     "Nesting",
         "Fixed nests run together on the GPU: one-way nesting, sub-cycling, two-way feedback, restart and what is not supported."),
    ]),
    ("Performance & validation", [
        ("performance",  "Performance & GPU Modes",     "Performance & GPU Modes",
         "Cold compile and warm-start cache, the batched-ensemble throughput mode, optional fp32, VRAM, the scaling law and energy."),
        ("validation",   "Validation against WRF",      "Validation against WRF",
         "How wrf_gpu is checked against the original CPU-WRF: same inputs, every hourly field, limits fixed in advance, and the results of each release."),
        ("limitations",  "Boundaries & Known Issues",   "Boundaries & Known Issues",
         "What v0.4.0 does not cover yet: untested combinations, nests, experimental features, unsupported WRF components and known numerical issues."),
    ]),
    ("Reference", [
        ("version-history", "Version History",          "Version History",
         "Release-by-release history of wrf_gpu, from the first GPU kernels to v0.4.0."),
        ("credits",      "Credits, License & Citation", "Credits, License & Citation",
         "Credit to the WRF/NCAR/UCAR team, the AI authorship of this rewrite, licensing notes, and how to cite the project."),
        ("glossary",     "Glossary",                    "Glossary",
         "Definitions of WRF, NWP and wrf_gpu-specific terms used throughout the guide."),
    ]),
]

FLAT = [(pid, nav, title, desc) for _, pages in SITE for (pid, nav, title, desc) in pages]
IDS = [p[0] for p in FLAT]


class SectionExtractor(HTMLParser):
    """Split a content fragment into sections at h2/h3 headings that carry an id,
    collecting heading text and body text for the sidebar sub-nav + search index."""

    SKIP = {"script", "style"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections = []          # list of dicts: level, anchor, heading, text
        self.subnav = []            # (anchor, heading) for h2 only
        self._cur = {"level": 1, "anchor": "", "heading": "", "text": []}
        self._in_heading = False
        self._heading_level = 0
        self._heading_anchor = ""
        self._heading_buf = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if tag in ("h2", "h3"):
            a = dict(attrs)
            self._flush()
            self._in_heading = True
            self._heading_level = int(tag[1])
            self._heading_anchor = a.get("id", "")
            self._heading_buf = []

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in ("h2", "h3") and self._in_heading:
            heading = "".join(self._heading_buf).strip()
            self._cur = {"level": self._heading_level, "anchor": self._heading_anchor,
                         "heading": heading, "text": []}
            if self._heading_level == 2 and self._heading_anchor:
                self.subnav.append((self._heading_anchor, heading))
            self._in_heading = False

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_heading:
            self._heading_buf.append(data)
        else:
            self._cur["text"].append(data)

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._cur["text"])).strip()
        if self._cur["heading"] or text:
            self.sections.append({**self._cur, "text": text})
        self._cur = {"level": 1, "anchor": "", "heading": "", "text": []}

    def close(self):
        self._flush()
        super().close()


def render_sidebar(active_id, subnav):
    out = ['<nav class="toc">']
    for group, pages in SITE:
        out.append(f'<div class="group-title">{html.escape(group)}</div>')
        for pid, nav, _title, _desc in pages:
            active = " active" if pid == active_id else ""
            out.append(f'<a class="{active.strip()}" href="{pid}.html">{html.escape(nav)}</a>')
            if pid == active_id and subnav:
                out.append('<div class="subnav">')
                for anchor, heading in subnav:
                    out.append(f'<a href="{pid}.html#{anchor}">{html.escape(heading)}</a>')
                out.append("</div>")
    out.append("</nav>")
    return "\n".join(out)


def render_breadcrumb(pid, title):
    group = next(g for g, pages in SITE for (p, *_r) in pages if p == pid)
    home = "" if pid == "index" else '<a href="index.html">Overview</a> › '
    if pid == "index":
        return '<div class="breadcrumb">wrf_gpu User\'s Guide</div>'
    return (f'<div class="breadcrumb"><a href="index.html">User\'s Guide</a> › '
            f'{html.escape(group)} › {html.escape(title)}</div>')


def render_prevnext(pid):
    i = IDS.index(pid)
    parts = ['<nav class="page-nav">']
    if i > 0:
        p = FLAT[i - 1]
        parts.append(f'<a class="prev" href="{p[0]}.html"><span class="pn-dir">← Previous</span>'
                     f'<span class="pn-title">{html.escape(p[1])}</span></a>')
    else:
        parts.append('<span></span>')
    if i < len(FLAT) - 1:
        n = FLAT[i + 1]
        parts.append(f'<a class="next" href="{n[0]}.html"><span class="pn-dir">Next →</span>'
                     f'<span class="pn-title">{html.escape(n[1])}</span></a>')
    else:
        parts.append('<span></span>')
    parts.append("</nav>")
    return "\n".join(parts)


def add_headerlinks(content):
    """Give every h2/h3 with an id a hover ¶ anchor link."""
    def repl(m):
        tag, attrs, inner = m.group(1), m.group(2), m.group(3)
        mid = re.search(r'id="([^"]+)"', attrs)
        if not mid:
            return m.group(0)
        return f'<{tag}{attrs}>{inner}<a class="headerlink" href="#{mid.group(1)}" aria-label="Permalink">¶</a></{tag}>'
    return re.sub(r'<(h[23])([^>]*)>(.*?)</\1>', repl, content, flags=re.S)


GITHUB_BLOB = "https://github.com/wrf-gpu/wrf_gpu/blob/main/"
_HREF = re.compile(r'href="([^"]*)"')


def page_anchors(content):
    return set(re.findall(r'\bid="([^"]+)"', content))


def rewrite_links(content, pid, repo, anchors, problems):
    """Pages serves docs/ as plain files (.nojekyll): a .md target arrives as raw text and a
    path outside docs/ is a 404. Point those at the rendered GitHub page instead, and record
    every link whose target file or in-guide anchor does not exist."""
    def repl(m):
        url = m.group(1)
        if not url or url.startswith(("#", "/", "mailto:")) or re.match(r"[a-z][a-z0-9+.-]*://", url, re.I):
            if url.startswith("#") and url[1:] and url[1:] not in anchors[pid]:
                problems.append(f"{pid}: missing anchor {url}")
            return m.group(0)
        path, _, frag = url.partition("#")
        target = os.path.normpath(os.path.join("docs", path))
        if "/" not in path and path.endswith(".html") and path[:-5] in anchors:  # another guide page
            if frag and frag not in anchors[Path(path).stem]:
                problems.append(f"{pid}: missing anchor {url}")
            return m.group(0)
        if not (repo / target).exists():
            problems.append(f"{pid}: missing target {url}")
            return m.group(0)
        if target.startswith("docs" + os.sep) and not path.endswith(".md"):
            return m.group(0)  # served by Pages as is (evidence galleries, images)
        return f'href="{GITHUB_BLOB}{Path(target).as_posix()}{"#" + frag if frag else ""}"'
    return _HREF.sub(repl, content)


def build(out_dir: Path):
    root = Path(__file__).resolve().parent
    template = (root / "template.html").read_text()
    content_dir = root / "content"
    assets_src = root / "assets"
    repo = root.parents[1]
    fragments = {pid: (content_dir / f"{pid}.html").read_text()
                 for pid in IDS if (content_dir / f"{pid}.html").exists()}
    anchors = {pid: page_anchors(c) for pid, c in fragments.items()}
    problems = []

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_out = out_dir / "_assets"
    assets_out.mkdir(exist_ok=True)
    for f in ("style.css", "app.js", "logo.svg"):
        shutil.copy2(assets_src / f, assets_out / f)
    (out_dir / ".nojekyll").write_text("")

    search_index = []
    missing = []

    for pid, nav, title, desc in FLAT:
        frag_path = content_dir / f"{pid}.html"
        if not frag_path.exists():
            missing.append(pid)
            content = f"<h1>{html.escape(title)}</h1><p><em>Content pending.</em></p>"
            subnav = []
        else:
            content = rewrite_links(fragments[pid], pid, repo, anchors, problems)
            ex = SectionExtractor()
            ex.feed(content)
            ex.close()
            subnav = ex.subnav
            # search entries: page + each id-bearing section
            page_text = " ".join(s["text"] for s in ex.sections)
            search_index.append({"url": f"{pid}.html", "title": title,
                                  "crumb": nav, "text": page_text[:1200]})
            for s in ex.sections:
                if s["anchor"] and s["heading"]:
                    search_index.append({
                        "url": f"{pid}.html#{s['anchor']}",
                        "title": s["heading"],
                        "crumb": f"{title}",
                        "text": (s["text"] or s["heading"])[:600],
                    })
            content = add_headerlinks(content)

        page = (template
                .replace("__BASE__", "")
                .replace("__VERSION__", VERSION)
                .replace("__TITLE__", html.escape(title))
                .replace("__DESC__", html.escape(desc))
                .replace("__SIDEBAR__", render_sidebar(pid, subnav))
                .replace("__BREADCRUMB__", render_breadcrumb(pid, title))
                .replace("__CONTENT__", content)
                .replace("__PREVNEXT__", render_prevnext(pid) if pid != "index" else render_prevnext(pid)))
        (out_dir / f"{pid}.html").write_text(page)

    # ---- dedicated search results page ----
    search_body = (
        '<h1>Search the guide</h1>'
        '<form class="search-page-form" role="search">'
        '<input id="full-search-input" type="search" placeholder="Search…" '
        'style="width:100%;max-width:520px;padding:.6rem .8rem;font-size:1rem;'
        'border:1px solid var(--border-strong);border-radius:8px;" aria-label="Search">'
        '</form>'
        '<h2 id="results-heading" style="border:none;margin-top:1.4rem;">Type a query to search the guide.</h2>'
        '<div id="full-results"></div>'
        '<style>.result{padding:.8rem 0;border-bottom:1px solid var(--border);}'
        '.result-link{font-weight:600;font-size:1.05rem;} .result .sr-crumb{color:var(--fg-soft);font-size:.82rem;}'
        '.result .sr-snip{margin-top:.25rem;color:var(--fg-soft);} mark{background:#f4b41a;color:#1b2330;border-radius:2px;}</style>'
    )
    page = (template
            .replace("__BASE__", "")
            .replace("__VERSION__", VERSION)
            .replace("__TITLE__", "Search")
            .replace("__DESC__", "Search the wrf_gpu User's Guide.")
            .replace("__SIDEBAR__", render_sidebar("", []))
            .replace("__BREADCRUMB__", '<div class="breadcrumb"><a href="index.html">User\'s Guide</a> › Search</div>')
            .replace("__CONTENT__", search_body)
            .replace("__PREVNEXT__", ""))
    (out_dir / "search.html").write_text(page)

    search_payload = json.dumps(search_index, ensure_ascii=False)
    (assets_out / "search-index.json").write_text(search_payload)
    (out_dir / "search-index.json").write_text(search_payload)

    print(f"Built {len(FLAT)} pages + search.html into {out_dir}")
    print(f"Search index: {len(search_index)} entries")
    if missing:
        print(f"WARNING: {len(missing)} pages have no content fragment yet: {', '.join(missing)}")
    if problems:
        raise SystemExit("broken guide links:\n  " + "\n  ".join(problems))


def main():
    ap = argparse.ArgumentParser()
    default_out = Path(__file__).resolve().parents[2] / "docs"
    ap.add_argument("--out", type=Path, default=default_out)
    args = ap.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
