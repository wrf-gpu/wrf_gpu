# wrf_gpu User's Guide — content fragment style guide

You are writing **one HTML content fragment** per page, saved to
`tools/docsite/content/<id>.html`. A Python generator wraps each fragment in the
site shell (sidebar, search, header, footer) — so **write only the inner
`<main>` content**, starting with a single `<h1>`. Do NOT write `<!doctype>`,
`<html>`, `<head>`, `<body>`, the sidebar, or the footer.

## Golden rules (non-negotiable)

1. **Accuracy over completeness.** Use ONLY facts present in the repository
   sources you are told to read (the public `README.md`, `src/gpuwrf/cli.py`,
   `docs/*.md`, `examples/`, source code). **Never invent numbers, flags, scheme
   names, or behavior.** If something is unclear or unverifiable, omit it or
   describe it qualitatively — do not fabricate.
2. **Honesty.** This is an AI-written rewrite validated against WRF as an oracle,
   not a Fortran port. Keep the project's honest framing: what is proven vs. open,
   opt-in vs. default, measured vs. projected. Never overclaim. Priority order is
   `stability > identity > speed > memory`. No masking/clamps/nan_to_num on results.
3. **No PII / no absolute local paths.** Never write `/home/<user>`, `/mnt/...`,
   real usernames, hostnames, or emails. Use placeholders: `<WRF_ROOT>`,
   `/path/to/WRF`, `/path/to/case`, `runs/<case>`, `/fast/nvme/scratch`.
4. **Style: WRF-Users-Guide-like — short, clear, complete.** Terse, technical,
   example-bearing. Prefer tables and short paragraphs over prose walls.
5. **Usage parity with WRF.** Where wrf_gpu intentionally maps to a WRF concept
   (namelist keys, `met_em`, `wrfout`, nesting), say so explicitly so a WRF user
   feels at home.

## Required HTML conventions

- One `<h1>Page Title</h1>` at the top, then a `<p class="lead">…</p>` intro.
- Section headings: `<h2 id="kebab-slug">Heading</h2>` (every h2 MUST have an
  `id` — the generator builds the sidebar sub-nav and search anchors from them).
  Sub-headings: `<h3 id="kebab-slug">…</h3>`.
- **Tables** must be wrapped:
  `<div class="table-scroll"><table><thead>…</thead><tbody>…</tbody></table></div>`
- **Code blocks**: `<pre><code>…</code></pre>`. You MUST HTML-escape `<`, `>`,
  `&` inside code (`&lt;`, `&gt;`, `&amp;`). Inline code: `<code>…</code>`.
- **Admonitions** (call-out boxes):
  `<div class="admon TYPE"><div class="admon-title">Label</div><p>…</p></div>`
  where TYPE ∈ `note` (blue), `warn` (amber), `danger` (red), `measured` (green,
  for MEASURED facts), `honest` (clay, for scope/limitation honesty).
- **Status pills** (inline): `<span class="pill op">Operational</span>`,
  `pill optin` (Opt-in), `pill ref` (Reference-only), `pill closed` (Fail-closed),
  `pill oos` (Out of scope).
- **Cross-links** between pages are relative: `<a href="physics.html">Physics
  Options</a>`, optionally with an anchor `physics.html#microphysics`.
  Page ids: index, concepts, installation, quickstart, running, input-data,
  namelist, physics, environment, output, dynamics, nesting, performance,
  validation, limitations, version-history, credits, glossary.
- External links open normally: `<a href="https://…">…</a>`.

## Voice
Second person ("you run…"), imperative for steps, present tense. Assume the reader
knows WRF/NWP basics but not this port. Aim for ~150–400 lines per page; complete
but not padded. Look at `content/index.html` for the exact house style.
