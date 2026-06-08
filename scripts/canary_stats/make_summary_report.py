#!/usr/bin/env python3
"""Generate a lightweight Markdown/plot summary from canary_stats artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _plot_counts(counts: Counter[str], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(counts)
    values = [counts[k] for k in labels]
    plt.figure(figsize=(max(6, len(labels) * 1.2), 4))
    plt.bar(labels, values, color="#4c78a8")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--pairability", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--plot-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    pair = json.loads(args.pairability.read_text(encoding="utf-8"))
    prov_counts = Counter(r.get("provenance", "unknown") for r in inv["records"])
    pair_counts = Counter(p["classification"] for p in pair["pairs"])
    by_domain = defaultdict(Counter)
    for p in pair["pairs"]:
        by_domain[p["domain"]][p["classification"]] += 1

    prov_plot = args.plot_dir / "inventory_by_provenance.png"
    pair_plot = args.plot_dir / "pairability_by_class.png"
    _plot_counts(prov_counts, prov_plot, "Inventory by provenance")
    _plot_counts(pair_counts, pair_plot, "Raw pairability by class")

    lines = [
        "# Canary Existing-Data Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Inventory records: {inv['record_count']}",
        f"- Raw CPU/GPU pair candidates: {pair['pair_count']}",
        "",
        "## Inventory By Provenance",
        "",
    ]
    for key, val in sorted(prov_counts.items()):
        lines.append(f"- {key}: {val}")
    lines += ["", f"![Inventory by provenance]({prov_plot.relative_to(args.out_md.parent) if prov_plot.is_relative_to(args.out_md.parent) else prov_plot})", ""]
    lines += ["## Pairability", ""]
    for key, val in sorted(pair_counts.items()):
        lines.append(f"- {key}: {val}")
    lines += ["", f"![Pairability by class]({pair_plot.relative_to(args.out_md.parent) if pair_plot.is_relative_to(args.out_md.parent) else pair_plot})", ""]
    lines += ["## Pairability By Domain", ""]
    for dom, counts in sorted(by_domain.items()):
        bits = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"- {dom}: {bits}")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary report={args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
