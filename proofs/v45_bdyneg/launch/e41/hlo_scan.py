"""Count optimized HLO instructions in kInput transpose fusion bodies (E41).

The threshold is 500 HLO instructions, including parameters and the root. It is
not a SASS instruction threshold. Compact AOT placeholders cannot prove zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

HEADER = re.compile(r"^\s*(?:ENTRY\s+)?%?([\w.\-]+)\s*(?:\(.*)?\{\s*$")
INSTR = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.+)$")
OP = re.compile(r"\b([a-z][a-z0-9-]*)\(")
CALLS = re.compile(r"\bcalls\s*=\s*%?([\w.\-]+)")
KIND = re.compile(r"\bkind\s*=\s*(k\w+)")


def scan_text(text: str) -> dict:
    computations, name, instructions, start = {}, None, [], 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if name is None:
            if not line.startswith("HloModule") and (head := HEADER.match(line)):
                name, start, instructions = head.group(1), lineno, []
            continue
        if line.strip() == "}":
            if name in computations or not instructions:
                raise ValueError(f"duplicate/empty optimized HLO computation: {name}")
            computations[name] = {"instructions": instructions, "lines": [start, lineno]}
            name = None
        elif match := INSTR.match(line):
            if re.match(r"slim\.(?:zero|convert|out|root)(?:\.|$)", match.group(1)):
                raise ValueError("compact AOT metadata cannot measure optimized HLO")
            op = OP.search(match.group(2))
            if not op:
                raise ValueError(f"unrecognized HLO instruction at {lineno}")
            instructions.append((match.group(1), op.group(1), line, lineno))
    if name is not None or not computations:
        raise ValueError("incomplete/empty optimized HLO computations")
    fusions, n_fusions = [], 0
    for parent, computation in computations.items():
        for fusion, opcode, line, lineno in computation["instructions"]:
            if opcode != "fusion":
                continue
            target, kind = CALLS.search(line), KIND.search(line)
            if not target or not kind or target.group(1) not in computations:
                raise ValueError(f"missing optimized HLO fusion body/kind at {lineno}")
            n_fusions += 1
            body = computations[target.group(1)]
            ops = [ins[1] for ins in body["instructions"]]
            if "transpose" in ops:
                fusions.append({"fusion": fusion, "kind": kind.group(1), "parent": parent,
                                "call_line": lineno, "comp": target.group(1),
                                "comp_lines": body["lines"], "n_instr": len(ops),
                                "n_transpose": ops.count("transpose"), "n_params": ops.count("parameter")})
    return {"computations": len(computations), "n_fusions": n_fusions,
            "transpose_hero_fusions": fusions}


def main(path: str, label: str = "scan") -> dict:
    result = dict(scan_text(Path(path).read_text()), file=str(path), label=label)
    Path(f"hlo_scan_{label}.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hlo")
    parser.add_argument("label", nargs="?", default="scan")
    args = parser.parse_args()
    main(args.hlo, args.label)
