"""CPU-only native E41 scanner. No CUDA context is created.

STACK is measured by cuobjdump, SASS instructions by disassembly. A large root
caller is diagnostic only: the 1000-instruction limit applies to a *callee*
which itself calls another routine (E189). Unresolved calls/control flow are
UNKNOWN, never evidence of a passing gate.

CLI compatibility: cubin_scan.py CUBIN [TIMING_DB] [LABEL] writes
cubin_scan_LABEL.json. The timing DB is accepted for old launchers but unused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)")
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+([^/]+)", re.I)
RESOURCE = re.compile(r"^\s*Function\s+(\S+):\s*$")
OPCODE = re.compile(r"^(?:@!?U?P(?:\d+|T)\s+)?([A-Z][A-Z0-9]*)(?:\.[A-Z0-9]+)*\b")
DIRECT = re.compile(r"\b(?:CALL|BRA)(?:\.[A-Z0-9]+)*\s+(?:[^;]*,\s*)?(0x[0-9a-f]+|`?\(?[A-Za-z_$][\w.$]*\)?`?)\s*;", re.I)
BARRIER_SET = re.compile(r"\bBSSY(?:\.[A-Z0-9]+)*\s+(B\d+),\s*0x([0-9a-f]+)\s*;", re.I)
BARRIER_SYNC = re.compile(r"\bBSYNC(?:\.[A-Z0-9]+)*\s+(B\d+)\s*;", re.I)


def parse_resources(text: str) -> dict[str, dict]:
    rows, current = {}, None
    for line in text.splitlines():
        if match := RESOURCE.match(line):
            current = match.group(1)
            if current in rows:
                raise ValueError(f"duplicate resource function: {current}")
            rows[current] = {"kernel": current}
        if current is not None:
            for key, value in re.findall(r"\b(STACK|REG|LOCAL|SHARED):(\d+)\b", line):
                rows[current][{"REG": "regs", "STACK": "stack", "LOCAL": "local", "SHARED": "shared"}[key]] = int(value)
    if not rows or any("stack" not in row for row in rows.values()):
        raise ValueError("missing measured STACK resource rows")
    return rows


def parse_sass(text: str) -> dict[str, dict[int, str]]:
    functions, current = {}, None
    for line in text.splitlines():
        if match := FUNCTION.match(line):
            current = functions.setdefault(match.group(1), {})
            continue
        match = INSTRUCTION.match(line)
        if current is not None and match:
            pc, op = int(match.group(1), 16), match.group(2).strip()
            if pc in current:
                raise ValueError("duplicate SASS instruction address")
            current[pc] = op
    if not functions or not all(functions.values()):
        raise ValueError("empty SASS function/instruction inventory")
    return functions


def _opcode(op: str) -> str:
    match = OPCODE.match(op)
    if not match:
        raise ValueError(f"unrecognized SASS opcode: {op}")
    return match.group(1)


def _target(op: str, name: str, functions: dict):
    match = DIRECT.search(op)
    if not match:
        return None
    target = match.group(1).strip("`()")
    if target.startswith("0x"):
        pc = int(target, 16)
        return (name, pc) if pc in functions[name] else None
    if target in functions:
        return target, min(functions[target])
    return None


def analyze_calls(functions: dict[str, dict[int, str]]) -> dict[str, dict]:
    """Walk each direct callee's CFG without including its nested call bodies.

    cuobjdump typically emits local CALL.REL.NOINC targets inside one Function,
    so counting Function blocks alone mistakes a large root for a mega-callee.
    BRA targets and predicated returns are followed; indirect control flow is
    explicitly UNKNOWN. Function-symbol calls are supported too. BSSY barrier
    bindings are local to each traversal path, never shared between unrelated
    callees which reuse barrier registers. BSYNC is accepted as fallthrough
    only when its bound convergence target is the following instruction.
    NVIDIA's instruction names/roles: CUDA Binary Utilities 13.1, section 6.
    https://docs.nvidia.com/cuda/archive/13.1.2/pdf/CUDA_Binary_Utilities.pdf
    """
    results = {}
    for name, instructions in functions.items():
        targets, unknown = set(), []
        calls = [(pc, op) for pc, op in instructions.items() if _opcode(op) == "CALL"]
        for pc, op in calls:
            target = _target(op, name, functions)
            if target is None:
                unknown.append({"pc": pc, "op": op, "reason": "unresolved CALL"})
            else:
                targets.add(target)
        rows = []
        for function, start in sorted(targets):
            code = functions[function]
            pcs = sorted(code)
            following = dict(zip(pcs, pcs[1:]))
            pending, visited, seen, returned = [(start, ())], set(), set(), False
            while pending:
                pc, bindings = pending.pop()
                if (pc, bindings) in visited:
                    continue
                visited.add((pc, bindings))
                if len(visited) > 100000:
                    unknown.append({"function": function, "start": start, "reason": "callee CFG state limit"})
                    break
                if pc not in code:
                    unknown.append({"function": function, "start": start, "pc": pc,
                                    "reason": "control flow outside disassembly"})
                    continue
                seen.add(pc)
                op = code[pc]
                opcode = _opcode(op)
                conditional = op.startswith("@") or bool(re.search(r"\bBRA(?:\.[A-Z0-9]+)*\s+!?U?P(?:\d+|T),", op))
                if opcode == "CALL":
                    if _target(op, function, functions) is None:
                        unknown.append({"function": function, "pc": pc, "reason": "unresolved nested CALL"})
                if opcode == "BSSY":
                    setup = BARRIER_SET.search(op)
                    if setup is None or int(setup.group(2), 16) not in code:
                        unknown.append({"function": function, "pc": pc, "op": op,
                                        "reason": "unresolved convergence setup"})
                    else:
                        updated = dict(bindings)
                        updated[setup.group(1)] = int(setup.group(2), 16)
                        if conditional and pc in following:
                            pending.append((following[pc], bindings))
                        bindings = tuple(sorted(updated.items()))
                elif opcode == "BSYNC":
                    barrier = BARRIER_SYNC.search(op)
                    if barrier is None or dict(bindings).get(barrier.group(1)) != following.get(pc):
                        unknown.append({"function": function, "pc": pc, "op": op,
                                        "reason": "unresolved convergence barrier"})
                if opcode in {"BRX", "BRXU", "JMX", "JMXU", "JMP", "BMOV", "BREAK", "BRK", "CONT", "KILL", "RTT"}:
                    unknown.append({"function": function, "pc": pc, "op": op,
                                    "reason": "indirect/unimplemented callee control flow"})
                if opcode in {"RET", "EXIT"}:
                    returned = True
                    if not conditional:
                        continue
                if opcode == "BRA":
                    target = _target(op, function, functions)
                    if target is None or target[0] != function:
                        unknown.append({"function": function, "pc": pc, "reason": "unresolved BRA"})
                    else:
                        pending.append((target[1], bindings))
                    if not conditional:
                        continue
                if pc in following:
                    pending.append((following[pc], bindings))
                else:
                    unknown.append({"function": function, "start": start,
                                    "reason": "callee falls past function end"})
            if not returned:
                unknown.append({"function": function, "start": start, "reason": "no reachable return"})
            rows.append({"function": function, "target_pc": start, "instructions": len(seen),
                         "nested_calls": sum(_opcode(code[pc]) == "CALL" for pc in seen)})
        failures = [row for row in rows if row["instructions"] > 1000 and row["nested_calls"]]
        results[name] = {"sass": len(instructions), "calls": len(calls), "callees": len(rows),
                         "stl": sum(_opcode(op) == "STL" for op in instructions.values()),
                         "callee_rows": rows, "nonleaf_over1k": failures,
                         "callgraph_unknown": unknown,
                         "callgraph_status": "FAIL" if failures else ("UNKNOWN" if unknown else "PASS")}
    return results


def scan_text(resources: str, sass: str) -> list[dict]:
    rows, functions = parse_resources(resources), parse_sass(sass)
    if missing := rows.keys() - functions.keys():
        raise ValueError(f"resource functions absent from SASS: {sorted(missing)}")
    # A separately disassembled device function still needs a measured STACK.
    if missing := functions.keys() - rows.keys():
        raise ValueError(f"SASS functions absent from resources: {sorted(missing)}")
    calls = analyze_calls(functions)
    return [dict(row, **calls[name]) for name, row in rows.items()]


def scan(path: Path, cuobjdump: str | None = None) -> list[dict]:
    tool = cuobjdump or os.environ.get("CUOBJDUMP") or shutil.which("cuobjdump")
    if not tool:
        raise ValueError("cuobjdump is required (set CUOBJDUMP or PATH)")
    outputs = []
    for option in ("-res-usage", "-sass"):
        result = subprocess.run([tool, option, str(path)], capture_output=True, text=True,
                                check=True, timeout=180)
        outputs.append(result.stdout)
    return scan_text(*outputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cubin", type=Path)
    parser.add_argument("timing_db", nargs="?")
    parser.add_argument("label", nargs="?", default="scan")
    args = parser.parse_args(argv)
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        parser.error("set JAX_PLATFORMS=cpu")
    rows = scan(args.cubin)
    Path(f"cubin_scan_{args.label}.json").write_text(json.dumps(rows, indent=1, allow_nan=False) + "\n")
    # Scanner records failures; wave_cubin_gate applies the gate and exit status.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
