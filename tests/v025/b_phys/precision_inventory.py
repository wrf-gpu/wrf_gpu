"""CPU-only, source-attributed fp64 inventory of a frozen operational graph.

This lowers HLO without compiling or executing a forecast. Native Pallas calls
use their CPU interpreter lowering, so counts are pre-optimization structural
evidence, not optimized GPU instruction counts or performance measurements.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def family(source_file, function, op_name):
    name = (source_file + "/" + function + "/" + op_name).lower()
    for token, label in (("mynn", "MYNN"), ("thompson", "Thompson"),
                         ("cumulus_kf", "KF"), ("kf_adapter", "KF"),
                         ("rrtmg", "RRTMG"), ("radiation", "radiation_glue"),
                         ("noahmp", "Noah-MP"), ("surface_layer", "sfclay"),
                         ("sfclay", "sfclay")):
        if token in name:
            return label
    if "/coupling/" in source_file:
        return "shared_coupling"
    if "/dynamics/" in source_file or "/kernels/dyn_" in source_file:
        return "dynamics"
    if "/runtime/" in source_file:
        return "runtime"
    return "other"


def inventory(hlo, source_root):
    """Count f64 leaf outputs and f64 operands, including predicate outputs.

    A carry tuple containing f64 is not itself an f64 arithmetic operation.
    Parameters, constants, data movements and converts are reported separately.
    """
    nodes = []
    symbols = {}
    functions = {}
    # Current XLA interns source locations into these header tables rather
    # than attaching source_file/source_line to every instruction.
    locations = {}
    frames = {}
    filenames = {}
    functionnames = {}
    section = ""
    for line in hlo.splitlines():
        if line.strip() in {"FileNames", "FunctionNames", "FileLocations", "StackFrames"}:
            section = line.strip()
            continue
        entry = re.match(r"^\s*(\d+) (.+)$", line)
        if not entry:
            continue
        index, value = int(entry.group(1)), entry.group(2)
        if section in {"FileNames", "FunctionNames"} and value.startswith('"'):
            (filenames if section == "FileNames" else functionnames)[index] = json.loads(value)
        elif section in {"FileLocations", "StackFrames"} and value.startswith("{"):
            fields = {k:int(v) for k,v in re.findall(r"(\w+)=(\d+)", value)}
            (locations if section == "FileLocations" else frames)[index] = fields
    pattern = re.compile(r"^\s*(?:ROOT\s+)?([\w.%-]+) = (.+?) ([\w-]+)\((.*?)\)(.*)$")
    for line in hlo.splitlines():
        if line.rstrip().endswith("{") or line.strip() == "}":
            symbols = {}
        match = pattern.match(line)
        if not match:
            continue
        symbol, dtype, opcode, operands, tail = match.groups()
        result64 = bool(re.match(r"f64\[", dtype))
        operand64 = any(symbols.get(token, False) for token in re.findall(r"[\w.%-]+", operands))
        symbols[symbol] = result64
        if not (result64 or operand64) or opcode == "tuple":
            continue
        source = re.search(r'source_file="([^"]+)"', tail)
        source_file = source.group(1) if source else ""
        number = re.search(r"source_line=(\d+)", tail)
        lineno = int(number.group(1)) if number else 0
        scope = re.search(r'op_name="([^"]+)"', tail)
        op_name = scope.group(1) if scope else ""
        function = ""
        frame_id = re.search(r"stack_frame_id=(\d+)", tail)
        if frame_id and not source_file:
            location = locations.get(frames.get(int(frame_id.group(1)), {}).get("file_location_id"), {})
            source_file = filenames.get(location.get("file_name_id"), "")
            lineno = location.get("line", 0)
            function = functionnames.get(location.get("function_name_id"), "")
        path = Path(source_file)
        if not path.is_absolute():
            path = source_root / path
        if source_file and path.is_file():
            if source_file not in functions:
                tree = ast.parse(path.read_text())
                functions[source_file] = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree)
                                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            containing = [(b-a, n) for a, b, n in functions[source_file] if a <= lineno <= b]
            if containing:
                function = min(containing)[1]
        if opcode in ("parameter", "constant"):
            category = opcode
        elif opcode == "convert":
            category = "convert"
        elif opcode in {"get-tuple-element", "reshape", "bitcast", "transpose", "broadcast",
                        "slice", "dynamic-slice", "dynamic-update-slice", "concatenate", "copy"}:
            category = "data"
        else:
            category = "compute"
        nodes.append(dict(symbol=symbol, opcode=opcode, category=category,
                          result64=result64, operand64=operand64,
                          source_file=source_file, source_line=lineno, function=function,
                          op_name=op_name, family=family(source_file, function, op_name)))
    by_family = defaultdict(Counter)
    by_function = defaultdict(Counter)
    for node in nodes:
        by_family[node["family"]][node["category"]] += 1
        by_function[node["source_file"] + ":" + node["function"]][node["category"]] += 1
    return dict(f64_nodes=len(nodes), by_family=dict(by_family), by_function=dict(by_function)), nodes


def quiet_check():
    if Path("/tmp/wrf_gpu2_quiet").exists():
        raise SystemExit("quiet window: defer CPU inventory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=["d01", "d02"])
    args = parser.parse_args()
    quiet_check()
    import jax
    import jax.numpy as jnp
    import importlib
    from jax.experimental import pallas as pl
    assert jax.devices()[0].platform == "cpu"
    # Radiation and acoustic entry points explicitly request native lowering.
    # CPU has no native Pallas backend: preserve each kernel body, grid, shape
    # and dtype, selecting only its interpreter for this analysis process.
    native_pallas_call = pl.pallas_call
    def cpu_pallas_call(*positional, **keywords):
        return native_pallas_call(*positional, **dict(keywords, interpret=True))
    pl.pallas_call = cpu_pallas_call
    # Native fp32 dycore/boundary/SINT kernels pin WRF REAL rounding with PTX
    # inline asm (``{add,sub,mul,div}.<rnd>.<f32|f64> $0, $1, $2;``), which has no
    # CPU lowering. For this dtype inventory substitute the same IEEE operation
    # in the same dtype (the kernels' own interpret branch); anything else fails.
    from jax.experimental.pallas import triton as pallas_triton
    asm_form = re.compile(r"(add|sub|mul|div)\.\w+\.(f32|f64) \$0, \$1, \$2;")

    def cpu_inline_asm(asm, *, args, constraints, pack, result_shape_dtypes):
        del constraints, pack
        match = asm_form.fullmatch(asm.strip())
        if match is None or len(args) != 2 or len(result_shape_dtypes) != 1:
            raise NotImplementedError(f"no CPU stand-in for inline asm {asm!r}")
        a, b = args
        value = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[match.group(1)]
        return [value.astype(result_shape_dtypes[0].dtype)]

    pallas_triton.elementwise_inline_asm = cpu_inline_asm
    from gpuwrf.integration import nested_pipeline as NP
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori, build_clock_base

    source_root = Path.cwd()
    args.out.mkdir(parents=True, exist_ok=True)
    # Initial radiative VALUES do not affect a shape-only graph. Trace the exact
    # initializer to get its real pytree/dtypes, avoiding a CPU radiation solve.
    # The measured operational graph and all static settings stay unmodified.
    seed = NP.noahmp_initial_rad

    def abstract_seed(state, namelist, *, land_state, **keywords):
        return jax.eval_shape(lambda s, land: seed(
            s, namelist, land_state=land, **keywords), state, land_state)

    NP.noahmp_initial_rad = abstract_seed
    NP._commit_to_operational_device = lambda value: value
    config = NP.NestedPipelineConfig(args.input_dir, args.out/"unused", args.out/"proof",
                                     hours=3, max_dom=2)
    started = time.monotonic()
    # State.zeros normally insists on GPU placement, including during setup.
    # Change only that allocation target while loading, then restore it before
    # tracing the operational function. Shape and dtype contracts are retained.
    state_module = importlib.import_module("gpuwrf.contracts.state")
    gpu_device = state_module._gpu_device
    state_module._gpu_device = lambda: jax.devices("cpu")[0]
    try:
        _hierarchy, bundles, _meta, _start, _dts, carries = NP._load_domains(config, ("d01", "d02"))
    finally:
        state_module._gpu_device = gpu_device
    print("domain setup", time.monotonic()-started, flush=True)
    manifest = dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    backend="cpu", scope="pre-optimization HLO; all Pallas calls forced to CPU interpreter without changing bodies",
                    initializer="CPU allocation placement; exact shape-only radiation initializer; no forecast execution",
                    flags={k:v for k,v in os.environ.items() if k.startswith("GPUWRF_")},
                    input_dir=str(args.input_dir), domains={})
    for domain in args.domains:
        quiet_check()
        nl = bundles[domain].namelist
        carry = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carries[domain])
        lowered = _advance_chunk_fori.lower(carry, nl, jnp.int32(0), build_clock_base(nl),
                                             n_steps=1, cadence=int(nl.radiation_cadence_steps))
        from jaxlib import _jax
        print_options = _jax.HloPrintOptions()
        print_options.print_metadata = True
        print_options.print_large_constants = False
        hlo = lowered.compiler_ir(dialect="hlo").as_hlo_module().to_string(print_options)
        (args.out/(domain+".hlo.txt")).write_text(hlo)
        counts, nodes = inventory(hlo, source_root)
        (args.out/(domain+".nodes.json")).write_text(json.dumps(nodes, indent=2)+"\n")
        counts.update(shape=list(carries[domain].state.theta.shape),
                      carry_dtypes=dict(Counter(str(v.dtype) for v in jax.tree.leaves(carry))),
                      hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest())
        manifest["domains"][domain] = counts
        (args.out/"inventory.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(domain, counts["f64_nodes"], counts["by_family"], flush=True)


if __name__ == "__main__":
    main()
