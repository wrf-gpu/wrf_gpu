"""Host-only nsys/HLO byte estimate while ncu counters are blocked (B18).

Byte footprints are inferred, not measured DRAM traffic. Gathers, strides,
aliases and cache reuse can change actual bytes transferred.
"""
import argparse
import json
import math
from pathlib import Path
import re

TYPE_BYTES = dict(f64=8, f32=4, f16=2, bf16=2, s64=8, u64=8,
                  s32=4, u32=4, s16=2, u16=2, s8=1, u8=1, pred=1)


def shape_bytes(text):
    total, shapes = 0, []
    for dtype, dims in re.findall(
            r"\b(f64|f32|f16|bf16|s64|u64|s32|u32|s16|u16|s8|u8|pred)\[([0-9,]*)\]", text):
        size = math.prod(int(d) for d in dims.split(",") if d)
        count = TYPE_BYTES[dtype] * size
        total += count
        shapes.append(dict(dtype=dtype, dims=dims, bytes=count))
    return total, shapes


def estimate(directory):
    metrics = json.loads((directory / "device_metrics.json").read_text())
    records = []
    for domain in ("d01", "d02"):
        for arm in ("baseline", "candidate"):
            hlo = (directory / f"day_{domain}_pair_{arm}.hlo").read_text()
            parts = hlo.split("\n\n")
            bodies = {}
            for part in parts:
                first = part.strip().splitlines()[0] if part.strip() else ""
                match = re.match(r"%([^ ]+) .*\{$", first)
                if match:
                    bodies[match.group(1)] = part
            main = next(part for part in parts if part.lstrip().startswith("ENTRY "))
            rows = metrics[f"day_{domain}_{arm}"]
            cells = 8400 * 44 if domain == "d01" else 31239 * 44
            columns = cells // 44
            selected = list({row["name"]: row for row in rows["kernels"][:3] +
                             [row for row in rows["kernels"]
                              if row["name"].startswith("thompson_")]}.values())
            for kernel in selected:
                name = kernel["name"]
                record = dict(domain=domain, arm=arm, kernel=name,
                              measured_ms_per_launch=kernel["device_ms"] / kernel["count"],
                              launches_per_call=kernel["count"] / rows["calls"])
                if name == "thompson_fill_down":
                    record.update(estimated_bytes_per_launch=9 * cells,
                                  model="read fp32 velocity+bool activity; write fp32 velocity; masking/reuse can reduce reads")
                elif name == "thompson_sediment_column":
                    record.update(estimated_bytes_per_launch=32 * cells + 8 * columns + 8,
                                  model="six fp32 profiles read; two written; column nstep+precip; ignores cache/alias reuse")
                elif name.startswith("thompson_full_column"):
                    record.update(estimated_bytes_per_launch=240 * cells + 40 * columns,
                                  estimated_input_bytes=120 * cells,
                                  estimated_output_bytes=120 * cells + 40 * columns,
                                  model="declared 15 fp64 storage profiles read+written, 5 fp64 column sinks; table gathers/spills excluded; cache/unused reads can reduce traffic; internal REAL arithmetic remains fp32")
                else:
                    line = next((line for line in main.splitlines()
                                 if re.search(r"%" + re.escape(name) + r"\s*=", line)), None)
                    if line is None:
                        raise ValueError(f"no exact HLO fusion match: {name}")
                    called = re.search(r"calls=%([^, ]+)", line).group(1)
                    body = bodies[called]
                    params = [line for line in body.splitlines() if " parameter(" in line]
                    reads = sum(shape_bytes(line.split(" = ", 1)[1].split(" parameter(", 1)[0])[0]
                                for line in params)
                    header = body.strip().splitlines()[0]
                    output = header.split(" -> ", 1)[1].rsplit(" {", 1)[0]
                    writes, outputs = shape_bytes(output)
                    record.update(estimated_bytes_per_launch=reads + writes,
                                  estimated_input_bytes=reads, estimated_output_bytes=writes,
                                  parameter_count=len(params), output_shapes=outputs,
                                  model="full parameter read+output write; not physical traffic for gathers/cache/strides/aliases")
                record["estimated_effective_GB_s"] = (record["estimated_bytes_per_launch"] /
                                                       record["measured_ms_per_launch"] / 1.e6)
                record["estimated_memory_time_ms_at_peak"] = record["estimated_bytes_per_launch"] / 1.792e9
                record["peak_to_estimated_effective_ratio"] = 1792 / record["estimated_effective_GB_s"]
                records.append(record)
    return dict(scope="B18 manager-authorized estimate; no DRAM/FLOP/occupancy counters; no final-maximum-speed proof",
                timing_label="[M] nsys", byte_bandwidth_labels="[I]", peak_memory_bandwidth_GB_s=1792,
                peak_source="https://www.nvidia.com/en-us/geforce/graphics-cards/compare/",
                limitations=["parameter footprints differ from actual accesses",
                             "above-DRAM effective bandwidth indicates cache/reuse/overestimated traffic",
                             "FP64 pow/exp/software arithmetic is absent from a memory-only bound"], records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = estimate(args.directory)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["records"]:
        print(row["domain"], row["arm"], row["kernel"], row["estimated_effective_GB_s"])
