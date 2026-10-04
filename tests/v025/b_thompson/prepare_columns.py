"""Prepare real PROD day/night Thompson workloads through the retained coupler."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/v025"))
from real_state import load_real_snapshot
import jax
from gpuwrf.contracts import state as state_contract
from gpuwrf.coupling import physics_couplers as coupler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError("host preparation requires JAX_PLATFORMS=cpu")
    state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    args.output.mkdir(parents=True, exist_ok=True)
    for label, time in [("night", "2026-07-26_00:00:00"),
                        ("day", "2026-07-26_12:00:00")]:
        for domain in ("d01", "d02"):
            snapshot = load_real_snapshot(args.case, domain=domain,
                                         wrfout_name=f"wrfout_{domain}_{time}")
            column = coupler._thompson_column_from_state(
                snapshot.state, SimpleNamespace(metrics=snapshot.metrics))
            arrays = {key: np.asarray(getattr(column, key)) for key in column.__slots__}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise RuntimeError("nonfinite input")
            name = f"{label}_{domain}"
            np.savez(args.output / f"{name}.npz", **arrays)
            metadata = dict(provenance=snapshot.provenance, shape=list(column.qv.shape),
                            dt=54. if domain == "d01" else 18.,
                            coupler_source=str(Path(coupler.__file__).resolve()),
                            coupler_sha256=hashlib.sha256(Path(coupler.__file__).read_bytes()).hexdigest(),
                            active_columns={key: int(np.any(arrays[key] > 1.e-12, axis=-1).sum())
                                            for key in ("qc", "qr", "qi", "qs", "qg")},
                            oracle_scope="real workload inputs, no independent expected output")
            (args.output / f"{name}.json").write_text(json.dumps(metadata, indent=2)+"\n")
            print(name, column.qv.shape, flush=True)


if __name__ == "__main__":
    main()
