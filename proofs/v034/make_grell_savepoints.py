#!/usr/bin/env python3
"""Stage the v0.3.4 Grell-family pristine-WRF tile savepoints.

Usage (CPU only; build the oracles first with oracle/cumulus_grell/build.sh):
    python3 proofs/v034/make_grell_savepoints.py BUILD_DIR

For every case the deterministic tile generator (grell_oracle_io.make_tile) is
run, the UNMODIFIED pristine-WRF G3DRV+conv_grell_spread3d / GRELLDRV oracle
executable is called, and the oracle outputs plus the case metadata and the
sha256 of the binary input are written to savepoints/cumulus_grell/<case>.npz.
Inputs are not stored: tests regenerate them from the seed and check the hash.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grell_oracle_io as gio  # noqa: E402

OUT = HERE / "savepoints" / "cumulus_grell"

CASES = {
    # name: (scheme, precision, seed, kx, opts)
    "g3_deep_r8": ("g3", "r8", 1, 44, dict(dt=54.0, dx=9000.0)),
    "g3_shallow_r8": ("g3", "r8", 2, 44, dict(dt=54.0, dx=9000.0, ishallow=1)),
    "g3_highres_r8": ("g3", "r8", 5, 44, dict(dt=18.0, dx=3000.0, ishallow=1, cugd_avedx=3)),
    "g3_ichoice1_kx33_r8": ("g3", "r8", 7, 33, dict(dt=60.0, dx=12000.0, ishallow=1, ichoice=1)),
    "g3_deep_r4": ("g3", "r4", 1, 44, dict(dt=54.0, dx=9000.0)),
    "gd_r8_s1": ("gd", "r8", 1, 44, dict(dt=54.0, dx=9000.0)),
    "gd_r8_s2": ("gd", "r8", 2, 44, dict(dt=54.0, dx=9000.0)),
    "gd_r4_s2": ("gd", "r4", 2, 44, dict(dt=54.0, dx=9000.0)),
}
NX = NY = 12


def input_sha256(tile, opts) -> str:
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        gio.write_input(fh.name, tile, **opts)
        return hashlib.sha256(Path(fh.name).read_bytes()).hexdigest()


def main() -> int:
    build = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    sources = (build / "sources.sha256").read_text()
    for name, (scheme, prec, seed, kx, opts) in CASES.items():
        tile = gio.make_tile(seed, nx=NX, ny=NY, kx=kx)
        out = gio.run_oracle(build / f"{scheme}_{prec}.exe", tile, workdir=str(build), **opts)
        meta = dict(case=name, scheme=scheme, precision=prec, seed=seed, nx=NX, ny=NY, kx=kx,
                    opts=opts, input_sha256=input_sha256(tile, opts),
                    oracle_sources_sha256=sources,
                    generator="proofs/v034/grell_oracle_io.make_tile")
        np.savez_compressed(OUT / f"{name}.npz", meta=json.dumps(meta),
                            **{k: v for k, v in out.items()})
        print(name, "rain cols", int((out["RAINCV"] > 0).sum()), "->", OUT / f"{name}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
