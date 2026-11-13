"""Command registration stays independent of NumPy, NetCDF and JAX."""
from pathlib import Path
import json
import sys


def register(subparsers):
    parser = subparsers.add_parser("perturb", help="prepare offline IC/BC ensemble member inputs "
                                   "(uniform wind factors, or seeded boundary-layer theta/QV noise)")
    parser.add_argument("--case", type=Path, required=True, help="WRF input case directory")
    parser.add_argument("--output", type=Path, required=True, help="new member case directory")
    parser.add_argument("--members", type=int, required=True)
    recipe = parser.add_mutually_exclusive_group(required=True)
    recipe.add_argument("--wind-std", type=float,
                        help="sample standard deviation of a spatially uniform fractional U/V multiplier")
    recipe.add_argument("--thermo-config", type=Path,
                        help="JSON recipe (schema gpuwrf.perturb.thermo.v1: seed, theta_std_k, qv_rel_std, "
                             "full_m, top_m[, clip_sigma, domains]) for seeded PBL theta/QV noise in wrfinput")
    parser.add_argument("--seed", type=int, help="wind recipe seed (the thermo recipe takes its seed from the file)")
    parser.add_argument("--ic-only", action="store_true", help="explicitly leave lateral boundaries unchanged")
    parser.set_defaults(func=command)


def command(args):
    try:
        if args.thermo_config is not None:
            if args.seed is not None or args.ic_only:
                raise ValueError("--thermo-config takes its seed from the file and never edits boundaries; "
                                 "drop --seed/--ic-only")
            from .thermo import prepare as prepare_thermo
            result = prepare_thermo(args.case, args.output, members=args.members, config=args.thermo_config)
        else:
            if args.seed is None:
                raise ValueError("--wind-std requires --seed")
            from .inputs import prepare
            result = prepare(args.case, args.output, members=args.members, seed=args.seed,
                             wind_std=args.wind_std, ic_only=args.ic_only)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"gpuwrf perturb: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0
