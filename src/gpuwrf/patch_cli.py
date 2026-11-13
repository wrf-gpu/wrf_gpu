"""Backend-dark registration of ``gpuwrf patch`` (static land, z0 control, geo_em terrain, post-real check)."""
from pathlib import Path


def register(subparsers):
    parser = subparsers.add_parser("patch", help="checked static-field patches: land use/soil, z0 control, geo_em terrain")
    actions = parser.add_subparsers(dest="patch_action", required=True)
    land = actions.add_parser("land", help="land-use class / soil patches on new wrfinput+wrflowinp copies")
    land.add_argument("--input-dir", required=True, type=Path)
    land.add_argument("--output-dir", required=True, type=Path, help="new directory; must not exist")
    land.add_argument("--spec", required=True, type=Path, help="JSON/YAML patches and explicit waivers")
    land.add_argument("--namelist", type=Path, default=None, help="namelist.input (max_dom, spec_bdy_width); required for non-empty specs")
    z0 = actions.add_parser("z0", help="masked z0mvt multipliers -> surface-control file (traced, no recompile)")
    z0.add_argument("--input-dir", required=True, type=Path, help="case whose wrfinput defines the masks (the basis case)")
    z0.add_argument("--basis", required=True, type=Path, help="gpuwrf surface-params basis output of the same case")
    z0.add_argument("--spec", required=True, type=Path)
    z0.add_argument("--output", required=True, type=Path, help="new .npz control file for gpuwrf run --surface-controls")
    geo = actions.add_parser("geo-em", help="HGT_M patch on new geo_em copies for metgrid/real.exe")
    geo.add_argument("--geo-dir", required=True, type=Path)
    geo.add_argument("--output-dir", required=True, type=Path, help="new directory; must not exist")
    geo.add_argument("--spec", required=True, type=Path)
    geo.add_argument("--namelist", type=Path, default=None, help="namelist.input (max_dom, nest geometry, gwd_opt, spec_bdy_width, blend_width); required for non-empty specs")
    chk = actions.add_parser("check-real", help="check real.exe output against the patched geo_em (and a reference run)")
    chk.add_argument("--geo-dir", required=True, type=Path)
    chk.add_argument("--real-dir", required=True, type=Path, help="directory with the new wrfinput_dNN")
    chk.add_argument("--reference-dir", type=Path, default=None, help="unpatched real.exe output of the same case")
    chk.add_argument("--namelist", type=Path, required=True, help="namelist.input of the real.exe run (max_dom, spec_bdy_width, blend_width)")
    chk.add_argument("--output", type=Path, default=None, help="write the JSON report here (new file)")
    ver = actions.add_parser("verify", help="re-audit an edited case against its source files")
    ver.add_argument("--input-dir", required=True, type=Path, help="unedited source case")
    ver.add_argument("--patched-dir", required=True, type=Path)
    ver.add_argument("--namelist", type=Path, default=None)
    ver.add_argument("--waiver", action="append", default=[], metavar="CHECK=REASON",
                     help="waive a waivable check (e.g. dependent_keep=...); recorded as WAIVED")
    parser.set_defaults(func=command)


def command(args):
    import json
    import sys
    from gpuwrf.io.static_patch import StaticPatchError, load_spec
    try:
        if args.patch_action == "land":
            from gpuwrf.io.static_patch import patch_land
            spec, base = load_spec(args.spec)
            report = patch_land(args.input_dir, args.output_dir, spec, spec_base=base, namelist=args.namelist)
        elif args.patch_action == "z0":
            from gpuwrf.io.static_patch import patch_z0
            spec, base = load_spec(args.spec)
            report = patch_z0(args.input_dir, args.basis, spec, args.output, spec_base=base)
        elif args.patch_action == "geo-em":
            from gpuwrf.io.geo_em_patch import patch_geo_em
            spec, base = load_spec(args.spec)
            report = patch_geo_em(args.geo_dir, args.output_dir, spec, spec_base=base, namelist=args.namelist)
        elif args.patch_action == "verify":
            from gpuwrf.io.static_patch import verify_patched
            waivers = dict(w.split("=", 1) for w in args.waiver if "=" in w and w.split("=", 1)[1].strip())
            if len(waivers) != len(args.waiver):
                raise ValueError("--waiver needs CHECK=REASON with a nonempty reason")
            report = verify_patched(args.input_dir, args.patched_dir, namelist=args.namelist, waivers=waivers)
            print(json.dumps(report, sort_keys=True, allow_nan=False, default=str))
            return 0 if report["passed"] else 2
        else:
            from gpuwrf.io.geo_em_patch import check_real
            report = check_real(args.geo_dir, args.real_dir, reference_dir=args.reference_dir, namelist=args.namelist)
            if args.output is not None:
                if args.output.exists():
                    raise ValueError(f"{args.output} exists")
                args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps(report, sort_keys=True, allow_nan=False))
            return 0 if report["passed"] else 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        if isinstance(exc, StaticPatchError):
            failed = [c for c in exc.report["checks"] if c["status"] == "FAIL"]
            print(json.dumps({"error": str(exc), "failed_checks": failed}, sort_keys=True, default=str), file=sys.stderr)
        else:
            print(f"gpuwrf patch: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False, default=str))
    return 0
