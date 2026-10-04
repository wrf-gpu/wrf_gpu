"""Share imports/CUDA initialization across the operator validation matrix.

No GPU subprocesses; the only child in the oracle is the CPU Fortran compiler.
Each case still emits its own device assertion, source/fixture hashes and gate.
"""
import argparse
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--identity',action='store_true')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--arm',choices=('native','legacy'),default='native')
    args=parser.parse_args()
    mode,arm=('identity','legacy') if args.identity else ('profile',args.arm) if args.profile else ('verify','native')
    # The import-time precision flag is frozen once for this whole process.
    sys.argv=['probe_rk','--mode',mode,'--arm',arm]
    import probe_rk as probe
    original=(probe.rkmod,probe.prepmod,probe.finishmod)
    for domain in ('d01','d02'):
        families=('prep',) if args.identity else ('prep','finish','rk','pgf','coriolis','curvature','eos')
        for family in families:
            sys.argv=['probe_rk','--mode',mode,'--arm',arm,'--domain',domain,
                      '--family',family,'--out',str(args.out/(mode+'_'+domain+'_'+family+'.json'))]
            probe.main()
            if args.identity:
                probe.rkmod,probe.prepmod,probe.finishmod=original
                probe.jax.clear_caches()


if __name__=='__main__':main()
