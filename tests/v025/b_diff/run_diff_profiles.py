"""One GPU process, twelve separately captured warm diffusion cases."""
import argparse
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import probe_sixth
    import probe_horizontal
    for name,probe in (('sixth',probe_sixth),('horizontal',probe_horizontal)):
        for domain in ('d01','d02'):
            for arm in ('legacy','jax32','native'):
                sys.argv=['probe','--mode','profile','--domain',domain,'--arm',arm,
                          '--out',str(args.out/(name+'_'+domain+'_'+arm+'_bench.json'))]
                probe.main()


if __name__=='__main__':main()
