"""Build a private pristine Noah-MP REAL oracle; never write the WRF checkout.

SOILWATER is PRIVATE in WRF, so only its accessibility declaration is exposed
in a local preprocessed module. Every routine body remains verbatim. Runtime
stubs come unchanged from the existing offline-driver build recipe.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--wrf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--compiler', type=Path, default=Path('<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'))
    args = p.parse_args()
    if Path('/tmp/wrf_gpu2_quiet').exists():
        return 125
    args.out.mkdir(parents=True, exist_ok=True)
    phys = args.wrf/'phys'
    original = phys/'module_sf_noahmplsm.f90'
    text = original.read_text()
    local, n = re.subn(r'(?im)^(\s*)private\s*::\s*SOILWATER\s*$',
                       r'\1public :: SOILWATER', text)
    assert n == 1
    (args.out/original.name).write_text(local)
    recipe = (args.repo/'proofs/noahmp/build_driver.sh').read_text()
    for name in ['_wrfstubs','_modstubs']:
        marker = f"cat > {name}.F90 <<'EOF'\n"
        start = recipe.index(marker)+len(marker)
        end = recipe.index('\nEOF',start)
        (args.out/f'{name}.F90').write_text(recipe[start:end]+'\n')
    driver = args.repo/'tests/v025/b_noahmp/water_real_driver.F90'
    (args.out/driver.name).write_text(driver.read_text())
    fixtures = args.repo/'proofs/noahmp/savepoints_all.json'
    cols = json.loads(fixtures.read_text())['columns']
    lines = [str(len(cols))]
    for c in cols:
        s = c['state_in']
        lines += [' '.join(map(str,[c['vegtyp'],c['isltyp'],c['dt'],c['dx'],s['tg'],s['sneqv'],c['wrf']['et']['edir'],s['smcwtd']])),
                  ' '.join(map(str,c['zsoil'])), '0.05 0.20 0.45 0.80',
                  ' '.join(map(str,s['smc'])), ' '.join(map(str,s['sh2o']))]
    (args.out/'water_columns.in').write_text('\n'.join(lines)+'\n')
    for name in ['MPTABLE.TBL','SOILPARM.TBL','GENPARM.TBL']:
        link = args.out/name
        if not link.exists():
            link.symlink_to(args.wrf/'run'/name)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(args.compiler.parent.parent/'lib')
    flags = ['-w','-ffree-form','-ffree-line-length-none','-O2',f'-I{phys}']
    subprocess.run([str(args.compiler),*flags,'-c',original.name,'-o','module_sf_noahmplsm.o'],
                   cwd=args.out,env=env,timeout=120,check=True)
    if Path('/tmp/wrf_gpu2_quiet').exists():
        return 125
    objects = [phys/f'{name}.o' for name in ['module_sf_noahmpdrv','module_sf_gecros',
               'module_sf_noahmp_glacier','module_sf_noahmp_groundwater']]
    subprocess.run([str(args.compiler),*flags,'-I.','_wrfstubs.F90','_modstubs.F90',
                    driver.name,'module_sf_noahmplsm.o',*map(str,objects),'-o','water_real_driver.exe'],
                   cwd=args.out,env=env,timeout=120,check=True)
    subprocess.run(['./water_real_driver.exe'],cwd=args.out,env=env,timeout=60,check=True)
    provenance = dict(compiler=str(args.compiler),flags=flags,real_bits=32,
                      only_source_change='PRIVATE SOILWATER -> PUBLIC SOILWATER',
                      files={str(f):hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in [original,driver,fixtures,*objects]},
                      module_modified_sha256=hashlib.sha256(local.encode()).hexdigest())
    (args.out/'build_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
