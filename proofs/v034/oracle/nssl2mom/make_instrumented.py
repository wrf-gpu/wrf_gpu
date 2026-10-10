#!/usr/bin/env python3
"""Generate the PRINT-ONLY instrumented copy of WRF module_mp_nssl_2mom.F.

Inserts (never edits existing statements):
  * a public subroutine o1_dump_module_vars (prints every module-level variable after init),
  * o1_dump_* calls after each nssl_2mom_driver stage (pack/denscale, calcnfromq,
    sediment1d, nssl_2mom_gs, NUCOND, smallvalues, radardd02).
Anchors are verified by exact line content; the build script proves the final outputs
are bit-identical to the un-instrumented pristine build (FINDINGS E26).
"""
import subprocess, sys

SRC = sys.argv[1]
OUT = sys.argv[2]
VARS = sys.argv[3]
GSVARS = sys.argv[4]

lines = open(SRC).read().split('\n')

def at(lineno, expect):
    got = lines[lineno - 1]
    if got.strip() != expect.strip():
        raise SystemExit(f'anchor mismatch at {lineno}: {got!r} != {expect!r}')
    return lineno

DUMPA = ("      CALL o1_dump_an('{tag}', nx, nz, na, an)\n")
def dump_t(tag, names):
    return ''.join(f"      CALL o1_dump_col('{tag}', '{n}', nx, nz, {n})\n" for n in names)

inserts = {}  # after line -> text
inserts[at(192, '  public nssl_2mom_init')] = '  public o1_dump_module_vars\n'
inserts[at(3104, '    ENDIF')] = DUMPA.format(tag='S0') + dump_t('S0', ['t0', 't7', 't00', 't77', 'pn', 'wn', 'dn1', 'dz2d'])
inserts[at(3117, '       ENDIF')] = DUMPA.format(tag='S1')
inserts[at(3146, "     &   ,timesed1,timesed2,timesed3,zmaxsed,timesetvt)")] = (
    DUMPA.format(tag='S2') + "      CALL o1_dump_xfall('S2', nx, na, xfall)\n")
inserts[at(3236, '     & )')] = DUMPA.format(tag='S3') + dump_t('S3', ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9'])
inserts[at(3255, '     &  ,ssat,t00,t77,flag_qndrop)')] = DUMPA.format(tag='S4') + dump_t('S4', ['t0', 't9', 'ssat'])
inserts[at(3263, '     &  ,t77,flag_qndrop)')] = DUMPA.format(tag='S5')
inserts[at(3329, '     &    dbz2d,dn1,nz,cnoh,rho_qh,ipconc,kediagloc, 0)')] = dump_t('S6', ['dbz2d'])

# module-var dump subroutine, inserted right after CONTAINS (line 1221)
vars_ = [l.split() for l in open(VARS).read().split('\n') if l.strip()]
SKIP = {'dab0lu', 'dab1lu'}  # 3-moment-only collection tables (ipconc>=6), 85M elements each
body = ['', '      SUBROUTINE o1_dump_module_vars()', '      IMPLICIT NONE',
        '      DOUBLE PRECISION, ALLOCATABLE :: o1flat(:)', '      INTEGER :: o1i']
for name, typ, kind, par in vars_:
    if name.lower() in SKIP:
        continue
    if typ in ('real', 'doubleprecision'):
        fmt = "'(A,*(ES26.17E3,1X))'"
    elif typ == 'integer':
        fmt = "'(A,*(I0,1X))'"
    else:
        fmt = "'(A,*(L1,1X))'"
    if kind == 'array' and typ in ('real', 'doubleprecision'):
        body.append(f"      IF (SIZE({name}) <= 4096) THEN")
        body.append(f"        WRITE(*,{fmt}) 'MV:{name}=', {name}")
        body.append(f"      ELSE")
        body.append(f"        o1flat = PACK(DBLE({name}), .TRUE.)")
        body.append(f"        WRITE(*,'(A,I0,1X,*(ES26.17E3,1X))') 'MVS:{name}=', SIZE(o1flat), SUM(o1flat), (o1flat(o1i), o1i=1,SIZE(o1flat),997)")
        body.append(f"      ENDIF")
    else:
        body.append(f"      WRITE(*,{fmt}) 'MV:{name}=', {name}")
body += ['      END SUBROUTINE o1_dump_module_vars', '',
         '      SUBROUTINE o1_dump_an(tag, nx, nz, na, an)',
         '      IMPLICIT NONE',
         '      CHARACTER(LEN=*) :: tag', '      INTEGER :: nx, nz, na, il, kz',
         '      REAL :: an(nx,1,nz,na)',
         '      DO il = 1, na', '        DO kz = 1, nz',
         "          WRITE(*,'(A,A,I0,A,I0,A,ES26.17E3)') tag, ':AN:', il, ':', kz, '=', an(1,1,kz,il)",
         '        ENDDO', '      ENDDO', '      END SUBROUTINE o1_dump_an', '',
         '      SUBROUTINE o1_dump_col(tag, name, nx, nz, arr)',
         '      IMPLICIT NONE',
         '      CHARACTER(LEN=*) :: tag, name', '      INTEGER :: nx, nz, kz',
         '      REAL :: arr(nx,1,nz)',
         '      DO kz = 1, nz',
         "        WRITE(*,'(A,A,A,A,I0,A,ES26.17E3)') tag, ':', name, ':', kz, '=', arr(1,1,kz)",
         '      ENDDO', '      END SUBROUTINE o1_dump_col', '',
         '      SUBROUTINE o1_dump_xfall(tag, nx, na, xfall)',
         '      IMPLICIT NONE',
         '      CHARACTER(LEN=*) :: tag', '      INTEGER :: nx, na, il',
         '      REAL :: xfall(nx,1,na)',
         '      DO il = 1, na',
         "        WRITE(*,'(A,A,I0,A,ES26.17E3)') tag, ':XFALL:', il, '=', xfall(1,1,il)",
         '      ENDDO', '      END SUBROUTINE o1_dump_xfall', '']
inserts[at(1221, ' CONTAINS')] = '\n'.join(body) + '\n'

# --- gs-internal dumps (G1 after collection efficiencies, G2 after the qlimit/deposition block) ---
import re as _re
hdr = ' '.join(l.split('!')[0] for l in lines[12619:12637])
args = set(x.lower() for x in _re.findall(r'[A-Za-z_][A-Za-z0-9_]*', hdr.split('(', 1)[1]))
gsv = [l.split() for l in open(GSVARS).read().split('\n') if l.strip()]
def gs_block(tag):
    out = [f"      WRITE(*,'(A,I0)') '{tag}:ngscnt=', ngscnt",
           f"      WRITE(*,'(A,*(I0,1X))') '{tag}:kgs=', kgs(1:ngscnt)"]
    for name, typ, kind, par in gsv:
        if par == '1' or name.lower() in args or name.lower() in ('galpha', 'dgalpha'):
            continue
        if typ in ('real', 'doubleprecision'):
            fmt = "'(A,*(ES26.17E3,1X))'"
        elif typ == 'integer':
            fmt = "'(A,*(I0,1X))'"
        else:
            fmt = "'(A,*(L1,1X))'"
        if kind == 'array':
            out.append(f"      IF (SIZE({name}) <= 20000) WRITE(*,{fmt}) '{tag}:{name}=', {name}")
        else:
            out.append(f"      WRITE(*,{fmt}) '{tag}:{name}=', {name}")
    return '\n'.join(out) + '\n'
inserts[at(16326, '      ENDDO  ! mgs loop for collection efficiencies')] = gs_block('G1')
inserts[at(19695, '      ENDIF')] = gs_block('G2')

out = []
for i, l in enumerate(lines, start=1):
    out.append(l)
    if i in inserts:
        out.append(inserts[i].rstrip('\n'))
open(OUT, 'w').write('\n'.join(out))
print(f'wrote {OUT}: {len(inserts)} insertion points, {len(vars_)} module vars')
