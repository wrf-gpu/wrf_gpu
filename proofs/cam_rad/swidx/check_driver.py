import numpy as np
c = np.load('<USER_HOME>/wrf_gpu2_lanes/o1-camrad/CAM01/cam01_fixture.npz')
m0 = np.load('<USER_HOME>/wrf_gpu2_lanes/o1-camrad/SWIDX/swidx_fixture_m0.npz')
m1 = np.load('<USER_HOME>/wrf_gpu2_lanes/o1-camrad/SWIDX/swidx_fixture_m1.npz')
n = int(m0['ncam01']); s = np.float64(np.float32(1e-3))
chk = {'rh': (m0['o_rh'][:n], c['swi_rh']), 'aerosol': (m0['o_aerosol'][:n], c['swi_aerosol'])}
for k in ['qrs', 'qrscs', 'fsdndir', 'fsdndif', 'sols', 'soll', 'solsd', 'solld', 'tauxcl', 'tauxci']:
    chk[k] = (m0['o_' + k][:n], c['r8s_' + k])
for k in ['fsup', 'fsupc', 'fsdn', 'fsdnc', 'fsns', 'fsds', 'fsdsdir', 'fsdsdif']:
    chk[k] = (m0['o_' + k][:n] * s, c['r8s_' + k])
chk['swcftoa'] = (m0['o_fsntoa'][:n] * s - m0['o_fsntoac'][:n] * s, c['r8s_swcftoa'])
print('mode0 vs CAM01 bitwise:', {k: bool(np.array_equal(a, b)) for k, (a, b) in chk.items()})
cols = m1['columns']
fb = (m1['o_nmx_after'] == 1) & (m1['in_nmxrgn'] > 1)
print('second-pass columns', int(fb.sum()), 'labels', sorted(set(str(x).split('_src')[0] for x in cols[fb])))
day = m1['in_coszrs'] > 0
print('day columns', int(day.sum()), 'of', len(cols))
a1 = m1['o_aerosol']; print('mode1 nonzero aerosol slots', [m for m in range(13) if np.any(a1[:, :, m] != 0)])
print('mode1 vs mode0 fsns differ (day cols):', int(np.sum(m1['o_fsns'][day] != m0['o_fsns'][day])), 'max rel',
      float(np.max(np.abs(m1['o_fsns'][day] - m0['o_fsns'][day]) / m0['o_fsns'][day])))
print('rh range', float(m1['o_rh'].min()), float(m1['o_rh'].max()))
