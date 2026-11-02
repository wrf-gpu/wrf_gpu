"""Deploy only committed diagnostic helpers; no DATA writes or input copies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from bdyneg_common import ROOT, sha, write

NAMES = ('bdyneg_common.py', 'control.py', 'model.py', 'post.py',
         'bdyneg_supervisor.py', 'bdyneg_registry.py', 'bdyneg_pristine.py', 'bdyneg_cpu_lease_check.py',
         'bdyneg_bundle_consumer.py', 'wave_cubin_gate.py', 'e41/cubin_scan.py', 'e41/hlo_scan.py')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--commit', required=True)
    p.add_argument('--deploy', type=Path, required=True)
    p.add_argument('--cache-seed', type=Path)
    a = p.parse_args()
    a.commit = subprocess.check_output(['git', 'rev-parse', a.commit + '^{commit}'], text=True).strip()
    if a.deploy.resolve() != a.deploy or not a.deploy.is_relative_to(ROOT):
        raise ValueError('deployment must stay under the ROOT lane directory')
    a.deploy.mkdir(exist_ok=False)
    tools = a.deploy / 'tools'
    tools.mkdir()
    def blob(name):
        return subprocess.check_output(['git', 'show', a.commit + ':proofs/v45_bdyneg/launch/' + name])
    for name in NAMES:
        (tools / name).parent.mkdir(parents=True, exist_ok=True)
        (tools / name).write_bytes(blob(name))
    binding = json.loads((ROOT / 'C_INPUT_BINDING.json').read_text())
    b = binding['C']
    inputs = dict(b['input_data_sha256'], **{'namelist.input': b['namelist_sha256']})
    bundle = Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/rc_bundle_3e5219bc8088')
    source = json.loads((bundle / 'meta/PIN_MANIFEST.json').read_text())
    assert source['src_tree'] == subprocess.check_output(
        ['git', 'rev-parse', 'v0.4.4:src'], text=True).strip()
    assert source['src_tree'] == 'a22dea711202a48d00c86eafa34391ff2cb60fe2'
    wrf = Path('<USER_HOME>/src/wrf_pristine/WRF')
    immutable = [bundle / 'bundle.env', bundle / 'sealed/BUNDLE_MANIFEST.json',
                 bundle / 'meta/PIN_MANIFEST.json']
    immutable += [wrf / n for n in ('share/module_bc.F', 'dyn_em/module_bc_em.F',
                                    'dyn_em/module_em.F', 'dyn_em/solve_em.F')]
    fixture = ROOT / 'event_ring_operands_20261028/wrfout_d02_2026-01-11_03:00:00.npz'
    cfg = dict(schema='v45-bdyneg.launch-seal.v1', tools_commit=a.commit,
               model_commit='fcf12bc229', model_src_tree=source['src_tree'],
               bundle=str(bundle), release_bundle='<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.4',
               bundle_consumer=str(tools / 'bdyneg_bundle_consumer.py'),
               python='<USER_HOME>/miniconda3/bin/python',
               gpu_lock='<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh',
               real_utc='<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh',
               input_dir=b['input_dir'], input_sha256=inputs,
               hours=3, checkpoint_interval_steps=400,
               max_bytes=5 * 1024**3, model_seconds=840, total_seconds=1170,
               C_d02_h3_fixture=str(fixture), C_d02_h3_sha256=sha(fixture),
               input_binding_source_sha256=binding['source_sha256'],
               tools={n: sha(tools / n) for n in NAMES},
               immutable_files={str(n): sha(n) for n in immutable},
               cache_seed=[] if a.cache_seed is None else json.loads(a.cache_seed.read_text()),
               precision={'JAX_ENABLE_X64': 'true', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'},
               physics='C namelist exact SHA5963ad19; only duration/output/checkpoint operational controls differ',
               scope='acquire actual nested boundary operands; transient root PD not guaranteed')
    upstream_consumer = Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.4/tools/bundle_consumer.py')
    cfg['consumer_upstream'] = dict(path=str(upstream_consumer), sha256=sha(upstream_consumer),
        host_delta='read-only payload mounts/cwd in the existing consumer namespace; model package unchanged')
    cfg['CPU_RAM_grant'] = 'manager2026-10-28T18:00:51Z: one exact fused CPU smoke,32G/noSwap; GPU remains16G'
    cfg['CPU_final_grant'] = 'manager2026-10-30T23:51:07Z V046 exact SEQ_SMOKE slot32;model<=60min,HOME scratch'
    cfg.update(CPU_model_seconds=3600, CPU_total_seconds=3900,
               CPU_cache=str(ROOT / 'smoke_cache_C_v044'))
    for n in (cfg['bundle_consumer'], cfg['gpu_lock'], cfg['real_utc'], str(fixture)):
        cfg['immutable_files'][n] = sha(n)
    write(a.deploy / 'seal.json', cfg)
    seal_sha = sha(a.deploy / 'seal.json')
    entry = blob('entry.sh.in').decode().replace('@DEPLOY@', str(a.deploy)).replace('@SEAL_SHA@', seal_sha)
    (a.deploy / 'entry.sh').write_text(entry)
    write(a.deploy / 'plan.json', dict(schema='v45-bdyneg.night-plan.v1',
          model_commit=cfg['model_commit'], helpers_commit=a.commit, seal_sha256=seal_sha,
          steps=[dict(lane='v45-bdyneg', path=str(a.deploy / 'entry.sh'),
                      sha256=sha(a.deploy / 'entry.sh'), cap_min=20)],
          order='after ALISIOS Nightly DONE and fresh manager GO; next window proposed, not armed',
          hard_end_utc='2026-10-31T18:45:00Z', data_floor_GiB=430, data_start_GiB=435,
          data_writes_before_GA_START=0, peak_data_GiB=5, input_copies=0,
          manager_receipt=dict(path=str(a.deploy / 'NIGHTLY_DONE.json'),
              schema='v45-bdyneg.nightly-done.v1', Nightly_DONE=True, by='manager',
              completed_utc='actual Nightly DONE UTC <= GA_START_UTC'),
          manager_environment=dict(GA_GATE_ID='v45-bdyneg-night', CPU_LEASE_DIR='manager lease directory')))
    cpu_plan = json.loads((a.deploy / 'plan.json').read_text())
    cpu_plan['schema'] = 'v45-bdyneg.CPU-only-smoke-plan.v1'
    cpu_plan['steps'][0]['cap_min'] = 65
    cpu_plan['CPU_only_authority'] = cfg['CPU_final_grant']
    cpu_plan['native_window_plan_cap_min'] = 20
    write(a.deploy / 'CPU_SMOKE_PLAN.json', cpu_plan)
    write(a.deploy / 'PREPARED.json', dict(seal_sha256=seal_sha, plan_sha256=sha(a.deploy / 'plan.json'),
                                          entry_sha256=sha(a.deploy / 'entry.sh'), model_source='unchanged v0.4.4'))
    print(json.dumps(dict(deploy=str(a.deploy), seal_sha256=seal_sha, plan_sha256=sha(a.deploy / 'plan.json'))))


if __name__ == '__main__':
    raise SystemExit(main())
