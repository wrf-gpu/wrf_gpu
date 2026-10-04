"""Compilation overlap, worker context and deterministic serial fallback."""
from types import SimpleNamespace
import threading

import jax
import pytest

from gpuwrf.runtime import aot_executable as ax, aot_precompile as ap


@pytest.mark.parametrize('x64', [False, True])
def test_lowering_stays_on_caller_and_overlaps_compile(monkeypatch, x64):
    caller = threading.get_ident()
    first_started = threading.Event()
    second_lowered = threading.Event()
    observations = []
    monkeypatch.setattr(ax, "export_compile_options", lambda: {"export": True})

    class Lowered:
        def __init__(self, name): self.name = name
        def compile(self, options):
            assert threading.get_ident() != caller
            assert jax.config.jax_enable_x64 is x64
            assert options == {"export": True}
            if self.name == "d01":
                first_started.set()
                assert second_lowered.wait(5)
            return self.name

    def lower(name):
        assert threading.get_ident() == caller
        if name == "d02":
            assert first_started.wait(5)
            second_lowered.set()
        observations.append(name)
        return Lowered(name)

    jobs = [{"name":n, "lower":lambda n=n:lower(n),
             "finish":lambda compiled, lowered:compiled} for n in ("d01","d02")]
    saved=bool(jax.config.jax_enable_x64)
    try:
        # Opposite global default makes losing the caller's thread-local context
        # observable, even when the enclosing pytest suite enables x64 globally.
        jax.config.update('jax_enable_x64',not x64)
        with jax.enable_x64(x64):
            results, errors = ap._compile_lowered_threads(jobs)
    finally:
        jax.config.update('jax_enable_x64',saved)
    assert results == ["d01","d02"] and not errors
    assert observations == ["d01","d02"]


@pytest.mark.parametrize("phase", ["lower", "compile", "capture"])
def test_failure_joins_workers_and_returns_serial_fallback(phase, monkeypatch):
    finished = threading.Event()
    monkeypatch.setattr(ax, "export_compile_options", lambda: None)
    def lower(name):
        if name == "d02" and phase == "lower": raise ValueError("lower failed")
        def compile(options):
            if name == "d02" and phase == "compile": raise ValueError("compile failed")
            return name
        return SimpleNamespace(compile=compile)
    def finish(name, compiled, lowered):
        if name == "d02" and phase == "capture": raise ValueError("capture failed")
        finished.set()
        return name
    jobs=[{"name":n,"lower":lambda n=n:lower(n),
           "finish":lambda c,l,n=n:finish(n,c,l)} for n in ("d01","d02")]
    results, errors = ap._compile_lowered_threads(jobs)
    assert results == ["d01"] and finished.is_set()
    assert errors and errors[0]["name"] == "d02"
    assert not any(t.name.startswith("wrf-cold") for t in threading.enumerate())


def test_worker_limit_caps_large_override_at_two(monkeypatch):
    observed=[]
    real=ap.ThreadPoolExecutor
    def executor(*args, **kwargs):
        observed.append(kwargs["max_workers"])
        return real(*args,**kwargs)
    monkeypatch.setattr(ap,"ThreadPoolExecutor",executor)
    results, errors=ap._compile_lowered_threads([],max_workers=128)
    assert observed == [2] and results == errors == []


def test_runtime_keys_warm_once_then_skip_without_lowering(tmp_path, monkeypatch):
    from gpuwrf.runtime import operational_mode as op, aot_cheap_key as ck, domain_tree as dt
    calls=[]
    monkeypatch.setattr(jax,"devices",lambda:[SimpleNamespace(platform="gpu")])
    monkeypatch.setattr(ap,"_aot_enabled",lambda:True)
    monkeypatch.setattr(ap,"aot_dir",lambda:tmp_path)
    monkeypatch.setattr(ap,"cheap_key_is_quarantined",lambda *a:False)
    monkeypatch.setattr(ax,"export_compile_options",lambda:None)
    monkeypatch.setattr(ax,"hlo_sha256_from_lowered",lambda l:"h"*64)
    monkeypatch.setattr(op,"build_clock_base",lambda nl:None)
    monkeypatch.setattr(ck,"cheap_key",lambda fn,args,kw,nl: nl.key)
    def lower(*args,**kwargs):
        calls.append((args[0],kwargs))
        return SimpleNamespace(compile=lambda opts:"compiled")
    monkeypatch.setattr(op,"_advance_chunk_fori",SimpleNamespace(lower=lower))
    def paths(name,**kw):
        return (tmp_path/(name+".xlaexec"),tmp_path/(name+".meta"))
    monkeypatch.setattr(ap,"_aot_blob_paths",paths)
    def capture(name,compiled,cache_dir,**kwargs):
        assert compiled=="compiled" and kwargs['cheap_key']==name
        assert cache_dir is None
        for path in paths(name):path.write_bytes(b"attested-by-runtime-loader")
        return {"aot_written":True}
    monkeypatch.setattr(ap,"_serialize_domain_blob",capture)
    domains={n:SimpleNamespace(namelist=SimpleNamespace(radiation_cadence_steps=5,key=n))
             for n in ("d01","d02","d03")}
    edges={"d01":(SimpleNamespace(child="d02",parent_grid_ratio=3),),
           "d02":(SimpleNamespace(child="d03",parent_grid_ratio=3),)}
    forced=[]
    def force(edge,parent,child):
        forced.append((parent,child))
        return child+"-live"
    monkeypatch.setattr(dt,"_operational_force",force)
    tree=SimpleNamespace(domains=domains,edges=edges,
        hierarchy=SimpleNamespace(children=lambda name:edges.get(name,())))
    carries={n:n for n in domains}
    first=ap.prewarm_runtime_threads(tree,carries=carries)
    assert first['warm_all'] and first['active'] and len(calls)==3
    assert forced == [("d01","d02"),("d02-live","d03")]
    assert [(args,kw['n_steps']) for args,kw in calls] == [
        ("d01",1),("d02-live",1),("d03-live",3)]
    assert carries == {n:n for n in domains}
    second=ap.prewarm_runtime_threads(tree,carries=carries)
    assert not second['active'] and second['source']=='skip:artifacts-present'
    assert len(calls)==3


def test_shared_cache_lease_compiles_once_for_simultaneous_cases(tmp_path, monkeypatch):
    """The waiting case must recheck keys instead of doubling compiler memory."""
    import fcntl
    from concurrent.futures import ThreadPoolExecutor
    from gpuwrf.runtime import operational_mode as op, aot_cheap_key as ck
    second_lease = threading.Event()
    attempts=[]
    real_flock=fcntl.flock
    def flock(fd, mode):
        attempts.append(fd)
        if len(attempts)==2:second_lease.set()
        return real_flock(fd,mode)
    monkeypatch.setattr(fcntl,'flock',flock)
    monkeypatch.setattr(jax,'devices',lambda:[SimpleNamespace(platform='gpu')])
    monkeypatch.setattr(ap,'_aot_enabled',lambda:True)
    monkeypatch.setattr(ap,'aot_dir',lambda:tmp_path)
    monkeypatch.setattr(ap,'cheap_key_is_quarantined',lambda *a:False)
    monkeypatch.setattr(ck,'cheap_key',lambda fn,args,kw,nl:nl.key)
    monkeypatch.setattr(op,'build_clock_base',lambda nl:None)
    monkeypatch.setattr(ax,'export_compile_options',lambda:None)
    monkeypatch.setattr(ax,'hlo_sha256_from_lowered',lambda l:'h'*64)
    lowered=[]
    def lower(carry,*args,**kwargs):
        assert second_lease.wait(5), 'second case never reached lease'
        lowered.append(carry)
        return SimpleNamespace(compile=lambda opts:carry)
    monkeypatch.setattr(op,'_advance_chunk_fori',SimpleNamespace(lower=lower))
    def paths(name,**kwargs):return tmp_path/(name+'.xlaexec'),tmp_path/(name+'.meta')
    monkeypatch.setattr(ap,'_aot_blob_paths',paths)
    def capture(name,compiled,cache_dir,**kwargs):
        for path in paths(name):path.write_bytes(b'native')
        return {'aot_written':True}
    monkeypatch.setattr(ap,'_serialize_domain_blob',capture)
    domains={n:SimpleNamespace(namelist=SimpleNamespace(key=n,radiation_cadence_steps=5))
             for n in ('d01','d02')}
    tree=SimpleNamespace(domains=domains,edges={})
    with ThreadPoolExecutor(2) as cases:
        reports=list(cases.map(lambda _:ap.prewarm_runtime_threads(tree,carries={n:n for n in domains}),range(2)))
    assert lowered==['d01','d02']
    assert len(attempts)==2
    assert sum(r['active'] for r in reports)==1
    assert all(r['warm_all'] and not r['errors'] for r in reports)


def test_orchestration_exception_returns_to_serial_caller(monkeypatch):
    from gpuwrf.runtime import domain_tree as dt
    monkeypatch.setattr(dt,'_nested_fuse_default_enabled',lambda:False)
    monkeypatch.setattr(jax,'default_backend',lambda:'gpu')
    monkeypatch.setenv('GPUWRF_NESTED_THREADED_COMPILE','2')
    def fail(*args,**kwargs):raise RuntimeError('injected compiler failure')
    monkeypatch.setattr(ap,'prewarm_runtime_threads',fail)
    report=dt.maybe_prewarm_defused_nest(object(),carries={})
    assert not report['active'] and 'injected compiler failure' in report['error']
