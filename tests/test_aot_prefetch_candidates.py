"""CPU-only artifact selection; no loader, key prediction, or executable reuse."""
from pathlib import Path
from types import SimpleNamespace

import jax
import pytest
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.integration.nested_pipeline import _aot_prefetch_artifact_candidates


def _artifact(root, name, key):
    path=root/name
    path.mkdir(exist_ok=True)
    (path/('k_'+key+'.meta')).write_bytes(b'metadata')
    (path/('k_'+key+'.xlaexec')).write_bytes(b'guarded-by-loader')


def test_select_only_complete_unique_artifact_identity(tmp_path):
    _artifact(tmp_path,'d01','a'*64)
    _artifact(tmp_path,'d02','b'*64)
    assert _aot_prefetch_artifact_candidates(('d01','d02','d03'),directory=tmp_path)=={'d01':'a'*64,'d02':'b'*64}
    _artifact(tmp_path,'d02','c'*64)
    assert _aot_prefetch_artifact_candidates(('d01','d02'),directory=tmp_path)=={'d01':'a'*64}


def test_missing_empty_and_legacy_addresses_do_not_become_hints(tmp_path):
    _artifact(tmp_path,'d01','a'*64)
    p=tmp_path/'d01'
    (p/('k_'+'a'*64+'.xlaexec')).write_bytes(b'')
    (p/'legacy.meta').write_bytes(b'metadata')
    (p/'legacy.xlaexec').write_bytes(b'old-address')
    (p/('k_'+'b'*64+'.meta')).write_bytes(b'missing-blob')
    _artifact(tmp_path,'d01','not-a-sha')
    assert _aot_prefetch_artifact_candidates(('d01',),directory=tmp_path)=={}


def test_pipeline_joins_loaders_when_domain_init_fails(tmp_path, monkeypatch):
    import pytest
    from types import SimpleNamespace
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import aot_precompile
    import jax

    events = []
    device = SimpleNamespace(platform="gpu")
    handle = SimpleNamespace(close=lambda: events.append("joined"))
    monkeypatch.setattr(jax, "devices", lambda: events.append("backend") or [device])
    monkeypatch.setattr(pipeline, "_aot_prefetch_artifact_candidates",
                        lambda names: {"d01": "a"*64})

    def start(keys, **kwargs):
        assert kwargs["dev"] is device
        assert keys == {"d01": "a"*64}
        events.append("load-start")
        return handle

    def fail_init(*args, **kwargs):
        events.append("init")
        raise ValueError("deliberate input error")

    monkeypatch.setattr(aot_precompile, "prefetch_domain_blobs", start)
    monkeypatch.setattr(pipeline, "_load_domains", fail_init)
    monkeypatch.setattr(pipeline, "_batch_ensemble_size_from_env", lambda: 1)
    config = pipeline.NestedPipelineConfig(
        input_dir=tmp_path, output_dir=tmp_path/"out", proof_dir=tmp_path/"proof",
        hours=3, max_dom=2, aot_prefetch=True,
    )
    with pytest.raises(ValueError, match="deliberate input error"):
        pipeline.execute_nested_pipeline(config)
    assert events == ["backend", "load-start", "init", "joined"]


def test_default_pipeline_never_starts_prefetch(tmp_path, monkeypatch):
    import pytest
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import aot_precompile

    def forbidden(*args, **kwargs):
        raise AssertionError("default pipeline started prefetch")

    def fail_init(*args, **kwargs):
        raise ValueError("deliberate input error")

    monkeypatch.setattr(aot_precompile, "prefetch_domain_blobs", forbidden)
    monkeypatch.setattr(pipeline, "_load_domains", fail_init)
    monkeypatch.setattr(pipeline, "_batch_ensemble_size_from_env", lambda: 1)
    config = pipeline.NestedPipelineConfig(
        input_dir=tmp_path, output_dir=tmp_path/"out", proof_dir=tmp_path/"proof",
        hours=3, max_dom=2,
    )
    with pytest.raises(ValueError, match="deliberate input error"):
        pipeline.execute_nested_pipeline(config)


def test_pipeline_candidate_scan_error_must_fail_open(tmp_path, monkeypatch):
    monkeypatch.setattr(jax, 'devices', lambda: [SimpleNamespace(platform='gpu')])
    monkeypatch.setattr(pipeline, '_batch_ensemble_size_from_env', lambda: 1)
    def scan_error(names):
        raise FileNotFoundError('cache blob removed between is_file and stat')
    monkeypatch.setattr(pipeline, '_aot_prefetch_artifact_candidates', scan_error)
    def reached_init(*args, **kwargs):
        raise ValueError('initialization reached')
    monkeypatch.setattr(pipeline, '_load_domains', reached_init)
    config = pipeline.NestedPipelineConfig(input_dir=tmp_path, output_dir=tmp_path/'out',
        proof_dir=tmp_path/'proof', hours=1, max_dom=3, aot_prefetch=True)
    with pytest.raises(ValueError, match='initialization reached'):
        pipeline.execute_nested_pipeline(config)


def test_selector_has_is_file_stat_deletion_race(tmp_path, monkeypatch):
    key = 'a' * 64
    directory = tmp_path/'d01'
    directory.mkdir()
    (directory/f'k_{key}.meta').write_bytes(b'meta')
    blob = directory/f'k_{key}.xlaexec'
    blob.write_bytes(b'blob')
    original = Path.is_file
    def delete_after_check(path):
        result = original(path)
        if path == blob:
            path.unlink()
        return result
    monkeypatch.setattr(Path, 'is_file', delete_after_check)
    with pytest.raises(FileNotFoundError):
        pipeline._aot_prefetch_artifact_candidates(('d01',), directory=tmp_path)


def test_ambiguous_keys_prefer_runtime_last_used_key(tmp_path):
    from gpuwrf.runtime import aot_precompile as ap

    for key in ('a' * 64, 'b' * 64):
        _artifact(tmp_path, 'd01', key)
    assert _aot_prefetch_artifact_candidates(('d01',), directory=tmp_path) == {}
    (tmp_path / 'd01' / ap._LAST_CHEAP_KEY).write_text('b' * 64)
    assert _aot_prefetch_artifact_candidates(('d01',), directory=tmp_path) == {'d01': 'b' * 64}
    # A pointer to an incomplete/missing artifact is never a hint.
    (tmp_path / 'd01' / ap._LAST_CHEAP_KEY).write_text('c' * 64)
    assert _aot_prefetch_artifact_candidates(('d01',), directory=tmp_path) == {}
