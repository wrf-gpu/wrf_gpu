from gpuwrf.runtime.operational_mode import _StaticHolder


def test_static_holder_none_hash_is_stable_across_flatten_rebuilds():
    """Disabled static bundles must not fragment JIT cache keys."""

    first = _StaticHolder(None)
    second = _StaticHolder(None)

    assert first == second
    assert hash(first) == hash(second)


def test_static_holder_real_bundle_hashes_by_identity():
    bundle = object()
    same = _StaticHolder(bundle)
    again = _StaticHolder(bundle)
    different = _StaticHolder(object())

    assert same == again
    assert hash(same) == hash(again)
    assert same != different
