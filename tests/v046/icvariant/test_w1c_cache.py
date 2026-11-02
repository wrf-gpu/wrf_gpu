"""W1c cache admission: extra compiled entries in the paid layer are accepted only as a pinned superset of the prime."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location('w1c_cache', Path(__file__).resolve().parents[3] / 'scripts/w1c/launcher/w1c_cache.py')
wc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(wc)

K1, K2 = 'b' * 64, 'a' * 64
PRIME = {f'jax/aot/v/d01/k_{K1}.xlaexec': {'sha256': '1' * 64, 'bytes': 10},
         f'jax/aot/v/fused_d02/k_{K2}.xlaexec': {'sha256': '2' * 64, 'bytes': 20}}
PROGRAMS = {'domains': {'d01': {'cheap_key': K1}, 'fused/d02': {'cheap_key': K2}}}
EXTRA = {f'jax/aot/v/d01/k_{"c" * 64}.xlaexec': {'sha256': '3' * 64, 'bytes': 30},
         f'jax/jit_fused_jit-{"d" * 64}-cache': {'sha256': '4' * 64, 'bytes': 40}}


def test_superset_accepted():
    frozen, added = wc.frozen_inventory(PRIME, {**PRIME, **EXTRA}, PROGRAMS)
    assert frozen == {**PRIME, **EXTRA} and added == sorted(EXTRA)


def test_identity_without_additions():
    frozen, added = wc.frozen_inventory(PRIME, dict(PRIME), PROGRAMS)
    assert frozen == PRIME and added == []


@pytest.mark.parametrize('mut', ['missing_prime', 'changed_prime', 'shadow_key', 'outside_jax', 'odd_suffix', 'malformed', 'empty'])
def test_refusals(mut):
    inv = {**PRIME, **EXTRA}
    if mut == 'missing_prime':
        inv.pop(next(iter(PRIME)))
    elif mut == 'changed_prime':
        inv[next(iter(PRIME))] = {'sha256': '9' * 64, 'bytes': 10}
    elif mut == 'shadow_key':
        inv[f'jax/aot/v/other/k_{K1}.meta'] = {'sha256': '5' * 64, 'bytes': 1}
    elif mut == 'outside_jax':
        inv['triton/x.xlaexec'] = {'sha256': '6' * 64, 'bytes': 1}
    elif mut == 'odd_suffix':
        inv['jax/aot/v/d01/notes.txt'] = {'sha256': '7' * 64, 'bytes': 1}
    elif mut == 'malformed':
        inv[f'jax/aot/v/d01/k_{"e" * 64}.meta'] = {'sha256': 'x', 'bytes': 1}
    else:
        inv = {}
    with pytest.raises(ValueError):
        wc.frozen_inventory(PRIME, inv, PROGRAMS)


def test_sealed_pin_matches_inventory_file():
    """w1c.py pins W1C_CACHE_INVENTORY.json by SHA: exactly the prime (1863) + the 10 ALP1 C3 additions (manager (A))."""
    import hashlib, json, re
    here = Path(__file__).resolve().parents[3] / 'scripts/w1c/launcher'
    src = (here / 'w1c.py').read_text()
    pin = re.search(r"CACHE_INVENTORY_SHA = '([0-9a-f]{64})'", src).group(1)
    assert hashlib.sha256((here / 'W1C_CACHE_INVENTORY.json').read_bytes()).hexdigest() == pin
    inv = json.loads((here / 'W1C_CACHE_INVENTORY.json').read_text())
    assert inv['prime_entries'] == 1863 and len(inv['added_after_prime']) == 10 == int(re.search(r'CACHE_ADDED_AFTER_PRIME = (\d+)', src).group(1))
    assert len(inv['compiled']) == 1873 and all(a in inv['compiled'] for a in inv['added_after_prime'])
    assert "'W1C_CACHE_INVENTORY.json'" in (here / 'make_seal.py').read_text()
