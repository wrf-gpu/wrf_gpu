"""W1c cache admission (pure): the paid W1a layer may hold EXTRA compiled entries written by other runs after the prime
(e.g. ALP1 C3, 2026-10-31 03:54-04:00Z). W1c accepts a SHA-pinned full inventory only if every prime entry is present and
byte-identical, no prime program key is shadowed, and every addition is a plain JAX cache/AOT entry; the run then freezes
on that full inventory (any further change refuses, exactly like the W1a unchanged() guard)."""
import re


def frozen_inventory(prime_compiled, inventory, prime_programs):
    """Return the inventory to freeze on, or raise ValueError."""
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('empty cache inventory')
    missing = sorted(set(prime_compiled) - set(inventory))
    if missing:
        raise ValueError(f'prime entries missing from the cache inventory: {missing[:3]}')
    changed = sorted(k for k in prime_compiled if inventory[k] != prime_compiled[k])
    if changed:
        raise ValueError(f'prime entries changed in the cache inventory: {changed[:3]}')
    added = sorted(set(inventory) - set(prime_compiled))
    keys = {p['cheap_key'] for p in prime_programs['domains'].values()}
    for rel in added:
        if not rel.startswith('jax/') or not (rel.endswith('.xlaexec') or rel.endswith('.meta') or rel.endswith('-cache')):
            raise ValueError(f'unexpected added cache entry: {rel}')
        m = re.search(r'/k_([0-9a-f]{64})\.(xlaexec|meta)$', rel)
        if m and m.group(1) in keys:
            raise ValueError(f'added entry shadows a prime program key: {rel}')
        v = inventory[rel]
        if not (isinstance(v, dict) and re.fullmatch('[0-9a-f]{64}', v.get('sha256', '')) and isinstance(v.get('bytes'), int)):
            raise ValueError(f'malformed inventory row: {rel}')
    return dict(inventory), added
