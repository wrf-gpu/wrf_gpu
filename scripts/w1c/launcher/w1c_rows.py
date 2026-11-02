"""Member rows of a signed W1c GO (pure; used by w1c.py and w1c_raw_release.py).

The issued ALISIOS GO lists only the identity fields per member and pins its full row draft by SHA in `go_draft`:
rows come from that pinned draft, whose identity fields must equal the signed GO's members one for one, same order.
A document whose members carry full rows (no identity-only list) is used as is."""

IDENTITY_FIELDS = ('ordinal', 'row_id', 'our_id', 'registry_run_id')


def signed_rows(doc, load_pinned):
    """load_pinned(ref) -> parsed JSON of ref {'path','sha256'} after an exact SHA check (raises otherwise)."""
    members = doc.get('members') or doc.get('rows') or []
    if not members:
        raise ValueError('signed GO has no members')
    if all(set(m) <= set(IDENTITY_FIELDS) for m in members):
        ref = doc.get('go_draft')
        if not (isinstance(ref, dict) and ref.get('path') and ref.get('sha256')):
            raise ValueError('identity-only GO members without a SHA-pinned go_draft')
        full = (load_pinned(ref).get('members')) or []
        if [tuple(m.get(k) for k in IDENTITY_FIELDS) for m in members] != [tuple(f.get(k) for k in IDENTITY_FIELDS) for f in full]:
            raise ValueError('signed GO members != pinned GO-draft rows')
        return [dict(f) for f in full]
    return [dict(m) for m in members]
