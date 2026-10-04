"""An explicit frozen WRF root remains valid with the required checkout link."""

from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/v025"))

import wrf_source_authority as authority  # noqa: E402


@pytest.fixture
def linked_checkout(tmp_path, monkeypatch):
    canonical = tmp_path / "pristine" / "WRF"
    canonical.mkdir(parents=True)
    checkout = tmp_path / "checkout"
    (checkout / "data").mkdir(parents=True)
    (checkout / "data/wrf_pristine").symlink_to(canonical.parent)
    monkeypatch.setattr(authority, "REPO_ROOT", checkout)
    return canonical, checkout / authority.CHECKOUT_DEFAULT_REL


def test_explicit_canonical_root_survives_checkout_symlink(linked_checkout):
    canonical, checkout_default = linked_checkout
    assert checkout_default.resolve() == canonical
    assert authority.resolve_authority_root(
        environ={name: str(canonical) for name in authority.ROOT_ENV_VARS},
        canonical_root=canonical,
    ) == canonical


@pytest.mark.parametrize("normalize", [False, True])
def test_checkout_default_spelling_remains_forbidden(linked_checkout, normalize):
    canonical, checkout_default = linked_checkout
    candidate = (
        checkout_default.parent / ".." / "wrf_pristine" / "WRF"
        if normalize else checkout_default
    )
    with pytest.raises(authority.SourceAuthorityRefusal, match="checkout-relative"):
        authority.resolve_authority_root(
            environ={authority.ENV_VAR: str(candidate)}, canonical_root=canonical,
        )


def test_unfrozen_root_remains_forbidden(linked_checkout, tmp_path):
    canonical, _ = linked_checkout
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(authority.SourceAuthorityRefusal, match="not frozen authority"):
        authority.resolve_authority_root(
            environ={authority.ENV_VAR: str(other)}, canonical_root=canonical,
        )


def test_split_root_environment_remains_forbidden(linked_checkout, tmp_path):
    canonical, _ = linked_checkout
    with pytest.raises(authority.SourceAuthorityRefusal, match="conflicts"):
        authority.resolve_authority_root(
            environ={authority.ENV_VAR: str(canonical),
                     authority.SRC_ENV_VAR: str(tmp_path)},
            canonical_root=canonical,
        )
