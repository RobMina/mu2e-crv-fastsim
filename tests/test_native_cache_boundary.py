"""A native cache cannot bypass seven-file or construction-support admission."""
import json

import pytest

from crvfastsim import tier1b_assets
from crvfastsim.analyses.context_audit import content_fingerprint
from test_native_builder_integration import _driver, _patterns, native_config, inputs


def test_real_native_cache_miss_hit_and_bundle_drift(tmp_path, native_config, inputs, monkeypatch):
    bundle = tmp_path / "library"
    _driver().run(*_patterns(tmp_path), bundle, native_config.name, [101])
    spec = tier1b_assets.native_bundle(bundle, native_config)
    geom = native_config.geometry()
    expected = content_fingerprint(tier1b_assets.load(spec, geom), workers=1)
    monkeypatch.setenv("CRVFS_PREPARED_LIBRARY_CACHE", str(tmp_path / "cache"))
    first = tier1b_assets.load(spec, geom)
    assert content_fingerprint(first, workers=1) == expected
    def forbidden(*args):
        pytest.fail("cache hit called loader")
    monkeypatch.setattr(tier1b_assets, "_load_uncached", forbidden)
    second = tier1b_assets.load(spec, geom)
    assert content_fingerprint(second, workers=1) == expected
    with (bundle / "build_check.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="changed since resolution"):
        tier1b_assets.load(spec, geom)


def test_cached_numeric_payload_does_not_admit_missing_construction_support(tmp_path, native_config, inputs, monkeypatch):
    bundle = tmp_path / "library"
    _driver().run(*_patterns(tmp_path), bundle, native_config.name, [101])
    spec = tier1b_assets.native_bundle(bundle, native_config)
    monkeypatch.setenv("CRVFS_PREPARED_LIBRARY_CACHE", str(tmp_path / "cache"))
    tier1b_assets.load(spec, native_config.geometry())
    for filename in ("library.json", "build_check.json"):
        path = bundle / filename
        data = json.loads(path.read_text())
        data.pop("construction_support")
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="construction support"):
        tier1b_assets.native_bundle(bundle, native_config)
