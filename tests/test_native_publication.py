"""Real driver publication failures stay failures; unsupported events are not filtered."""
import json

import numpy as np
import pandas as pd
import pytest

from crvfastsim import native_training, tier1b_assets
from test_native_builder_integration import _driver, _frames, _patterns, _write, native_config, inputs


@pytest.mark.parametrize("role", ["donor", "full"])
def test_multi_primary_refuses_before_output_creation(tmp_path, native_config, role):
    donor, full = _frames(native_config.geometry())
    if role == "donor":
        added = donor["particles"].iloc[:1].copy().assign(key=np.int64(17))
        donor["particles"] = pd.concat([donor["particles"], added], ignore_index=True)
        donor["events"].loc[0, "n_particles"] += 1
    else:
        added = full["gen"].iloc[:1].copy().assign(idx=np.int64(1))
        full["gen"] = pd.concat([full["gen"], added], ignore_index=True)
        full["events"].loc[0, "n_gen"] += 1
    d = _write(tmp_path / "crvsteps.001400_00000101.root", donor)
    g = _write(tmp_path / "gen.001400_00000101.root", full)
    assert native_training.validate([d], [g])["census_admitted"] is True
    out = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="multiple primary muons"):
        _driver().run(*_patterns(tmp_path), out, native_config.name, [101])
    assert not out.exists()


def test_two_shards_with_reused_particle_keys_build_and_attach(tmp_path, native_config):
    for shard in (101, 103):
        donor, full = _frames(native_config.geometry(), subrun=shard)
        _write(tmp_path / f"crvsteps.001400_{shard:08d}.root", donor)
        _write(tmp_path / f"gen.001400_{shard:08d}.root", full)
    out = tmp_path / "two-shards"
    _driver().run(*_patterns(tmp_path), out, native_config.name, [101, 103])
    spec = tier1b_assets.native_bundle(out, native_config)
    lib = tier1b_assets.load(spec, native_config.geometry())
    assert spec.native["training_census"]["count"] == 12
    assert lib["libdesc"].root.tolist() == [8, 8]
    assert lib["anchor"]["n_roots"] == 2
    assert lib["xgeo"]["mu"].tolist() == [0, 1]
    assert spec.native["construction_support"]["events_filtered"] == 0


def test_shared_root_chords_unvisited_and_nearmiss_serialize_together(tmp_path, native_config):
    donor, full = _frames(native_config.geometry())
    geom = native_config.geometry()
    next_bar = geom.bar_index(0, 0, 1, 0)
    uv_bar = geom.bar_index(0, 0, 3, 0)
    move = geom.center[next_bar] - geom.center[int(donor["steps"].bar.iloc[0])]
    extra = donor["steps"].iloc[:2].copy().assign(bar=np.int64(next_bar))
    for j, coordinate in enumerate("xyz"):
        for endpoint in ("0", "1"):
            extra[coordinate + endpoint] += move[j]
    extra["t0"] += 10.
    extra["t1"] += 10.
    unvisited = extra.iloc[[1]].copy().assign(bar=np.int64(uv_bar))
    offset = geom.center[uv_bar] - geom.center[next_bar]
    for j, coordinate in enumerate("xyz"):
        for endpoint in ("0", "1"):
            unvisited[coordinate + endpoint] += offset[j]
    donor["steps"] = pd.concat([donor["steps"], extra, unvisited], ignore_index=True)
    donor["events"].loc[0, "n_steps"] = np.int64(5)
    near_event = donor["events"].iloc[[0]].copy().assign(event=np.uint32(5), n_particles=np.int64(2),
        n_steps=np.int64(1))
    donor["events"] = pd.concat([donor["events"], near_event], ignore_index=True)
    donor["gen"] = pd.concat([donor["gen"], full["gen"].loc[full["gen"].event == 5]], ignore_index=True)
    near_parts = donor["particles"].iloc[:2].copy().assign(event=np.uint32(5))
    donor["particles"] = pd.concat([donor["particles"], near_parts], ignore_index=True)
    near_step = unvisited.copy().assign(event=np.uint32(5))
    donor["steps"] = pd.concat([donor["steps"], near_step], ignore_index=True)
    _write(tmp_path / "crvsteps.001400_00000101.root", donor)
    _write(tmp_path / "gen.001400_00000101.root", full)
    out = tmp_path / "full-attachments"
    _driver().run(*_patterns(tmp_path), out, native_config.name, [101])
    spec = tier1b_assets.native_bundle(out, native_config)
    lib = tier1b_assets.load(spec, geom)
    assert lib["anchor"]["n_roots"] == 1
    assert lib["anchor"]["n_multi"] == 1
    assert lib["xgeo"]["nxt"].tolist() == [1, -1]
    assert len(lib["libdesc"]) == 3 and lib["libdesc"].root.tolist() == [8, 8, 8]
    metadata = json.loads((out / "templates_v3.npz.json").read_text())
    assert metadata["unvisited_rows"] == 1
    assert metadata["nearmiss_templates"] == metadata["nearmiss_steps"] == 1


def test_dangling_output_symlink_is_refused(tmp_path, native_config):
    out = tmp_path / "dangling"
    out.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        _driver().run(*_patterns(tmp_path), out, native_config.name, [101])
    assert out.is_symlink() and not out.exists()


@pytest.mark.parametrize("failure", ["sidecar", "attachment", "final_admission", "source_drift", "geometry_drift"])
def test_driver_never_prints_success_after_publication_failure(tmp_path, native_config, inputs, monkeypatch, capsys, failure):
    if failure == "geometry_drift":
        from dataclasses import replace
        from pathlib import Path
        from crvfastsim import runconfig
        counters = tmp_path / "mutable-counters.txt"
        counters.write_bytes(Path(native_config.counters_path()).read_bytes())
        native_config = replace(native_config, counters=str(counters))
        monkeypatch.setitem(runconfig.REGISTRY, native_config.name, native_config)
    driver = _driver()
    def fail(*args, **kw):
        raise ValueError("injected publication failure")
    if failure == "sidecar":
        monkeypatch.setattr(driver.descendant_templates, "sidecar", fail)
    elif failure == "attachment":
        monkeypatch.setattr(driver.tier1b, "attach_anchors", fail)
    elif failure == "final_admission":
        monkeypatch.setattr(driver.tier1b_assets, "native_bundle", fail)
    else:
        original = driver.descendant_templates.sidecar
        def mutate(*args, **kw):
            result = original(*args, **kw)
            path = inputs[0] if failure == "source_drift" else native_config.counters_path()
            with open(path, "ab") as stream:
                stream.write(b"\n# injected drift\n")
            return result
        monkeypatch.setattr(driver.descendant_templates, "sidecar", mutate)
    out = tmp_path / "failed-publication"
    with pytest.raises(ValueError, match="publication failure|training inputs changed|build-time geometry"):
        driver.run(*_patterns(tmp_path), out, native_config.name, [101])
    lines = capsys.readouterr().out.splitlines()
    assert not any('"status": "built and attachments checked' in line for line in lines)
    assert out.exists()


def test_checker_output_is_exclusive_and_real_bundle_loads(tmp_path, native_config, inputs, monkeypatch):
    from scripts import check_native_bundle
    out = tmp_path / "bundle"
    _driver().run(*_patterns(tmp_path), out, native_config.name, [101])
    report = tmp_path / "check.json"
    monkeypatch.setattr("sys.argv", ["check_native_bundle.py", "--directory", str(out),
        "--run-config", native_config.name, "--out", str(report)])
    check_native_bundle.main()
    before = report.read_bytes()
    assert json.loads(before)["tier1b"]["native"]["construction_support"]["supported"] is True
    with pytest.raises(FileExistsError):
        check_native_bundle.main()
    assert report.read_bytes() == before
