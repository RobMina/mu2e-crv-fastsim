"""Real schema-v2 donors through the native driver, sidecars and attached bundle."""
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim import genassoc, native_training, runconfig, tier1b, tier1b_assets


ROOT = Path(__file__).resolve().parents[1]


def _driver():
    spec = importlib.util.spec_from_file_location("native_builder_integration_driver", ROOT / "scripts/build_native_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, frames):
    with uproot.recreate(path) as root:
        for name, frame in frames.items():
            columns = {column: frame[column].to_numpy() for column in frame}
            root.mktree("dump/" + name, {column: values.dtype for column, values in columns.items()})
            if len(frame):
                root["dump/" + name].extend(columns)
    return path


def _tiny_counters(path):
    path.write_text("\n".join([
        'vector<string> crs.sectorNames = {"T1", "T2"};',
        'int crs.nLayers = 4;',
        'double crs.scintillatorBarThickness = 19.8;',
        'double crs.scintillatorBarWidth = 51.34;',
        'double crs.gapSmall = 0;',
        'double crs.gapLarge = 0.2;',
        'double crs.gapBetweenModules = 3.66;',
        'vector<double> crs.gapBetweenLayers = {9.525, 9.525, 9.525};',
        'double crs.layerOffset = 0;',
        *[line for sector, z in (("T1", 0), ("T2", 1000)) for line in (
            f'double crs.scintillatorBarLength{sector} = 6000;',
            f'int crs.nModules{sector} = 1;',
            f'int crs.nCountersPerModule{sector} = 2;',
            f'vector<double> crs.firstCounter{sector} = {{0, 0, {z}}};',
            f'vector<double> crs.layerDirection{sector} = {{0, 1, 0}};',
            f'vector<double> crs.gapDirection{sector} = {{0, 0, 1}};',
            f'vector<double> crs.offsetDirection{sector} = {{0, 0, -1}};',
        )],
    ]) + "\n")
    return path


@pytest.fixture(params=["portable", "local_run1a"])
def native_config(tmp_path, monkeypatch, request):
    monkeypatch.setenv("CRVFS_NO_FETCH", "1")
    monkeypatch.delenv("CRVFS_PREPARED_LIBRARY_CACHE", raising=False)
    if request.param == "portable":
        counters = _tiny_counters(tmp_path / "tiny-run1a-counters.txt")
    else:
        suffix = Path("Offline/Mu2eG4/geom/crv_counters_run1a_v01.txt")
        roots = [Path(p) for p in os.environ.get("MU2E_SEARCH_PATH", "").split(":") if p]
        if os.environ.get("CRVFS_OFFLINE_ROOT"):
            roots.append(Path(os.environ["CRVFS_OFFLINE_ROOT"]))
        roots += [Path("/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00")]
        candidates = [root / rel for root in roots for rel in (suffix, Path(*suffix.parts[1:]))]
        counters = next((path for path in candidates if path.is_file()), None)
        if counters is None:
            pytest.skip("local Run 1A counters unavailable; portable real-geometry arm remains covered")
    rc = replace(runconfig.get("run1a_v01"), counters=str(counters.resolve()))
    monkeypatch.setitem(runconfig.REGISTRY, rc.name, rc)
    return rc


def _frames(geom, subrun=101):
    # Donors retain crossing, shower and gen-empty events; full census also retains two rejected primaries and an empty event.
    events = pd.DataFrame(dict(run=np.full(6, 1400, dtype=np.uint32),
        subrun=np.full(6, subrun, dtype=np.uint32), event=np.arange(1, 7, dtype=np.uint32),
        schema_version=np.uint32(2), associations_checked=np.uint32(0), full_generator_stream=np.uint32(1),
        gen_product_id=np.uint32(99), sim_product_id=np.uint32(0),
        n_gen=np.array([1, 1, 0, 1, 1, 0], dtype=np.int64), n_particles=np.int64(0), n_steps=np.int64(0)))
    bar = geom.bar_index(0, 0, 0, 0)
    center = geom.center[bar]
    axis = np.eye(3)[geom.local_to_world[bar, 0]]
    half = geom.halflen[bar, geom.local_to_world[bar, 0]]
    gen = events.iloc[[0, 1, 3, 4]][genassoc.KEY + ["gen_product_id"]].copy().reset_index(drop=True)
    gen = gen.assign(idx=np.int64(0), pdg=np.array([-13, 22, 22, -13], dtype=np.int32), genid=np.int32(1),
                     e=np.array([200., 20., 20., 200.]), t=0.)
    position = np.tile(center + (half + 100.) * axis, (4, 1))
    momentum = -np.array([np.sqrt(200.**2 - tier1b.MASS[13]**2), 20., 20., 100.])[:, None] * axis
    for j, column in enumerate(("x", "y", "z")):
        gen[column] = position[:, j]
    for j, column in enumerate(("px", "py", "pz")):
        gen[column] = momentum[:, j]
    donor_events = events.iloc[:3].copy().assign(associations_checked=np.uint32(1),
        full_generator_stream=np.uint32(0), sim_product_id=np.uint32(202))
    donor_events["n_particles"] = np.array([2, 1, 0], dtype=np.int64)
    donor_events["n_steps"] = np.array([2, 1, 0], dtype=np.int64)
    particles = donor_events.iloc[[0, 0, 1]][genassoc.KEY + ["gen_product_id", "sim_product_id"]].copy().reset_index(drop=True)
    particles = particles.assign(key=np.array([7, 8, 9], dtype=np.int64), parent=np.array([-1, 7, -1], dtype=np.int64),
        parent_product_id=np.array([0, 202, 0], dtype=np.uint32), genidx=np.int64(0),
        isprim=np.array([1, 0, 1], dtype=np.int32), pdg=np.array([-13, 11, 22], dtype=np.int32),
        se=np.array([200., 2., 20.]), st=0.)
    steps = particles[genassoc.KEY + ["key", "sim_product_id"]].rename(
        columns={"key": "sp", "sim_product_id": "sp_product_id"}).copy()
    starts = np.array([center + half * axis, center, center])
    ends = np.array([center - half * axis, center - 0.25 * axis, center - 0.5 * axis])
    momenta = -np.array([np.sqrt(200.**2 - tier1b.MASS[13]**2), 1., 20.])[:, None] * axis
    steps = steps.assign(bar=np.int64(bar), t0=np.array([10., 10.1, 20.]), t1=np.array([10.01, 10.11, 20.01]),
                         len=np.linalg.norm(ends - starts, axis=1), edep=np.array([2., 0.2, 0.5]),
                         p1=np.linalg.norm(momenta, axis=1) * 0.99)
    for j, column in enumerate(("x", "y", "z")):
        steps[column + "0"], steps[column + "1"] = starts[:, j], ends[:, j]
        steps["p" + column + "0"] = momenta[:, j]
    donor = dict(events=donor_events, gen=gen.iloc[:2].copy(), particles=particles, steps=steps)
    full = dict(events=events, gen=gen)
    return donor, full


@pytest.fixture
def inputs(tmp_path, native_config):
    donor, full = _frames(native_config.geometry())
    steps = _write(tmp_path / "crvsteps.001400_00000101.root", donor)
    generator = _write(tmp_path / "gen.001400_00000101.root", full)
    return steps, generator


def _patterns(directory):
    return str(directory / "crvsteps.{s}.root"), str(directory / "gen.{s}.root")


def _check_bundle(outdir, rc, inputs):
    check = json.loads((outdir / "build_check.json").read_text())
    manifest = json.loads((outdir / "library.json").read_text())
    templates = json.loads((outdir / "templates_v3.npz.json").read_text())
    receipt = check["training_census"]
    assert {k: v for k, v in receipt.items() if k != "declared_shards"} == {
        k: v for k, v in manifest["training_census"].items() if k != "declared_shards"}
    assert receipt["schema"] == native_training.SCHEMA
    assert receipt["total_exposure"] == check["generated"] == manifest["generated_scaled_to"] == 6
    assert receipt["actual_shards"] == receipt["declared_shards"] == [{"run": 1400, "subrun": 101}]
    assert receipt["pairs"][0]["donor_events"]["count"] == 3
    assert receipt["pairs"][0]["generator_rows"] == {"donor": 2, "full": 4}
    keys = np.array([[1400, 101, event] for event in range(1, 7)], dtype="<i8")
    assert receipt["full_generator_events"] == {"count": 6, "sha256": hashlib.sha256(keys.tobytes()).hexdigest()}
    for role, path in zip(("steps", "generator"), inputs):
        record = receipt["pairs"][0][role]
        assert record["path"] == str(path.resolve())
        assert record["sha256"] == tier1b_assets.sha256(path)
    assert templates["generated"] == templates["census_events"] == 6
    assert templates["census_muons"] == 2 and templates["census_no_crossing_muons"] == 1
    assert templates["census"][0]["steps"] == str(inputs[0].resolve())
    assert templates["census"][0]["gen"] == str(inputs[1].resolve())
    spec = tier1b_assets.native_bundle(outdir, rc)
    lib = tier1b_assets.load(spec, rc.geometry())
    assert spec.name is None and spec.native["anchor_scope"] == "module"
    assert len(spec.paths()) == 7 and all(Path(path).is_file() for path in spec.paths())
    assert spec.options == dict(tier1b_assets._STRICT, chord=True, anchor=True, showers=True)
    assert check["counts"] == {name: len(lib[name]) for name in tier1b.LIB_TREES}
    assert {name: len(lib[name]) for name in tier1b.LIB_TREES if name != "libbins"} == dict(
        libxing=1, libdesc=1, libshower=1, libshowersteps=1, libgen=1)
    assert lib["libbins"].group.unique().tolist() == [0]
    assert lib["libdesc"].root.tolist() == [8] and lib["libdesc"].samebar.tolist() == [1]
    assert "tvis" in lib["libdesc"] and "uv" in lib["libdesc"]
    assert lib["xgeo"]["nxt"].tolist() == [-1]
    assert np.all(np.linalg.norm(lib["xgeo"]["exit"] - lib["xgeo"]["entry"], axis=1) > 0)
    assert lib["libgen"].ngen.tolist() == [2] and lib["libgen"].nshower.tolist() == [1]
    assert check["anchor"] == {"n_roots": 1, "n_multi": 0, "forward_edep_share": 0.}
    assert lib["anchor"]["n_roots"] == 1 and lib["anchor"]["axid"].tolist() == [0]
    assert "physics validation pending" in check["status"]
    assert tier1b_assets.default_for(rc.name) is None
    assert tier1b_assets.DEFAULT == "strict_v5" and tier1b_assets.default_for("extracted_v04") == "strict_v5"
    with pytest.raises(ValueError, match="no tier-1b asset set"):
        tier1b_assets.resolve(run_config=rc.name)
    return spec


def test_driver_sidecars_and_bundle_load(tmp_path, native_config, inputs, monkeypatch):
    outdir = tmp_path / "new-library"
    donors, generators = _patterns(tmp_path)
    monkeypatch.setattr("sys.argv", ["build_native_library.py", "--donors", donors, "--generators", generators,
        "--run-config", native_config.name, "--shards", "101", "--outdir", str(outdir)])
    _driver().main()
    spec = _check_bundle(outdir, native_config, inputs)
    before = {path: tier1b_assets.sha256(path) for path in spec.paths()}
    with pytest.raises(FileExistsError):
        _driver().run(donors, generators, outdir, native_config.name, [101])
    assert before == {path: tier1b_assets.sha256(path) for path in spec.paths()}


def test_relative_patterns_are_canonical_across_driver_and_sidecars(tmp_path, native_config, inputs, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _driver().run("crvsteps.{s}.root", "gen.{s}.root", "relative-library", native_config.name, [101])
    _check_bundle(tmp_path / "relative-library", native_config, inputs)


def test_declared_shard_cannot_override_actual_subrun(tmp_path, native_config):
    donor, full = _frames(native_config.geometry(), subrun=102)
    _write(tmp_path / "crvsteps.001400_00000101.root", donor)
    _write(tmp_path / "gen.001400_00000101.root", full)
    outdir = tmp_path / "wrong-shard-library"
    with pytest.raises(ValueError, match="declared_shards do not match actual run/subrun"):
        _driver().run(*_patterns(tmp_path), outdir, native_config.name, [101])
    assert not outdir.exists()


def test_existing_output_is_refused_without_touching_it(tmp_path, native_config):
    outdir = tmp_path / "occupied"
    outdir.mkdir()
    sentinel = outdir / "keep.txt"
    sentinel.write_text("untouched\n")
    with pytest.raises(FileExistsError):
        _driver().run(*_patterns(tmp_path), outdir, native_config.name, [101])
    assert list(outdir.iterdir()) == [sentinel] and sentinel.read_text() == "untouched\n"
