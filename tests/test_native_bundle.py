"""Explicit native bundle integrity and ordinary chain wiring; no physics certification."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from crvfastsim import runconfig, tier1b_assets as T
from crvfastsim.analyses import runchain as R
from test_e2e_bench import pipeline


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    from types import SimpleNamespace
    counters = tmp_path / "counters.txt"
    counters.write_text("synthetic counters\n")
    geom = SimpleNamespace(nbars=2, sector_names=["T1"], sector=np.zeros(2, int), lib_group=np.array(["EX"]))
    monkeypatch.setattr(runconfig.RunConfig, "counters_path", lambda self: str(counters))
    monkeypatch.setattr(runconfig.RunConfig, "geometry", lambda self: copy.deepcopy(geom))
    rc = runconfig.get("run1a_v01")
    directory = tmp_path / "library"
    directory.mkdir()
    paths = {name: directory / name for name in ("library.root", "topology.npz", "templates_v3.npz")}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    donors, generators = ["training-donor.root"], ["training-generator.root"]
    from crvfastsim.analyses.context_audit import content_fingerprint
    manifest = {"run_config": rc.name, "generated_scaled_to": 17, "inputs": donors, "gen": generators,
                "counters_sha256": T.sha256(counters), "geometry_fingerprint": content_fingerprint(geom, workers=1),
                "generator_associations": [{"schema_version": 2, "associations_checked": True,
                                             "path": donors[0], "stream_role": "checked donor"}],
                "generator_streams": [{"schema_version": 2, "path": generators[0],
                                        "stream_role": "full generator stream", "counts": {"events": 17}}]}
    from crvfastsim import native_training
    actual = [{"run": 1400, "subrun": 101}]
    training = dict(schema=native_training.SCHEMA, receipt_version=native_training.RECEIPT_VERSION,
                    total_exposure=17, full_generator_events=dict(count=17, sha256="a" * 64), actual_shards=actual,
                    declared_shards=actual, census_admitted=True, per_stage_associations_validated=True,
                    cross_stream_generator_pointer_join="not_established",
                    shared_generator_provenance="not_established_by_census", physics_admitted=False,
                    pairs=[dict(steps=dict(path=donors[0], sha256="b" * 64),
                                generator=dict(path=generators[0], sha256="c" * 64), generator_events=dict(count=17))])
    manifest["training_census"] = copy.deepcopy(training)
    from crvfastsim import native_construction
    counts = dict(primary_muons=1, events_with_muons=1, max_muons_per_event=1, multiple_muon_events=0)
    construction = dict(schema=native_construction.SCHEMA, supported=True, max_supported_primary_muons_per_event=1,
        events_filtered=0, physics_admitted=False, input_sha256={donors[0]: "b" * 64, generators[0]: "c" * 64},
        pairs=[dict(steps=donors[0], generator=generators[0], steps_multiplicity=counts, generator_multiplicity=counts)])
    manifest["construction_support"] = copy.deepcopy(construction)
    (directory / "library.json").write_text(json.dumps(manifest))
    check = {"run_config": rc.name, "generated": 17, "shards": [101], "library_sha256": T.sha256(paths["library.root"]),
             "training_census": training, "construction_support": copy.deepcopy(construction)}
    (directory / "build_check.json").write_text(json.dumps(check))
    for name in ("topology.npz", "templates_v3.npz"):
        meta = {"library_sha256": check["library_sha256"], "sidecar_sha256": T.sha256(paths[name]),
                "geometry_sha256": T.sha256(counters), "donors": [{"path": donors[0], "sha256": "b" * 64}]}
        if name == "templates_v3.npz":
            meta.update(generated=17, census_events=17, census=[{"steps": donors[0], "gen": generators[0],
                        "steps_sha256": "b" * 64, "gen_sha256": "c" * 64}])
        Path(str(paths[name]) + ".json").write_text(json.dumps(meta))
    return directory, rc, geom


def alter(path, key, value):
    meta = json.loads(path.read_text())
    meta[key] = value
    path.write_text(json.dumps(meta))


def test_explicit_bundle_does_not_install_a_default(bundle):
    directory, rc, geom = bundle
    spec = T.native_bundle(directory, rc)
    assert spec.name is None and T.default_for(rc.name) is None
    assert sorted(T.SETS) == ["strict_v4", "strict_v5"]
    assert spec.options == dict(T._STRICT, chord=True, anchor=True, showers=True)
    assert spec.native["anchor_scope"] == "module"
    assert len(spec.paths()) == 7
    assert T.resolve(library=spec.library, run_config=rc.name).options == {}
    T._check_native(spec, geom)
    geom.lib_group[:] = "T1"
    with pytest.raises(ValueError, match="geometry or group map"):
        T._check_native(spec, geom)


@pytest.mark.parametrize("name", ["extracted_v04", "run2_v10"])
def test_bundle_requires_matching_named_configuration(bundle, name):
    directory, rc, geom = bundle
    with pytest.raises(ValueError, match="named run configuration|configuration mismatch"):
        T.native_bundle(directory, runconfig.get(name))


@pytest.mark.parametrize("file,key,value,match", [
    ("build_check.json", "library_sha256", "0" * 64, "does not bind"),
    ("build_check.json", "generated", 18, "exposure mismatch"),
    ("build_check.json", "shards", [101, 101], "distinct integer"),
    ("library.json", "generator_associations", [{"schema_version": 1}], "schema-v2"),
    ("library.json", "generator_streams", [{"schema_version": 2, "path": "wrong.root"}], "declared training"),
    ("library.json", "geometry_fingerprint", "0" * 64, "build-time geometry"),
    ("library.json", "counters_sha256", "0" * 64, "build-time geometry"),
    ("topology.npz.json", "geometry_sha256", "0" * 64, "counters geometry"),
    ("templates_v3.npz.json", "library_sha256", "0" * 64, "content mismatch"),
    ("topology.npz.json", "donors", [{"path": "held-out.root"}], "donor population"),
    ("templates_v3.npz.json", "census", [{"steps": "held-out.root", "gen": "held-out-gen.root"}], "training population"),
])
def test_bundle_rejects_mixed_evidence(bundle, file, key, value, match):
    directory, rc, geom = bundle
    alter(directory / file, key, value)
    with pytest.raises(ValueError, match=match):
        T.native_bundle(directory, rc)


@pytest.mark.parametrize("file", ["library.root", "topology.npz", "topology.npz.json", "templates_v3.npz",
                                 "templates_v3.npz.json", "library.json", "build_check.json"])
def test_load_rejects_drift_before_reading(bundle, monkeypatch, file):
    directory, rc, geom = bundle
    spec = T.native_bundle(directory, rc)
    with (directory / file).open("ab") as stream:
        stream.write(b"drift")
    monkeypatch.setattr("crvfastsim.tier1b.read_library", lambda *a: pytest.fail("loaded changed bundle"))
    with pytest.raises(ValueError, match="changed since resolution"):
        T.load(spec, geom)


def test_load_attaches_in_order_and_checks_root_support(bundle, monkeypatch):
    from crvfastsim.analyses import tier1b as A
    directory, rc, geom = bundle
    spec = T.native_bundle(directory, rc)
    seen = []
    monkeypatch.setattr("crvfastsim.tier1b.read_library", lambda *a: {})
    monkeypatch.setattr(A, "read_topology_sidecar", lambda *a, **kw: (None, None))
    monkeypatch.setattr(A, "read_template_sidecar", lambda *a, **kw: None)
    monkeypatch.setattr("crvfastsim.tier1b.attach_topology", lambda lib, *a: seen.append("topology") or lib)
    monkeypatch.setattr("crvfastsim.tier1b.attach_templates", lambda lib, *a: seen.append("templates") or lib)
    def anchors(lib, geom):
        seen.append("anchors")
        raise ValueError("anchors need a library built with roots")
    monkeypatch.setattr("crvfastsim.tier1b.attach_anchors", anchors)
    with pytest.raises(ValueError, match="built with roots"):
        T.load(spec, geom)
    assert seen == ["topology", "templates", "anchors"]


@pytest.mark.parametrize("extra", [{"library": "x"}, {"tier1b_set": "strict_v5"}, {"tier1b_dir": "x"}])
def test_resolver_rejects_ambiguous_choice(bundle, extra):
    directory, rc, geom = bundle
    opts = dict(library=None, calib="nominal", deviation=None, gdml="world", jax=False,
                counters=None, run_config=rc.name, window=None, native_library_dir=str(directory))
    with pytest.raises(SystemExit, match="cannot be combined"):
        R._resolve(**{**opts, **extra})


def test_ordinary_chain_forwards_native_anchors_and_native_showers(pipeline, bundle, monkeypatch, tmp_path):
    directory, rc, geom = bundle
    seen = []
    monkeypatch.setattr(T, "load", lambda spec, g: seen.append(spec) or None)
    monkeypatch.setattr(runconfig.RunConfig, "counters_cfg",
                        lambda self: {"crs.sipmsAtSide0T1": True, "crs.sipmsAtSide1T1": True})
    monkeypatch.setattr(R.tr, "load_below", lambda *a, **kw: None)
    monkeypatch.setattr(runconfig.RunConfig, "tagger_bar", lambda *a: None)
    monkeypatch.setattr(runconfig, "sector_type_names", lambda *a: {1: "T1"})
    options = []
    monkeypatch.setattr(R, "_native_generator_input", lambda *a: (pipeline["gen"], pipeline["gen"]))
    monkeypatch.setattr(R.tier1b, "sample", lambda st, pa, *a, **kw: options.append(kw) or (st, pa, None))
    result = R.run(pipeline["inputs"], str(tmp_path / "out"), calib="nominal", gdml="world",
                   run_config=rc.name, native_library_dir=str(directory))
    assert options == [dict(T._STRICT, chord=True, anchor=True, showers=True)]
    assert result["tier1b"]["native"] == seen[0].native
    assert result["tier1b"]["options"]["showers"] is True
    assert rc.showers is False


def test_bundle_rejected_with_context_before_resolution(monkeypatch):
    monkeypatch.setattr(R, "_resolve", lambda *a, **kw: pytest.fail("resolved before rejection"))
    for extra in ({"prepared": object()}, {"benchmark": object()}):
        with pytest.raises(ValueError, match="native library bundles are not supported"):
            R.run([], "unused", native_library_dir="bundle", **extra)


def test_cli_exposes_explicit_not_default_bundle():
    from crvfastsim.cli import build_parser
    args = build_parser().parse_args(["runchain", "gen.root", "--run-config", "run1a_v01",
                                      "--native-library-dir", "/assets/native"])
    assert args.native_library_dir == "/assets/native" and args.tier1b_set is None and args.library is None


def test_native_input_uses_full_census_with_empty_events(bundle, monkeypatch):
    import pandas as pd
    from crvfastsim import genassoc
    directory, rc, geom = bundle
    spec = T.native_bundle(directory, rc)
    universe = pd.DataFrame(dict(run=[1400] * 3, subrun=[102] * 3, event=[1, 2, 3]))
    gen = universe.iloc[[0, 0, 2]].assign(idx=[0, 1, 0])
    monkeypatch.setattr(genassoc, 'validate_generator_stream', lambda p: dict(schema_version=2, stream_role='full generator stream'))
    monkeypatch.setattr(genassoc, 'generator_event_keys', lambda p: universe.copy())
    monkeypatch.setattr(R.tables, 'read', lambda *a: gen.copy())
    rows, census = R._native_generator_input('gen.root', spec)
    assert len(census) == 3 and len(rows) == 3
    rows, census = R._native_generator_input('gen.root', spec, 2)
    assert census.event.tolist() == [1, 2] and rows.event.tolist() == [1, 1]
    universe.loc[:, 'subrun'] = 101
    with pytest.raises(ValueError, match='overlaps training'):
        R._native_generator_input('gen.root', spec)


@pytest.mark.parametrize('change,match', [('legacy', 'schema-v2'), ('runs', 'one run'), ('capacity', '20-bit'),
                                         ('feb', 'FEB-label')])
def test_native_input_rejects_unchecked_or_ambiguous_population(bundle, monkeypatch, change, match):
    import pandas as pd
    from crvfastsim import genassoc
    directory, rc, geom = bundle
    spec = T.native_bundle(directory, rc)
    universe = pd.DataFrame(dict(run=[1400, 1400], subrun=[102, 102], event=[1, 2]))
    if change == 'runs':
        universe.loc[1, 'run'] = 1401
    if change == 'capacity':
        universe.loc[1, 'event'] = 2**20
    if change == 'feb':
        universe.loc[1, 'subrun'] = 2**24
    report = dict(schema_version=None if change == 'legacy' else 2, stream_role='full generator stream')
    monkeypatch.setattr(genassoc, 'validate_generator_stream', lambda p: report)
    monkeypatch.setattr(genassoc, 'generator_event_keys', lambda p: universe)
    with pytest.raises(ValueError, match=match):
        R._native_generator_input('gen.root', spec)


def test_native_record_captures_effective_transport_flags(pipeline, bundle, monkeypatch, tmp_path):
    from crvfastsim import native_record, named_conditions
    directory, rc, geom = bundle
    monkeypatch.setattr(T, 'load', lambda *a: None)
    monkeypatch.setattr(runconfig.RunConfig, 'counters_cfg',
                        lambda self: {'crs.sipmsAtSide0T1': True, 'crs.sipmsAtSide1T1': True})
    monkeypatch.setattr(runconfig.RunConfig, 'tagger_bar', lambda *a: None)
    monkeypatch.setattr(runconfig, 'sector_type_names', lambda *a: {1: 'T1'})
    monkeypatch.setattr(named_conditions, 'load_conditions', lambda *a: (runconfig.nominal_conditions(geom), {}))
    monkeypatch.setattr(R.tr, 'load_below', lambda *a, **kw: (object(), None))
    monkeypatch.setattr(R, '_native_generator_input', lambda *a: (pipeline['gen'], pipeline['gen']))
    monkeypatch.setattr(R.tier1b, 'sample', lambda st, pa, *a, **kw: (st, pa, {name: 0 for name in
                        ('desc_steps', 'desc_dropped', 'desc_clamped', 'showers', 'shower_steps', 'nonmu_primaries')}))
    monkeypatch.setattr(native_record, 'snapshot', lambda *a: {})
    seen = []
    monkeypatch.setattr(native_record, 'write', lambda *a, **kw: seen.append(kw) or {})
    gdml = tmp_path / 'world.gdml'
    gdml.write_text('synthetic')
    R.run(pipeline['inputs'], str(tmp_path / 'result'), run_config=rc.name, native_library_dir=str(directory),
          conditions_manifest='synthetic', gdml=str(gdml), native_output=str(tmp_path / 'out.npz'),
          below_msc=True, below_margin=4000., seed=7)
    assert seen[0]['seed'] == 7
    assert seen[0]['effective_transport'] == dict(range_out=False, below_msc=True, below_margin=4000.,
                                                backend='numpy', gdml_sha256=T.sha256(gdml))


@pytest.mark.parametrize('change', ['missing', 'exposure', 'actual', 'claim', 'hash', 'paths'])
def test_training_census_is_build_bound_and_scope_limited(bundle, change):
    directory, rc, _ = bundle
    path = directory / 'library.json'
    meta = json.loads(path.read_text())
    if change == 'missing':
        meta.pop('training_census')
    elif change == 'exposure':
        meta['training_census']['total_exposure'] = 18
    elif change == 'actual':
        meta['training_census']['actual_shards'][0]['subrun'] = 102
    elif change == 'claim':
        meta['training_census']['cross_stream_generator_pointer_join'] = 'established'
    elif change == 'hash':
        meta['training_census']['pairs'][0]['steps']['sha256'] = 'not-a-hash'
    else:
        meta['training_census']['pairs'][0]['generator']['path'] = 'heldout.root'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='training census'):
        T.native_bundle(directory, rc)


def test_training_overlap_is_by_actual_run_and_subrun(bundle, monkeypatch):
    import pandas as pd
    from crvfastsim import genassoc
    directory, rc, _ = bundle
    spec = T.native_bundle(directory, rc)
    universe = pd.DataFrame(dict(run=[1401], subrun=[101], event=[1]))
    gen = universe.assign(idx=0)
    monkeypatch.setattr(genassoc, 'validate_generator_stream', lambda p: dict(schema_version=2, stream_role='full generator stream'))
    monkeypatch.setattr(genassoc, 'generator_event_keys', lambda p: universe.copy())
    monkeypatch.setattr(R.tables, 'read', lambda *a: gen.copy())
    _, keys = R._native_generator_input('gen.root', spec)
    assert keys.run.tolist() == [1401]
    universe.loc[:, 'run'] = 1400
    with pytest.raises(ValueError, match='overlaps training'):
        R._native_generator_input('gen.root', spec)


@pytest.mark.parametrize('file,kind', [('topology.npz.json', 'donor'), ('templates_v3.npz.json', 'donor'),
                                     ('templates_v3.npz.json', 'steps'), ('templates_v3.npz.json', 'gen')])
def test_sidecar_source_hashes_bind_the_build_training_population(bundle, file, kind):
    directory, rc, _ = bundle
    path = directory / file
    meta = json.loads(path.read_text())
    if kind == 'donor':
        meta['donors'][0]['sha256'] = 'd' * 64
    else:
        meta['census'][0][kind + '_sha256'] = 'd' * 64
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='hashes differ from build-time training'):
        T.native_bundle(directory, rc)
