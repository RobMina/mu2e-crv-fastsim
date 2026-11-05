"""Saved payload comparison uses real small ROOT/NPZ; attachment/load is synthetic."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import uproot

from crvfastsim import runconfig, tier1b, tier1b_assets as T
from crvfastsim.analyses.context_audit import content_fingerprint


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_native_rebuild", ROOT / "scripts/check_native_rebuild.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def write_root(path, dtype="float64", values=(0., 1., 2.), omit=None):
    with uproot.recreate(path) as root:
        for name in tier1b.LIB_TREES:
            columns = dict(value=np.asarray(values, dtype=dtype), index=np.arange(len(values), dtype=np.int64))
            if name == omit:
                columns.pop("index")
            root.mktree("dump/" + name, {key: value.dtype for key, value in columns.items()})
            root["dump/" + name].extend(columns)


def refresh(directory):
    digest = T.sha256(directory / "library.root")
    check = json.loads((directory / "build_check.json").read_text())
    check["library_sha256"] = digest
    (directory / "build_check.json").write_text(json.dumps(check))
    for name in ("topology.npz", "templates_v3.npz"):
        path = directory / (name + ".json")
        meta = json.loads(path.read_text())
        meta.update(library_sha256=digest, sidecar_sha256=T.sha256(directory / name))
        path.write_text(json.dumps(meta))


@pytest.fixture
def bundles(tmp_path, monkeypatch):
    monkeypatch.delenv("CRVFS_PREPARED_LIBRARY_CACHE", raising=False)
    geom = SimpleNamespace(nbars=2, sector_names=["T1"], sector=np.zeros(2, int), lib_group=np.array(["EX"]))
    counters = tmp_path / "counters.txt"
    counters.write_text("synthetic local counters\n")
    monkeypatch.setattr(runconfig.RunConfig, "counters_path", lambda self: str(counters))
    monkeypatch.setattr(runconfig.RunConfig, "geometry", lambda self: copy.deepcopy(geom))
    rc = runconfig.get("run1a_v01")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    write_root(candidate / "library.root")
    np.savez(candidate / "topology.npz", crossings_bar=np.array([0, 1], dtype=np.int64),
             flows=np.array([2, 3, 4, 5], dtype=np.int64), extra=np.arange(6).reshape(2, 3))
    np.savez(candidate / "templates_v3.npz", xing_p=np.array([1., 2., 3.]),
             empty=np.array([], dtype=np.float32), metadata=np.array("not numeric metadata"))
    donors, generators = ["donor.root"], ["generator.root"]
    actual = [dict(run=1400, subrun=101)]
    from crvfastsim import native_training
    training = dict(schema=native_training.SCHEMA, receipt_version=native_training.RECEIPT_VERSION,
                    total_exposure=17, full_generator_events=dict(count=17, sha256="a" * 64),
                    actual_shards=actual, declared_shards=actual, census_admitted=True,
                    per_stage_associations_validated=True, cross_stream_generator_pointer_join="not_established",
                    shared_generator_provenance="not_established_by_census", physics_admitted=False,
                    pairs=[dict(steps=dict(path=donors[0], sha256="b" * 64),
                                generator=dict(path=generators[0], sha256="c" * 64), generator_events=dict(count=17))])
    from crvfastsim import native_construction
    counts = dict(primary_muons=1, events_with_muons=1, max_muons_per_event=1, multiple_muon_events=0)
    construction = dict(schema=native_construction.SCHEMA, supported=True, max_supported_primary_muons_per_event=1,
        events_filtered=0, physics_admitted=False, input_sha256={donors[0]: "b" * 64, generators[0]: "c" * 64},
        pairs=[dict(steps=donors[0], generator=generators[0], steps_multiplicity=counts, generator_multiplicity=counts)])
    manifest = dict(run_config=rc.name, generated_scaled_to=17, inputs=donors, gen=generators,
                    counters_sha256=T.sha256(counters), geometry_fingerprint=content_fingerprint(geom, workers=1),
                    training_census=training, construction_support=construction,
                    generator_associations=[dict(path=donors[0], schema_version=2, associations_checked=True,
                                                 stream_role="checked donor")],
                    generator_streams=[dict(path=generators[0], schema_version=2, counts=dict(events=17),
                                            stream_role="full generator stream")])
    (candidate / "library.json").write_text(json.dumps(manifest))
    (candidate / "build_check.json").write_text(json.dumps(dict(run_config=rc.name, generated=17, shards=[101],
        training_census=training, construction_support=construction,
        anchor=dict(n_roots=99, n_multi=98, forward_edep_share=0.75))))
    for name in ("topology.npz", "templates_v3.npz"):
        meta = dict(geometry_sha256=T.sha256(counters), donors=[dict(path=donors[0], sha256="b" * 64)])
        if name == "templates_v3.npz":
            meta.update(generated=17, census_events=17,
                        census=[dict(steps=donors[0], gen=generators[0], steps_sha256="b" * 64, gen_sha256="c" * 64)])
        (candidate / (name + ".json")).write_text(json.dumps(meta))
    refresh(candidate)
    reference = tmp_path / "reference"
    shutil.copytree(candidate, reference)
    for name in ("library.json", "build_check.json"):
        path = reference / name
        obj = json.loads(path.read_text())
        obj.pop("training_census")
        path.write_text(json.dumps(obj))
    seen = []
    def anchor(lib, geom):
        return {**lib, "anchor": dict(axid=lib["libdesc"].index.to_numpy(), n_roots=1, n_multi=0,
                                       forward_edep_share=0.)}
    def load(spec, geom):
        seen.append(spec)
        assert spec.native["training_census"]["count"] == 17
        return anchor(tier1b.read_library(spec.library), geom)
    monkeypatch.setattr(T, "load", load)
    monkeypatch.setattr(tier1b, "attach_topology", lambda lib, *args: lib)
    monkeypatch.setattr(tier1b, "attach_templates", lambda lib, *args: lib)
    monkeypatch.setattr(tier1b, "attach_anchors", anchor)
    return reference, candidate, rc, seen


def invoke(bundles, tmp_path):
    a, b, rc, _ = bundles
    return D.run(a, b, rc.name, tmp_path / "report.json")


def test_strict_candidate_only_and_old_anchor_declarations(bundles, tmp_path):
    a, b, _, seen = bundles
    # Recontainerization changes the ROOT hash but not the numeric payload.
    write_root(b / "library.root")
    refresh(b)
    assert T.sha256(a / "library.root") != T.sha256(b / "library.root")
    result = invoke(bundles, tmp_path)
    assert len(seen) == 1 and Path(seen[0].library).parent == b
    assert result["identical_numeric_payload"] and result["mismatch_count"] == 0
    assert result["candidate_strictly_loaded"] and not result["reference_admitted"]
    assert not result["physics_admitted"]
    assert result["reference_build_anchor_declared"]["n_roots"] == 99
    assert result["current_code_anchor_counts"]["reference"]["n_roots"] == 1
    assert result["hashes_before"] == result["hashes_after"]
    assert len([key for key in result["hashes_before"] if key.startswith("reference/")]) == 7
    assert "source/scripts/build_native_library.py" in result["hashes_before"]
    assert result["npz_metadata_excluded"]["candidate"]["templates_v3.npz"] == ["metadata"]
    assert json.loads((tmp_path / "report.json").read_text())["identical_numeric_payload"]


@pytest.mark.parametrize("change", ["dtype", "values", "signedzero", "missingfield", "shape", "flows", "newvariable"])
def test_real_saved_numeric_mismatches(bundles, tmp_path, change):
    _, b, _, _ = bundles
    if change in ("dtype", "values", "signedzero", "missingfield"):
        write_root(b / "library.root", dtype="float32" if change == "dtype" else "float64",
                   values=(-0., 1., 2.) if change == "signedzero" else (0., 9., 2.) if change == "values" else (0., 1., 2.),
                   omit="libdesc" if change == "missingfield" else None)
        expected = "library/libdesc/" + ("index" if change == "missingfield" else "value")
    else:
        path = b / "topology.npz"
        with np.load(path) as archive:
            data = dict(archive)
        if change == "shape":
            data["extra"] = data["extra"].reshape(3, 2)
            expected = "topology.npz/extra"
        elif change == "flows":
            data["flows"][-1] += 1  # Overflow is not silently discarded.
            expected = "topology.npz/flows"
        else:
            data["newvariable"] = np.array([1])
            expected = "topology.npz/newvariable"
        np.savez(path, **data)
    refresh(b)
    result = invoke(bundles, tmp_path)
    assert not result["identical_numeric_payload"]
    assert expected in {row["name"] for row in result["mismatches"]}
    assert not result["physics_admitted"]


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("name", D.BUNDLE_FILES)
def test_corrupt_or_missing_integrity_is_fatal(bundles, tmp_path, arm, name):
    directory = bundles[arm]
    # Missing any one of the seven files must be fatal before publication.
    (directory / name).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        invoke(bundles, tmp_path)
    assert not (tmp_path / "report.json").exists()


def test_stale_sidecar_hash_is_fatal(bundles, tmp_path):
    a, _, _, _ = bundles
    with (a / "topology.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        invoke(bundles, tmp_path)


def test_candidate_missing_current_census_is_not_diagnostic_fallback(bundles, tmp_path):
    _, b, _, seen = bundles
    path = b / "library.json"
    obj = json.loads(path.read_text())
    obj.pop("training_census")
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="training census"):
        invoke(bundles, tmp_path)
    assert not seen and not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("key", ["run_config", "counters_sha256", "geometry_fingerprint"])
def test_reference_build_binding_required(bundles, tmp_path, key):
    a, _, _, _ = bundles
    path = a / "library.json"
    obj = json.loads(path.read_text())
    obj[key] = "wrong"
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="reference"):
        invoke(bundles, tmp_path)


def test_drift_during_load_is_fatal(bundles, tmp_path, monkeypatch):
    a, _, _, _ = bundles
    original = T.load
    def drift(*args):
        lib = original(*args)
        with (a / "library.json").open("a") as stream:
            stream.write(" ")
        return lib
    monkeypatch.setattr(T, "load", drift)
    with pytest.raises(ValueError, match="changed during diagnostic"):
        invoke(bundles, tmp_path)
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_symlink"])
def test_exclusive_output(tmp_path, monkeypatch, kind):
    path = tmp_path / "occupied"
    if kind == "file":
        path.write_text("keep")
    elif kind == "directory":
        path.mkdir()
    else:
        path.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(D, "source_paths", lambda: pytest.fail("read inputs before exclusive-output check"))
    with pytest.raises(FileExistsError):
        D.run("missing-reference", "missing-candidate", "run1a_v01", path)
    if kind == "file":
        assert path.read_text() == "keep"


def test_source_drift_checked(bundles, tmp_path, monkeypatch):
    # Mutate only a temporary synthetic source, never real builder sources.
    source = tmp_path / "critical.py"
    source.write_text("before")
    monkeypatch.setattr(D, "source_paths", lambda: {"critical.py": source})
    original = T.load
    def drift(*args):
        source.write_text("after")
        return original(*args)
    monkeypatch.setattr(T, "load", drift)
    with pytest.raises(ValueError, match="changed during diagnostic"):
        invoke(bundles, tmp_path)


def test_anchor_arrays_are_compared(bundles, tmp_path, monkeypatch):
    original = T.load
    def changed(*args):
        lib = original(*args)
        lib["anchor"]["axid"] = lib["anchor"]["axid"].astype(np.int32)
        return lib
    monkeypatch.setattr(T, "load", changed)
    result = invoke(bundles, tmp_path)
    assert [row["name"] for row in result["mismatches"]] == ["anchor/axid"]


def test_primitive_fingerprint_retains_nan_bits_and_empty_shape():
    a = np.array([0x7ff8000000000001], dtype=np.uint64).view(np.float64)
    b = np.array([0x7ff8000000000002], dtype=np.uint64).view(np.float64)
    assert D.array_identity(a) != D.array_identity(b)
    assert D.array_identity(np.empty((0, 2))) != D.array_identity(np.empty((0, 3)))
    with pytest.raises(ValueError, match="non-primitive"):
        D.array_identity(np.array([object()]))


def test_integer_storage_change_keeps_strict_mismatch_and_reports_exact_values(bundles, tmp_path):
    _, candidate, _, _ = bundles
    path = candidate / 'topology.npz'
    with np.load(path) as archive:
        data = dict(archive)
    data['crossings_bar'] = data['crossings_bar'].astype(np.int32)
    np.savez(path, **data)
    refresh(candidate)
    result = invoke(bundles, tmp_path)
    assert not result['identical_numeric_payload'] and result['mismatch_count'] == 1
    row = result['integer_storage_diagnostics'][0]
    assert row['name'] == 'topology.npz/crossings_bar'
    assert row['exact_int64_values_equal'] is True
    assert row['reference_dtype'] == '<i8' and row['candidate_dtype'] == '<i4'
