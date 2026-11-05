"""Conditional source-key software evidence, not proof of a source relationship."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import native_identity
from test_native_builder_integration import _frames, _write

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit():
    spec = importlib.util.spec_from_file_location("native_source_candidates", ROOT / "scripts/audit_native_source_candidates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gen(rows):
    frame = pd.DataFrame(rows, columns=["event", "idx", "pdg"], dtype=np.int64)
    return frame.assign(run=np.int64(1400), subrun=np.int64(101), genid=np.int64(1),
                        e=200., t=0., x=1., y=2., z=3., px=4., py=5., pz=6.)


def _parts(rows):
    frame = pd.DataFrame(rows, columns=["event", "key", "parent", "genidx", "isprim", "pdg"], dtype=np.int64)
    return frame.assign(run=np.int64(1400), subrun=np.int64(101))


def _assert_not_admitted(result):
    for key in ("physics_admitted", "source_identity_established", "filter_completeness_established"):
        assert result[key] is False


def test_compression_reindex_not_source_idx_all_pdgs_and_two_primaries(audit):
    parts = _parts([(1, 8, -1, 0, 1, 13), (1, 10, -1, 1, 1, 22), (1, 11, 8, 0, 0, 11)])
    donor = _gen([(1, 0, 13), (1, 1, 22), (1, 2, 13)])
    full = _gen([(1, 7, 13), (1, 9, 22), (1, 0, 11)])
    original = [frame.copy(deep=True) for frame in (parts, donor, full)]
    result = audit.analyze(parts, donor, full)
    _assert_not_admitted(result)
    assert result["selected_primaries"] == result["source_idx_differs_from_donor_genidx"] == 2
    assert result["selected_primary_muons"] == result["multi_primary_events"] == 1
    assert result["payload_exact_equal"] == result["distinct_full_generator_candidates"] == 2
    assert result["full_generator_rows"] == 3
    assert result["unreferenced_full_generator_rows"] == 1
    assert result["repeated_candidate_references"] == 0
    for before, after in zip(original, (parts, donor, full)):
        pd.testing.assert_frame_equal(before, after)
    with pytest.raises((ValueError, NotImplementedError)):
        native_identity.require_full_nearmiss_bridge()


def test_adjacent_large_integers_in_all_identity_dimensions(audit):
    large = 2**53 + 20
    parts = _parts([(large, large + 1, -1, large, 1, 13),
                    (large, large + 2, -1, large + 1, 1, -13),
                    (large + 1, large + 1, -1, large, 1, 22)])
    donor = _gen([(large, large, 13), (large, large + 1, -13), (large + 1, large, 22)])
    full = donor.copy(deep=True)
    for frame in (parts, donor, full):
        frame["run"] = np.int64(large)
        frame["subrun"] = np.int64(large + 1)
    result = audit.analyze(parts, donor, full)
    assert result["payload_exact_equal"] == result["distinct_full_generator_candidates"] == 3
    full = full.iloc[[0, 2]]
    result = audit.analyze(parts, donor, full)
    assert result["missing_source_candidate"] == 1
    assert result["payload_exact_equal"] == 2


@pytest.mark.parametrize("field", ["pdg", "genid", "e", "t", "x", "y", "z", "px", "py", "pz"])
def test_exact_mismatch_and_missing_candidates_never_substitute(audit, field):
    parts = _parts([(1, 8, -1, 0, 1, 13), (1, 9, -1, 1, 1, 13)])
    donor = _gen([(1, 0, 13), (1, 1, 13)])
    full = _gen([(1, 7, 13), (1, 0, 13)])
    old = full.loc[0, field]
    full.loc[0, field] = old + 1 if field in ("pdg", "genid") else np.nextafter(old, np.inf)
    result = audit.analyze(parts, donor, full)
    assert result["missing_source_candidate"] == result["payload_mismatched"] == 1
    assert result["payload_exact_equal"] == 0
    assert result["payload_mismatches_by_field"][field] == 1
    assert sum(result["payload_mismatches_by_field"].values()) == 1


@pytest.mark.parametrize("scope", ["run", "subrun", "event"])
def test_candidate_and_local_pointer_scopes_use_entire_event(audit, scope):
    parts = _parts([(1, 8, -1, 0, 1, 13)])
    donor = _gen([(1, 0, 13)])
    full = _gen([(1, 7, 13)])
    full[scope] += 1
    assert audit.analyze(parts, donor, full)["missing_source_candidate"] == 1
    donor[scope] += 1
    assert audit.analyze(parts, donor, full)["missing_stage_local_generator"] == 1


@pytest.mark.parametrize("column,value", [("key", 0), ("key", -1), ("parent", 3), ("genidx", -1), ("isprim", 2)])
def test_invalid_actual_primary_refused(audit, column, value):
    parts = _parts([(1, 8, -1, 0, 1, 13)])
    parts[column] = value
    with pytest.raises(ValueError):
        audit.analyze(parts, _gen([(1, 0, 13)]), _gen([(1, 7, 13)]))


def test_float_and_duplicate_identities_refused(audit):
    parts = _parts([(1, 8, -1, 0, 1, 13)])
    donor, full = _gen([(1, 0, 13)]), _gen([(1, 7, 13)])
    for frame, column in ((parts, "key"), (donor, "idx"), (full, "event")):
        bad = frame.copy()
        bad[column] = bad[column].astype(float)
        args = [bad if f is frame else f for f in (parts, donor, full)]
        with pytest.raises(ValueError, match="must be integer"):
            audit.analyze(*args)
    for i, frame in enumerate((parts, donor, full)):
        args = [parts, donor, full]
        args[i] = pd.concat([frame, frame], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate keys"):
            audit.analyze(*args)


def test_large_adjacent_integer_payload_is_compared_without_float(audit):
    parts = _parts([(1, 8, -1, 0, 1, 13)])
    donor, full = _gen([(1, 0, 13)]), _gen([(1, 7, 13)])
    donor["genid"] = np.int64(2**53)
    full["genid"] = np.int64(2**53 + 1)
    result = audit.analyze(parts, donor, full)
    assert result["payload_mismatches_by_field"]["genid"] == result["payload_mismatched"] == 1


def test_no_primary_is_valid_and_does_not_select_parentless_nonprimary(audit):
    result = audit.analyze(_parts([(1, 8, -1, 0, 0, 13)]), _gen([(1, 0, 13)]), _gen([(1, 7, 13)]))
    assert result["selected_primaries"] == result["payload_compared"] == 0


@pytest.fixture
def inputs(tmp_path):
    geom = SimpleNamespace(bar_index=lambda *args: 0, center=np.zeros((1, 3)),
                           local_to_world=np.array([[0, 1, 2]]), halflen=np.ones((1, 3)))
    donor, full = _frames(geom)
    donor["particles"]["key"] = np.array([1, 2, 1], dtype=np.int64)
    donor["particles"]["parent"] = np.array([-1, 1, -1], dtype=np.int64)
    donor["steps"]["sp"] = np.array([1, 2, 1], dtype=np.int64)
    alias = donor["gen"].iloc[[0]].copy().assign(idx=np.int64(1))
    donor["gen"] = pd.concat([donor["gen"], alias], ignore_index=True)
    donor["particles"].loc[0, "genidx"] = 1
    donor["events"].loc[donor["events"].event.eq(1), "n_gen"] = 2
    paths = [_write(tmp_path / "donor.root", donor), _write(tmp_path / "full.root", full)]
    return paths, donor, full


def test_real_schema2_aliases_probe_rejected_events_hashes_and_exclusive_output(audit, inputs, tmp_path):
    paths, _, _ = inputs
    before = {str(path): audit._sha256(path) for path in paths}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(donors=[str(paths[0])], generators=[str(paths[1])],
                                     probe_events=[[1400, 101, 5], [1400, 101, 3], [1400, 101, 16483]])))
    out = tmp_path / "report.json"
    result = audit.run(config, out)
    assert json.loads(out.read_text()) == result
    _assert_not_admitted(result)
    measurement = result["files"][0]["measurements"]
    assert measurement["selected_primaries"] == measurement["distinct_full_generator_candidates"] == 2
    assert measurement["payload_exact_equal"] == 2
    assert measurement["source_idx_differs_from_donor_genidx"] == 1
    assert measurement["full_generator_rows"] == 4
    assert result["training_census"]["pairs"][0]["generator_rows"] == {"donor": 3, "full": 4}
    probes = result["files"][0]["probe_events"]
    assert probes[0] == dict(run=1400, subrun=101, event=5, in_full_census=True,
                             absent_donor_event=True, fullgen_muons=1)
    assert probes[1]["fullgen_muons"] == 0 and probes[1]["absent_donor_event"] is False
    assert probes[2]["in_full_census"] is False
    assert result["inputs_sha256"][str(config)] == audit._sha256(config)
    assert result["hashes_verified_after"] is True
    assert before == {str(path): audit._sha256(path) for path in paths}
    for source, digest in result["sources_sha256"].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
    saved = out.read_bytes()
    with pytest.raises(FileExistsError):
        audit.run(config, out)
    assert out.read_bytes() == saved
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        audit.run(config, link)


@pytest.mark.parametrize("change", ["unknown", "empty", "string", "unpaired", "badprobe", "boolprobe", "duplicateprobe", "wrongrole", "alias"])
def test_malformed_config_and_pair_refusals(audit, inputs, tmp_path, change):
    paths, _, _ = inputs
    cfg = dict(donors=[str(paths[0])], generators=[str(paths[1])])
    if change == "unknown":
        cfg["run_config"] = "not-used"
    elif change == "empty":
        cfg["donors"] = []
    elif change == "string":
        cfg["generators"] = str(paths[1])
    elif change == "unpaired":
        cfg["generators"] *= 2
    elif change == "badprobe":
        cfg["probe_events"] = [[1400, 101, 5.0]]
    elif change == "boolprobe":
        cfg["probe_events"] = [[1400, 101, True]]
    elif change == "duplicateprobe":
        cfg["probe_events"] = [[1400, 101, 5]] * 2
    elif change == "wrongrole":
        cfg["donors"], cfg["generators"] = cfg["generators"], cfg["donors"]
    else:
        cfg["generators"] = cfg["donors"]
    out = tmp_path / "refused.json"
    with pytest.raises((ValueError, TypeError)):
        audit.run(cfg, out)
    assert not out.exists()


def test_wrong_census_pair_legacy_and_missing_payload_refused(audit, inputs, tmp_path):
    paths, donor, full = inputs
    for variant in ("wrong_pair", "legacy", "missing_payload"):
        changed = deepcopy(full)
        if variant == "wrong_pair":
            for frame in changed.values():
                frame["subrun"] += 1
        elif variant == "legacy":
            changed.pop("events")
            changed["gen"] = changed["gen"].drop(columns="gen_product_id")
        else:
            changed["gen"] = changed["gen"].drop(columns="x")
        full_path = _write(tmp_path / f"{variant}.root", changed)
        out = tmp_path / f"{variant}.json"
        with pytest.raises(ValueError):
            audit.run(dict(donors=[paths[0]], generators=[full_path]), out)
        assert not out.exists()


@pytest.mark.parametrize("target", ["input", "source", "config"])
def test_hash_drift_refused_before_output(audit, inputs, tmp_path, monkeypatch, target):
    paths, _, _ = inputs
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(donors=[str(paths[0])], generators=[str(paths[1])])))
    original_analyze, original_hash = audit.analyze, audit._sha256
    selected = str(paths[0]) if target == "input" else str(config) if target == "config" else str(Path(audit.__file__).resolve())
    drift = False

    def analyze(*args):
        nonlocal drift
        result = original_analyze(*args)
        drift = True
        return result

    def hashed(path):
        return "0" * 64 if drift and str(path) == selected else original_hash(path)

    monkeypatch.setattr(audit, "analyze", analyze)
    monkeypatch.setattr(audit, "_sha256", hashed)
    out = tmp_path / "drift.json"
    with pytest.raises(ValueError, match="hash drift"):
        audit.run(config, out)
    assert not out.exists()
