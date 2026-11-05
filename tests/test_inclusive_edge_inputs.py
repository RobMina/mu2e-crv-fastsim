"""Synthetic scanner evidence only; census order is not verified art source order."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import uproot

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/inspect_inclusive_edge_inputs.py"


@pytest.fixture
def scanner():
    spec = importlib.util.spec_from_file_location("inclusive_edge_inputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames():
    # Deliberately not sorted: first empty must be census entry 1, event 7, not event 2.
    events = pd.DataFrame(dict(run=np.full(4, 1400, dtype=np.uint32),
        subrun=np.full(4, 101, dtype=np.uint32), event=np.array([9, 7, 8, 2], dtype=np.uint32),
        schema_version=np.uint32(2), associations_checked=np.uint32(0),
        full_generator_stream=np.uint32(1), gen_product_id=np.uint32(99), sim_product_id=np.uint32(0),
        n_gen=np.array([1, 0, 3, 0], dtype=np.int64), n_particles=np.int64(0), n_steps=np.int64(0)))
    gen = events.iloc[[0, 2, 2, 2]][["run", "subrun", "event", "gen_product_id"]].reset_index(drop=True)
    gen = gen.assign(idx=np.array([0, 0, 1, 2], dtype=np.int64),
                     pdg=np.array([22, 13, -13, 11], dtype=np.int32), e=200., t=0.)
    return {"events": events, "gen": gen}


def _write(path, frames, update=False):
    with (uproot.update(path) if update else uproot.recreate(path)) as root:
        for name, frame in frames.items():
            columns = {column: frame[column].to_numpy() for column in frame}
            tree = root.mktree("dump/" + name, {column: values.dtype for column, values in columns.items()})
            if len(frame):
                tree.extend(columns)
    return path


@pytest.fixture
def full(tmp_path):
    return _write(tmp_path / "full.root", _frames())


def test_schema_summary_probes_hashes_and_exclusive_output(scanner, full, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(generators=[str(full)], probe_events=[
        [1400, 101, 7], [1400, 101, 8], [1400, 101, 9], [1400, 101, 123], [1401, 101, 7], [1400, 102, 7]])))
    before = scanner._sha256(full)
    out = tmp_path / "report.json"
    report = scanner.run(config, out)
    assert json.loads(out.read_text()) == report
    assert report["schema"] == "crvfastsim.inclusive_edge_inputs.v1"
    for flag in ("physics_admitted", "source_art_order_verified", "transport_performed"):
        assert report[flag] is False
    assert report["files"][0]["summary"] == dict(n_events=4, n_gen=4, empty_event_count=2,
        first_empty_event=dict(run=1400, subrun=101, event=7, census_entry_index=1),
        fullgen_muons=2, multi_muon_event_count=1)
    assert report["first_empty_event"] == dict(generator=str(full), run=1400, subrun=101, event=7, census_entry_index=1)
    probes = report["probe_events"]
    assert probes[0] == dict(run=1400, subrun=101, event=7, in_full_census=True,
        n_gen=0, fullgen_muons=0, census_entry_index=1, generator=str(full))
    assert probes[1]["n_gen"] == 3 and probes[1]["fullgen_muons"] == 2
    assert probes[2]["census_entry_index"] == 0
    for probe in probes[3:]:
        assert probe["in_full_census"] is False
        assert all(probe[column] is None for column in ("n_gen", "fullgen_muons", "census_entry_index", "generator"))
    assert report["inputs_sha256"] == {str(full): before, str(config): scanner._sha256(config)}
    assert scanner._sha256(full) == before
    assert report["hashes_verified_after"] is True
    expected = hashlib.sha256(json.dumps(report["config"], sort_keys=True).encode()).hexdigest()
    assert report["config_sha256"] == expected
    assert set(report["sources_sha256"]) == {str(SCRIPT), str(Path(scanner.genassoc.__file__).resolve())}
    for path, digest in report["sources_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert "NOT an authoritative art entry" in " ".join(report["limitations"])
    saved = out.read_bytes()
    with pytest.raises(FileExistsError):
        scanner.run(config, out)
    assert out.read_bytes() == saved
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        scanner.run(config, link)


def test_highest_cycles_used_by_validation_and_scan(scanner, full, tmp_path):
    latest = _frames()
    latest["events"]["event"] += np.uint32(100)
    latest["gen"]["event"] += np.uint32(100)
    _write(full, latest, update=True)
    result = scanner.run(dict(generators=[full]), tmp_path / "cycles.json")
    assert result["first_empty_event"]["event"] == 107
    for name in ("events", "gen"):
        for trees in (result["files"][0]["trees"], result["files"][0]["validation"]["trees"]):
            assert trees[name]["cycles"] == [1, 2]
            assert trees[name]["selected_cycle"] == 2


def test_multiple_files_first_existing_empty_and_none(scanner, tmp_path):
    frames = _frames()
    frames["events"] = frames["events"].iloc[[0, 2]].copy()
    first = _write(tmp_path / "first.root", frames)
    none = scanner.run(dict(generators=[first]), tmp_path / "none.json")
    assert none["first_empty_event"] is None
    assert none["files"][0]["summary"]["empty_event_count"] == 0
    assert none["probe_events"] == []
    frames = _frames()
    for frame in frames.values():
        frame["subrun"] += np.uint32(1)
    second = _write(tmp_path / "second.root", frames)
    report = scanner.run(dict(generators=[first, second], probe_events=[[1400, 102, 7]]), tmp_path / "two.json")
    assert report["first_empty_event"]["generator"] == str(second)
    assert report["probe_events"][0]["generator"] == str(second)


def test_all_empty_and_adjacent_uint32_max_keys(scanner, tmp_path):
    frames = _frames()
    frames["events"] = frames["events"].iloc[:2].copy()
    for column in ("run", "subrun", "event"):
        frames["events"][column] = np.array([2**32 - 1, 2**32 - 2], dtype=np.uint32)
    frames["events"]["n_gen"] = np.int64(0)
    frames["gen"] = frames["gen"].iloc[:0]
    path = _write(tmp_path / "empty.root", frames)
    probes = [[2**32 - 1] * 3, [2**32 - 2] * 3, [2**32 - 3] * 3]
    report = scanner.run(dict(generators=[path], probe_events=probes), tmp_path / "empty.json")
    assert report["files"][0]["summary"]["n_gen"] == 0
    assert report["files"][0]["summary"]["empty_event_count"] == 2
    assert [row["census_entry_index"] for row in report["probe_events"]] == [0, 1, None]


@pytest.mark.parametrize("probes", [None, "1,2,3", [1, 2, 3], [[1, 2]], [[1, 2, 3, 4]],
    [[1, 2, 3.0]], [[1, 2, True]], [[1, 2, -1]], [[1, 2, 2**32]], [[1, 2, "3"]], [[1, 2, 3]] * 2])
def test_bad_probes_refused(scanner, full, tmp_path, probes):
    out = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        scanner.run(dict(generators=[full], probe_events=probes), out)
    assert not out.exists()


@pytest.mark.parametrize("config", [None, [], {}, {"generators": []}, {"generators": "path"},
    {"generators": [123]}, {"generators": [""]}, {"generators": [], "extra": 1}])
def test_bad_configs_refused(scanner, tmp_path, config):
    out = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        scanner.run(config, out)
    assert not out.exists()


def test_missing_directory_duplicate_alias_and_overlap_refused(scanner, full, tmp_path):
    alias = tmp_path / "alias.root"
    alias.symlink_to(full)
    overlap = _write(tmp_path / "overlap.root", _frames())
    for paths in ([tmp_path / "missing.root"], [tmp_path], [full, full], [full, alias], [full, overlap]):
        out = tmp_path / "bad.json"
        with pytest.raises((ValueError, FileNotFoundError)):
            scanner.run(dict(generators=paths), out)
        assert not out.exists()


@pytest.mark.parametrize("change", ["legacy", "missing_census", "missing_branch", "version", "role", "checked",
    "float_key", "float_idx", "negative_count", "count_mismatch", "duplicate_event", "duplicate_gen", "empty_census"])
def test_strict_schema_and_census_refusals(scanner, tmp_path, change):
    frames = deepcopy(_frames())
    if change == "legacy":
        frames.pop("events")
        frames["gen"] = frames["gen"].drop(columns="gen_product_id")
    elif change == "missing_census":
        frames.pop("events")
    elif change == "missing_branch":
        frames["gen"] = frames["gen"].drop(columns="pdg")
    elif change in ("version", "role", "checked"):
        column, value = {"version": ("schema_version", 1), "role": ("full_generator_stream", 0),
                         "checked": ("associations_checked", 1)}[change]
        frames["events"][column] = np.uint32(value)
    elif change in ("float_key", "float_idx"):
        name, column = ("events", "event") if change == "float_key" else ("gen", "idx")
        frames[name][column] = frames[name][column].astype(float)
    elif change in ("negative_count", "count_mismatch"):
        frames["events"].loc[0, "n_gen"] = -1 if change == "negative_count" else 2
    elif change in ("duplicate_event", "duplicate_gen"):
        name = "events" if change == "duplicate_event" else "gen"
        frames[name] = pd.concat([frames[name], frames[name].iloc[:1]], ignore_index=True)
    else:
        frames = {name: frame.iloc[:0] for name, frame in frames.items()}
    path = _write(tmp_path / "bad.root", frames)
    out = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        scanner.run(dict(generators=[path]), out)
    assert not out.exists()


@pytest.mark.parametrize("target", ["input", "config", "script", "genassoc"])
def test_hash_drift_refused_without_writing(scanner, full, tmp_path, monkeypatch, target):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(generators=[str(full)])))
    chosen = {"input": full, "config": config, "script": SCRIPT,
              "genassoc": Path(scanner.genassoc.__file__).resolve()}[target]
    original_hash, original_analyze = scanner._sha256, scanner.analyze
    drift = False

    def analyze(*args):
        nonlocal drift
        result = original_analyze(*args)
        drift = True
        return result

    def hashed(path):
        return "0" * 64 if drift and Path(path) == chosen else original_hash(path)

    monkeypatch.setattr(scanner, "analyze", analyze)
    monkeypatch.setattr(scanner, "_sha256", hashed)
    out = tmp_path / "drift.json"
    with pytest.raises(ValueError, match="hash drift"):
        scanner.run(config, out)
    assert not out.exists()


def test_cli(scanner, full, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(generators=[str(full)])))
    out = tmp_path / "cli.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config), "--out", str(out)],
                               cwd=ROOT, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(out.read_text())["source_art_order_verified"] is False
