"""Synthetic exact rawdump ROOT evidence only; never build, transport or run art.

Opaque synthetic art tests hash binding, not art provenance or physics. Mixed-PDG
primaries, empty events and uint64 keys exercise complete bounded audit behavior.
"""
from copy import deepcopy
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
from scripts import audit_inclusive_run1a_raw_integrity as audit


def _capture_fixtures():
    spec = importlib.util.spec_from_file_location("raw_integrity_capture_fixtures", ROOT / "tests/test_inclusive_run1a_capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pointer(prefix, key=None, pid=77):
    nonnull = key is not None
    return {prefix + "_" + field: (int(nonnull) if field in ("nonnull", "available", "dereferenced", "member", "matches_selected")
                                    else (0 if field == "error" or not nonnull else (pid if field == "product_id" else key)))
            for field in audit.POINTER}


def _frames(n=10, empty=False):
    fixtures = _capture_fixtures()
    original, _, _ = fixtures._frames(audit.capture, n, empty)
    census, gen = original["events"], original["gen"]
    e = census[audit.KEY + ["gen_product_id", "n_gen"]].copy()
    e = e.assign(schema_version=1, sim_product_id=77, raw_product_id=88, status_product_id=99,
                 gen_present=1, sim_present=1, raw_present=1, status_present=1, collection_error=0,
                 n_particles=0, n_raw=0, n_daughters=0, status=0, nG4Tracks=100,
                 overflowSimP=0, nStepLimit=0, nFieldPropagator=0, cpuTime=.01, realTime=.02, **audit.TAGS)
    particles, raw, daughters = [], [], []
    # Key zero is valid; adjacent full uint64 values must remain distinguishable.
    keys = [0, 2**32 + 9, 2**63 + 9, 2**64 - 1]
    for j, row in enumerate(gen.to_dict("records")):
        identity = {column: row[column] for column in audit.KEY}
        particles.append({**identity, "key": keys[j], "id": keys[j], "n_daughters": int(j == 0),
                          "sim_product_id": 77, "pdg": row["pdg"], "isprim": 1,
                          **_pointer("parent"), **_pointer("gen", row["idx"], 42)})
        raw.append({**identity, "idx": row["idx"], "raw_product_id": 88, "bar": 5, **_pointer("sim", keys[j])})
    if not empty:
        identity = {column: gen.iloc[0][column] for column in audit.KEY}
        particles.append({**identity, "key": 2**63 + 10, "id": 2**63 + 10, "n_daughters": 0,
                          "sim_product_id": 77, "pdg": 11, "isprim": 0,
                          **_pointer("parent", 0), **_pointer("gen", 0, 42)})
        daughters.append({**identity, "owner_key": 0, "idx": 0, "sim_product_id": 77,
                          **_pointer("daughter", 2**63 + 10)})
    frames = {"events": e, "gen": gen.copy(), "particles": pd.DataFrame(particles),
              "raw": pd.DataFrame(raw), "daughters": pd.DataFrame(daughters)}
    for name in ("particles", "raw", "daughters"):
        for event in e.event:
            count = sum(row["event"] == event for row in {"particles": particles, "raw": raw, "daughters": daughters}[name])
            e.loc[e.event.eq(event), "n_" + name] = count
    for name, schema in audit.SCHEMA.items():
        if frames[name].empty:
            frames[name] = pd.DataFrame({column: pd.Series(dtype="object" if dtype == "string" else dtype) for column, dtype in schema.items()})
        else:
            for column, dtype in schema.items():
                # Construct uint64 directly from Python ints; DataFrame inference can round sparse huge keys.
                if name in ("particles", "raw", "daughters"):
                    records = {"particles": particles, "raw": raw, "daughters": daughters}[name]
                    frames[name][column] = pd.Series([row[column] for row in records], dtype="object" if dtype == "string" else dtype)
                else:
                    frames[name][column] = frames[name][column].astype("object" if dtype == "string" else dtype)
    return original, frames, audit.parse_status(fixtures._log(n))


def _write(path, frames):
    with uproot.recreate(path) as root:
        for name, frame in frames.items():
            schema = {column: ("string" if dtype == "string" else np.dtype(dtype)) for column, dtype in audit.SCHEMA[name].items()}
            tree = root.mktree("rawdump/" + name, schema)
            if len(frame):
                tree.extend({column: frame[column].to_numpy() for column in frame})


def _directory(tmp_path, n=10, empty=False):
    raw = tmp_path / "raw"
    raw.mkdir()
    fixtures = _capture_fixtures()
    fixtures._directory(raw, audit.capture, n, empty)
    (raw / "inclusive-raw-run1a.art").write_bytes(b"synthetic opaque art evidence, not art validation\n")
    (raw / "capture-audit.json").write_text("old untrusted invalid JSON")
    original, frames, status = _frames(n, empty)
    export = tmp_path / "export.root"
    _write(export, frames)
    log = tmp_path / "exporter.log"
    log.write_text("Art has completed and will exit with status 0.\n")
    return raw, export, log, original, frames, status


def _run(tmp_path, n=10, **kwargs):
    raw, export, log, original, frames, status = _directory(tmp_path, n, **kwargs)
    return audit.run(raw, export, n, log, tmp_path / "audit.json")


def _flags(result, passed):
    assert result["raw_pointer_integrity_established"] is passed
    assert result["raw_integrity_checks_passed"] is passed
    for name in audit.FLAGS:
        if name != "raw_pointer_integrity_established":
            assert result[name] is False


@pytest.mark.parametrize("n,empty", [(1, False), (10, False), (1, True), (10, True)])
def test_complete_exact_root_empty_multi_all_pdg_uint64_and_no_mutation(tmp_path, n, empty):
    raw, export, log, original, frames, status = _directory(tmp_path, n, empty)
    before = deepcopy((original, frames, status))
    summary = audit.analyze_frames(original, frames, status, n)
    _flags(summary, True)
    assert summary["empty_generator_events"] == (n if empty else n - (2 if n == 10 else 1))
    assert summary["multi_muon_events"] == int(not empty)
    assert summary["unknown_crossing_events"] == n
    assert all(row["crossing_state"] == row["transport_state"] == "unknown" for row in summary["event_outcomes"])
    for actual, expected in zip((original, frames), before[:2]):
        for name in actual:
            pd.testing.assert_frame_equal(actual[name], expected[name])
    pd.testing.assert_frame_equal(status, before[2])
    result = audit.run(raw, export, n, log, tmp_path / "audit.json")
    _flags(result, True)
    assert result["summary"] == summary
    assert json.loads((tmp_path / "audit.json").read_text()) == result
    assert result["inputs_sha256_before"] == result["inputs_sha256_after"]
    assert result["file_identities_before"] == result["file_identities_after"]
    assert {Path(path).name for path in result["sources_sha256_before"]} == {
        "audit_inclusive_run1a_raw_integrity.py", "audit_inclusive_run1a_capture.py", "audit_inclusive_run1a_pilot.py", "genassoc.py"}


@pytest.mark.parametrize("problem,needle", [
    ("raw_null", "required pointer"), ("unavailable", "available"), ("wrong_product", "target"),
    ("out_of_range", "target"), ("no_resolve", "dereferenced"), ("no_member", "member"),
    ("wrong_address", "matches_selected"), ("pointer_error", "error"), ("null_metadata", "noncanonical"),
    ("internal_id", "internal ID"), ("parent_missing", "parent"), ("gen_missing", "gen"),
    ("daughter_missing", "daughter"), ("daughter_uint64_neighbor", "edges"),
    ("cycle", "ancestry"), ("missing_edge", "edges"),
    ("duplicate_edge", "duplicate edge"), ("nonprimary_root", "root/isprim"),
    ("nonmuon_pdg", "signed PDG"), ("primary_bijection", "bijection"),
    ("payload_ulp", "payload"), ("status", "status"), ("field_counter", "counters"),
    ("status_time", "cpuTime"), ("collection_missing", "raw_present")])
def test_well_formed_failures_are_whole_export_diagnostics(tmp_path, problem, needle):
    raw, export, log, original, f, status = _directory(tmp_path, 1)
    p, r, d, e = [f[name] for name in ("particles", "raw", "daughters", "events")]
    if problem == "raw_null":
        for column, value in _pointer("sim").items():
            r.loc[0, column] = value
    elif problem in ("unavailable", "no_resolve", "no_member", "wrong_address", "pointer_error"):
        column = {"unavailable": "available", "no_resolve": "dereferenced", "no_member": "member", "wrong_address": "matches_selected", "pointer_error": "error"}[problem]
        r.loc[0, "sim_" + column] = int(problem == "pointer_error")
    elif problem == "wrong_product":
        r.loc[0, "sim_product_id"] = 999
    elif problem == "out_of_range":
        r.loc[0, "sim_key"] = np.uint64(2**63 + 11)
    elif problem == "null_metadata":
        p.loc[0, "parent_product_id"] = 77
    elif problem == "internal_id":
        p.loc[2, "id"] = np.uint64(2**63 + 8)
    elif problem == "parent_missing":
        p.loc[3, "parent_key"] = np.uint64(2**63 + 11)
    elif problem == "gen_missing":
        p.loc[3, "gen_key"] = np.uint64(2**63 + 11)
    elif problem == "daughter_missing":
        d.loc[0, "daughter_key"] = np.uint64(2**63 + 11)
    elif problem == "daughter_uint64_neighbor":
        d.loc[0, "daughter_key"] = np.uint64(2**63 + 9)
    elif problem == "cycle":
        for column, value in _pointer("parent", 2**63 + 10).items():
            p.loc[3, column] = value
    elif problem == "missing_edge":
        f["daughters"] = d.iloc[:0]
        e["n_daughters"] = np.uint64(0)
        p["n_daughters"] = np.uint64(0)
    elif problem == "duplicate_edge":
        f["daughters"] = pd.concat([d, d], ignore_index=True)
        f["daughters"].loc[1, "idx"] = np.uint64(1)
        e["n_daughters"] = np.uint64(2)
        p.loc[0, "n_daughters"] = np.uint64(2)
    elif problem == "nonprimary_root":
        p.loc[2, "isprim"] = 0
    elif problem == "nonmuon_pdg":
        p.loc[2, "pdg"] = -22
    elif problem == "primary_bijection":
        p.loc[2, "gen_key"] = np.uint64(0)
    elif problem == "payload_ulp":
        f["gen"].loc[2, "x"] = np.nextafter(1., np.inf)
    elif problem == "status":
        e.loc[0, "status"] = -1
    elif problem == "field_counter":
        e.loc[0, "nFieldPropagator"] = 1
    elif problem == "status_time":
        e.loc[0, "cpuTime"] = .03
    elif problem == "collection_missing":
        e.loc[0, "raw_present"] = 0
        e.loc[0, "collection_error"] = 1
        e.loc[0, "n_raw"] = np.uint64(0)
        f["raw"] = r.iloc[:0]
    _write(export, f)
    result = audit.run(raw, export, 1, log, tmp_path / "failed.json")
    _flags(result, False)
    assert any(needle in violation for violation in result["violations"])
    assert result["summary"]["unknown_crossing_events"] == 1
    assert json.loads((tmp_path / "failed.json").read_text()) == result


@pytest.mark.parametrize("problem", ["width", "float_key", "duplicate_key", "duplicate_raw_idx", "raw_idx_gap",
    "daughter_idx", "daughter_owner", "count", "negative_counter", "nonfinite", "bad_flag", "tag", "missing_column", "missing_event"])
def test_malformed_schema_counts_or_identity_refuse(problem):
    original, f, status = _frames(1)
    if problem == "width":
        f["gen"]["pdg"] = f["gen"].pdg.astype("int64")
    elif problem == "float_key":
        f["particles"]["key"] = f["particles"].key.astype("float64")
    elif problem == "duplicate_key":
        f["particles"].loc[1, "key"] = np.uint64(0)
    elif problem == "duplicate_raw_idx":
        f["raw"].loc[1, "idx"] = np.uint64(0)
    elif problem == "raw_idx_gap":
        f["raw"].loc[1, "idx"] = np.uint64(4)
    elif problem == "daughter_idx":
        f["daughters"].loc[0, "idx"] = np.uint64(1)
    elif problem == "daughter_owner":
        f["daughters"].loc[0, "owner_key"] = np.uint64(99)
    elif problem == "count":
        f["events"].loc[0, "n_daughters"] = np.uint64(2)
    elif problem == "negative_counter":
        f["events"].loc[0, "nFieldPropagator"] = -1
    elif problem == "nonfinite":
        f["events"].loc[0, "realTime"] = np.inf
    elif problem == "bad_flag":
        f["raw"].loc[0, "sim_available"] = -1
    elif problem == "tag":
        f["events"].loc[0, "raw_tag"] = "g4run:CRV:Other"
    elif problem == "missing_column":
        f["events"] = f["events"].drop(columns="raw_present")
    else:
        f["events"] = f["events"].iloc[:0]
    with pytest.raises(ValueError):
        audit.analyze_frames(original, f, status, 1)


def test_real_root_width_empty_tree_missing_tree_and_ambiguous_cycle(tmp_path):
    _, export, _, _, f, _ = _directory(tmp_path, 1, True)
    with uproot.update(export) as root:
        root.mktree("rawdump/gen", {column: np.dtype(dtype) for column, dtype in audit.SCHEMA["gen"].items()})
    with pytest.raises(ValueError, match="cycle"):
        audit._read(export)
    with uproot.recreate(export) as root:
        for name, schema in audit.SCHEMA.items():
            if name == "raw":
                continue
            tree_schema = {column: ("string" if dtype == "string" else np.dtype(dtype)) for column, dtype in schema.items()}
            if name == "gen":
                tree_schema["idx"] = np.dtype("int64")
            root.mktree("rawdump/" + name, tree_schema)
    with pytest.raises(ValueError):
        audit._read(export)
    _write(export, f)
    assert audit._read(export)["raw"].empty


@pytest.mark.parametrize("problem", ["capture_warning", "capture_status", "exporter_incomplete"])
def test_fresh_capture_and_exporter_completion_gate_diagnostic(tmp_path, monkeypatch, problem):
    raw, export, log, _, _, _ = _directory(tmp_path, 1)
    if problem == "capture_warning":
        with (raw / "transport.log").open("a") as stream:
            stream.write("Maximum number of steps reached in detector\n")
    elif problem == "capture_status":
        path = raw / "status.log"
        path.write_text(path.read_text().replace("status: 0", "status: 2"))
    else:
        log.write_text("incomplete exporter\n")
    original_run = audit.capture.run
    receipts = []

    def fresh(directory, events, out, expected_event=None):
        assert Path(out).parent != raw and not Path(out).exists()
        receipts.append(Path(out))
        return original_run(directory, events, out, expected_event)

    monkeypatch.setattr(audit.capture, "run", fresh)
    result = audit.run(raw, export, 1, log, tmp_path / "audit.json")
    _flags(result, False)
    assert receipts and not receipts[0].exists()


@pytest.mark.parametrize("problem", ["missing_art", "missing_export", "missing_log", "hardlink", "symlink", "same_log", "identity_drift", "hash_drift", "source_drift"])
def test_input_source_bindings_aliases_and_drift_refuse(tmp_path, monkeypatch, problem):
    raw, export, log, _, _, _ = _directory(tmp_path, 1)
    if problem.startswith("missing_"):
        {"missing_art": raw / "inclusive-raw-run1a.art", "missing_export": export, "missing_log": log}[problem].unlink()
    elif problem in ("hardlink", "symlink"):
        log.unlink()
        if problem == "hardlink":
            os.link(raw / "transport.log", log)
        else:
            log.symlink_to(raw / "transport.log")
    elif problem == "same_log":
        log = raw / "transport.log"
    elif problem == "identity_drift":
        original_read = audit._read

        def replaced(path):
            frames = original_read(path)
            replacement = tmp_path / "replacement.root"
            replacement.write_bytes(export.read_bytes())
            replacement.replace(export)
            return frames

        monkeypatch.setattr(audit, "_read", replaced)
    else:
        original_hash = audit._sha256
        calls = {}
        target = "genassoc.py" if problem == "source_drift" else "inclusive-raw-run1a.art"

        def drift(path):
            key = str(path)
            calls[key] = calls.get(key, 0) + 1
            return "changed" if Path(path).name == target and calls[key] > 1 else original_hash(path)

        monkeypatch.setattr(audit, "_sha256", drift)
    out = tmp_path / "audit.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.run(raw, export, 1, log, out)
    assert not out.exists()


def test_expected_triplet_exclusive_output_and_cli(tmp_path):
    raw, export, log, _, _, _ = _directory(tmp_path, 1)
    out = tmp_path / "audit.json"
    for expected in ([1400, 101, 2], [True, 101, 1], [1400, 101, 2**32]):
        with pytest.raises(ValueError, match="expected_event"):
            audit.run(raw, export, 1, log, out, expected)
    command = [sys.executable, str(ROOT / "scripts/audit_inclusive_run1a_raw_integrity.py"),
               "--raw-directory", str(raw), "--export-root", str(export), "--events", "1",
               "--exporter-log", str(log), "--out", str(out), "--expected-event", "1400", "101", "1"]
    import mu2efastsim
    core_path = str(Path(mu2efastsim.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), core_path]),
           "PYTHONDONTWRITEBYTECODE": "1", "NO_FETCH": "1"}
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    _flags(json.loads(out.read_text()), True)
    with pytest.raises(FileExistsError):
        audit.run(raw, export, 1, log, out)
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "absent.json")
    with pytest.raises(FileExistsError):
        audit.run(raw, export, 1, log, dangling)


def test_empty_generator_product_identity_and_no_narrowing_wrap():
    original, frames, status = _frames(1, True)
    frames["events"]["gen_product_id"] = np.uint32(43)
    result = audit.analyze_frames(original, frames, status, 1)
    _flags(result, False)
    assert any("gen_product_id" in message for message in result["violations"])
    original, frames, status = _frames(1)
    # Original wider PDG integers must not silently wrap onto exported int32 PDGs.
    original["gen"].loc[2, "pdg"] += 2**32
    result = audit.analyze_frames(original, frames, status, 1)
    _flags(result, False)
    assert any("pdg" in message for message in result["violations"])


def test_empty_payload_root_exact_width_is_not_vacuous(tmp_path):
    _, export, _, _, frames, _ = _directory(tmp_path, 1, True)
    with uproot.recreate(export) as root:
        for name, schema in audit.SCHEMA.items():
            types = {column: ("string" if dtype == "string" else np.dtype(dtype)) for column, dtype in schema.items()}
            if name == "gen":
                types["idx"] = np.dtype("int64")
            tree = root.mktree("rawdump/" + name, types)
            if len(frames[name]):
                tree.extend({column: frames[name][column].to_numpy() for column in frames[name]})
    with pytest.raises(ValueError, match="idx: requires uint64"):
        audit._read(export)


@pytest.mark.parametrize("bound", [0, 2, True, 1.0])
def test_invalid_bound_refused(bound):
    original, frames, status = _frames(1)
    with pytest.raises(ValueError, match="1 or 10"):
        audit.analyze_frames(original, frames, status, bound)


def test_serialized_float_status_times_match_only_printer_precision():
    original, frames, status = _frames(1)
    frames["events"]["cpuTime"] = np.float64(np.float32(.010321))
    frames["events"]["realTime"] = np.float64(np.float32(.019789))
    status["cpuTime"] = .010
    status["realTime"] = .020
    assert audit.analyze_frames(original, frames, status, 1)["raw_integrity_checks_passed"] is True
    status["realTime"] = .021
    result = audit.analyze_frames(original, frames, status, 1)
    assert result["raw_integrity_checks_passed"] is False
    assert any("realTime" in message and "printer precision" in message for message in result["violations"])


@pytest.mark.parametrize("suffix", ["Art has completed and will exit with status 1.\n",
    "Art has completed and will exit with status 0.\n", "Art has completed with unknown state\n"])
def test_exporter_ambiguous_or_contradictory_completion_refuses(tmp_path, suffix):
    raw, export, log, _, _, _ = _directory(tmp_path, 1)
    log.write_text(log.read_text() + suffix)
    out = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="ambiguous art completion"):
        audit.run(raw, export, 1, log, out)
    assert not out.exists()
