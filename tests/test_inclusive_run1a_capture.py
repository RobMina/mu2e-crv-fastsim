"""Synthetic bounded raw capture evidence, including retained failed transport.

ROOT fixtures are generator-only schema-v2 tables, not simulated physics or raw
pointer tests. They establish audit behavior without transport, builds or fetching.
"""
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


@pytest.fixture
def audit():
    spec = importlib.util.spec_from_file_location("inclusive_capture", ROOT / "scripts/audit_inclusive_run1a_capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _typed(frame, unsigned):
    for column in frame:
        dtype = np.uint32 if column in unsigned else (np.float64 if column in ("e", "t", "x", "y", "z", "px", "py", "pz") else np.int64)
        frame[column] = frame[column].astype(dtype)
    return frame


def _frames(audit, n=10, empty=False, identity=(1400, 101, 1)):
    run, subrun, first = identity
    events = pd.DataFrame({"run": run, "subrun": subrun, "event": np.arange(first, first + n, dtype=np.uint32),
                           "schema_version": 2, "associations_checked": 0, "full_generator_stream": 1,
                           "gen_product_id": 42, "sim_product_id": 0, "n_gen": 0,
                           "n_particles": 0, "n_steps": 0})
    if not empty:
        events.loc[0, "n_gen"] = 3
        if n == 10:
            events.loc[1, "n_gen"] = 1
    events = _typed(events, audit.KEY + ["schema_version", "gen_product_id", "sim_product_id"])
    gen = pd.DataFrame({"event": [first, first, first] + ([first + 1] if n == 10 else []),
                        "idx": [0, 1, 2] + ([0] if n == 10 else []),
                        "pdg": [13, -13, 22] + ([11] if n == 10 else [])})
    gen = gen.assign(run=run, subrun=subrun, gen_product_id=42, genid=7, e=200., t=0.,
                     x=1., y=2., z=3., px=4., py=5., pz=6.)
    gen = _typed(gen, audit.KEY + ["gen_product_id"])
    if empty:
        gen = gen.iloc[:0]
    original = {"events": events, "gen": gen}
    return original, deepcopy(original), audit.parse_status(_log(n, identity))


def _log(n=10, identity=(1400, 101, 1)):
    run, subrun, first = identity
    return "startup\n" + "".join(
        f"###############  PrintModule Run/Subrun/Event [{run}] [{subrun}] [{event}]\n"
        "ProductPrint mu2e::StatusG4_g4run__R3Inclusive\n"
        "status: 0 nG4Tracks: 100 overflowSimP: 0 nStepLimit: 0 CPU: 0.01 Wall: 0.02\n"
        for event in range(first, first + n))


def _write(path, frames):
    with uproot.recreate(path) as root:
        for name, frame in frames.items():
            root.mktree("dump/" + name, {col: frame[col].to_numpy() for col in frame})


def _directory(tmp_path, audit, n=10, empty=False, identity=(1400, 101, 1)):
    original, retained, _ = _frames(audit, n, empty, identity)
    _write(tmp_path / "input-gen.root", original)
    _write(tmp_path / "raw-gen.root", retained)
    (tmp_path / "status.log").write_text(_log(n, identity))
    (tmp_path / "transport.log").write_text("Art has completed and will exit with status 0.\n")
    (tmp_path / "products.txt").write_text("\n".join(audit.PRODUCTS + ["art::TriggerResults_TriggerResults__R3Inclusive"]))
    return original, retained


def _flags(result):
    assert result["capture_checks_passed"] is True
    for flag in ("physics_admitted", "global_denominator_completeness_established", "conversion_eligible",
                 "raw_pointer_integrity_established", "downstream_conversion_authorized"):
        assert result[flag] is False


@pytest.mark.parametrize("n,empty", [(1, False), (10, False), (1, True), (10, True)])
def test_clean_root_empty_multi_and_no_mutation(tmp_path, audit, n, empty):
    original, retained = _directory(tmp_path, audit, n, empty)
    status = audit.parse_status(_log(n))
    before = deepcopy((original, retained, status))
    summary = audit.analyze_frames(original, retained, status, n)
    _flags(summary)
    assert summary["events"] == n
    assert summary["empty_generator_events"] == (n if empty else n - (2 if n == 10 else 1))
    assert summary["multi_muon_events"] == (0 if empty else 1)
    assert summary["counts"]["generators"] == (0 if empty else (4 if n == 10 else 3))
    assert summary["counts"]["unknown_crossing_events"] == n
    assert summary["counts"]["status_ok_events"] == n
    assert summary["raw_transport_status_ok"] is True
    assert all(row["crossing_state"] == "unknown" for row in summary["event_status"])
    result = audit.run(tmp_path, n, tmp_path / "audit.json")
    assert result["summary"] == summary
    assert result["original_generator_validation"]["schema_version"] == 2
    assert result["retained_generator_validation"]["schema_version"] == 2
    _flags(result)
    for actual, expected in zip((original, retained), before[:2]):
        for name in actual:
            pd.testing.assert_frame_equal(actual[name], expected[name])
    pd.testing.assert_frame_equal(status, before[2])


@pytest.mark.parametrize("field", ["status", "overflowSimP", "nStepLimit", "mixed", "signed_status"])
def test_failed_status_retains_diagnostic_json_not_zero(tmp_path, audit, field):
    _directory(tmp_path, audit)
    text = _log()
    if field == "mixed":
        text = text.replace("status: 0", "status: 2", 1).replace("overflowSimP: 0", "overflowSimP: 1", 1)
        text = text.replace("nStepLimit: 0", "nStepLimit: 3", 1)
        text = text.replace("status: 0", "status: 4", 1)
    else:
        column = "status" if field == "signed_status" else field
        text = text.replace(f"{column}: 0", f"{column}: {-1 if field == 'signed_status' else 1}", 1)
    (tmp_path / "status.log").write_text(text)
    result = audit.run(tmp_path, 10, tmp_path / "failed.json")
    _flags(result)
    assert result["raw_transport_status_ok"] is False
    summary = result["summary"]
    assert summary["counts"]["unknown_transport_events"] == (2 if field == "mixed" else 1)
    assert summary["event_status"][0]["transport_state"] == "unknown"
    assert summary["event_status"][0]["crossing_state"] == "unknown"
    assert summary["counts"]["unknown_crossing_events"] == 10
    assert json.loads((tmp_path / "failed.json").read_text()) == result


@pytest.mark.parametrize("field", ["nG4Tracks", "overflowSimP", "nStepLimit"])
def test_negative_counters_refused_no_report(tmp_path, audit, field):
    _directory(tmp_path, audit)
    path = tmp_path / "status.log"
    path.write_text(path.read_text().replace(f"{field}: {100 if field == 'nG4Tracks' else 0}", f"{field}: -1", 1))
    with pytest.raises(ValueError, match="negative counter"):
        audit.run(tmp_path, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("field", ["pdg", "genid", "e", "t", "x", "y", "z", "px", "py", "pz"])
def test_exact_payload_one_ulp_no_tolerance(tmp_path, audit, field):
    _, retained = _directory(tmp_path, audit)
    old = retained["gen"].loc[0, field]
    retained["gen"].loc[0, field] = old + 1 if field in ("pdg", "genid") else np.nextafter(old, np.inf)
    _write(tmp_path / "raw-gen.root", retained)
    with pytest.raises(ValueError, match="exact mismatch"):
        audit.run(tmp_path, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("field", ["run", "subrun", "event", "gen_product_id", "swapped_indices"])
def test_exact_identity_not_payload_matching(audit, field):
    original, retained, status = _frames(audit)
    if field == "swapped_indices":
        retained["gen"].loc[[0, 1], "idx"] = [1, 0]
    else:
        for frame in retained.values():
            frame[field] += np.uint32(1)
    with pytest.raises(ValueError, match="exact mismatch"):
        audit.analyze_frames(original, retained, status, 10)


@pytest.mark.parametrize("side", ["input-gen.root", "raw-gen.root"])
@pytest.mark.parametrize("problem", ["missing_census", "duplicate_census", "missing_empty", "missing_payload", "duplicate_gen",
                                      "wrong_role", "schema", "wrong_count", "float_identity", "missing_branch", "nonfinite"])
def test_bad_root_streams_refuse_without_report(tmp_path, audit, side, problem):
    original, retained = _directory(tmp_path, audit)
    frames = original if side == "input-gen.root" else retained
    if problem == "missing_census":
        del frames["events"]
    elif problem == "duplicate_census":
        frames["events"] = pd.concat([frames["events"], frames["events"].iloc[[0]]], ignore_index=True)
    elif problem == "missing_empty":
        frames["events"] = frames["events"].iloc[:-1]
    elif problem == "missing_payload":
        frames["gen"] = frames["gen"].iloc[:-1]
    elif problem == "duplicate_gen":
        frames["gen"] = pd.concat([frames["gen"], frames["gen"].iloc[[0]]], ignore_index=True)
    elif problem == "wrong_role":
        frames["events"]["associations_checked"] = np.int64(1)
    elif problem == "schema":
        frames["events"]["schema_version"] = np.uint32(1)
    elif problem == "wrong_count":
        frames["events"].loc[0, "n_gen"] += 1
    elif problem == "float_identity":
        frames["gen"]["idx"] = frames["gen"]["idx"].astype(float)
    elif problem == "missing_branch":
        frames["gen"] = frames["gen"].drop(columns="genid")
    else:
        frames["gen"].loc[0, "x"] = np.inf
    _write(tmp_path / side, frames)
    with pytest.raises(ValueError):
        audit.run(tmp_path, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("problem", ["missing", "duplicate", "mismatch", "no_file", "wrong_tag", "malformed", "truncated", "extra_status"])
def test_status_must_be_independent_complete_exact_once(tmp_path, audit, problem):
    _directory(tmp_path, audit)
    path = tmp_path / "status.log"
    text = path.read_text()
    if problem == "missing":
        text = _log(9)
    elif problem == "duplicate":
        text += _log(1)
    elif problem == "mismatch":
        text = text.replace("[10]", "[11]")
    elif problem == "no_file":
        path.unlink()
    elif problem == "wrong_tag":
        text = text.replace("g4run__R3Inclusive", "g4run__Primary")
    elif problem == "malformed":
        text = text.replace("status: 0", "status: unknown", 1)
    elif problem == "truncated":
        text = text.replace(" nStepLimit: 0", "", 1)
    else:
        text += "status: 0 nG4Tracks: 0 overflowSimP: 0 nStepLimit: 0 CPU: 0 Wall: 0\n"
    if problem != "no_file":
        path.write_text(text)
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.run(tmp_path, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("missing", range(5))
def test_raw_products_require_trigger_but_not_crvsteps(tmp_path, audit, missing):
    _directory(tmp_path, audit)
    path = tmp_path / "products.txt"
    assert "CrvSteps_CrvSteps" not in path.read_text()
    assert len(audit.validate_products(path.read_text())) == 5
    branches = path.read_text().splitlines()
    del branches[missing]
    path.write_text("\n".join(branches))
    with pytest.raises(ValueError, match="missing retained branches"):
        audit.run(tmp_path, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("n", [0, 2, 11, True, 1.0])
def test_bounds_refused(audit, n):
    original, retained, status = _frames(audit)
    with pytest.raises(ValueError, match="1 or 10"):
        audit.analyze_frames(original, retained, status, n)


@pytest.mark.parametrize("expected", [[1., 2, 3], [True, 2, 3], [-1, 2, 3], [1, 2, 2**32], [1, 2], "1:2:3"])
def test_expected_requires_exact_uint32_triplet(tmp_path, audit, expected):
    with pytest.raises(ValueError, match="expected_event"):
        audit.run(tmp_path, 1, tmp_path / "audit.json", expected)
    assert not (tmp_path / "audit.json").exists()


def test_expected_binds_original_and_retained_uint32_max(tmp_path, audit):
    identity = [2**32 - 1, 2**32 - 1, 2**32 - 1]
    original, retained = _directory(tmp_path, audit, 1, identity=identity)
    out = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="expected_event"):
        audit.run(tmp_path, 1, out, [*identity[:2], identity[2] - 1])
    assert not out.exists()
    result = audit.run(tmp_path, 1, out, identity)
    assert result["expected_event"] == identity
    assert [result["summary"]["event_status"][0][k] for k in audit.KEY] == identity
    retained["events"]["event"] -= np.uint32(1)
    retained["gen"]["event"] -= np.uint32(1)
    _write(tmp_path / "raw-gen.root", retained)
    with pytest.raises(ValueError):
        audit.run(tmp_path, 1, tmp_path / "mismatch.json", identity)
    assert not (tmp_path / "mismatch.json").exists()


def test_expected_not_allowed_for_ten(tmp_path, audit):
    with pytest.raises(ValueError, match="expected_event"):
        audit.run(tmp_path, 10, tmp_path / "audit.json", [1400, 101, 1])


@pytest.mark.parametrize("target", ["input-gen.root", "raw-gen.root", "status.log", "products.txt", "transport.log",
                                     "audit_inclusive_run1a_capture.py", "audit_inclusive_run1a_pilot.py", "genassoc.py"])
def test_prepost_input_and_imported_source_drift_no_report(tmp_path, audit, monkeypatch, target):
    _directory(tmp_path, audit)
    original = audit._sha256
    calls = {}

    def drift(path):
        key = str(path)
        calls[key] = calls.get(key, 0) + 1
        return "changed" if Path(path).name == target and calls[key] > 1 else original(path)

    monkeypatch.setattr(audit, "_sha256", drift)
    with pytest.raises(ValueError, match="hash drift"):
        audit.run(tmp_path, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


def test_hashes_sources_exclusive_output_and_readonly_inputs(tmp_path, audit):
    _directory(tmp_path, audit, 1)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    out = tmp_path / "audit.json"
    result = audit.run(tmp_path, 1, out)
    assert result["inputs_sha256_before"] == result["inputs_sha256_after"]
    assert result["sources_sha256_before"] == result["sources_sha256_after"]
    assert {Path(p).name for p in result["sources_sha256_before"]} == {
        "audit_inclusive_run1a_capture.py", "audit_inclusive_run1a_pilot.py", "genassoc.py"}
    for name, digest in before.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        audit.run(tmp_path, 1, out)
    link = tmp_path / "dangling.json"
    link.symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        audit.run(tmp_path, 1, link)


def test_cli_exclusive_failed_status_json(tmp_path, audit):
    _directory(tmp_path, audit, 1)
    path = tmp_path / "status.log"
    path.write_text(path.read_text().replace("status: 0", "status: 1"))
    out = tmp_path / "cli.json"
    command = [sys.executable, str(ROOT / "scripts/audit_inclusive_run1a_capture.py"),
               "--directory", str(tmp_path), "--events", "1", "--out", str(out),
               "--expected-event", "1400", "101", "1"]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "NO_FETCH": "1"}
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(out.read_text())
    _flags(result)
    assert result["raw_transport_status_ok"] is False
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0


@pytest.mark.parametrize("warning", ["Maximum number of steps reached in CRV: 101",
    "Only 100 are saved in output collection.", "Maximum number of particles reached in TrackingAction",
    "probably the SimParticleCollection exceeds its maximum allowed size"])
def test_warning_only_recording_failure_keeps_json_and_quarantines_all(tmp_path, audit, warning):
    _directory(tmp_path, audit)
    (tmp_path / "transport.log").write_text(warning + "\nArt has completed and will exit with status 0.\n")
    result = audit.run(tmp_path, 10, tmp_path / "warning.json")
    _flags(result)
    assert result["raw_transport_status_ok"] is True
    assert result["raw_capture_review_gate_passed"] is False
    assert result["recording_log"]["truncation_warning_patterns"]
    assert result["recording_log"]["native_raw_recording_completeness_established"] is False
    assert result["summary"]["counts"]["unknown_crossing_events"] == 10


def test_incomplete_or_missing_transport_log_is_not_clean_capture(tmp_path, audit):
    _directory(tmp_path, audit, 1)
    (tmp_path / "transport.log").write_text("truncated log")
    result = audit.run(tmp_path, 1, tmp_path / "incomplete.json")
    assert result["raw_capture_review_gate_passed"] is False
    assert result["recording_log"]["art_completed_status_zero"] is False
    (tmp_path / "transport.log").unlink()
    with pytest.raises(FileNotFoundError):
        audit.run(tmp_path, 1, tmp_path / "missing.json")
    assert not (tmp_path / "missing.json").exists()
