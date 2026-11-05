"""Synthetic retrospective evidence only: never transport, convert, build or run art.

Reuse the pilot's real schema-v2 ROOT fixtures (multi-muon, companion, descendants,
empty events). Art files here are explicitly opaque synthetic hash inputs, not art
validation. Production dumps/art provenance is a stated limitation of the audit.
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

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def audit():
    return _load("inclusive_conversion", ROOT / "scripts/audit_inclusive_run1a_conversion.py")


@pytest.fixture
def fixtures():
    return _load("conversion_pilot_fixtures", ROOT / "tests/test_inclusive_run1a_pilot.py")


def _directory(tmp_path, audit, fixtures, n=10, empty=False):
    raw, converted = tmp_path / "raw", tmp_path / "converted"
    raw.mkdir()
    converted.mkdir()
    donor, original, _ = fixtures._frames(audit.pilot, n)
    if empty:
        for frames in (donor, original):
            frames["events"][["n_gen", "n_particles", "n_steps"]] = np.int64(0)
            frames["gen"] = frames["gen"].iloc[:0]
        donor["particles"] = donor["particles"].iloc[:0]
        donor["steps"] = donor["steps"].iloc[:0]
    fixtures._write(raw / "input-gen.root", original)
    fixtures._write(raw / "raw-gen.root", original)
    fixtures._write(converted / "inclusive-converted-steps.root", donor)
    # Dotted branch names with size columns include repeated inventory summaries.
    branches = audit.capture.PRODUCTS + ["art::TriggerResults_TriggerResults__R3Inclusive"]
    raw_inventory = "\n".join(f"{branch}. 1234 42 1" for branch in branches)
    (raw / "products.txt").write_text(raw_inventory + "\n" + raw_inventory)
    converted_inventory = raw_inventory + "\n" + "\n".join([
        f"{audit.CONVERTED_STEPS}. 5678 24 1",
        "art::TriggerResults_TriggerResults__R3InclusiveSteps. 100 10 1"])
    (converted / "products.txt").write_text(converted_inventory)
    for directory in (raw, converted):
        (directory / "status.log").write_text(fixtures._log(n))
    (raw / "transport.log").write_text("Art has completed and will exit with status 0.\n")
    (converted / "conversion.log").write_text("Art has completed and will exit with status 0.\n")
    (raw / "inclusive-raw-run1a.art").write_bytes(b"synthetic opaque raw art hash evidence\n")
    (converted / "inclusive-converted-run1a.art").write_bytes(b"synthetic opaque converted art hash evidence\n")
    # A prior receipt, even malformed, is not an input, proof or launch token.
    (raw / "capture-audit.json").write_text("not trusted JSON")
    return raw, converted, donor, original


def _snapshot(*directories):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in directories for path in directory.iterdir()}


def _flags(result):
    for flag in ("physics_admitted", "global_denominator_completeness_established", "conversion_eligible",
                 "raw_pointer_integrity_established", "downstream_conversion_authorized",
                 "native_raw_completeness_established", "native_raw_recording_completeness_established"):
        assert result[flag] is False


@pytest.mark.parametrize("n,empty", [(1, False), (10, False), (1, True), (10, True)])
def test_success_empty_multi_exact_all_pdg_and_no_mutation(tmp_path, audit, fixtures, monkeypatch, n, empty):
    raw, converted, donor, original = _directory(tmp_path, audit, fixtures, n, empty)
    frames_before = deepcopy((donor, original))
    before = _snapshot(raw, converted)
    original_capture = audit.capture.run
    receipts = []

    def checked_capture(directory, events, out, expected_event=None):
        assert Path(out).parent != raw
        assert not Path(out).exists()
        receipts.append(Path(out))
        return original_capture(directory, events, out, expected_event)

    monkeypatch.setattr(audit.capture, "run", checked_capture)
    out = tmp_path / "audit.json"
    result = audit.run(raw, converted, n, out, [1400, 101, 1] if n == 1 else None)
    assert result["conversion_checks_passed"] is True
    _flags(result)
    _flags(result["summary"])
    assert result["summary"]["events"] == n
    assert result["summary"]["empty_generator_events"] == (n if empty else n - (2 if n == 10 else 1))
    assert result["summary"]["multi_muon_events"] == (0 if empty else 1)
    assert result["summary"]["counts"]["actual_primaries"] == (0 if empty else (4 if n == 10 else 3))
    assert result["summary"]["unknown_crossing_events"] == n
    assert result["capture_receipt"]["raw_capture_review_gate_passed"] is True
    assert result["capture_receipt"]["downstream_conversion_authorized"] is False
    assert result["converted_donor_validation"]["associations_checked"] is True
    assert result["converted_status_matches_raw"] is True
    assert result["inputs_sha256_before"] == result["inputs_sha256_after"]
    assert len(result["inputs_sha256_before"]) == 11
    assert not any(Path(path).name == "capture-audit.json" for path in result["inputs_sha256_before"])
    assert {Path(path).name for path in result["sources_sha256_before"]} == {
        "audit_inclusive_run1a_conversion.py", "audit_inclusive_run1a_capture.py",
        "audit_inclusive_run1a_pilot.py", "genassoc.py"}
    assert result["sources_sha256_before"] == result["sources_sha256_after"]
    assert receipts and not receipts[0].parent.exists()
    assert _snapshot(raw, converted) == before
    for frames, old_frames in zip((donor, original), frames_before):
        for name in frames:
            pd.testing.assert_frame_equal(frames[name], old_frames[name])
    assert json.loads(out.read_text()) == result


@pytest.mark.parametrize("target", ["input-gen.root", "raw-gen.root", "inclusive-converted-steps.root"])
@pytest.mark.parametrize("column", ["pdg", "genid", "e", "t", "x", "y", "z", "px", "py", "pz"])
def test_original_and_retained_payload_drift(tmp_path, audit, fixtures, target, column):
    raw, converted, donor, original = _directory(tmp_path, audit, fixtures)
    frames = donor if target == "inclusive-converted-steps.root" else original
    value = frames["gen"].loc[0, column]
    frames["gen"].loc[0, column] = value + 1 if column in ("pdg", "genid") else np.nextafter(value, np.inf)
    fixtures._write((converted if target.startswith("inclusive-") else raw) / target, frames)
    with pytest.raises(ValueError):
        audit.run(raw, converted, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("side,problem", [
    ("raw", "wrong_process"), ("raw", "ambiguous"), ("raw", "other_module"),
    ("converted", "wrong_process"), ("converted", "ambiguous"), ("converted", "missing_raw"),
    ("converted", "wrong_steps"), ("converted", "extra_steps"), ("raw", "extra_steps"),
    ("converted", "old_trigger"), ("converted", "new_trigger"),
    ("converted", "generator"), ("converted", "sim"), ("converted", "status")])
def test_inventory_is_process_exact_and_unambiguous(tmp_path, audit, fixtures, side, problem):
    raw, converted, _, _ = _directory(tmp_path, audit, fixtures)
    path = (raw if side == "raw" else converted) / "products.txt"
    text = path.read_text()
    if problem == "wrong_process":
        text = text.replace(audit.RAW_CRV, audit.RAW_CRV + "Wrong")
    elif problem == "ambiguous":
        text += "\nmu2e::StepPointMCs_g4run_CRV_Other. 1 1 1"
    elif problem == "other_module":
        text += "\nmu2e::StepPointMCs_other_CRV_R3Inclusive. 1 1 1"
    elif problem == "missing_raw":
        text = text.replace(audit.RAW_CRV, "irrelevant")
    elif problem == "wrong_steps":
        text = text.replace(audit.CONVERTED_STEPS, audit.CONVERTED_STEPS + "Wrong")
    elif problem == "extra_steps":
        text += "\nmu2e::CrvSteps_other__Other. 1 1 1"
    else:
        branch = {"old_trigger": "art::TriggerResults_TriggerResults__R3Inclusive.",
                  "new_trigger": "art::TriggerResults_TriggerResults__R3InclusiveSteps.",
                  "generator": "mu2e::GenParticles_cosmicTimeOffset__Primary.",
                  "sim": "mu2e::SimParticlemv_g4run__R3Inclusive.",
                  "status": "mu2e::StatusG4_g4run__R3Inclusive."}[problem]
        text = text.replace(branch, "irrelevant.")
    path.write_text(text)
    with pytest.raises(ValueError, match="products"):
        audit.run(raw, converted, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("field,old,new", [("status", 0, 1), ("nG4Tracks", 100, 101),
    ("overflowSimP", 0, 1), ("nStepLimit", 0, 1), ("event", 10, 11), ("malformed", 0, 0)])
def test_converted_status_must_match_raw_exactly(tmp_path, audit, fixtures, field, old, new):
    raw, converted, _, _ = _directory(tmp_path, audit, fixtures)
    path = converted / "status.log"
    text = path.read_text()
    text = (text.replace("[10]", "[11]") if field == "event" else
            text.replace("nStepLimit: 0", "nStepLimit: unknown", 1) if field == "malformed" else
            text.replace(f"{field}: {old}", f"{field}: {new}", 1))
    path.write_text(text)
    with pytest.raises(ValueError):
        audit.run(raw, converted, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("problem", ["status", "warning", "incomplete_raw", "incomplete_conversion"])
def test_prior_clean_json_never_overrides_actual_gate(tmp_path, audit, fixtures, problem):
    raw, converted, _, _ = _directory(tmp_path, audit, fixtures)
    (raw / "capture-audit.json").write_text(json.dumps({"raw_capture_review_gate_passed": True,
                                                       "conversion_eligible": True}))
    if problem == "status":
        path = raw / "status.log"
        path.write_text(path.read_text().replace("status: 0", "status: 1", 1))
    elif problem == "warning":
        path = raw / "transport.log"
        path.write_text("Only 1 are saved in output collection.\n" + path.read_text())
    else:
        path = (raw / "transport.log" if problem == "incomplete_raw" else converted / "conversion.log")
        path.write_text("partial output, no normal art completion")
    with pytest.raises(ValueError, match="clean raw capture|normal art completion"):
        audit.run(raw, converted, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("problem", ["step", "parent", "cycle", "nonprimary_root", "missing_primary",
                                    "repeat_primary", "wrong_generator_product", "missing_census"])
def test_strict_converted_pointer_ancestor_and_all_pdg_bijection(tmp_path, audit, fixtures, problem):
    raw, converted, donor, _ = _directory(tmp_path, audit, fixtures)
    parts = donor["particles"]
    if problem == "step":
        donor["steps"].loc[0, "sp"] = 999
    elif problem == "parent":
        parts.loc[3, "parent_product_id"] = 999
    elif problem == "cycle":
        parts.loc[3, "parent"] = 130
    elif problem == "nonprimary_root":
        parts.loc[3, "parent"] = -1
        parts.loc[3, "parent_product_id"] = 0
    elif problem == "missing_primary":
        parts.loc[2, "isprim"] = 0  # The photon companion, not just muons.
    elif problem == "repeat_primary":
        parts.loc[1, "genidx"] = 0
        parts.loc[1, "pdg"] = 13
    elif problem == "wrong_generator_product":
        parts.loc[0, "gen_product_id"] = 999
    else:
        del donor["events"]
    fixtures._write(converted / "inclusive-converted-steps.root", donor)
    with pytest.raises(ValueError):
        audit.run(raw, converted, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("side,name", [("raw", name) for name in (
    "input-gen.root", "raw-gen.root", "status.log", "products.txt", "transport.log", "inclusive-raw-run1a.art")]
    + [("converted", name) for name in ("inclusive-converted-steps.root", "status.log", "products.txt",
                                        "conversion.log", "inclusive-converted-run1a.art")])
def test_every_missing_input_refuses(tmp_path, audit, fixtures, side, name):
    raw, converted, _, _ = _directory(tmp_path, audit, fixtures)
    ((raw if side == "raw" else converted) / name).unlink()
    with pytest.raises(ValueError, match="missing or nonregular"):
        audit.run(raw, converted, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("side,target", [("raw", name) for name in (
    "input-gen.root", "raw-gen.root", "status.log", "products.txt", "transport.log", "inclusive-raw-run1a.art")]
    + [("converted", name) for name in ("inclusive-converted-steps.root", "status.log", "products.txt",
                                        "conversion.log", "inclusive-converted-run1a.art")]
    + [("source", name) for name in ("audit_inclusive_run1a_conversion.py", "audit_inclusive_run1a_capture.py",
                                     "audit_inclusive_run1a_pilot.py", "genassoc.py")])
def test_all_input_and_source_hash_drift_refuses(tmp_path, audit, fixtures, monkeypatch, side, target):
    raw, converted, _, _ = _directory(tmp_path, audit, fixtures)
    sha = audit._sha256
    calls = {}

    def drift(path):
        key = str(path)
        calls[key] = calls.get(key, 0) + 1
        matches = Path(path).name == target and (side == "source" or Path(path).parent == (
            raw if side == "raw" else converted))
        return "changed" if matches and calls[key] > 1 else sha(path)

    monkeypatch.setattr(audit, "_sha256", drift)
    with pytest.raises(ValueError, match="hash drift"):
        audit.run(raw, converted, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("problem", ["symlink", "hardlink", "same_directory", "directory_symlink", "empty"])
def test_alias_and_empty_evidence_refuse(tmp_path, audit, fixtures, problem):
    raw, converted, _, _ = _directory(tmp_path, audit, fixtures)
    path = converted / "status.log"
    if problem in ("symlink", "hardlink"):
        path.unlink()
        if problem == "symlink":
            path.symlink_to(raw / "status.log")
        else:
            os.link(raw / "status.log", path)
    elif problem == "same_directory":
        converted = raw
    elif problem == "directory_symlink":
        link = tmp_path / "linked"
        link.symlink_to(converted, target_is_directory=True)
        converted = link
    else:
        path.write_text("")
    with pytest.raises(ValueError, match="alias|empty input"):
        audit.run(raw, converted, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("problem", ["existing", "symlink", "dangling", "input_alias"])
def test_exclusive_output(tmp_path, audit, fixtures, problem):
    raw, converted, _, _ = _directory(tmp_path, audit, fixtures)
    out = tmp_path / "audit.json"
    if problem == "existing":
        out.write_text("keep")
    elif problem in ("symlink", "dangling"):
        out.symlink_to(raw / ("status.log" if problem == "symlink" else "missing"))
    else:
        out = raw / "input-gen.root"
    before = _snapshot(raw, converted)
    with pytest.raises(FileExistsError):
        audit.run(raw, converted, 10, out)
    assert _snapshot(raw, converted) == before
    if problem == "existing":
        assert out.read_text() == "keep"


@pytest.mark.parametrize("n,expected", [(0, None), (True, None), (1.0, None), (11, None),
    (10, [1400, 101, 1]), (1, [1400, 101, 2]), (1, [1400., 101, 1]), (1, [1400, 101, 2**32])])
def test_bounds_and_expected_event(tmp_path, audit, fixtures, n, expected):
    raw, converted, _, _ = _directory(tmp_path, audit, fixtures, 1)
    with pytest.raises(ValueError):
        audit.run(raw, converted, n, tmp_path / "audit.json", expected)
    assert not (tmp_path / "audit.json").exists()


def test_cli_success_exclusivity_and_invalid_bound(tmp_path, audit, fixtures):
    raw, converted, _, _ = _directory(tmp_path, audit, fixtures, 1)
    out = tmp_path / "cli.json"
    command = [sys.executable, str(ROOT / "scripts/audit_inclusive_run1a_conversion.py"),
               "--raw-directory", str(raw), "--conversion-directory", str(converted),
               "--events", "1", "--out", str(out), "--expected-event", "1400", "101", "1"]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "NO_FETCH": "1"}
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    _flags(json.loads(out.read_text()))
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0
    command[command.index("--events") + 1] = "2"
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0
