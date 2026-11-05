"""Bounded pilot software tests: exact original identities, never physics admission."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit():
    spec = importlib.util.spec_from_file_location("inclusive_pilot", ROOT / "scripts/audit_inclusive_run1a_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _typed(frame, unsigned):
    for column in frame:
        if column in unsigned:
            frame[column] = frame[column].astype(np.uint32)
        elif column not in ("e", "t", "x", "y", "z", "px", "py", "pz", "se", "st"):
            frame[column] = frame[column].astype(np.int64)
    return frame


def _frames(audit, n=10):
    census = pd.DataFrame({"run": 1400, "subrun": 101, "event": np.arange(1, n + 1),
                           "schema_version": 2, "associations_checked": 1, "full_generator_stream": 0,
                           "gen_product_id": 42, "sim_product_id": 73, "n_gen": 0,
                           "n_particles": 0, "n_steps": 0})
    census.loc[0, ["n_gen", "n_particles", "n_steps"]] = [3, 5, 3]
    if n == 10:
        census.loc[1, ["n_gen", "n_particles"]] = [1, 1]
    census = _typed(census, audit.KEY + ["schema_version", "gen_product_id", "sim_product_id"])
    gen = pd.DataFrame({"event": [1, 1, 1] + ([2] if n == 10 else []),
                        "idx": [0, 1, 2] + ([0] if n == 10 else []),
                        "pdg": [13, -13, 22] + ([11] if n == 10 else [])})
    gen = gen.assign(run=1400, subrun=101, gen_product_id=42, genid=7,
                     e=200., t=0., x=1., y=2., z=3., px=4., py=5., pz=6.)
    gen = _typed(gen, audit.KEY + ["gen_product_id"])
    # Arbitrary, non-source-index keys; two descendant steps share one ancestor.
    parts = pd.DataFrame([[1, 40, -1, 0, 1, 13], [1, 70, -1, 1, 1, -13],
                          [1, 90, -1, 2, 1, 22], [1, 120, 40, 0, 0, 11],
                          [1, 130, 120, -1, 0, 11]] +
                         ([[2, 9, -1, 0, 1, 11]] if n == 10 else []),
                         columns=["event", "key", "parent", "genidx", "isprim", "pdg"])
    parts = parts.assign(run=1400, subrun=101, sim_product_id=73, parent_product_id=0,
                         gen_product_id=42, se=200., st=0.)
    parts.loc[parts.parent.ge(0), "parent_product_id"] = 73
    parts.loc[parts.genidx.eq(-1), "gen_product_id"] = 0
    parts = _typed(parts, audit.KEY + ["sim_product_id", "parent_product_id", "gen_product_id"])
    steps = pd.DataFrame({"run": [1400] * 3, "subrun": [101] * 3, "event": [1] * 3,
                          "sp": [120, 130, 90], "sp_product_id": [73] * 3})
    steps = _typed(steps, audit.KEY + ["sp_product_id"])
    donor = {"events": census, "gen": gen, "particles": parts, "steps": steps}
    full = {"events": census.copy(), "gen": gen.copy()}
    full["events"]["associations_checked"] = np.int64(0)
    full["events"]["full_generator_stream"] = np.int64(1)
    full["events"]["sim_product_id"] = np.uint32(0)
    full["events"][["n_particles", "n_steps"]] = np.int64(0)
    return donor, full, audit.parse_status(_log(n))


def _log(n=10):
    # Only text generation/parser loops; frame analysis is vectorized.
    return "startup\n" + "".join(
        f"###############  PrintModule Run/Subrun/Event [1400] [101] [{event}]\n"
        "ProductPrint mu2e::StatusG4_g4run__R3Inclusive\n"
        "status: 0 nG4Tracks: 100 overflowSimP: 0 nStepLimit: 0 CPU: 1.2e-3 Wall: 0.01\n"
        for event in range(1, n + 1)) + "End processing\n"


def _products(audit):
    return "\n".join(audit.PRODUCTS + ["art::TriggerResults_TriggerResults__R3Inclusive"])


def _write(path, frames):
    with uproot.recreate(path) as output:
        for name, frame in frames.items():
            output.mktree("dump/" + name, {col: frame[col].to_numpy() for col in frame})


def _directory(tmp_path, audit, n=10):
    donor, full, _ = _frames(audit, n)
    _write(tmp_path / "inclusive-steps.root", donor)
    _write(tmp_path / "input-gen.root", full)
    (tmp_path / "status.log").write_text(_log(n))
    (tmp_path / "products.txt").write_text(_products(audit))
    return donor, full


def test_companions_multi_muon_empty_zero_steps_own_ancestry_and_no_mutation(audit):
    donor, full, status = _frames(audit)
    before = deepcopy((donor, full, status))
    result = audit.analyze_frames(donor, full, status, 10)
    assert result["events"] == 10
    assert result["empty_generator_events"] == 8
    assert result["multi_muon_events"] == 1
    assert result["primary_muons"] == 2
    assert result["distinct_crv_step_ancestor_identities"] == 2
    assert result["distinct_crv_step_muon_ancestor_identities"] == 1
    assert result["zero_step_events"] == 9
    assert result["counts"] == {"generators": 4, "actual_primaries": 4, "particles": 6,
                                "steps": 3, "status_records": 10}
    assert result["physics_admitted"] is False
    assert result["global_denominator_completeness_established"] is False
    for after, original in zip((donor, full), before[:2]):
        for name in after:
            pd.testing.assert_frame_equal(after[name], original[name])
    pd.testing.assert_frame_equal(status, before[2])


@pytest.mark.parametrize("field", ["pdg", "genid", "e", "t", "x", "y", "z", "px", "py", "pz"])
def test_exact_payload_one_ulp_not_tolerance(audit, field):
    donor, full, status = _frames(audit)
    old = full["gen"].loc[0, field]
    full["gen"].loc[0, field] = old + 1 if field in ("pdg", "genid") else np.nextafter(old, np.inf)
    with pytest.raises(ValueError, match="exact mismatch"):
        audit.analyze_frames(donor, full, status, 10)


@pytest.mark.parametrize("empty", [False, True])
def test_missing_event_including_empty_generator_event(audit, empty):
    donor, full, status = _frames(audit)
    event = 10 if empty else 2
    for name in donor:
        donor[name] = donor[name].loc[donor[name].event.ne(event)]
    with pytest.raises(ValueError, match="count mismatch"):
        audit.analyze_frames(donor, full, status, 10)


@pytest.mark.parametrize("scope", ["run", "subrun", "event", "gen_product_id"])
def test_full_integer_event_and_product_identity(audit, scope):
    donor, full, status = _frames(audit)
    full["events"][scope] += np.uint32(1)
    full["gen"][scope] += np.uint32(1)
    with pytest.raises(ValueError, match="exact mismatch"):
        audit.analyze_frames(donor, full, status, 10)


@pytest.mark.parametrize("failure", ["missing_companion", "missing_muon", "repeat_pointer", "wrong_pdg",
                                    "wrong_product", "wrong_index", "parentless_not_primary"])
def test_actual_primary_complete_bijection_all_pdgs(audit, failure):
    donor, full, status = _frames(audit)
    parts = donor["particles"]
    if failure in ("missing_companion", "missing_muon"):
        # Removing isprim is enough to expose an omitted input without breaking census.
        parts.loc[2 if failure == "missing_companion" else 1, "isprim"] = 0
    elif failure == "repeat_pointer":
        parts.loc[1, "genidx"] = 0
    elif failure == "wrong_pdg":
        parts.loc[0, "pdg"] = -13
    elif failure == "wrong_product":
        parts.loc[0, "gen_product_id"] = 999
    elif failure == "wrong_index":
        parts.loc[0, "genidx"] = 100
    else:
        parts.loc[0, "isprim"] = 0
    with pytest.raises(ValueError):
        audit.analyze_frames(donor, full, status, 10)


@pytest.mark.parametrize("which,tree", [("donor", "events"), ("donor", "gen"),
                                        ("donor", "particles"), ("full", "gen"), ("full", "events")])
def test_duplicate_identities(audit, which, tree):
    donor, full, status = _frames(audit)
    frames = donor if which == "donor" else full
    frames[tree] = pd.concat([frames[tree], frames[tree].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        audit.analyze_frames(donor, full, status, 10)


@pytest.mark.parametrize("change", ["missing", "duplicate", "mismatch", "nonzero", "overflow", "limit", "negative_tracks"])
def test_status_census_and_success_required(audit, change):
    donor, full, status = _frames(audit)
    if change == "missing":
        status = status.iloc[:-1]
    elif change == "duplicate":
        status = pd.concat([status, status.iloc[[0]]], ignore_index=True)
    elif change == "mismatch":
        status.loc[9, "event"] = 11
    else:
        column = {"nonzero": "status", "overflow": "overflowSimP", "limit": "nStepLimit",
                  "negative_tracks": "nG4Tracks"}[change]
        status.loc[0, column] = -1 if change == "negative_tracks" else 1
    with pytest.raises(ValueError):
        audit.analyze_frames(donor, full, status, 10)


def test_parser_bracketed_and_plain_headings(audit):
    status = audit.parse_status(_log(1))
    assert status.to_dict("records") == [{"run": 1400, "subrun": 101, "event": 1,
                                         "status": 0, "nG4Tracks": 100, "overflowSimP": 0, "nStepLimit": 0}]
    pd.testing.assert_frame_equal(status, audit.parse_status(_log(1).replace("[", "").replace("]", "")))


@pytest.mark.parametrize("change", ["no_heading", "bad_heading", "no_product", "wrong_product", "no_status",
                                    "unknown_status", "truncated", "duplicate_status", "duplicate_product", "bad_time"])
def test_parser_refuses_unknown_or_ambiguous_status(audit, change):
    text = _log(1)
    if change == "no_heading":
        text = text.split("\n", 2)[2]
    elif change == "bad_heading":
        text = text.replace("[1400]", "[unknown]")
    elif change == "no_product":
        text = text.replace("ProductPrint mu2e::StatusG4_g4run__R3Inclusive", "")
    elif change == "wrong_product":
        text = text.replace("R3Inclusive", "Primary")
    elif change == "no_status":
        text = text.replace("status: 0 nG4Tracks: 100 overflowSimP: 0 nStepLimit: 0 CPU: 1.2e-3 Wall: 0.01", "")
    elif change == "unknown_status":
        text = text.replace("status: 0", "status: unknown")
    elif change == "truncated":
        text = text.replace(" nStepLimit: 0", "")
    elif change == "duplicate_status":
        text += "status: 0 nG4Tracks: 100 overflowSimP: 0 nStepLimit: 0 CPU: 0 Wall: 0\n"
    elif change == "duplicate_product":
        text += "ProductPrint mu2e::StatusG4_g4run__R3Inclusive\n"
    else:
        text = text.replace("Wall: 0.01", "Wall: nan")
    with pytest.raises(ValueError):
        audit.parse_status(text)


@pytest.mark.parametrize("missing", range(6))
def test_mandatory_product_inventory(audit, missing):
    products = _products(audit).splitlines()
    del products[missing]
    with pytest.raises(ValueError, match="missing retained branches"):
        audit.validate_products("\n".join(products))
    assert len(audit.validate_products(_products(audit))) == 6
    with pytest.raises(ValueError):
        audit.validate_products(_products(audit).replace("g4run__R3Inclusive", "compressDetStepMCs__R3Inclusive"))


@pytest.mark.parametrize("n", [1, 10])
def test_run_real_root_validators_exclusive_output_and_hashes(tmp_path, audit, n):
    _directory(tmp_path, audit, n)
    out = tmp_path / "audit.json"
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}
    result = audit.run(tmp_path, n, out)
    assert result["pilot_checks_passed"] is True
    assert result["summary"]["events"] == n
    assert result["donor_validation"]["associations_checked"] is True
    assert result["generator_validation"]["schema_version"] == 2
    assert result["inputs_sha256_before"] == result["inputs_sha256_after"]
    assert result["sources_sha256_before"] == result["sources_sha256_after"]
    assert str(ROOT / "scripts/audit_inclusive_run1a_pilot.py") in result["sources_sha256_before"]
    assert result["physics_admitted"] is False
    assert result["global_denominator_completeness_established"] is False
    assert json.loads(out.read_text()) == result
    for name, digest in before.items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        audit.run(tmp_path, n, out)


def test_run_refuses_status_without_writing(tmp_path, audit):
    _directory(tmp_path, audit)
    (tmp_path / "status.log").write_text(_log().replace("status: 0", "status: 1", 1))
    out = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="unknown, not zero"):
        audit.run(tmp_path, 10, out)
    assert not out.exists()


def test_run_reuses_strict_parent_step_validator(tmp_path, audit):
    donor, _ = _directory(tmp_path, audit)
    donor["steps"].loc[0, "sp"] = 999
    _write(tmp_path / "inclusive-steps.root", donor)
    with pytest.raises(ValueError, match="strict generator association validation failed"):
        audit.run(tmp_path, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("target", ["status.log", "audit_inclusive_run1a_pilot.py"])
def test_hash_drift_refused_before_output(tmp_path, audit, monkeypatch, target):
    _directory(tmp_path, audit)
    original = audit._sha256
    calls = {}

    def drift(path):
        key = str(path)
        calls[key] = calls.get(key, 0) + 1
        return "changed" if str(path).endswith(target) and calls[key] > 1 else original(path)

    monkeypatch.setattr(audit, "_sha256", drift)
    with pytest.raises(ValueError, match="hash drift"):
        audit.run(tmp_path, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("n", [0, 2, 11, True, 1.0])
def test_only_bounded_pilots(audit, n):
    donor, full, status = _frames(audit)
    with pytest.raises(ValueError, match="1 or 10"):
        audit.analyze_frames(donor, full, status, n)


@pytest.mark.parametrize("n", [1, 10])
def test_all_empty_generators_real_root_pilot(tmp_path, audit, n):
    donor, full, status = _frames(audit, n)
    for frames in (donor, full):
        frames["events"][["n_gen", "n_particles", "n_steps"]] = np.int64(0)
        frames["gen"] = frames["gen"].iloc[:0]
    donor["particles"] = donor["particles"].iloc[:0]
    donor["steps"] = donor["steps"].iloc[:0]
    result = audit.analyze_frames(donor, full, status, n)
    assert result["empty_generator_events"] == result["zero_step_events"] == n
    assert result["counts"]["actual_primaries"] == result["primary_muons"] == 0
    assert result["distinct_crv_step_ancestor_identities"] == 0
    _write(tmp_path / "inclusive-steps.root", donor)
    _write(tmp_path / "input-gen.root", full)
    (tmp_path / "status.log").write_text(_log(n))
    (tmp_path / "products.txt").write_text(_products(audit))
    assert audit.run(tmp_path, n, tmp_path / "audit.json")["summary"] == result


@pytest.mark.parametrize("tree,count", [("gen", "n_gen"), ("particles", "n_particles"), ("steps", "n_steps")])
def test_event_payload_counts_exact(audit, tree, count):
    donor, full, status = _frames(audit)
    donor["events"].loc[0, count] += 1
    with pytest.raises(ValueError, match="payload count mismatch"):
        audit.analyze_frames(donor, full, status, 10)


@pytest.mark.parametrize("failure", ["cycle", "nonprimary_root", "missing_parent", "missing_step"])
def test_ancestry_is_never_approximated(audit, failure):
    donor, full, status = _frames(audit)
    parts = donor["particles"]
    if failure == "cycle":
        parts.loc[3, "parent"] = 130
    elif failure == "nonprimary_root":
        parts.loc[3, "parent"] = -1
        parts.loc[3, "parent_product_id"] = 0
    elif failure == "missing_parent":
        parts.loc[3, "parent"] = 999
    else:
        donor["steps"].loc[0, "sp"] = 999
    with pytest.raises(ValueError):
        audit.analyze_frames(donor, full, status, 10)


@pytest.mark.parametrize("tree,column", [("events", "event"), ("gen", "idx"), ("particles", "genidx"), ("steps", "sp")])
def test_float_identities_refused(audit, tree, column):
    donor, full, status = _frames(audit)
    donor[tree][column] = donor[tree][column].astype(float)
    with pytest.raises(ValueError, match="must be integer"):
        audit.analyze_frames(donor, full, status, 10)


def test_parser_refuses_extra_status_outside_product(audit):
    with pytest.raises(ValueError, match="exactly one status"):
        audit.parse_status(_log(1).replace("ProductPrint", "status: unknown\nProductPrint"))


def test_cli_exclusive_json(tmp_path, audit):
    import os
    import subprocess
    import sys

    _directory(tmp_path, audit, 1)
    out = tmp_path / "cli.json"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, str(ROOT / "scripts/audit_inclusive_run1a_pilot.py"),
               "--directory", str(tmp_path), "--events", "1", "--out", str(out)]
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(out.read_text())["requested_events"] == 1
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0


@pytest.mark.parametrize("prefix", ["status: unknown\n", "ProductPrint mu2e::StatusG4_g4run__R3Inclusive\n"])
def test_parser_refuses_orphan_status(audit, prefix):
    with pytest.raises(ValueError, match="without an event heading"):
        audit.parse_status(prefix + _log(1))


def test_run_legacy_external_generator_refused(tmp_path, audit):
    _, full = _directory(tmp_path, audit)
    # Legacy gen-only validation may succeed, but the pilot must require schema 2.
    full["gen"] = full["gen"].drop(columns=["gen_product_id"])
    _write(tmp_path / "input-gen.root", {"gen": full["gen"]})
    with pytest.raises(ValueError, match="strict schema v2 required"):
        audit.run(tmp_path, 10, tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


def test_expected_single_event_bound_to_actual_census(tmp_path, audit):
    _directory(tmp_path, audit, 1)
    out = tmp_path / "edge.json"
    with pytest.raises(ValueError, match="expected_event"):
        audit.run(tmp_path, 1, out, [1400, 101, 2])
    assert not out.exists()
    assert audit.run(tmp_path, 1, out, [1400, 101, 1])["expected_event"] == [1400, 101, 1]


@pytest.mark.parametrize("expected", [[1., 2, 3], [True, 2, 3], [-1, 2, 3], [1, 2, 2**32], [1, 2], "1:2:3"])
def test_expected_single_event_rejects_nonexact_identity(tmp_path, audit, expected):
    with pytest.raises(ValueError, match="expected_event"):
        audit.run(tmp_path, 1, tmp_path / "edge.json", expected)
