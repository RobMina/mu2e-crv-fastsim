"""Read-only bounded G4-only raw capture audit, never conversion or physics admission.

Run from the repository root with PYTHONPATH=. python3 scripts/audit_inclusive_run1a_capture.py
--directory RESULT --events 1 --out capture.json (or --events 10). Both generator
files must be independent schema-v2 full-generator dumps: input-gen.root from the
original input and raw-gen.root from retained raw art. Original ProductID/index
and all payload columns are compared exactly, never matched by payload.

Status failures are retained as unknown transport diagnostics, not refusals or zero
crossings. Even clean status cannot establish SimParticle/raw-pointer integrity,
raw preservation on overflow, or authorization for downstream conversion. This
checks bounded generator/census retention and independent status presence only.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import uproot

from crvfastsim import genassoc
from scripts import audit_inclusive_run1a_pilot as pilot

KEY = genassoc.KEY
PAYLOAD = pilot.PAYLOAD
GEN = pilot.GEN
STATUS = pilot.STATUS
PRODUCTS = [branch for branch in pilot.PRODUCTS if branch != "mu2e::CrvSteps_CrvSteps__R3Inclusive"]
FLAGS = {**pilot.FLAGS, "raw_pointer_integrity_established": False,
         "downstream_conversion_authorized": False, "conversion_eligible": False}
parse_status = pilot.parse_status
_sha256 = pilot._sha256


def validate_products(text):
    """Require raw retained inventory, without requiring downstream CrvSteps."""
    missing = [branch for branch in PRODUCTS
               if re.search(r"(?<![\w:])" + re.escape(branch) + r"(?=[\s.\"']|$)", text) is None]
    trigger = "art::TriggerResults_*__R3Inclusive"
    if re.search(r"(?<![\w:])art::TriggerResults_[A-Za-z0-9]+__R3Inclusive(?=[\s.\"']|$)", text) is None:
        missing.append(trigger)
    genassoc._require(not missing, f"products: missing retained branches {missing}")
    return PRODUCTS + [trigger]


def recording_log_gate(text):
    """Conservative whole-capture gate for warning-only sensitive-detector truncation.

    Absence of warnings is not a native hit-count certificate. Any ambiguous
    detector truncation warning quarantines the whole bounded capture for review.
    """
    patterns = [r"Maximum number of steps reached in", r"Only\s+\d+\s+are saved in output collection",
                r"Maximum number of particles reached in", r"SimParticleCollection exceeds its maximum"]
    markers = [pattern for pattern in patterns if re.search(pattern, text, re.I)]
    completed = "Art has completed and will exit with status 0." in text
    return {"art_completed_status_zero": completed, "truncation_warning_patterns": markers,
            "recording_log_gate_passed": completed and not markers,
            "native_raw_recording_completeness_established": False}


def _generator_frames(frames, events, label):
    """Validate complete full-generator roles and payloads without integer coercion."""
    census = pilot._frame(frames["events"], genassoc.EVENT_COLUMNS, [], KEY, label + " events")
    genassoc._integer(census, KEY + ["schema_version", "gen_product_id", "sim_product_id"],
                      label + " events", ("u", 4))
    genassoc._integer(census, ["associations_checked", "full_generator_stream"], label + " events")
    genassoc._integer(census, ["n_gen", "n_particles", "n_steps"], label + " events", ("i", 8))
    genassoc._require(len(census) == events, f"{label}: requested capture event count mismatch")
    genassoc._require(census.schema_version.eq(2), f"{label}: strict schema v2 required")
    genassoc._require(census.associations_checked.eq(0) & census.full_generator_stream.eq(1),
                      f"{label}: require full generator stream role")
    genassoc._require(census.gen_product_id.gt(0) & census.sim_product_id.eq(0),
                      f"{label}: invalid generator/SimParticle ProductID")
    genassoc._require(census.n_gen.ge(0) & census.n_particles.eq(0) & census.n_steps.eq(0),
                      f"{label}: invalid full generator counts")
    gen = pilot._frame(frames["gen"], GEN, [], KEY + ["idx"], label + " gen")
    genassoc._integer(gen, KEY + ["gen_product_id"], label + " gen", ("u", 4))
    genassoc._integer(gen, ["idx", "pdg", "genid"], label + " gen")
    rows = pilot._census(gen, census, "gen")
    genassoc._require(gen.gen_product_id.to_numpy() == census.gen_product_id.to_numpy()[rows],
                      f"{label}: generator ProductID differs from census")
    genassoc._require((gen.idx.to_numpy() >= 0) & (gen.idx.to_numpy() < census.n_gen.to_numpy()[rows]),
                      f"{label}: generator index out of range")
    genassoc._require(np.isfinite(gen[PAYLOAD[2:]].to_numpy()), f"{label}: nonfinite generator payload")
    return census, gen


def _expected_event(events, expected_event):
    if expected_event is not None:
        genassoc._require(events == 1 and isinstance(expected_event, (list, tuple)) and len(expected_event) == 3
                          and all(type(v) is int and 0 <= v <= np.iinfo(np.uint32).max for v in expected_event),
                          "expected_event requires one exact uint32 event triplet")


def analyze_frames(original, retained, status, events, expected_event=None):
    """Vectorized exact census/payload comparison with non-refusing failure diagnostics.

    Status, overflow and step-limit failures mark transport unknown. Negative track,
    overflow or step-limit counters are malformed; a signed nonzero status code is
    a failure diagnostic. No event has a measured crossing outcome in this audit.
    """
    genassoc._require(type(events) is int and events in (1, 10), "events must be 1 or 10")
    _expected_event(events, expected_event)
    oe, og = _generator_frames(original, events, "original")
    re_, rg = _generator_frames(retained, events, "retained")
    pilot._same_sorted(oe, re_, KEY, KEY + ["gen_product_id", "n_gen"], "event census")
    pilot._same_sorted(og, rg, KEY + ["idx"], GEN, "original generator identity/payload")
    if expected_event is not None:
        expected = np.asarray([expected_event], dtype=np.int64)
        genassoc._require(np.array_equal(oe[KEY].to_numpy(), expected)
                          and np.array_equal(re_[KEY].to_numpy(), expected),
                          "expected_event does not match original and retained event census")
    statuses = pilot._frame(status, KEY + STATUS, KEY + STATUS, KEY, "status")
    genassoc._require(statuses[KEY].to_numpy() <= np.iinfo(np.uint32).max,
                      "status: event identity exceeds uint32")
    pilot._same_sorted(statuses, oe, KEY, KEY, "status census")
    genassoc._require(statuses[["nG4Tracks", "overflowSimP", "nStepLimit"]].to_numpy() >= 0,
                      "status: negative counter")
    statuses = statuses.sort_values(KEY).reset_index(drop=True)
    ok = statuses[["status", "overflowSimP", "nStepLimit"]].eq(0).all(axis=1)
    statuses["raw_transport_status_ok"] = ok
    statuses["transport_state"] = np.where(ok, "status_ok", "unknown")
    statuses["crossing_state"] = "unknown"
    muons = og.loc[og.pdg.isin([-13, 13])].groupby(KEY).size()
    return {**FLAGS, "capture_checks_passed": True, "raw_transport_status_ok": bool(ok.all()),
            "events": len(oe), "empty_generator_events": int(oe.n_gen.eq(0).sum()),
            "multi_muon_events": int(muons.gt(1).sum()),
            "counts": {"generators": len(og), "status_records": len(statuses),
                       "status_ok_events": int(ok.sum()), "unknown_transport_events": int((~ok).sum()),
                       "nonzero_status_events": int(statuses.status.ne(0).sum()),
                       "overflow_events": int(statuses.overflowSimP.gt(0).sum()),
                       "step_limit_events": int(statuses.nStepLimit.gt(0).sum()),
                       "unknown_crossing_events": len(oe)},
            "event_status": statuses.to_dict("records")}


def run(directory, events, out, expected_event=None):
    """Validate immutable inputs and source hashes before exclusive JSON creation.

    Hashes detect pre/post drift, not provenance or immutable snapshots. Refuse
    malformed evidence without writing a report; valid failed transport still
    produces its diagnostic report. No SimParticle or raw-pointer audit runs here.
    """
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    genassoc._require(type(events) is int and events in (1, 10), "events must be 1 or 10")
    _expected_event(events, expected_event)
    directory = Path(directory).resolve(strict=True)
    inputs = [directory / name for name in ("input-gen.root", "raw-gen.root", "status.log", "products.txt", "transport.log")]
    sources = [Path(__file__).resolve(), Path(pilot.__file__).resolve(), Path(genassoc.__file__).resolve()]
    before = {str(path): _sha256(path) for path in inputs}
    source_before = {str(path): _sha256(path) for path in sources}
    original_receipt = genassoc.validate_generator_stream(inputs[0])
    retained_receipt = genassoc.validate_generator_stream(inputs[1])
    genassoc._require(original_receipt["schema_version"] == retained_receipt["schema_version"] == 2,
                      "both generator streams: strict schema v2 required")
    products = validate_products(inputs[3].read_text())
    summary = analyze_frames(pilot._read(inputs[0], False), pilot._read(inputs[1], False),
                             parse_status(inputs[2].read_text()), events, expected_event)
    recording = recording_log_gate(inputs[4].read_text())
    after = {path: _sha256(path) for path in before}
    source_after = {path: _sha256(path) for path in source_before}
    genassoc._require(before == after and source_before == source_after, "input/source hash drift")
    result = {"schema": "crvfastsim.inclusive_run1a_capture.v1", "scope": "bounded_g4_only_raw_capture",
              **FLAGS, "capture_checks_passed": True,
              "raw_transport_status_ok": summary["raw_transport_status_ok"],
              "recording_log": recording,
              "raw_capture_review_gate_passed": summary["raw_transport_status_ok"] and recording["recording_log_gate_passed"],
              "requested_events": events, "expected_event": expected_event, "directory": str(directory),
              "summary": summary, "original_generator_validation": original_receipt,
              "retained_generator_validation": retained_receipt, "retained_products": products,
              "inputs_sha256_before": before, "inputs_sha256_after": after,
              "sources_sha256_before": source_before, "sources_sha256_after": source_after,
              "hashes_verified_after": True,
              "limitations": [
                  "Capture checks establish only bounded original generator/census equality and independent status presence.",
                  "Failed/overflow/step-limit transport is unknown, never a zero crossing or a raw-preservation proof.",
                  "Even clean status establishes neither SimParticle nor raw-pointer integrity; these are not audited here.",
                  "Sensitive-detector hit truncation can be warning-only; the whole capture is quarantined on a matching log marker.",
                  "No native raw attempted/saved-hit counts or effective sensitive-detector limit are certified by this log gate.",
                  "Downstream conversion requires a separate integrity audit and explicit user approval; it is not authorized.",
                  "Full campaign completeness, physics admission and global denominator completeness are not established.",
                  "Input/source hashes detect drift, not provenance or immutable snapshots."],
              "software_versions": {"numpy": np.__version__, "pandas": pd.__version__, "uproot": uproot.__version__}}
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--events", required=True, type=int, choices=(1, 10))
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-event", nargs=3, type=int, metavar=("RUN", "SUBRUN", "EVENT"))
    args = parser.parse_args()
    run(args.directory, args.events, args.out, args.expected_event)


if __name__ == "__main__":
    main()
