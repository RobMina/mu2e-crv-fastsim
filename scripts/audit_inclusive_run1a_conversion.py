"""Read-only retrospective bounded saved-raw conversion checks, never a launch token.

No transport, conversion, art execution or build is performed. Independently dumped
schema-v2 tables and logs must already exist. A clean recomputed capture gate and
strict converted associations are necessary bounded checks, not proof of every raw
pointer/daughter, native attempted/saved-hit completeness or permission to convert.
"""
import argparse
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
import uproot

from crvfastsim import genassoc
from scripts import audit_inclusive_run1a_capture as capture
from scripts import audit_inclusive_run1a_pilot as pilot

KEY = genassoc.KEY
FLAGS = {**capture.FLAGS, "native_raw_completeness_established": False,
         "native_raw_recording_completeness_established": False}
RAW_FILES = ("input-gen.root", "raw-gen.root", "status.log", "products.txt", "transport.log",
             "inclusive-raw-run1a.art")
CONVERSION_FILES = ("inclusive-converted-steps.root", "status.log", "products.txt", "conversion.log",
                    "inclusive-converted-run1a.art")
RAW_CRV = "mu2e::StepPointMCs_g4run_CRV_R3Inclusive"
CONVERTED_STEPS = "mu2e::CrvSteps_CrvSteps__R3InclusiveSteps"
_sha256 = pilot._sha256


def _inventory(text, kind):
    # Match whole branch tokens and deduplicate repeated size summaries.
    return set(re.findall(r"(?<![\w:])" + re.escape(kind)
                          + r"_[^\s.\"'|,;()\[\]{}<>]+", text))


def validate_products(raw_text, converted_text):
    """Reject competing raw CRV or unexpected converted CrvSteps collections."""
    raw_required = capture.validate_products(raw_text)
    converted_required = capture.validate_products(converted_text)
    inventories = []
    for label, text in (("raw", raw_text), ("converted", converted_text)):
        crv = {name for name in _inventory(text, "mu2e::StepPointMCs") if "_CRV_" in name}
        genassoc._require(crv == {RAW_CRV},
                          f"{label} products: require exactly one unique raw CRV collection, R3Inclusive")
        steps = _inventory(text, "mu2e::CrvSteps")
        genassoc._require(steps == (set() if label == "raw" else {CONVERTED_STEPS}),
                          f"{label} products: unexpected or missing CrvSteps products")
        inventories.append(_inventory(text, "art::TriggerResults"))
    new_trigger = re.compile(r"art::TriggerResults_[A-Za-z0-9]+__R3InclusiveSteps\Z")
    genassoc._require(any(new_trigger.fullmatch(name) for name in inventories[1]),
                      "converted products: missing new R3InclusiveSteps TriggerResults")
    genassoc._require(inventories[0] <= inventories[1],
                      "converted products: missing retained original TriggerResults")
    return {"raw_required": raw_required, "converted_required": converted_required + [CONVERTED_STEPS],
            "raw_trigger_results": sorted(inventories[0]), "converted_trigger_results": sorted(inventories[1])}


def _input_identities(paths):
    """Require separate, regular, nonempty evidence files; refuse symlink/hardlink aliases."""
    identities = {}
    for path in paths:
        genassoc._require(not path.is_symlink(), f"input symlink alias: {path}")
        genassoc._require(path.is_file(), f"missing or nonregular input: {path}")
        stat = path.stat()
        genassoc._require(stat.st_size > 0, f"empty input: {path}")
        identities[str(path)] = (stat.st_dev, stat.st_ino)
    genassoc._require(len(set(identities.values())) == len(paths), "input file alias")
    return identities


def run(raw_directory, conversion_directory, events, out, expected_event=None):
    """Recompute evidence and bind inputs/sources pre/post, then exclusively write JSON.

    Any existing capture-audit.json is deliberately ignored. The recomputed receipt
    exists only in a temporary directory outside the raw inputs and is removed even
    on refusal. Hash binding detects drift, not provenance of art or its readbacks.
    """
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    genassoc._require(type(events) is int and events in (1, 10), "events must be 1 or 10")
    capture._expected_event(events, expected_event)
    directories = [Path(raw_directory), Path(conversion_directory)]
    for directory in directories:
        genassoc._require(directory.is_dir() and not directory.is_symlink(),
                          f"missing or symlink directory alias: {directory}")
    raw_directory, conversion_directory = [directory.resolve(strict=True) for directory in directories]
    genassoc._require(raw_directory != conversion_directory, "raw/conversion directory alias")
    inputs = ([raw_directory / name for name in RAW_FILES]
              + [conversion_directory / name for name in CONVERSION_FILES])
    identities_before = _input_identities(inputs)
    sources = [Path(module.__file__).resolve() for module in (capture, pilot, genassoc)] + [Path(__file__).resolve()]
    before = {str(path): _sha256(path) for path in inputs}
    source_before = {str(path): _sha256(path) for path in sources}
    products = validate_products((raw_directory / "products.txt").read_text(),
                                 (conversion_directory / "products.txt").read_text())
    with tempfile.TemporaryDirectory(prefix="inclusive-conversion-capture-") as temporary:
        capture_receipt = capture.run(raw_directory, events, Path(temporary) / "capture-audit.json", expected_event)
    genassoc._require(capture_receipt["raw_capture_review_gate_passed"] is True,
                      "recomputed capture: actual clean raw capture review gate required")
    raw_status = pilot.parse_status((raw_directory / "status.log").read_text())
    converted_status = pilot.parse_status((conversion_directory / "status.log").read_text())
    pilot._same_sorted(raw_status, converted_status, KEY, KEY + pilot.STATUS, "converted/raw independent status")
    completed = "Art has completed and will exit with status 0." in (conversion_directory / "conversion.log").read_text()
    genassoc._require(completed, "conversion: normal art completion with status zero required")
    donor_path = conversion_directory / "inclusive-converted-steps.root"
    original_path = raw_directory / "input-gen.root"
    donor_receipt = genassoc.validate_donor(donor_path)
    generator_receipt = genassoc.validate_generator_stream(original_path)
    genassoc._require(donor_receipt["schema_version"] == generator_receipt["schema_version"] == 2,
                      "converted donor and original generator: strict schema v2 required")
    summary = {**pilot.analyze_frames(pilot._read(donor_path, True), pilot._read(original_path, False),
                                     converted_status, events), **FLAGS,
               "unknown_crossing_events": events}
    identities_after = _input_identities(inputs)
    after = {path: _sha256(path) for path in before}
    source_after = {path: _sha256(path) for path in source_before}
    genassoc._require(identities_before == identities_after and before == after and source_before == source_after,
                      "input/source hash drift or file identity drift")
    result = {"schema": "crvfastsim.inclusive_run1a_conversion.v1",
              "scope": "retrospective_bounded_saved_raw_conversion_checks_only", **FLAGS,
              "conversion_checks_passed": True, "requested_events": events, "expected_event": expected_event,
              "raw_directory": str(raw_directory), "conversion_directory": str(conversion_directory),
              "capture_receipt": capture_receipt, "summary": summary,
              "converted_donor_validation": donor_receipt, "original_generator_validation": generator_receipt,
              "retained_products": products, "conversion_art_completed_status_zero": completed,
              "converted_status_matches_raw": True,
              "inputs_sha256_before": before, "inputs_sha256_after": after,
              "sources_sha256_before": source_before, "sources_sha256_after": source_after,
              "hashes_verified_after": True,
              "limitations": [
                  "conversion_checks_passed describes retrospective bounded readback checks only, never a launch token or authorization.",
                  "The capture receipt is freshly recomputed; a provided capture-audit.json is neither used nor trusted.",
                  "Strict converted donor checks establish exact all-PDG primary bijection, ancestors and converted step associations only.",
                  "Not every pre-conversion raw pointer, daughter link or raw-hit membership is audited.",
                  "Native raw attempted/saved-hit counts and native recording completeness remain unproven.",
                  "A descendant deposit or its primary ancestor is not a primary bar crossing; zero steps are not a zero-crossing claim.",
                  "No physics admission, global denominator completeness, conversion eligibility or downstream authorization is established.",
                  "Hashes detect pre/post drift, not provenance, immutable snapshots or independent proof that dumps came from the hashed art."],
              "software_versions": {"numpy": np.__version__, "pandas": pd.__version__, "uproot": uproot.__version__}}
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--conversion-directory", required=True)
    parser.add_argument("--events", required=True, type=int, choices=(1, 10))
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-event", nargs=3, type=int, metavar=("RUN", "SUBRUN", "EVENT"))
    args = parser.parse_args()
    run(args.raw_directory, args.conversion_directory, args.events, args.out, args.expected_event)


if __name__ == "__main__":
    main()
