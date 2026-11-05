"""Read-only, bounded 1/10-event existing-generator pilot audit; not library admission.

Original generator identities are compared directly, never inferred from SimParticle
keys, payload proximity, rank or timing. Failed/unknown transport is a refusal, not zero.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import uproot

from crvfastsim import genassoc

KEY = genassoc.KEY
PAYLOAD = ["pdg", "genid", "e", "t", "x", "y", "z", "px", "py", "pz"]
GEN = KEY + ["idx", "gen_product_id"] + PAYLOAD
STATUS = ["status", "nG4Tracks", "overflowSimP", "nStepLimit"]
PRODUCTS = ["mu2e::StatusG4_g4run__R3Inclusive",
            "mu2e::SimParticlemv_g4run__R3Inclusive",
            "mu2e::StepPointMCs_g4run_CRV_R3Inclusive",
            "mu2e::CrvSteps_CrvSteps__R3Inclusive",
            "mu2e::GenParticles_cosmicTimeOffset__Primary"]
FLAGS = {"physics_admitted": False, "global_denominator_completeness_established": False}


def parse_status(text):
    """Parse independent PrintModule event blocks, refusing absent/ambiguous status.

    Text-block iteration is not an event-dataframe analysis loop. CPU/Wall are
    checked as finite numeric fields, not used as evidence of successful transport.
    """
    heading = re.compile(r"^\s*#{15}\s+PrintModule Run/Subrun/Event\s+"
                         r"\[?(\d+)\]?\s+\[?(\d+)\]?\s+\[?(\d+)\]?\s*$", re.M)
    heads = list(heading.finditer(text))
    genassoc._require(bool(heads), "status: no PrintModule event headings")
    genassoc._require(len(heads) == text.count("PrintModule Run/Subrun/Event"),
                      "status: malformed event heading")
    genassoc._require(not re.search(r"\bstatus:|ProductPrint\s+mu2e::StatusG4", text[:heads[0].start()]),
                      "status: status/product without an event heading")
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    line = re.compile(r"^\s*status:\s*(-?\d+)\s+nG4Tracks:\s*(-?\d+)\s+"
                      r"overflowSimP:\s*(-?\d+)\s+nStepLimit:\s*(-?\d+)\s+"
                      rf"CPU:\s*({number})\s+Wall:\s*({number})\s*$", re.M)
    marker = re.compile(r"^\s*ProductPrint\s+mu2e::StatusG4_g4run__R3Inclusive\s*:?[ \t]*$", re.M)
    rows = []
    for i, head in enumerate(heads):
        block = text[head.end():heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        markers = list(marker.finditer(block))
        genassoc._require(len(markers) == 1, "status: require exactly one StatusG4 product per event")
        genassoc._require(len(re.findall(r"\bstatus:", block)) == 1,
                          "status: require exactly one status per event")
        tail = block[markers[0].end():]
        tail = re.split(r"^\s*ProductPrint\s", tail, maxsplit=1, flags=re.M)[0]
        matches = list(line.finditer(tail))
        genassoc._require(len(matches) == 1 and len(re.findall(r"\bstatus:", tail)) == 1,
                          "status: require exactly one complete status record per event (unknown is not zero)")
        match = matches[0]
        times = np.asarray(match.groups()[4:], dtype=float)
        genassoc._require(np.isfinite(times) & (times >= 0), "status: invalid CPU/Wall")
        rows.append([*map(int, head.groups()), *map(int, match.groups()[:4])])
    return pd.DataFrame(rows, columns=KEY + STATUS, dtype=np.int64)


def validate_products(text):
    missing = [branch for branch in PRODUCTS
               if re.search(r"(?<![\w:])" + re.escape(branch) + r"(?=[\s.\"']|$)", text) is None]
    if re.search(r"(?<![\w:])art::TriggerResults_[A-Za-z0-9]+__R3Inclusive(?=[\s.\"']|$)", text) is None:
        missing.append("art::TriggerResults_*__R3Inclusive")
    genassoc._require(not missing, f"products: missing retained branches {missing}")
    return PRODUCTS + ["art::TriggerResults_*__R3Inclusive"]


def _frame(frame, columns, integers, unique, label):
    genassoc._require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique
                      and set(columns) <= set(frame.columns), f"{label}: missing/duplicate columns")
    result = frame[columns].copy()
    genassoc._integer(result, integers, label)
    if unique:
        genassoc._unique(result, unique, label)
    genassoc._require(result[KEY].to_numpy() >= 0, f"{label}: negative event identity")
    return result


def _census(frame, events, name):
    # _frame has already normalized integers; genassoc._census expects raw uint32.
    rows = genassoc._lookup(frame, events, KEY, KEY, f"{name} event census")
    genassoc._require(np.bincount(rows, minlength=len(events)) == events["n_" + name].to_numpy(),
                      f"{name}: event census payload count mismatch (including zero rows)")
    return rows


def _same_sorted(left, right, keys, columns, label):
    a = left.sort_values(keys).reset_index(drop=True)
    b = right.sort_values(keys).reset_index(drop=True)
    genassoc._require(len(a) == len(b), f"{label}: row count mismatch")
    # Compare each column separately: mixed float payload must never coerce integer keys.
    for column in columns:
        genassoc._require(a[column].to_numpy() == b[column].to_numpy(),
                          f"{label}: exact mismatch in {column}")


def analyze_frames(donor, full, status, events):
    """Vectorized supplemental checks; runtime also uses both public genassoc validators.

    donor/full are tree-name -> DataFrame mappings. Empty events are defined only
    by the complete events trees, never reconstructed from generator/step rows.
    """
    genassoc._require(type(events) is int and events in (1, 10), "events must be 1 or 10")
    censuses, generators = [], []
    for frames, is_donor in ((donor, True), (full, False)):
        census = _frame(frames["events"], genassoc.EVENT_COLUMNS, genassoc.EVENT_COLUMNS,
                        KEY, "events")
        genassoc._require(len(census) == events, "events: requested pilot count mismatch")
        genassoc._require(census.schema_version.eq(2), "events: require schema_version=2")
        genassoc._require(census.associations_checked.eq(int(is_donor)) &
                          census.full_generator_stream.eq(int(not is_donor)), "events: wrong stream role")
        genassoc._require(census.gen_product_id.gt(0), "events: null generator ProductID")
        genassoc._require(census[["n_gen", "n_particles", "n_steps"]].to_numpy() >= 0,
                          "events: negative counts")
        genassoc._require(census.sim_product_id.gt(0) if is_donor else census.sim_product_id.eq(0),
                          "events: wrong SimParticle ProductID")
        if not is_donor:
            genassoc._require(census[["n_particles", "n_steps"]].to_numpy() == 0,
                              "full stream: nonzero particle/step count")
        gen = _frame(frames["gen"], GEN, KEY + ["idx", "gen_product_id", "pdg", "genid"],
                     KEY + ["idx"], "gen")
        erows = _census(gen, census, "gen")
        genassoc._require(gen.gen_product_id.to_numpy() == census.gen_product_id.to_numpy()[erows],
                          "gen: ProductID differs from census")
        genassoc._require((gen.idx.to_numpy() >= 0) & (gen.idx.to_numpy() < census.n_gen.to_numpy()[erows]),
                          "gen: index out of range")
        genassoc._require(np.isfinite(gen[PAYLOAD[2:]].to_numpy()), "gen: nonfinite payload")
        censuses.append(census)
        generators.append(gen)
    de, fe = censuses
    dg, fg = generators
    _same_sorted(de, fe, KEY, KEY + ["gen_product_id", "n_gen"], "event census")
    _same_sorted(dg, fg, KEY + ["idx"], GEN, "original generator payload")
    parts = _frame(donor["particles"], genassoc.PART_COLUMNS,
                   KEY + ["key", "sim_product_id", "parent", "parent_product_id", "genidx",
                          "gen_product_id", "isprim", "pdg"], KEY + ["key"], "particles")
    steps = _frame(donor["steps"], genassoc.STEP_COLUMNS, genassoc.STEP_COLUMNS, None, "steps")
    prows = _census(parts, de, "particles")
    srows = _census(steps, de, "steps")
    genassoc._require(parts.key.ge(0) & parts.isprim.isin([0, 1]), "particles: invalid key/isprim")
    genassoc._require(parts.sim_product_id.to_numpy() == de.sim_product_id.to_numpy()[prows],
                      "particles: wrong SimParticle ProductID")
    genassoc._require(steps.sp_product_id.to_numpy() == de.sim_product_id.to_numpy()[srows],
                      "steps: wrong SimParticle ProductID")
    for key, product in (("parent", "parent_product_id"), ("genidx", "gen_product_id")):
        genassoc._require(parts[key].ge(-1) & (parts[key].eq(-1) == parts[product].eq(0)),
                          f"particles: invalid {key}/{product} pointer")
    hasgen = parts.genidx.ge(0).to_numpy()
    genassoc._lookup(parts.loc[hasgen], fg, KEY + ["gen_product_id", "genidx"],
                     KEY + ["gen_product_id", "idx"], "original generator pointer")
    primary = parts.isprim.eq(1).to_numpy()
    prim = parts.loc[primary]
    genassoc._require(prim.parent.eq(-1) & prim.genidx.ge(0), "primary: missing original pointer or not actual root")
    genassoc._require(np.isfinite(prim[["se", "st"]].to_numpy()), "primary: nonfinite energy/time")
    identities = KEY + ["gen_product_id", "genidx"]
    genassoc._unique(prim, identities, "primary generator identities")
    grows = genassoc._lookup(prim, fg, identities, KEY + ["gen_product_id", "idx"], "primary original generator")
    genassoc._require(len(prim) == len(fg), "primary: missing input generator identities")
    genassoc._require(prim.pdg.to_numpy() == fg.pdg.to_numpy()[grows], "primary: signed PDG mismatch")
    hasparent = parts.parent.ge(0).to_numpy()
    ancestor = np.arange(len(parts), dtype=np.int64)
    ancestor[hasparent] = genassoc._lookup(parts.loc[hasparent], parts,
        KEY + ["parent_product_id", "parent"], KEY + ["sim_product_id", "key"], "parent")
    for _ in range(len(parts).bit_length()):
        advanced = ancestor[ancestor]
        if np.array_equal(advanced, ancestor):
            break
        ancestor = advanced
    genassoc._require(~hasparent[ancestor], "particles: ancestry cycle")
    used = genassoc._lookup(steps, parts, KEY + ["sp_product_id", "sp"],
                            KEY + ["sim_product_id", "key"], "steps")
    roots = np.unique(ancestor[used])
    genassoc._require(primary[roots], "steps: nonprimary/truncated root")
    statuses = _frame(status, KEY + STATUS, KEY + STATUS, KEY, "status")
    _same_sorted(statuses, de, KEY, KEY, "status census")
    genassoc._require(statuses[["status", "overflowSimP", "nStepLimit"]].to_numpy() == 0,
                      "status: failed/overflow/step-limit transport is unknown, not zero")
    genassoc._require(statuses.nG4Tracks.ge(0), "status: negative track count")
    muon_counts = fg.loc[fg.pdg.isin([-13, 13])].groupby(KEY).size()
    return {**FLAGS, "events": len(de), "empty_generator_events": int(de.n_gen.eq(0).sum()),
            "multi_muon_events": int(muon_counts.gt(1).sum()),
            "primary_muons": int(prim.pdg.isin([-13, 13]).sum()),
            "distinct_crv_step_ancestor_identities": len(roots),
            "distinct_crv_step_muon_ancestor_identities": int(parts.iloc[roots].pdg.isin([-13, 13]).sum()),
            "zero_step_events": int(de.n_steps.eq(0).sum()),
            "counts": {"generators": len(fg), "actual_primaries": len(prim), "particles": len(parts),
                       "steps": len(steps), "status_records": len(statuses)}}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path, donor):
    columns = {"events": genassoc.EVENT_COLUMNS, "gen": GEN}
    if donor:
        columns.update(particles=genassoc.PART_COLUMNS, steps=genassoc.STEP_COLUMNS)
    with uproot.open(path) as root:
        return {name: genassoc._read(root, name, names, {}) for name, names in columns.items()}


def run(directory, events, out, expected_event=None):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    genassoc._require(type(events) is int and events in (1, 10), "events must be 1 or 10")
    if expected_event is not None:
        genassoc._require(events == 1 and isinstance(expected_event, (list, tuple)) and len(expected_event) == 3
                          and all(type(v) is int and 0 <= v <= np.iinfo(np.uint32).max for v in expected_event),
                          "expected_event requires one exact uint32 event triplet")
    directory = Path(directory).resolve(strict=True)
    inputs = [directory / name for name in ("inclusive-steps.root", "input-gen.root", "status.log", "products.txt")]
    sources = [Path(__file__).resolve(), Path(genassoc.__file__).resolve()]
    before = {str(path): _sha256(path) for path in inputs}
    source_before = {str(path): _sha256(path) for path in sources}
    donor_receipt = genassoc.validate_donor(inputs[0])
    full_receipt = genassoc.validate_generator_stream(inputs[1])
    genassoc._require(full_receipt["schema_version"] == 2, "full generator stream: strict schema v2 required")
    retained = validate_products(inputs[3].read_text())
    donor, full = _read(inputs[0], True), _read(inputs[1], False)
    if expected_event is not None:
        genassoc._require(np.array_equal(full["events"][KEY].to_numpy(), np.asarray([expected_event], dtype=np.int64)),
                          "expected_event does not match actual input event census")
    summary = analyze_frames(donor, full, parse_status(inputs[2].read_text()), events)
    after = {path: _sha256(path) for path in before}
    source_after = {path: _sha256(path) for path in source_before}
    genassoc._require(before == after and source_before == source_after, "input/source hash drift")
    result = {"schema": "crvfastsim.inclusive_run1a_pilot.v1", "scope": "bounded_existing_generator_pilot_only",
              **FLAGS, "pilot_checks_passed": True, "requested_events": events, "directory": str(directory),
              "summary": summary, "donor_validation": donor_receipt, "generator_validation": full_receipt,
              "retained_products": retained, "inputs_sha256_before": before, "inputs_sha256_after": after,
              "sources_sha256_before": source_before, "sources_sha256_after": source_after,
              "hashes_verified_after": True, "expected_event": expected_event,
              "limitations": ["Only the supplied bounded input dump is compared; full campaign completeness is not established.",
                              "Hashes detect drift, not source provenance or immutable snapshots.",
                              "Zero-step events are bounded successful-transport observations, not historical recovery.",
                              "No physics admission, global denominator or runtime extrapolation is supplied."],
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
