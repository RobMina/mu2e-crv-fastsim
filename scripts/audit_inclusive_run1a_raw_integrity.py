"""Read-only exact bounded saved-raw integrity audit; never launches art or conversion.

Requires the schema-v1 rawdump export, independent exporter completion log and
existing capture files. Integer identities never pass through float or signed
int64 normalization. Malformed evidence is refused; valid failure diagnostics are
retained with every crossing outcome unknown. See docs/native-raw-integrity.md.
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
TAGS = {"gen_tag": "cosmicTimeOffset::Primary", "sim_tag": "g4run::R3Inclusive",
        "raw_tag": "g4run:CRV:R3Inclusive", "status_tag": "g4run::R3Inclusive"}
POINTER = {"nonnull": "int32", "product_id": "uint32", "key": "uint64",
           "available": "int32", "dereferenced": "int32", "member": "int32",
           "matches_selected": "int32", "error": "int32"}


def _schema():
    common = dict.fromkeys(KEY, "uint32")
    schemas = {
        "events": {**common, "schema_version": "uint32",
                   **dict.fromkeys([x + "_product_id" for x in ("gen", "sim", "raw", "status")], "uint32"),
                   **dict.fromkeys([x + "_present" for x in ("gen", "sim", "raw", "status")]
                                   + ["collection_error"] + pilot.STATUS + ["nFieldPropagator"], "int32"),
                   **dict.fromkeys(["n_gen", "n_particles", "n_raw", "n_daughters"], "uint64"),
                   "cpuTime": "float64", "realTime": "float64", **dict.fromkeys(TAGS, "string")},
        "gen": {**common, "idx": "uint64", "gen_product_id": "uint32", "pdg": "int32", "genid": "int32",
                **dict.fromkeys(pilot.PAYLOAD[2:], "float64")},
        "particles": {**common, **dict.fromkeys(["key", "id", "n_daughters"], "uint64"),
                      "sim_product_id": "uint32", "pdg": "int32", "isprim": "int32"},
        "raw": {**common, "idx": "uint64", "raw_product_id": "uint32", "bar": "uint32"},
        "daughters": {**common, "owner_key": "uint64", "idx": "uint64", "sim_product_id": "uint32"}}
    for tree, prefixes in (("particles", ("parent", "gen")), ("raw", ("sim",)), ("daughters", ("daughter",))):
        for prefix in prefixes:
            schemas[tree].update({prefix + "_" + name: dtype for name, dtype in POINTER.items()})
    return schemas


SCHEMA = _schema()
_require = genassoc._require
_sha256 = pilot._sha256


def _frames(frames):
    result = {}
    for tree, schema in SCHEMA.items():
        _require(tree in frames and isinstance(frames[tree], pd.DataFrame), f"missing rawdump/{tree}")
        frame = frames[tree]
        _require(frame.columns.is_unique and set(frame.columns) == set(schema), f"{tree}: exact schema columns required")
        for column, dtype in schema.items():
            a = frame[column].to_numpy()
            if dtype == "string":
                _require(pd.api.types.infer_dtype(frame[column], skipna=False) in ("string", "unicode"),
                         f"{tree}.{column}: string required")
            else:
                _require(a.dtype == np.dtype(dtype), f"{tree}.{column}: requires {dtype}, got {a.dtype}")
        result[tree] = frame.copy().reset_index(drop=True)
    return result


def _read(path):
    frames = {}
    with uproot.open(path) as root:
        for name, columns in SCHEMA.items():
            treepath = "rawdump/" + name
            cycles = [key for key in root.keys(recursive=True, cycle=True) if key.rsplit(";", 1)[0] == treepath]
            _require(len(cycles) == 1, f"{treepath}: exactly one unambiguous tree cycle required")
            tree = root[cycles[0]]
            _require(tree.classname == "TTree", f"{treepath}: TTree required")
            _require(set(tree.keys()) == set(columns), f"{treepath}: exact schema columns required")
            frames[name] = pd.DataFrame(tree.arrays(list(columns), library="np"))
    return _frames(frames)


def parse_status(text):
    """Retain printer CPU/Wall too; the existing parser first refuses malformed blocks."""
    frame = capture.parse_status(text)
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    times = re.findall(rf"CPU:\s*({number})\s+Wall:\s*({number})", text)
    _require(len(times) == len(frame), "status: ambiguous CPU/Wall records")
    frame[["cpuTime", "realTime"]] = np.asarray(times, dtype=np.float64)
    return frame


def _lookup(left, right, lcols, rcols):
    # Nullable row numbers prevent a missing match from coercing uint64 keys to float.
    target = right[rcols].rename(columns=dict(zip(rcols, lcols))).copy()
    for column in lcols:
        _require(left[column].dtype == target[column].dtype, f"join: unequal identity widths for {column}")
    target["_row"] = pd.array(np.arange(len(right), dtype=np.int64), dtype="Int64")
    joined = left[lcols].merge(target, on=lcols, how="left", sort=False, validate="many_to_one")
    return joined._row.fillna(-1).to_numpy(dtype=np.int64)


def _unique(frame, columns, label):
    _require(not frame.duplicated(columns).any(), f"{label}: ambiguous duplicate identity")


def analyze_frames(original, exported, status, events, expected_event=None):
    """Vectorized structural refusal and whole-export logical failure diagnostics.

    Only fixed tree/column/pointer-category loops and logarithmic pointer doubling
    are used; no per-event or per-particle Python analysis loops.
    """
    _require(type(events) is int and events in (1, 10), "events must be 1 or 10")
    capture._expected_event(events, expected_event)
    oe, og = capture._generator_frames(original, events, "original")
    frames = _frames(exported)
    e, g, p, r, d = [frames[name] for name in ("events", "gen", "particles", "raw", "daughters")]
    _unique(e, KEY, "events")
    _require(len(e) == events, "events: requested bounded census mismatch")
    pilot._same_sorted(e, oe, KEY, KEY, "original/export event census")
    if expected_event is not None:
        _require(np.array_equal(e[KEY].to_numpy(), np.asarray([expected_event], dtype=np.uint32)), "expected_event mismatch")
    _require(e.schema_version.eq(1), "events: require schema_version=1")
    for column, value in TAGS.items():
        _require(e[column].eq(value), f"events: exact process-qualified {column} required")
    violations = []

    def check(ok, message):
        if not bool(np.all(ok)):
            violations.append(message)

    original_events = oe.sort_values(KEY).reset_index(drop=True)
    exported_events = e.sort_values(KEY).reset_index(drop=True)
    for column in ("gen_product_id", "n_gen"):
        check(original_events[column].to_numpy() == exported_events[column].to_numpy(),
              f"events: original generator {column} mismatch")
    rows = {}
    for name, frame in (("gen", g), ("particles", p), ("raw", r), ("daughters", d)):
        erows = _lookup(frame, e, KEY, KEY)
        _require(erows >= 0, f"{name}: event absent from census")
        rows[name] = erows
        _require(np.bincount(erows, minlength=len(e)).astype(np.uint64) == e["n_" + name].to_numpy(),
                 f"{name}: census count mismatch")
        unique = KEY + (["key"] if name == "particles" else (["owner_key", "idx"] if name == "daughters" else ["idx"]))
        _unique(frame, unique, name)
        if name in ("gen", "raw"):
            _require(frame.idx.to_numpy() < e["n_" + name].to_numpy()[erows], f"{name}: incomplete range indices")
        product = {"gen": "gen", "particles": "sim", "raw": "raw", "daughters": "sim"}[name] + "_product_id"
        check(frame[product].to_numpy() == e[product].to_numpy()[erows], f"{name}: selected ProductID mismatch")
    for column in ["gen_present", "sim_present", "raw_present", "status_present", "collection_error"]:
        _require(e[column].isin([0, 1]), f"events: invalid switch {column}")
        check(e[column].eq(0 if column == "collection_error" else 1), f"events: failed {column}")
    for name in ("gen", "sim", "raw", "status"):
        check(e[name + "_product_id"].gt(0), f"events: null selected {name} ProductID")
    _require(p.isprim.isin([0, 1]), "particles: invalid isprim switch")
    _require(e[["nG4Tracks", "overflowSimP", "nStepLimit", "nFieldPropagator"]].to_numpy() >= 0, "events: negative status counter")
    check(e[["status", "overflowSimP", "nStepLimit", "nFieldPropagator"]].eq(0).to_numpy(), "events: nonzero transport status/error counters")
    _require(np.isfinite(e[["cpuTime", "realTime"]].to_numpy()) & (e[["cpuTime", "realTime"]].to_numpy() >= 0), "events: invalid serialized CPU/Wall")
    _require(np.isfinite(g[pilot.PAYLOAD[2:]].to_numpy()), "gen: nonfinite payload")
    statuses = pilot._frame(status, KEY + pilot.STATUS + ["cpuTime", "realTime"], KEY + pilot.STATUS, KEY, "status")
    _require(statuses[["nG4Tracks", "overflowSimP", "nStepLimit"]].to_numpy() >= 0, "status: negative counter")
    _require(np.isfinite(statuses[["cpuTime", "realTime"]].to_numpy()) & (statuses[["cpuTime", "realTime"]].to_numpy() >= 0), "status: invalid CPU/Wall")
    pilot._same_sorted(statuses, e, KEY, KEY, "independent status census")
    ordered_status = statuses.sort_values(KEY).reset_index(drop=True)
    ordered_events = e.sort_values(KEY).reset_index(drop=True)
    for column in pilot.STATUS:
        check(ordered_status[column].to_numpy() == ordered_events[column].to_numpy(), f"status: independent {column} mismatch")
    for column in ("cpuTime", "realTime"):
        printed = ordered_status[column].to_numpy()
        rendered = np.char.mod("%.3f", ordered_events[column].to_numpy()).astype(np.float64)
        check(printed == rendered, f"status: independent {column} mismatch at printer precision")
    # Original indices are validated signed-safe; cast each integer column explicitly.
    original_gen = og.copy()
    for column in KEY + ["idx", "gen_product_id"]:
        original_gen[column] = original_gen[column].astype(g[column].dtype)
    left, right = g.sort_values(KEY + ["idx"]).reset_index(drop=True), original_gen.sort_values(KEY + ["idx"]).reset_index(drop=True)
    check(len(left) == len(right), "gen: original generator row count mismatch")
    if len(left) == len(right):
        for column in pilot.GEN:
            check(left[column].to_numpy() == right[column].to_numpy(), f"gen: exact original identity/payload mismatch in {column}")
    check(p.key.eq(p.id), "particles: map key/internal ID mismatch")
    owner = _lookup(d, p, KEY + ["sim_product_id", "owner_key"], KEY + ["sim_product_id", "key"])
    _require(owner >= 0, "daughters: absent or ambiguous owner identity")
    _require(np.bincount(owner, minlength=len(p)).astype(np.uint64) == p.n_daughters.to_numpy(), "daughters: owner count mismatch")
    _require(d.idx.to_numpy() < p.n_daughters.to_numpy()[owner], "daughters: incomplete owner-bound range indices")

    def pointer(frame, prefix, target, product, key, mandatory=False):
        for field in ("nonnull", "available", "dereferenced", "member", "matches_selected", "error"):
            _require(frame[prefix + "_" + field].isin([0, 1]), f"{prefix}: invalid {field} flag")
        nonnull = frame[prefix + "_nonnull"].eq(1).to_numpy()
        if mandatory:
            check(nonnull, f"{prefix}: required pointer is null")
        null_columns = [prefix + "_" + field for field in POINTER if field != "nonnull"]
        check(frame.loc[~nonnull, null_columns].eq(0).to_numpy(), f"{prefix}: noncanonical null metadata")
        for field in ("available", "dereferenced", "member", "matches_selected", "error"):
            check(frame.loc[nonnull, prefix + "_" + field].eq(0 if field == "error" else 1), f"{prefix}: failed {field}")
        found = np.full(len(frame), -1, dtype=np.int64)
        found[nonnull] = _lookup(frame.loc[nonnull], target, KEY + [prefix + "_product_id", prefix + "_key"], KEY + [product, key])
        check(found[nonnull] >= 0, f"{prefix}: missing full ProductID/key target")
        return nonnull, found

    _, raw_target = pointer(r, "sim", p, "sim_product_id", "key", True)
    hasparent, parent_rows = pointer(p, "parent", p, "sim_product_id", "key")
    hasgen, gen_rows = pointer(p, "gen", g, "gen_product_id", "idx")
    _, daughter_rows = pointer(d, "daughter", p, "sim_product_id", "key", True)
    primary = p.isprim.eq(1).to_numpy()
    check(primary == ~hasparent, "particles: actual root/isprim equivalence failed")
    check(~primary | hasgen, "primary: missing original generator pointer")
    primrows = gen_rows[primary]
    check((primrows >= 0).all() and len(primrows) == len(g)
          and np.array_equal(np.sort(primrows), np.arange(len(g))), "primary: all-PDG original generator bijection failed")
    valid_primary = primary & (gen_rows >= 0)
    check(p.pdg.to_numpy()[valid_primary] == g.pdg.to_numpy()[gen_rows[valid_primary]], "primary: signed PDG mismatch")
    ancestor = np.arange(len(p), dtype=np.int64)
    valid_parent = hasparent & (parent_rows >= 0)
    ancestor[valid_parent] = parent_rows[valid_parent]
    for _ in range(len(p).bit_length()):
        advanced = ancestor[ancestor]
        if np.array_equal(advanced, ancestor):
            break
        ancestor = advanced
    check(~hasparent[ancestor] & primary[ancestor], "particles: ancestry cycle, missing parent or nonprimary root")
    # Edges use exact row indices only after exact ProductID/uint64-key joins.
    child = np.flatnonzero(hasparent)
    parent_edges = pd.DataFrame({"owner": parent_rows[child], "child": child})
    daughter_edges = pd.DataFrame({"owner": owner, "child": daughter_rows})
    check(~daughter_edges.duplicated(["owner", "child"]), "daughters: duplicate edge")
    check(parent_edges.sort_values(["owner", "child"]).reset_index(drop=True).equals(
          daughter_edges.sort_values(["owner", "child"]).reset_index(drop=True)), "particles: nonreciprocal or incomplete parent/daughter edges")
    passed = not violations
    muons = og.loc[og.pdg.isin([-13, 13])].groupby(KEY).size()
    outcomes = e[KEY].copy()
    outcomes["transport_state"] = "unknown"
    outcomes["crossing_state"] = "unknown"
    return {**FLAGS, "raw_pointer_integrity_established": passed, "raw_integrity_checks_passed": passed,
            "violations": violations, "events": events, "empty_generator_events": int(oe.n_gen.eq(0).sum()),
            "multi_muon_events": int(muons.gt(1).sum()), "unknown_crossing_events": events,
            "counts": {name: len(frame) for name, frame in frames.items()}, "event_outcomes": outcomes.to_dict("records")}


def _identities(paths):
    identities = {}
    for path in paths:
        _require(not path.is_symlink() and path.is_file(), f"missing/nonregular input or symlink alias: {path}")
        stat = path.stat()
        _require(stat.st_size > 0, f"empty input: {path}")
        identities[str(path)] = [stat.st_dev, stat.st_ino]
    _require(len({tuple(value) for value in identities.values()}) == len(paths), "input/source file alias")
    return identities


def run(raw_directory, export_root, events, exporter_log, out, expected_event=None):
    """Bind all readbacks/art/logs and sources; fresh capture gate, exclusive receipt."""
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    _require(type(events) is int and events in (1, 10), "events must be 1 or 10")
    capture._expected_event(events, expected_event)
    directory = Path(raw_directory)
    _require(directory.is_dir() and not directory.is_symlink(), "missing or symlink raw directory")
    directory = directory.resolve(strict=True)
    inputs = [directory / name for name in RAW_FILES] + [Path(export_root).absolute(), Path(exporter_log).absolute()]
    sources = [Path(__file__).resolve()] + [Path(module.__file__).resolve() for module in (capture, pilot, genassoc)]
    identities_before = _identities(inputs + sources)
    before = {str(path): _sha256(path) for path in inputs}
    source_before = {str(path): _sha256(path) for path in sources}
    with tempfile.TemporaryDirectory(prefix="inclusive-raw-integrity-capture-") as temporary:
        capture_receipt = capture.run(directory, events, Path(temporary) / "capture.json", expected_event)
    summary = analyze_frames(pilot._read(inputs[0], False), _read(inputs[-2]),
                             parse_status((directory / "status.log").read_text()), events, expected_event)
    export_text = inputs[-1].read_text()
    completions = re.findall(r"^Art has completed and will exit with status ([0-9]+)\.$", export_text, re.M)
    _require(len(completions) <= 1 and export_text.count("Art has completed") == len(completions),
             "exporter: malformed or ambiguous art completion records")
    completed = completions == ["0"]
    if not completed:
        summary["violations"].append("exporter: normal art completion with status zero absent")
    if capture_receipt["raw_capture_review_gate_passed"] is not True:
        summary["violations"].append("capture: fresh clean raw capture gate failed")
    passed = not summary["violations"]
    summary.update(raw_pointer_integrity_established=passed, raw_integrity_checks_passed=passed)
    identities_after = _identities(inputs + sources)
    after = {path: _sha256(path) for path in before}
    source_after = {path: _sha256(path) for path in source_before}
    _require(identities_before == identities_after and before == after and source_before == source_after,
             "input/source hash drift or file identity drift")
    result = {"schema": "crvfastsim.inclusive_run1a_raw_integrity.v1", "scope": "bounded_saved_raw_pointer_integrity_only",
              **FLAGS, "raw_pointer_integrity_established": passed, "raw_integrity_checks_passed": passed,
              "requested_events": events, "expected_event": expected_event, "raw_directory": str(directory),
              "export_root": str(inputs[-2]), "exporter_log": str(inputs[-1]), "exporter_art_completed_status_zero": completed,
              "capture_receipt": capture_receipt, "summary": summary, "violations": summary["violations"],
              "inputs_sha256_before": before, "inputs_sha256_after": after,
              "sources_sha256_before": source_before, "sources_sha256_after": source_after,
              "file_identities_before": identities_before, "file_identities_after": identities_after,
              "hashes_verified_after": True,
              "limitations": [
                  "Success establishes exact saved-pointer integrity only for this bounded export and original generator stream.",
                  "Every crossing outcome remains unknown; descendant deposits are not primary crossings.",
                  "No historical trajectory recovery, native attempted/saved-hit recording completeness, conversion authorization, physics admission or global completeness is established.",
                  "Hashes and file identities detect drift/aliases, not provenance, immutable snapshots or proof that readbacks originated from the hashed art.",
                  "The capture gate is freshly recomputed; preexisting receipts are not trusted."],
              "software_versions": {"numpy": np.__version__, "pandas": pd.__version__, "uproot": uproot.__version__}}
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--events", required=True, type=int, choices=(1, 10))
    parser.add_argument("--exporter-log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-event", nargs=3, type=int, metavar=("RUN", "SUBRUN", "EVENT"))
    args = parser.parse_args()
    run(args.raw_directory, args.export_root, args.events, args.exporter_log, args.out, args.expected_event)


if __name__ == "__main__":
    main()
