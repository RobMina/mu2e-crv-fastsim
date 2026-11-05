"""Read-only reconstruction comparison: source-derived cap gate, never native completeness."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat

import numpy as np
import uproot

EXPECTED = "Production/JobConfig/cosmic/geom_cosmic_run1_a.txt"
SCHEMA = {"schema_version": "uint32", "input_lines": "uint64", "input_hash": "uint64",
          **dict.fromkeys(("steps_limit", "steps_limit_explicit", "allow_replacement", "steps_limit_default"), "int32"),
          **dict.fromkeys(("input_file", "requested_input_file", "resolved_text"), "string")}
FLAGS = dict.fromkeys(("attempted_hit_counts_measured", "attempted_saved_equality_measured",
                       "native_raw_recording_completeness_established", "physics_admitted",
                       "global_denominator_completeness_established", "downstream_conversion_authorized",
                       "conversion_eligible"), False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bindings(paths):
    result, identities = {}, set()
    for path in paths:
        require(not path.is_symlink(), f"symlink input: {path}")
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_size > 0, f"nonregular/empty input: {path}")
        identity = (info.st_dev, info.st_ino)
        require(identity not in identities, f"aliased input/source: {path}")
        identities.add(identity)
        result[str(path)] = {"sha256": _sha256(path), "device_inode": identity, "bytes": info.st_size}
    return result


def _read(path):
    with uproot.open(path) as root:
        keys = [key for key in root.keys(recursive=True, cycle=True) if key.rsplit(";", 1)[0] == "recording/config"]
        require(bool(keys), "missing recording/config")
        key = max(keys, key=lambda value: int(value.rsplit(";", 1)[1]))
        tree = root[key]
        require(tree.classname == "TTree" and tree.num_entries == 1, "config: exactly one TTree row required")
        require(set(tree.keys()) == set(SCHEMA), "config: exact schema columns required")
        row = {}
        for column, dtype in SCHEMA.items():
            values = tree[column].array(library="np")
            if dtype == "string":
                require(isinstance(values[0], str), f"{column}: string required")
                row[column] = values[0]
            else:
                require(values.dtype == np.dtype(dtype), f"{column}: {dtype} required")
                row[column] = int(values[0])
    require(row["schema_version"] == 1, "config: schema_version=1 required")
    require(row["steps_limit_explicit"] in (0, 1) and row["allow_replacement"] in (0, 1), "invalid boolean flag")
    require(bool(row["input_file"]) and bool(row["requested_input_file"]), "empty geometry path")
    return row, key


def _completed(text, label):
    require(len(re.findall(r"^Art has completed and will exit with status 0\.$", text, re.M)) == 1
            and text.count("Art has completed") == 1, f"{label}: normal Art completion required")


def parse_original(text):
    _completed(text, "original")
    files = re.findall(r"^Geometry file: ([^\r\n]+)$", text, re.M)
    counts = re.findall(r"^Geometry lines: ([0-9]+)[ \t]+hash: ([0-9]+)$", text, re.M)
    require(len(files) == len(counts) == 1 and len(re.findall(r"^Geometry (?:file|lines):", text, re.M)) == 2,
            "original: exactly one well-formed Geometry file and lines/hash required")
    lines, hash_value = map(int, counts[0])
    require(lines < 2**64 and hash_value < 2**64, "original: uint64 overflow")
    return {"input_file": files[0], "input_lines": lines, "input_hash": hash_value}


def compare(row, original):
    diagnostics = [f"{name}: original/reconstruction mismatch" for name in original if row[name] != original[name]]
    expected = {"requested_input_file": EXPECTED, "allow_replacement": 1, "steps_limit_default": 0}
    diagnostics += [f"{name}: expected {value!r}" for name, value in expected.items() if row[name] != value]
    text, limit = row["resolved_text"], row["steps_limit"]
    if row["steps_limit_explicit"] == 0:
        consistent = limit == 0 and "g4.stepsSizeLimit" not in text
    else:
        # SimpleConfigRecord::toString prints active scalar records as 'int name = value;'.
        assignments = re.findall(r"^[ \t]*int[ \t]+g4\.stepsSizeLimit[ \t]*=[ \t]*([+-]?[0-9]+)[ \t]*;[ \t]*$", text, re.M)
        consistent = len(assignments) == 1 and text.count("g4.stepsSizeLimit") == 1 and int(assignments[0]) == limit
    if not consistent:
        diagnostics.append("resolved_text/explicit/limit inconsistent")
    if limit > 0:
        diagnostics.append("steps_limit: positive cap")
    return {**FLAGS, "source_derived_cap_gate_passed": not diagnostics, "diagnostics": diagnostics}


def run(export_root, transport_log, exporter_log, out):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    inputs = [Path(path).absolute() for path in (export_root, transport_log, exporter_log)]
    sources = [Path(__file__).absolute(), Path(__file__).absolute().parents[1] / "art/CrvFastSim/src/CrvRecordingConfigDumper_module.cc"]
    before = _bindings(inputs + sources)
    row, key = _read(inputs[0])
    original = parse_original(inputs[1].read_text())
    _completed(inputs[2].read_text(), "exporter")
    result = {"schema": "crvfastsim.inclusive_run1a_recording.v1", **compare(row, original),
              "original": original, "reconstructed": row, "selected_tree_cycle": key,
              "limitations": ["Source-derived cap gate only, not native recording completeness or authorization.",
                              "Legacy Boost hash is noncryptographic and has a continuation-line bug; equality is not contents proof.",
                              "Pre/post file/source SHA256 bindings detect drift, not provenance or immutable snapshots."]}
    after = _bindings(inputs + sources)
    require(before == after, "input/source drift")
    result.update(bindings_before=before, bindings_after=after, hashes_verified_after=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export-root", "transport-log", "exporter-log", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.export_root, args.transport_log, args.exporter_log, args.out)


if __name__ == "__main__":
    main()
