"""Read-only conditional source-key diagnostic, never an identity bridge or admission.

The candidate source_idx = actual primary SimParticle key - 1 is conditional on
source-defined direct GenParticles, zero offset, one input and no SimParticle
rekeying during compression. This audit cannot establish those assumptions.
Compressed donor genidx is only a stage-local pointer, not a source index.
Payload equality can corroborate or reject a candidate, never establish identity.

run(config, out) accepts JSON or a mapping containing explicit paired donors and
generators, plus optional probe_events as integer [run, subrun, event] arrays.
Paths are relative to the caller. No geometry, fetching, response or construction
is used. Files are analyzed separately; only an exclusive JSON report is written.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from crvfastsim import genassoc, native_training

KEY = genassoc.KEY
PAYLOAD = ["pdg", "genid", "e", "t", "x", "y", "z", "px", "py", "pz"]
PART_COLUMNS = KEY + ["key", "parent", "genidx", "isprim", "pdg"]
GEN_COLUMNS = KEY + ["idx"] + PAYLOAD
SCHEMA = "crvfastsim.native_source_candidates.v1"
FLAGS = {"physics_admitted": False, "source_identity_established": False,
         "filter_completeness_established": False}
ASSUMPTIONS = ["direct source GenParticles", "zero SimParticle source-key offset",
               "single generator input", "compression preserves SimParticle keys"]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame(frame, columns, integers, unique, label):
    if not isinstance(frame, pd.DataFrame) or not set(columns) <= set(frame.columns):
        raise ValueError(f"{label}: missing required columns {columns}")
    if not frame.columns.is_unique:
        raise ValueError(f"{label}: duplicate column names")
    result = frame[columns].copy()
    genassoc._integer(result, integers, label)
    genassoc._unique(result, unique, label)
    genassoc._require(result[KEY].to_numpy() >= 0, f"{label}: negative event identity")
    return result


def _rows(left, right, left_columns, right_columns):
    target = pd.MultiIndex.from_frame(right[right_columns])
    source = pd.MultiIndex.from_frame(left[left_columns])
    return target.get_indexer(source)


def analyze(parts, donor_gen, full_gen):
    """Count exact conditional candidates for all actual isprim primaries, all PDGs.

    Integer row indexers preserve adjacent large identities even on missing joins.
    Missing candidates and exact payload disagreements are counted, not repaired.
    No time/rank/kinematic search, tolerance, alternative candidate or bridge exists.
    """
    parts = _frame(parts, PART_COLUMNS, PART_COLUMNS, KEY + ["key"], "particles")
    donor = _frame(donor_gen, GEN_COLUMNS, KEY + ["idx", "pdg", "genid"],
                   KEY + ["idx"], "donor gen")
    full = _frame(full_gen, GEN_COLUMNS, KEY + ["idx", "pdg", "genid"],
                  KEY + ["idx"], "full gen")
    genassoc._require(parts.isprim.isin([0, 1]), "particles: isprim must be 0 or 1")
    for frame, label in ((donor, "donor gen"), (full, "full gen")):
        genassoc._require(frame.idx >= 0, f"{label}: negative index")
        genassoc._require(np.isfinite(frame[PAYLOAD[2:]].to_numpy()), f"{label}: nonfinite payload")
    primary = parts.loc[parts.isprim.eq(1)].copy()
    genassoc._require(primary.parent.eq(-1), "primary: actual parentless isprim required")
    genassoc._require(primary.key > 0, "primary: positive source candidate key required")
    genassoc._require(primary.genidx >= 0, "primary: missing stage-local generator pointer")
    primary["source_idx"] = primary.key.to_numpy(dtype=np.int64) - np.int64(1)
    local_rows = _rows(primary, donor, KEY + ["genidx"], KEY + ["idx"])
    source_rows = _rows(primary, full, KEY + ["source_idx"], KEY + ["idx"])
    local_found, source_found = local_rows >= 0, source_rows >= 0
    both = local_found & source_found
    local = donor.iloc[local_rows[both]]
    candidate = full.iloc[source_rows[both]]
    equal = np.ones(int(both.sum()), dtype=bool)
    mismatch = {}
    for column in PAYLOAD:
        same = local[column].to_numpy() == candidate[column].to_numpy()
        mismatch[column] = int((~same).sum())
        equal &= same
    unique_candidates = np.unique(source_rows[source_found])
    local_pdg_bad = (primary.pdg.to_numpy()[local_found]
                     != donor.pdg.to_numpy()[local_rows[local_found]])
    multiplicity = primary.groupby(KEY, sort=False).size()
    return {**FLAGS, "selected_primaries": int(len(primary)),
            "selected_primary_muons": int(primary.pdg.isin([-13, 13]).sum()),
            "primary_events": int(len(multiplicity)),
            "multi_primary_events": int((multiplicity > 1).sum()),
            "missing_stage_local_generator": int((~local_found).sum()),
            "primary_stage_local_pdg_mismatches": int(local_pdg_bad.sum()),
            "missing_source_candidate": int((~source_found).sum()),
            "source_candidates_found": int(source_found.sum()),
            "distinct_full_generator_candidates": int(len(unique_candidates)),
            "repeated_candidate_references": int(source_found.sum() - len(unique_candidates)),
            "full_generator_rows": int(len(full)),
            "full_generator_muons": int(full.pdg.isin([-13, 13]).sum()),
            "unreferenced_full_generator_rows": int(len(full) - len(unique_candidates)),
            "source_idx_equals_donor_genidx": int(primary.source_idx.eq(primary.genidx).sum()),
            "source_idx_differs_from_donor_genidx": int(primary.source_idx.ne(primary.genidx).sum()),
            "payload_compared": int(both.sum()), "payload_exact_equal": int(equal.sum()),
            "payload_mismatched": int((~equal).sum()), "payload_mismatches_by_field": mismatch}


def _paths(values, label):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label}: nonempty explicit path array required")
    if not all(isinstance(value, (str, Path)) for value in values):
        raise ValueError(f"{label}: paths must be strings or Paths")
    return native_training._paths(values, label)


def _probes(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError("probe_events must be integer [run, subrun, event] arrays")
    if not values:
        return pd.DataFrame({key: pd.Series(dtype=np.int64) for key in KEY})
    if not all(isinstance(row, (list, tuple)) and len(row) == 3
               and all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                       and 0 <= v <= np.iinfo(np.uint32).max for v in row) for row in values):
        raise ValueError("probe_events must be integer uint32 [run, subrun, event] arrays")
    frame = pd.DataFrame(values, columns=KEY, dtype=np.int64)
    genassoc._unique(frame, KEY, "probe_events")
    return frame


def _read(path, name, columns):
    with uproot.open(path) as root:
        return genassoc._read(root, name, columns, {})


def _probe_report(probes, donor_events, full_events, full):
    result = probes.copy()
    result["in_full_census"] = _rows(probes, full_events, KEY, KEY) >= 0
    result["absent_donor_event"] = _rows(probes, donor_events, KEY, KEY) < 0
    counts = full.loc[full.pdg.isin([-13, 13])].groupby(KEY).size()
    result["fullgen_muons"] = counts.reindex(pd.MultiIndex.from_frame(probes), fill_value=0).to_numpy(dtype=np.int64)
    return result.to_dict("records")


def run(config, out):
    """Strict paired census validation and pre/post hashes; no output on refusal.

    Hash rechecks detect drift, not immutable snapshots or source provenance.
    Mapping configs are recorded in full; file configs additionally bind bytes.
    """
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    config_path = Path(config).resolve(strict=True) if isinstance(config, (str, Path)) else None
    config_hash = _sha256(config_path) if config_path else None
    cfg = json.loads(config_path.read_text()) if config_path else config
    if not isinstance(cfg, dict) or not {"donors", "generators"} <= set(cfg) or set(cfg) - {
            "donors", "generators", "probe_events"}:
        raise ValueError("config requires donors/generators and optional probe_events only")
    donors, generators = _paths(cfg["donors"], "donors"), _paths(cfg["generators"], "generators")
    probes = _probes(cfg.get("probe_events", []))
    normalized = {"donors": list(map(str, donors)), "generators": list(map(str, generators)),
                  "probe_events": probes.to_numpy().tolist()}
    config_digest = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()
    source_paths = [Path(__file__).resolve(), Path(genassoc.__file__).resolve(),
                    Path(native_training.__file__).resolve()]
    sources = {str(path): _sha256(path) for path in source_paths}
    before = {str(path): _sha256(path) for path in [*donors, *generators]}
    if config_path:
        before[str(config_path)] = config_hash
    receipt = native_training.validate(donors, generators)
    if any(before[item["path"]] != item["sha256"] for pair in receipt["pairs"]
           for item in (pair["steps"], pair["generator"])):
        raise ValueError("source candidates input hash drift before validation")
    files = []
    for donor_path, full_path in zip(donors, generators):
        parts = _read(donor_path, "particles", PART_COLUMNS)
        donor = _read(donor_path, "gen", GEN_COLUMNS)
        full = _read(full_path, "gen", GEN_COLUMNS)
        donor_events = _read(donor_path, "events", KEY)
        full_events = _read(full_path, "events", KEY)
        files.append({"donor": str(donor_path), "generator": str(full_path),
                      "measurements": analyze(parts, donor, full),
                      "probe_events": _probe_report(probes, donor_events, full_events, full)})
    result = {"schema": SCHEMA, "scope": "conditional_source_key_candidate_diagnostic_only",
              **FLAGS, "candidate": "source_idx = actual isprim SimParticle key - 1",
              "unestablished_source_assumptions": ASSUMPTIONS,
              "config": normalized, "config_sha256": config_digest,
              "training_census": receipt, "files": files,
              "inputs_sha256": before, "sources_sha256": sources,
              "software_versions": {"numpy": np.__version__, "pandas": pd.__version__, "uproot": uproot.__version__},
              "limitations": ["Exact payload equality is corroboration only, not proof of source identity.",
                              "Missing/mismatched candidates are not replaced by payload, rank or time matches.",
                              "Compressed genidx equality and ProductID equality are not source identity.",
                              "No bridge, construction, filter completeness or physics admission is supplied."]}
    if any(_sha256(path) != digest for path, digest in {**before, **sources}.items()):
        raise ValueError("source candidates input/config/source hash drift")
    result["hashes_verified_after"] = True
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
