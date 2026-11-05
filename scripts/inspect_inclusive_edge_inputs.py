"""Read-only existing full-generator boundary-case census; no transport or admission.

Config contains explicit existing schema-v2 generators and optional probe_events
as exact uint32 [run, subrun, event] arrays. Paths are relative to the caller.
Only files/columns are iterated, never events. The first empty event is selected
from existing census rows in configured file order, without synthesizing an event.
A census_entry_index is a zero-based diagnostic source-order probe, NOT an art
RootInput entry/skip until independently checked against art RootInput/log evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from crvfastsim import genassoc

KEY = genassoc.KEY
SCHEMA = "crvfastsim.inclusive_edge_inputs.v1"
FLAGS = {"physics_admitted": False, "source_art_order_verified": False,
         "transport_performed": False}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("generators: nonempty explicit path array required")
    if not all(isinstance(value, (str, Path)) and str(value) for value in values):
        raise ValueError("generators: paths must be nonempty strings or Paths")
    paths = [Path(value).resolve(strict=True) for value in values]
    if not all(path.is_file() for path in paths):
        raise ValueError("generators: regular files required")
    if len(set(paths)) != len(paths):
        raise ValueError("generators: duplicate paths/aliases")
    return paths


def _probes(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError("probe_events must be integer uint32 [run, subrun, event] arrays")
    if not all(isinstance(row, (list, tuple)) and len(row) == 3
               and all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                       and 0 <= v <= np.iinfo(np.uint32).max for v in row) for row in values):
        raise ValueError("probe_events must be integer uint32 [run, subrun, event] arrays")
    frame = pd.DataFrame(values, columns=KEY, dtype=np.int64)
    genassoc._unique(frame, KEY, "probe_events")
    return frame


def analyze(events, gen):
    """Vectorized census counts after strict stream validation; no event loop."""
    events, gen = events.copy(), gen.copy()
    genassoc._integer(events, KEY, "events", ("u", 4))
    genassoc._integer(events, ["n_gen"], "events", ("i", 8))
    genassoc._integer(gen, KEY, "gen", ("u", 4))
    genassoc._integer(gen, ["idx", "pdg"], "gen")
    genassoc._unique(events, KEY, "events")
    genassoc._unique(gen, KEY + ["idx"], "gen")
    genassoc._require(events.n_gen >= 0, "events: negative generator count")
    rows = genassoc._lookup(gen, events, KEY, KEY, "generator census")
    counts = np.bincount(rows, minlength=len(events))
    genassoc._require(counts == events.n_gen.to_numpy(), "generator census count mismatch")
    genassoc._require((gen.idx.to_numpy() >= 0) & (gen.idx.to_numpy() < counts[rows]),
                     "gen: index out of range")
    muons = gen.pdg.isin([-13, 13]).to_numpy()
    events["census_entry_index"] = np.arange(len(events), dtype=np.int64)
    events["fullgen_muons"] = np.bincount(rows[muons], minlength=len(events))
    empty = np.flatnonzero(events.n_gen.to_numpy() == 0)
    first = (events.iloc[empty[:1]][KEY + ["census_entry_index"]].to_dict("records")[0]
             if len(empty) else None)
    summary = {"n_events": int(len(events)), "n_gen": int(len(gen)),
               "empty_event_count": int(len(empty)), "first_empty_event": first,
               "fullgen_muons": int(muons.sum()),
               "multi_muon_event_count": int((events.fullgen_muons > 1).sum())}
    return summary, events


def _probe_report(probes, census):
    rows = pd.MultiIndex.from_frame(census[KEY]).get_indexer(pd.MultiIndex.from_frame(probes))
    found = rows >= 0
    result = probes.copy()
    result["in_full_census"] = found
    # Nullable integer columns preserve exact counts/indices on missing probes.
    for column in ("n_gen", "fullgen_muons", "census_entry_index"):
        values = pd.array(np.zeros(len(probes), dtype=np.int64), dtype="Int64")
        values[found] = census[column].to_numpy(dtype=np.int64)[rows[found]]
        values[~found] = pd.NA
        result[column] = values
    result["generator"] = None
    result.loc[found, "generator"] = census.generator.to_numpy()[rows[found]]
    # pandas versions differ in whether to_dict normalizes pd.NA to None.
    return result.astype(object).where(result.notna(), None).to_dict("records")


def run(config, out):
    """Validate/read highest cycles, hash before/after, write one exclusive JSON.

    Hash checks detect drift, not immutable snapshots or art input provenance.
    Mapping configs bind normalized content; file configs additionally bind bytes.
    """
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    config_path = Path(config).resolve(strict=True) if isinstance(config, (str, Path)) else None
    config_hash = _sha256(config_path) if config_path else None
    cfg = json.loads(config_path.read_text()) if config_path else config
    if not isinstance(cfg, dict) or "generators" not in cfg or set(cfg) - {"generators", "probe_events"}:
        raise ValueError("config requires generators and optional probe_events only")
    paths = _paths(cfg["generators"])
    probes = _probes(cfg.get("probe_events", []))
    normalized = {"generators": list(map(str, paths)), "probe_events": probes.to_numpy().tolist()}
    config_digest = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()
    sources = {str(path): _sha256(path) for path in
               (Path(__file__).resolve(), Path(genassoc.__file__).resolve())}
    before = {str(path): _sha256(path) for path in paths}
    if config_path:
        before[str(config_path)] = config_hash
    files, censuses = [], []
    for path in paths:
        validation = genassoc.validate_generator_stream(path)
        if validation["schema_version"] != 2 or validation["stream_role"] != "full generator stream":
            raise ValueError(f"{path}: require existing schema-v2 full generator stream")
        trees = {}
        with uproot.open(path) as root:
            events = genassoc._read(root, "events", KEY + ["n_gen"], trees)
            gen = genassoc._read(root, "gen", KEY + ["idx", "pdg"], trees)
        summary, census = analyze(events, gen)
        census["generator"] = str(path)
        censuses.append(census)
        files.append({"generator": str(path), "validation": validation, "trees": trees,
                      "summary": summary})
    census = pd.concat(censuses, ignore_index=True)
    genassoc._unique(census, KEY, "overlapping generator event census")
    first_empty = None
    for item in files:
        if item["summary"]["first_empty_event"] is not None:
            first_empty = {"generator": item["generator"], **item["summary"]["first_empty_event"]}
            break
    result = {"schema": SCHEMA, "scope": "existing_generator_boundary_case_scan_only", **FLAGS,
              "config": normalized, "config_sha256": config_digest,
              "inputs_sha256": before, "sources_sha256": sources, "files": files,
              "probe_events": _probe_report(probes, census), "first_empty_event": first_empty,
              "software_versions": {"numpy": np.__version__, "pandas": pd.__version__, "uproot": uproot.__version__},
              "limitations": [
                  "census_entry_index is zero-based diagnostic census row order, NOT an authoritative art entry/skip.",
                  "Independently check art RootInput/log order before any source-order probe transport.",
                  "Only existing supplied fullgen events are scanned; no empty event is synthesized when none exists.",
                  "Dump hashes do not establish original art provenance; no transport or physics admission is supplied."]}
    if any(_sha256(path) != digest for path, digest in {**before, **sources}.items()):
        raise ValueError("inclusive edge inputs input/config/source hash drift")
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
