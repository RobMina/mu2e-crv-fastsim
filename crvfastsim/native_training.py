"""Stage-specific schema-v2 census admission, not a cross-stream generator join.

``validate(steps, generators, declared_shards=None)`` accepts ordered explicit
path iterables. Each stream must pass its own strict association/role validator;
each donor's complete event triples must be a subset of its paired full census.
Full censuses must be disjoint, including events with zero generator rows.

Donors dump ``compressDetStepMCs``; full streams dump ``cosmicTimeOffset``.
Compression copies/reindexes generator rows and can duplicate them. ProductIDs,
indices, multiplicities and generator content are therefore NOT compared across
streams. Census agreement establishes neither shared generator provenance nor a
cross-stream pointer join, source provenance, or filter completeness.

Integer shard declarations denote the exact subrun set of a single-run census.
For multiple runs use dictionaries with exactly ``run`` and ``subrun`` fields.
Declarations describe the union, not file order/count. Exposure counts full-census
events, never generator rows, accepted donor events or supplied declarations.
"""
import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from . import genassoc

RECEIPT_VERSION = 1
SCHEMA = "crvfastsim.native_training_census.v1"
EVENT_HASH_ENCODING = "lexicographically sorted (run,subrun,event), C-order little-endian int64"
KEY = genassoc.KEY


def _require(ok, message):
    if not bool(np.all(ok)):
        raise ValueError("native training: " + message)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(values, label):
    _require(not isinstance(values, (str, bytes, Path)), label + " must be an explicit path sequence")
    paths = [Path(value).resolve(strict=True) for value in values]
    _require(all(path.is_file() for path in paths), label + " must contain regular files")
    return paths


def _read(path):
    with uproot.open(path) as root:
        return genassoc._read(root, "events", genassoc.EVENT_COLUMNS, {})


def _index(frame):
    return pd.MultiIndex.from_frame(frame[KEY])


def _summary(events):
    keys = events[KEY].sort_values(KEY).to_numpy(dtype="<i8")
    return {"count": int(len(keys)), "sha256": hashlib.sha256(keys.tobytes(order="C")).hexdigest()}


def _shards(events):
    return events[["run", "subrun"]].drop_duplicates().sort_values(["run", "subrun"]).to_dict("records")


def _declarations(declared, actual):
    if declared is None:
        return None
    _require(not isinstance(declared, (str, bytes, dict)), "declared_shards must be a sequence")
    declared = list(declared)
    _require(bool(declared), "declared_shards cannot be empty")
    integers = all(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) for value in declared)
    if integers:
        _require(len({row["run"] for row in actual}) == 1,
                 "integer declared_shards are ambiguous for multiple runs; declare run/subrun dictionaries")
        run = actual[0]["run"]
        normalized = [{"run": run, "subrun": int(value)} for value in declared]
    else:
        _require(all(isinstance(value, dict) and set(value) == {"run", "subrun"} for value in declared),
                 "declared_shards require integers or run/subrun dictionaries")
        _require(all(isinstance(value[key], Integral) and not isinstance(value[key], (bool, np.bool_))
                     for value in declared for key in ("run", "subrun")),
                 "declared_shards identities must be integers")
        normalized = [{key: int(value[key]) for key in ("run", "subrun")} for value in declared]
    frame = pd.DataFrame(normalized)
    _require(not frame.duplicated().any(), "declared_shards must be distinct")
    _require((frame.to_numpy() >= 0).all() and (frame.to_numpy() <= np.iinfo(np.uint32).max).all(),
             "declared_shards identities must fit uint32")
    normalized = frame.sort_values(["run", "subrun"]).to_dict("records")
    _require(normalized == actual, "declared_shards do not match actual run/subrun contents")
    return normalized


def _compare(donor_events, full_events):
    rows = _index(full_events).get_indexer(_index(donor_events))
    _require((rows >= 0).all(), "donor events outside paired full-generator census (wrong pair)")


def validate(step_paths, gen_paths, declared_shards=None):
    """Return a JSON-safe v1 census receipt after strict stage-local admission.

    Ordered ``pairs`` retain paths, hashes, local validation reports and event
    summaries. ``total_exposure`` counts the disjoint full-event census, including
    gen-empty events. Generator row counts are stage-local diagnostics only.
    Highest ROOT cycles are used. No generator payload equality is required.

    Hashes are rechecked before return, not an immutable snapshot or execution
    attestation. Consumers must recheck recorded hashes before later use.
    Geometry must be bound separately; this API never admits physics training.
    """
    steps, generators = _paths(step_paths, "step_paths"), _paths(gen_paths, "gen_paths")
    _require(len(steps) == len(generators) and len(steps) > 0, "need equal nonzero paired path counts")
    paths = steps + generators
    _require(len(set(paths)) == len(paths), "paired paths must be distinct (including resolved aliases)")
    identities = [(path.stat().st_dev, path.stat().st_ino) for path in paths]
    _require(len(set(identities)) == len(paths), "paired paths must be distinct (including hardlink aliases)")
    before = {str(path): _sha256(path) for path in paths}
    pairs, censuses = [], []
    for step, generator in zip(steps, generators):
        donor_report = genassoc.validate_donor(step)
        _require(donor_report.get("schema_version") == genassoc.SCHEMA_VERSION
                 and donor_report.get("stream_role") == "checked donor"
                 and donor_report.get("associations_checked") is True,
                 "need schema-v2 checked donor")
        full_report = genassoc.validate_generator_stream(generator)
        _require(full_report.get("schema_version") == genassoc.SCHEMA_VERSION
                 and full_report.get("stream_role") == "full generator stream",
                 "need schema-v2 full generator stream, not an unverified legacy denominator")
        donor_events = _read(step)
        full_events = _read(generator)
        _compare(donor_events, full_events)
        censuses.append(full_events[KEY])
        pairs.append({"steps": {"path": str(step), "sha256": before[str(step)], "validation": donor_report},
                      "generator": {"path": str(generator), "sha256": before[str(generator)], "validation": full_report},
                      "donor_events": _summary(donor_events), "generator_events": _summary(full_events),
                      "generator_rows": {"donor": donor_report["counts"]["gen"], "full": full_report["counts"]["gen"]},
                      "actual_shards": _shards(full_events)})
    census = pd.concat(censuses, ignore_index=True)
    _require(not census.duplicated(KEY).any(), "overlapping full-generator event censuses")
    actual = _shards(census)
    declared = _declarations(declared_shards, actual)
    _require(all(_sha256(path) == before[str(path)] for path in paths), "input drift during admission")
    return {"schema": SCHEMA, "receipt_version": RECEIPT_VERSION,
            "scope": "stage_specific_census_not_geometry_or_physics_validation",
            "census_admitted": True, "per_stage_associations_validated": True,
            "cross_stream_generator_pointer_join": "not_established",
            "shared_generator_provenance": "not_established_by_census",
            "physics_admitted": False,
            "event_hash_encoding": EVENT_HASH_ENCODING, "pairs": pairs,
            "full_generator_events": _summary(census), "total_exposure": int(len(census)),
            "actual_shards": actual, "declared_shards": declared}
