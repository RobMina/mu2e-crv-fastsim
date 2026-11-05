"""Conservative construction-domain checks, separate from valid stage-local censuses.

Event/bar crossing aggregation and near-miss event masks require at most one
primary muon per event in both the donor and full-generator streams. Unsupported
multiplicity refuses the whole build; events are never selected or discarded.
This is a construction restriction, not a cross-stream pointer join or closure.
"""
import uproot

from . import genassoc, native_training


SCHEMA = "crvfastsim.native_construction_support.v1"


def multiplicity(frame, primary=False):
    """Count actual primary-muon identities, not compressed generator multiplicity."""
    chosen = frame.loc[frame.pdg.abs().eq(13)]
    identity = "idx"
    if primary:
        chosen = chosen.loc[chosen.isprim.eq(1)]
        identity = "key"
    keys = genassoc.KEY + [identity]
    values = chosen[keys].to_numpy()
    if values.dtype.kind not in "iu" or chosen.duplicated(keys).any():
        raise ValueError("native construction requires unique integer primary identities")
    counts = chosen.groupby(genassoc.KEY, sort=False).size()
    return {"primary_muons": int(len(chosen)), "events_with_muons": int(len(counts)),
            "max_muons_per_event": int(counts.max()) if len(counts) else 0,
            "multiple_muon_events": int((counts > 1).sum())}


def check_receipt(manifest, check):
    """Require build-bound construction-domain evidence without rewriting old bundles."""
    evidence = manifest.get("construction_support")
    if not isinstance(evidence, dict) or evidence != check.get("construction_support"):
        raise ValueError("native library lacks matching construction support; rebuild after identity audit")
    training = manifest["training_census"]
    expected = {item[role]["path"]: item[role]["sha256"] for item in training["pairs"]
                for role in ("steps", "generator")}
    if (evidence.get("schema") != SCHEMA or evidence.get("supported") is not True
            or evidence.get("max_supported_primary_muons_per_event") != 1
            or evidence.get("events_filtered") != 0 or evidence.get("physics_admitted") is not False
            or evidence.get("input_sha256") != expected):
        raise ValueError("native library invalid construction support scope/source evidence")
    reports = evidence.get("pairs", [])
    if len(reports) != len(training["pairs"]):
        raise ValueError("native library construction support pair mismatch")
    for report, item in zip(reports, training["pairs"]):
        for role in ("steps", "generator"):
            data = report.get(role + "_multiplicity", {})
            if (report.get(role) != item[role]["path"]
                    or set(data) != {"primary_muons", "events_with_muons", "max_muons_per_event", "multiple_muon_events"}
                    or any(type(n) is not int or n < 0 for n in data.values())
                    or data["multiple_muon_events"] != 0 or data["max_muons_per_event"] > 1
                    or data["primary_muons"] != data["events_with_muons"]
                    or data["max_muons_per_event"] != int(data["primary_muons"] > 0)):
                raise ValueError("native library unsupported primary multiplicity evidence")
    return evidence


def require_supported(training):
    """Recheck source-bound multiplicity before named construction; never filter input."""
    if training.get("schema") != native_training.SCHEMA or training.get("census_admitted") is not True:
        raise ValueError("native construction requires an admitted training census")
    pairs = training["pairs"]
    paths = {item[role]["path"]: item[role]["sha256"] for item in pairs for role in ("steps", "generator")}
    if not pairs or any(native_training._sha256(path) != digest for path, digest in paths.items()):
        raise ValueError("native construction input drift before support check")
    reports = []
    for item in pairs:
        data = {}
        for role, tree, fields in (("steps", "particles", genassoc.KEY + ["key", "isprim", "pdg"]),
                                   ("generator", "gen", genassoc.KEY + ["idx", "pdg"])):
            with uproot.open(item[role]["path"]) as root:
                frame = genassoc._read(root, tree, fields, {})
            data[role] = multiplicity(frame, primary=role == "steps")
        reports.append({"steps": item["steps"]["path"], "generator": item["generator"]["path"], **{
            role + "_multiplicity": data[role] for role in data}})
    if any(native_training._sha256(path) != digest for path, digest in paths.items()):
        raise ValueError("native construction input drift during support check")
    if any(report[role + "_multiplicity"]["multiple_muon_events"] for report in reports
           for role in ("steps", "generator")):
        raise ValueError("native construction does not support multiple primary muons per event; "
                         "event/bar crossing aggregation and event-only near-miss masks need identity-safe reconstruction; "
                         "no events were filtered")
    return {"schema": SCHEMA, "scope": "single_primary_muon_construction_domain_not_physics_validation",
            "supported": True, "max_supported_primary_muons_per_event": 1, "pairs": reports,
            "input_sha256": paths, "events_filtered": 0, "physics_admitted": False}
