"""Saved-only, exact numeric native-rebuild diagnostic; never certifies the historical bundle.

The candidate must pass current native_bundle/load admission. The reference may lack
new census receipts: it is read diagnostically, not passed through admission or repaired.
Both anchor arms run today's algorithm; historical build_check anchor counts are only
recorded declarations, not evidence of historical execution. No baseline is rebuilt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from crvfastsim import runconfig, tier1b, tier1b_assets
from crvfastsim.analyses.context_audit import content_fingerprint
from crvfastsim.analyses.tier1b import read_template_sidecar, read_topology_sidecar


BUNDLE_FILES = ("library.root", "library.json", "build_check.json", "topology.npz",
                "topology.npz.json", "templates_v3.npz", "templates_v3.npz.json")
ANCHOR_COUNTS = ("n_roots", "n_multi", "forward_edep_share")
# Sidecar JSON is integrity evidence, not numeric payload. No prefix/substring exclusions.
NPZ_METADATA = frozenset(("meta", "metadata"))


def array_identity(value):
    """Hash shape, primitive dtype and C-order bytes, including signed zeros/NaN bits."""
    a = np.asarray(value)
    if a.dtype.kind not in "biufc" or a.dtype.fields or a.dtype.metadata:
        raise ValueError("non-primitive numeric payload: " + str(a.dtype))
    header = json.dumps([a.dtype.str, list(a.shape)], separators=(",", ":")).encode()
    digest = hashlib.sha256(header + b"\0" + a.tobytes(order="C")).hexdigest()
    return dict(dtype=a.dtype.str, shape=list(a.shape), count=int(a.size), sha256=digest)


def numeric_payload(directory):
    """Read actual LIB_TREES frames and every NPZ variable (not fixed array names)."""
    directory = Path(directory)
    lib = tier1b.read_library(directory / "library.root")
    payload, tables, excluded = {}, {}, {}
    for name in tier1b.LIB_TREES:
        frame = lib[name]
        if not frame.columns.is_unique:
            raise ValueError("duplicate library fields: " + name)
        fields = list(frame.columns)
        tables[name] = dict(rows=len(frame), fields=fields)
        for field in fields:
            payload[f"library/{name}/{field}"] = array_identity(frame[field].to_numpy())
    for name in ("topology.npz", "templates_v3.npz"):
        with np.load(directory / name, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("duplicate NPZ variables: " + name)
            excluded[name] = sorted(set(archive.files) & NPZ_METADATA)
            for field in sorted(set(archive.files) - NPZ_METADATA):
                payload[f"{name}/{field}"] = array_identity(archive[field])
    return payload, tables, excluded


def integer_storage_changes(reference, candidate, mismatches):
    """Report exact integer values separately; storage mismatches remain strict mismatches."""
    result = []
    for row in mismatches:
        parts = row["name"].split("/")
        if len(parts) != 2 or parts[0] not in ("topology.npz", "templates_v3.npz"):
            continue
        with np.load(Path(reference) / parts[0], allow_pickle=False) as left, np.load(Path(candidate) / parts[0], allow_pickle=False) as right:
            if parts[1] not in left.files or parts[1] not in right.files:
                continue
            a, b = left[parts[1]], right[parts[1]]
            if a.dtype.kind not in "iu" or b.dtype.kind not in "iu" or a.shape != b.shape:
                continue
            safe = all(not (x.dtype.kind == "u" and np.any(x > np.iinfo(np.int64).max)) for x in (a, b))
            equal = bool(np.array_equal(a.astype(np.int64), b.astype(np.int64))) if safe else False
            result.append(dict(name=row["name"], exact_int64_values_equal=equal,
                               reference_dtype=a.dtype.str, candidate_dtype=b.dtype.str))
    return result


def compare_payloads(reference, candidate):
    """Missing fields, shapes, dtypes or byte changes are mismatches, not tolerances."""
    return [dict(name=name, reference=reference.get(name), candidate=candidate.get(name))
            for name in sorted(reference.keys() | candidate.keys())
            if reference.get(name) != candidate.get(name)]


def diagnostic_reference(directory, rc, geom):
    """Bind old build/sidecars to this geometry without admitting an old census."""
    directory = Path(directory)
    manifest = json.loads((directory / "library.json").read_text())
    check = json.loads((directory / "build_check.json").read_text())
    digest = tier1b_assets.sha256(directory / "library.root")
    counters = tier1b_assets.sha256(rc.counters_path())
    if manifest.get("run_config") != rc.name or check.get("run_config") != rc.name:
        raise ValueError("reference run configuration mismatch")
    if (manifest.get("counters_sha256") != counters
            or manifest.get("geometry_fingerprint") != content_fingerprint(geom, workers=1)):
        raise ValueError("reference build-time counters/geometry fingerprint mismatch")
    if check.get("library_sha256") != digest:
        raise ValueError("reference build check does not bind library")
    generated = check.get("generated")
    if type(generated) is not int or generated <= 0 or generated != manifest.get("generated_scaled_to"):
        raise ValueError("reference build exposure mismatch")
    for name in ("topology.npz", "templates_v3.npz"):
        meta = json.loads((directory / (name + ".json")).read_text())
        if (meta.get("library_sha256") != digest
                or meta.get("sidecar_sha256") != tier1b_assets.sha256(directory / name)
                or meta.get("geometry_sha256") != counters):
            raise ValueError("reference sidecar/library/geometry hash mismatch: " + name)
    declared = check.get("anchor")
    if not isinstance(declared, dict) or any(key not in declared for key in ANCHOR_COUNTS):
        raise ValueError("reference lacks declared build anchor counts")
    lib = tier1b.read_library(directory / "library.root")
    lib = tier1b.attach_topology(lib, *read_topology_sidecar(directory / "topology.npz", library_sha256=digest), geom)
    lib = tier1b.attach_templates(lib, read_template_sidecar(directory / "templates_v3.npz", library_sha256=digest), geom)
    lib = tier1b.attach_anchors(lib, geom)
    return lib, {key: declared[key] for key in ANCHOR_COUNTS}


def source_paths():
    """Current implementation and builder-critical sources, using installed locations."""
    from mu2efastsim import tables as core_tables
    root = Path(__file__).resolve().parents[1]
    paths = {"scripts/" + name: root / "scripts" / name for name in
             ("check_native_rebuild.py", "check_native_bundle.py", "build_native_library.py")}
    # Includes geometry, group maps, association/census readers, builder and attachment code.
    for package, directory in (("crvfastsim", Path(tier1b.__file__).resolve().parent),
                               ("valhist", root / "valhist"),
                               ("mu2efastsim", Path(core_tables.__file__).resolve().parent)):
        for path in sorted(directory.rglob("*.py")):
            paths[package + "/" + path.relative_to(directory).as_posix()] = path
    return paths


def snapshot(paths):
    # Retain lexical paths for rechecking; target identity changes also fail, even for equal bytes.
    return {name: dict(sha256=tier1b_assets.sha256(path), target=str(Path(path).resolve()))
            for name, path in paths.items()}


def run(reference, candidate, run_config, out):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if os.environ.get("CRVFS_PREPARED_LIBRARY_CACHE"):
        raise ValueError("unset CRVFS_PREPARED_LIBRARY_CACHE for saved-only diagnostic")
    reference, candidate = Path(reference).absolute(), Path(candidate).absolute()
    if reference.resolve() == candidate.resolve():
        raise ValueError("reference and candidate bundles must be distinct")
    paths = {f"{arm}/{name}": directory / name for arm, directory in
             (("reference", reference), ("candidate", candidate)) for name in BUNDLE_FILES}
    paths.update({"source/" + key: path for key, path in source_paths().items()})
    before = snapshot(paths)
    rc = runconfig.get(run_config)
    paths["configuration/counters"] = Path(rc.counters_path())
    before.update(snapshot({"configuration/counters": paths["configuration/counters"]}))
    geom = rc.geometry()
    geometry_before = content_fingerprint(geom, workers=1)
    # Only the candidate goes through admission. No fake or retrofitted old census.
    spec = tier1b_assets.native_bundle(candidate, rc)
    candidate_lib = tier1b_assets.load(spec, geom)
    reference_lib, declared = diagnostic_reference(reference, rc, geom)
    a, ta, ea = numeric_payload(reference)
    b, tb, eb = numeric_payload(candidate)
    for name, lib, payload in (("reference", reference_lib, a), ("candidate", candidate_lib, b)):
        if "anchor" not in lib or any(key not in lib["anchor"] for key in ANCHOR_COUNTS):
            raise ValueError(name + " lacks current anchor payload")
        payload.update({"anchor/" + key: array_identity(value) for key, value in lib["anchor"].items()})
    mismatches = compare_payloads(a, b)
    table_mismatches = compare_payloads(ta, tb)
    storage_changes = integer_storage_changes(reference, candidate, mismatches)
    after = snapshot(paths)
    if before != after or content_fingerprint(geom, workers=1) != geometry_before:
        raise ValueError("native rebuild inputs/implementation/geometry changed during diagnostic")
    result = dict(scope="saved_native_rebuild_numeric_diagnostic_only", run_config=rc.name,
                  candidate_strictly_loaded=True, reference_admitted=False, physics_admitted=False,
                  identical_numeric_payload=not mismatches and not table_mismatches,
                  mismatch_count=len(mismatches) + len(table_mismatches), mismatches=mismatches,
                  integer_storage_diagnostics=storage_changes,
                  table_mismatches=table_mismatches, tables=dict(reference=ta, candidate=tb),
                  fingerprints=dict(reference=a, candidate=b),
                  payload_fingerprints={arm: content_fingerprint(
                      {"arrays": {key: arrays[key] for key in sorted(arrays)}, "tables": tables}, workers=1)
                      for arm, arrays, tables in (("reference", a, ta), ("candidate", b, tb))},
                  npz_metadata_excluded=dict(reference=ea, candidate=eb),
                  reference_build_anchor_declared=declared,
                  current_code_anchor_counts={arm: {key: lib["anchor"][key] for key in ANCHOR_COUNTS}
                                              for arm, lib in (("reference", reference_lib), ("candidate", candidate_lib))},
                  geometry_fingerprint=geometry_before,
                  hashes_before={key: value["sha256"] for key, value in before.items()},
                  hashes_after={key: value["sha256"] for key, value in after.items()},
                  limitations=["Same-current-code anchor comparison is NOT historical execution attestation.",
                               "Old build anchor counts are declarations only; the old bundle is not certified.",
                               "ROOT container hashes may differ despite identical numeric payloads.",
                               "No baseline rebuild, transport, response replay, or physics admission."])
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with out.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in ("identical_numeric_payload", "mismatch_count", "physics_admitted")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.reference, args.candidate, args.run_config, args.out)


if __name__ == "__main__":
    main()
