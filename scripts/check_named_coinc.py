"""Conditional saved-stock-pulse coincidence check with strict named conditions; no E2E admission."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np

from crvfastsim import readout, response_receipt, runconfig
from crvfastsim.named_conditions import load_conditions
from crvfastsim.reco import coinc, coinc_jax
from crvfastsim.validation import reco_detail
from crvfastsim.validation.pilot import digest
from mu2efastsim import match


def signatures(ids, sector, members, width):
    """Exact collection-order-independent cluster identities with sorted member pulse indices."""
    lengths = ak.to_numpy(ak.num(members, axis=1))
    padded = ak.to_numpy(ak.fill_none(ak.pad_none(ak.sort(members, axis=1), width, clip=True), -1))
    result = np.column_stack([ids, sector, lengths, padded]).astype(np.int64)
    if len(np.unique(result, axis=0)) != len(result):
        raise ValueError("duplicate cluster identity")
    return result


def compare(events, stock, ids, output):
    """Compare full event triples and memberships, including the saved zero-cluster universe."""
    ec = events[["run", "subrun", "event"]].to_numpy(dtype=np.int64)
    labels = (ec[:, 1] << 20) | ec[:, 2]
    order = np.argsort(labels)
    key = np.asarray(output["cl_event"], np.int64)
    pos = np.searchsorted(labels[order], key)
    if np.any(pos >= len(labels)) or not np.array_equal(labels[order][pos], key):
        raise ValueError("candidate event outside stock universe")
    midx, mcl = np.asarray(output["mem_idx"], np.int64), np.asarray(output["mem_cl"], np.int64)
    if np.any(mcl < 0) or np.any(mcl >= len(key)):
        raise ValueError("invalid candidate cluster member")
    mo = np.lexsort((midx, mcl))
    members = ak.unflatten(midx[mo], np.bincount(mcl, minlength=len(key)))
    width = max(int(ak.max(ak.num(ids, axis=1), initial=0, mask_identity=False)),
                int(ak.max(ak.num(members, axis=1), initial=0, mask_identity=False)))
    a = signatures(stock[["run", "subrun", "event"]].to_numpy(), stock.sectortype.to_numpy(), ids, width)
    b = signatures(ec[order][pos], np.asarray(output["cl_type"]), members, width)
    ia, ib = match.join(a, b) if len(a) and len(b) else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    missing = np.setdiff1d(np.arange(len(a)), ia)
    extra = np.setdiff1d(np.arange(len(b)), ib)
    return {"stock_only_cluster_signatures": a[missing].tolist(), "fast_only_cluster_signatures": b[extra].tolist(),
            "signature_columns": ["run", "subrun", "event", "sectortype", "member_count", "sorted_members_padded_minus1"],
            "events": len(ec), "clusters_stock": len(a), "clusters_fast": len(b), "matched": len(ia),
            "identical_cluster_lists": bool(len(a) == len(b) == len(ia)),
            "stock_sector_counts": {str(k): int(v) for k, v in zip(*np.unique(a[:, 3], return_counts=True))},
            "candidate_sector_counts": {str(k): int(v) for k, v in zip(*np.unique(b[:, 3], return_counts=True))}}


def run(reference, manifest, name, out):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    sources = [reference, manifest, __file__, coinc.__file__, coinc_jax.__file__]
    before = {str(Path(p).resolve()): digest(p) for p in sources}
    rc = runconfig.get(name)
    geom = rc.geometry()
    cond, evidence = load_conditions(manifest, rc, geom)
    cond = readout.apply(cond, geom, rc.counters_cfg())
    identity = response_receipt.event_identity(reference)
    if identity["count"] == 0:
        raise ValueError("empty stock universe")
    frames, links, metadata = reco_detail._read(reference)
    events, pulses, clusters = (frames[k] for k in ("events", "pulses", "clusters"))
    if events.run.nunique() != 1 or (events[["subrun", "event"]] >= 2**20).any().any():
        raise ValueError("require one run and fast event-label capacity")
    ev = (pulses.subrun.to_numpy(dtype=np.int64) << 20) | pulses.event.to_numpy(dtype=np.int64)
    G = coinc.sector_geom(geom, rc.counters_cfg())
    output = coinc_jax.find(ev, *[pulses[k].to_numpy() for k in ("idx", "bar", "sipm", "time", "pe", "seq")],
                            G, readout.effective_status(cond), rc.finder())
    import uproot
    with uproot.open(reference) as root:
        rows = root["crvdump/clusters"].arrays(["run", "subrun", "event", "idx", "pulses"], library="ak")
        order = np.lexsort(tuple(ak.to_numpy(rows[k]) for k in ("idx", "event", "subrun", "run")))
        members = rows.pulses[order]
    result = compare(events, clusters, members, output)
    unmatched = result["stock_only_cluster_signatures"] + result["fast_only_cluster_signatures"]
    if unmatched:
        bad = np.asarray(unmatched, dtype=np.int64)
        redo = np.unique((bad[:, 1] << 20) | bad[:, 2])
        corrected = {k: np.asarray(v).copy() for k, v in output.items()}
        coinc_jax._replace_with_numpy(corrected, redo, ev,
                                     *[pulses[k].to_numpy() for k in ("idx", "bar", "sipm", "time", "pe", "seq")],
                                     G, readout.effective_status(cond), rc.finder())
        result["numpy_on_mismatch_events"] = {"events_replayed": len(redo),
                                              "comparison": compare(events, clusters, members, corrected),
                                              "scope": "NumPy replacement only on JAX-stock mismatch events"}
    from dataclasses import replace
    legacy_cfg = replace(rc.finder(), add_opposite_side_pulses=False)
    legacy = coinc_jax.find(ev, *[pulses[k].to_numpy() for k in ("idx", "bar", "sipm", "time", "pe", "seq")],
                            G, readout.effective_status(cond), legacy_cfg)
    result["no_opposite_attachment_diagnostic"] = compare(events, clusters, members, legacy)
    result["no_opposite_attachment_diagnostic"]["scope"] = "historical_backing_control_not_a_default_change"
    import jax
    result.update(run_config=name, event_identity=identity, conditions=evidence,
                  fallback_events=len(output["fallback_events"]), backend=jax.default_backend(),
                  metadata=metadata, sources_sha256=before,
                  scope="conditional_saved_stock_pulses_not_named_VR3_or_E2E_certification")
    if {str(Path(p).resolve()): digest(p) for p in sources} != before:
        raise ValueError("source changed during check")
    if response_receipt.event_identity(reference) != identity:
        raise ValueError("event identity changed during check")
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("events", "clusters_stock", "clusters_fast", "matched",
                                              "identical_cluster_lists", "fallback_events", "backend")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--conditions-manifest", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.reference, args.conditions_manifest, args.run_config, args.out)


if __name__ == "__main__":
    main()
