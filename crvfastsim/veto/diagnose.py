"""Explain saved C4 decision differences without changing cuts or matching truth by proximity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from . import VetoCard, vetoed
from .apply import coinc_mask, gate
from .paired import KEY, align
from .primaries import extract


def arm_columns(times, coincs, card):
    """Describe each track's candidate set; the representative maximises signed window margin."""
    dt = times[:, :, None] - coincs.time[:, None, :]
    eligible = ak.broadcast_arrays(coinc_mask(coincs, card)[:, None, :], dt)[0]
    margin = np.minimum(dt - card.dt_lo, card.dt_hi - dt)
    score = ak.where(eligible, margin, -np.inf)
    best = ak.argmax(score, axis=2, keepdims=True)
    available = ak.any(eligible, axis=2)
    def pick(values):
        broad = ak.broadcast_arrays(values, dt)[0]
        return ak.mask(ak.firsts(broad[best], axis=2), available)
    out = dict(veto=vetoed(times, coincs, card), n_eligible=ak.sum(eligible, axis=2),
               dt=pick(dt), margin=pick(margin))
    fields = ("time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType", "pos.fCoordinates.fZ", "nHits")
    for key in fields:
        if key in coincs.fields:
            out[key] = pick(coincs[key][:, None, :])
    for key, value in dict(n_coincs=ak.num(coincs), gate=gate(coincs, card)).items():
        out[key] = ak.broadcast_arrays(ak.Array(value)[:, None], times)[0]
    return out


def diagnose(selection, fast, card, primary=None, pmin=85., pmax=200.):
    """Return one row per fixed stock-selected track, with independent candidate summaries per arm."""
    fast = align(selection, fast)
    p, t = selection.p_mid, selection.t_mid
    if not ak.all(ak.num(p) == ak.num(t)):
        raise ValueError("cached track momentum and time are not aligned")
    use = (p > pmin) & (p < pmax) & gate(selection.coincs, card)
    def flat(x):
        values = ak.to_numpy(ak.flatten(x[use]))
        return values.astype(float).filled(np.nan) if np.ma.is_masked(values) else np.asarray(values)
    def broadcast(x):
        return flat(ak.broadcast_arrays(ak.Array(x)[:, None], p)[0])
    data = {k: broadcast(selection[k]) for k in KEY}
    data.update(p_mid=flat(p), t_mid=flat(t), track_index=flat(
        selection.track_index if "track_index" in selection.fields else ak.local_index(p, axis=1)))
    for key in ("shard", "pbtime", "pbtime_mc"):
        if key in selection.fields:
            data[key] = broadcast(selection[key])
    primaries, _ = extract(selection, [0., 0., 0.])
    counts = primaries.groupby(list(KEY)).size().rename("represented_primaries")
    keys = pd.DataFrame({k: ak.to_numpy(selection[k]) for k in KEY})
    counts = keys.merge(counts, on=list(KEY), how="left", validate="one_to_one").represented_primaries
    data["represented_primaries"] = broadcast(counts.fillna(0).to_numpy(np.int64))
    sims = selection.trkmcsim
    data["track_has_primary"] = flat(ak.any(sims.valid & (sims.gen == 38) & (abs(sims.pdg) == 13), axis=2))
    for label, coincs in (("stock", selection.coincs), ("fast", fast.coincs)):
        data.update({label + "_" + k: flat(v) for k, v in arm_columns(t, coincs, card).items()})
    table = pd.DataFrame(data)
    sv, fv = table.stock_veto, table.fast_veto
    table["decision"] = np.select([sv & ~fv, fv & ~sv, sv & fv],
                                  ["stock_only", "fast_only", "both_veto"], default="neither_veto")
    table["mechanism"] = "agreement"
    for label, decision in (("fast", "stock_only"), ("stock", "fast_only")):
        mechanism = np.select([table[label + "_n_coincs"] == 0, table[label + "_n_eligible"] == 0],
                              ["no_coincidence", "no_eligible_coincidence"], default="outside_time_window")
        table.loc[table.decision == decision, "mechanism"] = mechanism[table.decision == decision]
    if primary is not None:
        primary = align(selection, primary)
        table["primary_veto"] = flat(vetoed(t, primary.coincs, card))
    return table


def summarise(table, edges=(1., 5., 10.)):
    """Summarise diagnostic categories and timing margins, never interpret them as truth associations."""
    summary = dict(selected=len(table), stock_veto=int(table.stock_veto.sum()), fast_veto=int(table.fast_veto.sum()),
                   stock_only=int((table.decision == "stock_only").sum()),
                   fast_only=int((table.decision == "fast_only").sum()),
                   no_represented_primary=int((table.represented_primaries == 0).sum()),
                   fast_gate_failures=int((~table.fast_gate).sum()),
                   representative="independent per-arm coincidence with maximum signed dt-window margin; not a truth match",
                   categories={})
    for decision in ("stock_only", "fast_only", "both_veto", "neither_veto"):
        sub = table[table.decision == decision]
        row = dict(n=len(sub), mechanisms={k: int(v) for k, v in sub.mechanism.value_counts().items()},
                   no_represented_primary=int((sub.represented_primaries == 0).sum()),
                   track_without_primary=int((~sub.track_has_primary).sum()),
                   multiplicity_differs=int((sub.stock_n_coincs != sub.fast_n_coincs).sum()),
                   fast_gate_failures=int((~sub.fast_gate).sum()))
        if decision in ("stock_only", "fast_only"):
            loser, winner = ("fast", "stock") if decision == "stock_only" else ("stock", "fast")
            gap = -sub[loser + "_margin"]
            row["nonveto_near_boundary"] = {str(edge): int(((gap >= 0) & (gap <= edge)).sum()) for edge in edges}
            margin = sub[winner + "_margin"]
            row["veto_near_boundary"] = {str(edge): int(((margin > 0) & (margin <= edge)).sum()) for edge in edges}
        summary["categories"][decision] = row
    if "primary_veto" in table:
        summary["primary_vs_library"] = dict(primary_veto=int(table.primary_veto.sum()),
            gained=int((~table.primary_veto & table.fast_veto).sum()),
            lost=int((table.primary_veto & ~table.fast_veto).sum()),
            stock_only_shared=int((table.stock_veto & ~table.primary_veto & ~table.fast_veto).sum()),
            note="separate stochastic realizations; switches are not a pure secondary-response effect")
    return summary


def lineage_rows(selection, table):
    """Flatten saved selected-track genealogy with exact event and original-track identifiers."""
    sims = selection.trkmcsim
    data = {}
    for key in KEY:
        data[key] = ak.to_numpy(ak.flatten(ak.broadcast_arrays(selection[key], sims.id)[0], axis=None))
    index = selection.track_index if "track_index" in selection.fields else ak.local_index(selection.p_mid, axis=1)
    data["track_index"] = ak.to_numpy(ak.flatten(ak.broadcast_arrays(index, sims.id)[0], axis=None))
    for key in ("valid", "id", "pdg", "gen", "startCode", "stopCode", "rank", "nhits", "nactive", "time"):
        if key in sims.fields:
            data[key] = ak.to_numpy(ak.flatten(sims[key], axis=None))
    for key in ("prirel", "trkrel"):
        if key in sims.fields:
            data[key] = ak.to_numpy(ak.flatten(sims[key]._rel, axis=None))
    for field in ("pos", "mom", "endpos", "endmom"):
        if field in sims.fields:
            for axis in ("X", "Y", "Z"):
                data[field + "_" + axis.lower()] = ak.to_numpy(ak.flatten(sims[field].fCoordinates["f" + axis], axis=None))
    rows = pd.DataFrame(data)
    return rows.merge(table[list(KEY) + ["track_index", "decision", "represented_primaries"]],
                      on=list(KEY) + ["track_index"], how="inner", validate="many_to_one")


def geometry_diagnostic(lineage, table, geom, origin):
    """Truth-start straight rays and stock longitudinal positions, not a corrected response model."""
    from ..crossings import crossings
    roots = lineage[lineage.valid & (lineage.gen == 38)].drop_duplicates(list(KEY) + ["track_index", "id"]).copy()
    momentum = roots[["mom_x", "mom_y", "mom_z"]].to_numpy(float)
    magnitude = np.linalg.norm(momentum, axis=1)
    position = roots[["pos_x", "pos_y", "pos_z"]].to_numpy(float) + np.asarray(origin)
    direction = np.divide(momentum, magnitude[:, None], out=np.zeros_like(momentum), where=magnitude[:, None] > 0)
    rays = crossings(position, direction, geom, backend="numpy")
    ir, ib = rays[:, 0].astype(np.int64), rays[:, 1].astype(np.int64)
    roots["p_start"] = magnitude
    roots["ray_bars"] = np.bincount(ir, minlength=len(roots))
    for sector, name in enumerate(geom.sector_names):
        use = geom.sector[ib] == sector
        pairs = pd.DataFrame(dict(root=ir[use], layer=geom.layer[ib[use]])).drop_duplicates()
        roots["ray_layers_" + name] = pairs.groupby("root").size().reindex(np.arange(len(roots)), fill_value=0).to_numpy()
        mask = geom.sector == sector
        lo = (geom.center[mask] - geom.halflen[mask]).min(axis=0)
        hi = (geom.center[mask] + geom.halflen[mask]).max(axis=0)
        distance = np.divide((lo[1] + hi[1]) / 2 - position[:, 1], direction[:, 1],
                             out=np.full(len(roots), np.nan), where=direction[:, 1] != 0)
        hit = position + distance[:, None] * direction
        roots["ray_x_" + name], roots["ray_z_" + name] = hit[:, 0], hit[:, 2]
        miss = np.maximum(np.maximum(lo[[0, 2]] - hit[:, [0, 2]], hit[:, [0, 2]] - hi[[0, 2]]), 0.)
        roots["footprint_distance_" + name] = np.where(distance >= 0, np.linalg.norm(miss, axis=1), np.nan)
    mu = roots[roots.pdg.abs() == 13].copy()
    cols = ["ray_layers_" + name for name in geom.sector_names]
    mu["ray_max_layers"] = mu[cols].max(axis=1)
    count = mu.groupby(list(KEY) + ["track_index"]).ray_max_layers.max().rename("ray_max_layers")
    tracks = table.merge(count, on=list(KEY) + ["track_index"], how="left", validate="one_to_one")
    roots = roots.merge(table[[*KEY, "track_index", "stock_PEs", "stock_nLayers", "stock_margin"]],
                        on=[*KEY, "track_index"], how="left", validate="many_to_one")
    generated = roots[[*KEY, "id", "pdg", "decision"]].drop_duplicates()
    species = generated.groupby(["decision", "pdg"]).size().rename("primaries").reset_index()
    summary = {"scope": "truth-start straight-ray diagnostic; excludes material, scattering and field deflection",
               "generator_species": species.to_dict(orient="records"),
               "sector_types_are_not_physical_sectors": True, "decisions": {}}
    for decision in ("stock_only", "fast_only", "both_veto", "neither_veto"):
        sub = tracks[tracks.decision == decision]
        row = dict(no_muon_ray=int(sub.ray_max_layers.isna().sum()),
                   zero_layers=int((sub.ray_max_layers == 0).sum()),
                   one_or_two_layers=int(sub.ray_max_layers.isin([1, 2]).sum()),
                   three_or_four_layers=int((sub.ray_max_layers >= 3).sum()))
        zkey = "stock_pos.fCoordinates.fZ"
        if zkey in sub:
            z = sub[zkey] + origin[2]
            row["stock_z_in_physical_sector"] = {}
            for sector, name in enumerate(geom.sector_names):
                mask = geom.sector == sector
                lo = (geom.center[mask, 2] - geom.halflen[mask, 2]).min()
                hi = (geom.center[mask, 2] + geom.halflen[mask, 2]).max()
                row["stock_z_in_physical_sector"][name] = int(((z >= lo) & (z <= hi)).sum())
        summary["decisions"][decision] = row
    return roots, tracks, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("selection", "fast", "card", "outdir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--primary", help="optional primary-only response on the same selected events")
    parser.add_argument("--counters", help="optional source-matching counters file for a truth-ray diagnostic")
    parser.add_argument("--detector-origin", nargs=3, type=float)

    args = parser.parse_args(argv)
    saved = json.loads(Path(args.card).read_text())
    values = dict(saved.get("card", saved))
    values.pop("sha", None)
    selection = ak.from_parquet(args.selection)
    table = diagnose(selection, ak.from_parquet(args.fast), VetoCard.from_dict(values),
                     ak.from_parquet(args.primary) if args.primary else None)
    summary = summarise(table)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "tracks.parquet", index=False)
    table[table.decision.isin(["stock_only", "fast_only"])].to_csv(out / "discordant.tsv", sep="\t", index=False)
    lineage = lineage_rows(selection, table)
    lineage.to_parquet(out / "lineage.parquet", index=False)
    lineage[lineage.decision.isin(["stock_only", "fast_only"])].to_csv(out / "discordant_lineage.tsv", sep="\t", index=False)
    lineage[lineage.represented_primaries == 0].groupby(["decision", "valid", "pdg", "gen"]).size().rename("rows").to_csv(
        out / "missing_primary_species.tsv", sep="\t")
    if args.counters:
        if args.detector_origin is None:
            parser.error("--counters requires --detector-origin")
        from ..geometry import from_offline_config
        roots, tracks, geo = geometry_diagnostic(lineage, table, from_offline_config(args.counters), args.detector_origin)
        roots.to_parquet(out / "roots.parquet", index=False)
        roots[roots.decision.isin(["stock_only", "fast_only"])].to_csv(out / "discordant_roots.tsv", sep="\t", index=False)
        tracks.to_parquet(out / "tracks_geometry.parquet", index=False)
        summary["geometry_diagnostic"] = geo
    if "shard" in table:
        shards = table.assign(stock_only=table.decision == "stock_only", fast_only=table.decision == "fast_only")
        shards.groupby("shard")[["stock_only", "fast_only", "stock_veto", "fast_veto"]].sum().to_csv(out / "shards.tsv", sep="\t")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
