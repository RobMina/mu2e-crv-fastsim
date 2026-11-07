"""Read saved CRV ancestry and event primaries for the fixed C4 development population."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .paired import KEY, align


def source_rows(path, wanted):
    """Join source events by exact identity and retain native reco/MC cluster alignment."""
    with uproot.open(path) as root:
        tree = root["EventNtuple/ntuple"]
        keys = tree.arrays(["evtinfo/" + k for k in KEY], library="np")
        events = pd.DataFrame({k: keys["evtinfo/" + k] for k in KEY})
        events["row"] = np.arange(len(events))
        req = pd.DataFrame({k: ak.to_numpy(wanted[k]) for k in KEY})
        joined = req.merge(events, on=list(KEY), how="left", validate="one_to_one", sort=False)
        if joined.row.isna().any():
            raise ValueError("source is missing cached events")
        rows = joined.row.to_numpy(np.int64)
        fields = {k: wanted[k] for k in KEY}
        for name in ("crvcoincs", "crvcoincsmc", "crvsummarymc", "crvcoincsmcplane"):
            leaves = [k for k in tree.keys() if k.startswith(name + "/") and not tree[k].keys()]
            arrays = tree.arrays(leaves, library="ak", how=dict)
            fields[name] = ak.zip({k.split("/")[-1].removeprefix(name + "."): v[rows] for k, v in arrays.items()},
                                  depth_limit=1 if name == "crvsummarymc" else 2)
        fields["all_trkmcsim"] = tree["trkmcsim"].array(library="ak")[rows]
    for key in ("time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType"):
        if not ak.all(ak.num(fields["crvcoincs"][key]) == ak.num(wanted.coincs[key])):
            raise ValueError("source coincidence counts differ from cache")
        if not ak.all(fields["crvcoincs"][key] == wanted.coincs[key]):
            raise ValueError("source coincidence values differ from cache")
    if not ak.all(ak.num(fields["crvcoincs"].time) == ak.num(fields["crvcoincsmc"].valid)):
        raise ValueError("source MC and reco clusters are not aligned")
    return ak.zip(fields, depth_limit=1)


def collect(paths, selection, out):
    """Read only source files represented by the selected development population."""
    out = Path(out)
    parts = []
    for shard in np.unique(ak.to_numpy(selection.shard)):
        wanted = selection[selection.shard == shard]
        dest = out / f"truth_{shard:03d}.parquet"
        rows = align(wanted, ak.from_parquet(dest)) if dest.exists() else source_rows(paths[int(shard)], wanted)
        if not dest.exists():
            ak.to_parquet(rows, dest)
        parts.append(rows)
        print(json.dumps(dict(shard=int(shard), events=len(rows), clusters=int(ak.sum(ak.num(rows.crvcoincs.time))))), flush=True)
    combined = align(selection, ak.concatenate(parts))
    ak.to_parquet(combined, out / "truth.parquet")
    return combined


def inspect_source(path):
    """List available truth branches without claiming that ancestry kinematics carry particle identity."""
    with uproot.open(path) as root:
        tree = root["EventNtuple/ntuple"]
        names = tree.typenames()
        return dict(entries=tree.num_entries, branches={k: v for k, v in names.items()
                    if any(word in k.lower() for word in ("crv", "primary", "gen", "sim", "evtinfo"))})


def generator_roots(events, genealogy):
    """Unique represented CRY roots by event/SimParticle ID, including non-muons."""
    flat = ak.flatten(ak.flatten(genealogy, axis=2), axis=1)
    event = ak.to_numpy(ak.flatten(ak.broadcast_arrays(ak.local_index(events, axis=0), genealogy.id)[0], axis=None))
    good = ak.to_numpy(flat.valid & (flat.gen == 38))
    roots = flat[good]
    data = {k: ak.to_numpy(events[k])[event[good]] for k in KEY}
    data.update(id=ak.to_numpy(roots.id), pdg=ak.to_numpy(roots.pdg), time=ak.to_numpy(roots.time))
    for field in ("pos", "mom"):
        for axis in ("X", "Y", "Z"):
            data[field + "_" + axis.lower()] = ak.to_numpy(roots[field].fCoordinates["f" + axis])
    result = pd.DataFrame(data)
    state = [k for k in result if k not in [*KEY, "id"]]
    if (result.groupby([*KEY, "id"])[state].nunique() > 1).any().any():
        raise ValueError("one represented generator identity carries inconsistent states")
    return result.drop_duplicates([*KEY, "id"]).reset_index(drop=True)


def ancestry_table(selection, truth, card, decisions):
    """Native cluster index links its MC summary; no kinematic or nearest-neighbour identity matching."""
    from .apply import coinc_mask
    truth = align(selection, truth)
    co = truth.crvcoincs
    times = selection.t_mid
    dt = times[:, :, None] - co.time[:, None, :]
    hit = (dt > card.dt_lo) & (dt < card.dt_hi) & coinc_mask(co, card)[:, None, :]
    score = ak.where(hit, co.PEs[:, None, :], -1.)
    best = ak.argmax(score, axis=2, keepdims=True)
    veto = ak.any(hit, axis=2)
    index = selection.track_index if "track_index" in selection.fields else ak.local_index(times, axis=1)
    data = {k: ak.to_numpy(ak.flatten(ak.broadcast_arrays(selection[k], times)[0])) for k in KEY}
    data["track_index"] = ak.to_numpy(ak.flatten(index))
    def pick(value):
        values = ak.mask(ak.firsts(ak.broadcast_arrays(value[:, None, :], dt)[0][best], axis=2), veto)
        values = ak.to_numpy(ak.flatten(values)).astype(float)
        return np.ma.filled(values, np.nan)
    data["cluster_index"] = pick(ak.local_index(co.time, axis=1))
    data["attributed_stock_PEs"] = pick(co.PEs)
    data["attributed_stock_time"] = pick(co.time)
    for key in truth.crvcoincsmc.fields:
        data["crv_mc_" + key] = pick(truth.crvcoincsmc[key])
    for layer in range(4):
        data[f"stock_PE_layer{layer}"] = pick(co["PEsPerLayer[4]"][:, :, layer])
    for key in truth.crvsummarymc.fields:
        data["crv_summary_" + key] = ak.to_numpy(ak.flatten(ak.broadcast_arrays(truth.crvsummarymc[key], times)[0]))
    table = decisions.merge(pd.DataFrame(data), on=[*KEY, "track_index"], how="left", validate="one_to_one")
    all_roots = generator_roots(selection, truth.all_trkmcsim)
    selected_roots = generator_roots(selection, selection.trkmcsim)
    selected_ids = pd.MultiIndex.from_frame(selected_roots[[*KEY, "id"]])
    all_roots["in_selected_genealogy"] = pd.MultiIndex.from_frame(all_roots[[*KEY, "id"]]).isin(selected_ids)
    all_roots["in_fast_input"] = all_roots.in_selected_genealogy & (all_roots.pdg.abs() == 13)
    count = all_roots.groupby(list(KEY)).agg(represented_all_roots=("id", "size"), fast_roots=("in_fast_input", "sum"))
    table = table.merge(count, on=list(KEY), how="left", validate="many_to_one")
    species = all_roots[[*KEY, "pdg"]].drop_duplicates()
    table = table.merge(species.assign(primary_species_represented=True), left_on=[*KEY, "crv_mc_primaryPdgId"],
                        right_on=[*KEY, "pdg"], how="left", validate="many_to_one").drop(columns="pdg")
    table["primary_species_represented"] = table.primary_species_represented.eq(True)
    table[["represented_all_roots", "fast_roots"]] = table[["represented_all_roots", "fast_roots"]].fillna(0).astype(np.int64)
    columns = [*KEY, "track_index", "crv_mc_primaryPdgId", *["crv_mc_primary.fCoordinates.f" + a for a in "XYZ"]]
    signatures = table[columns].merge(all_roots, on=list(KEY), how="left", validate="many_to_many")
    compatible = signatures.crv_mc_primaryPdgId == signatures.pdg
    for axis in "XYZ":
        compatible &= signatures["crv_mc_primary.fCoordinates.f" + axis] == signatures["pos_" + axis.lower()]
    signatures["start_state_compatible"] = compatible
    signatures["fast_start_state_compatible"] = compatible & signatures.in_fast_input.eq(True)
    flags = signatures.groupby([*KEY, "track_index"])[["start_state_compatible", "fast_start_state_compatible"]].any()
    table = table.merge(flags, on=[*KEY, "track_index"], validate="one_to_one")
    return table, all_roots


def report(selection, truth, card, decisions, out, support=None):
    """Write ancestry summaries for saved decisions; species agreement is not particle identity."""
    table, roots = ancestry_table(selection, truth, card, decisions)
    table["coverage_diagnostic"] = np.select(
        [~table.stock_veto, ~table.start_state_compatible, ~table.fast_start_state_compatible],
        ["no_stock_veto", "start_signature_absent", "represented_nonmuon_excluded"],
        default="muon_start_signature_present")
    if support is not None:
        table = table.merge(support[[*KEY, "track_index", "step_support"]], on=[*KEY, "track_index"], validate="one_to_one")
    out = Path(out)
    table.to_parquet(out / "tracks_truth.parquet", index=False)
    roots.to_parquet(out / "represented_roots.parquet", index=False)
    table[table.decision.isin(["stock_only", "fast_only"])].to_csv(out / "discordant_truth.tsv", sep="\t", index=False)
    roots.merge(decisions[[*KEY, "track_index", "decision"]], on=list(KEY), validate="many_to_many").query(
        "decision == 'stock_only' or decision == 'fast_only'").to_csv(out / "discordant_roots.tsv", sep="\t", index=False)
    summary = dict(events=len(selection), roots=len(roots), excluded_roots=int((~roots.in_fast_input).sum()),
                   extra_roots_from_other_tracks=int((~roots.in_selected_genealogy).sum()),
                   identity_limit="CRV MC branch has no SimParticle IDs; species/kinematics are not an identity match",
                   population_limit="all reconstructed-track genealogy union, not a complete generated-primary census",
                   decisions={})
    for decision in ("stock_only", "fast_only", "both_veto", "neither_veto"):
        sub = table[table.decision == decision]
        summary["decisions"][decision] = dict(n=len(sub),
            valid_crv_mc=int(sub.crv_mc_valid.eq(1).sum()),
            primary_species={str(int(k)): int(v) for k, v in sub.crv_mc_primaryPdgId.value_counts().items()},
            depositing_species={str(int(k)): int(v) for k, v in sub.crv_mc_pdgId.value_counts().items()},
            crv_primary_species_absent=int((sub.stock_veto & ~sub.primary_species_represented).sum()),
            crv_primary_start_absent=int((sub.stock_veto & ~sub.start_state_compatible).sum()),
            crv_primary_start_absent_from_fast=int((sub.stock_veto & ~sub.fast_start_state_compatible).sum()),
            events_with_excluded_roots=int((sub.represented_all_roots > sub.fast_roots).sum()))
    loss = table[table.decision == "stock_only"]
    grouping = ["coverage_diagnostic"] + (["step_support"] if support is not None else [])
    summary["stock_only_coverage"] = loss.groupby(grouping).size().rename("tracks").reset_index().to_dict(orient="records")
    summary["start_signature_definition"] = "exact stored PDG and start coordinates; incompatible signature excludes a root, compatible does not prove identity"
    (out / "ancestry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def retained_generators(path, selection, out, origin=None):
    """Census independent generators; optional state compatibility does not establish particle identity."""
    with uproot.open(path) as root:
        events = pd.DataFrame(root["dump/events"].arrays(library="np"))
        gen = pd.DataFrame(root["dump/gen"].arrays(library="np"))
    keys = list(KEY)
    if events.duplicated(keys).any() or gen.duplicated(keys + ["gen_product_id", "idx"]).any():
        raise ValueError("duplicate event or generator identity in upstream dump")
    counts = gen.groupby(keys).size().rename("observed_gen")
    census = events.merge(counts, on=keys, how="left", validate="one_to_one")
    if not np.array_equal(census.observed_gen.fillna(0).to_numpy(), census.n_gen.to_numpy()):
        raise ValueError("generator rows disagree with event census")
    represented = gen[keys].drop_duplicates().merge(events[keys], on=keys, how="left", indicator=True)
    if (represented._merge != "both").any():
        raise ValueError("generator row outside event census")
    wanted = pd.DataFrame({k: ak.to_numpy(selection[k]) for k in KEY})
    if wanted.duplicated(keys).any():
        raise ValueError("duplicate selected event identity")
    overlap = wanted.merge(events[keys], on=keys, how="inner", validate="one_to_one")
    summary = dict(events=len(events), generators=len(gen), empty_events=int((events.n_gen == 0).sum()),
                   pdg_counts={str(k): int(v) for k, v in gen.pdg.value_counts().items()},
                   generator_codes={str(k): int(v) for k, v in gen.genid.value_counts().items()},
                   subruns=events.groupby(["run", "subrun"]).size().rename("events").reset_index().to_dict("records"),
                   selected_events=len(wanted), exact_event_key_overlap=len(overlap),
                   full_generator_stream=bool(events.full_generator_stream.eq(1).all()) if len(events) else False,
                   collection_tags=sorted(events.gen_tag.unique()),
                   mapping_limit="exact event keys only; no upstream/downstream resampling identity or time map established",
                   population_limit="all generators in the retained collection; not a complete generated-primary census")
    out = Path(out)
    gen.to_parquet(out / "retained_generators.parquet", index=False)
    events.to_parquet(out / "upstream_events.parquet", index=False)
    overlap.to_csv(out / "event_key_overlap.tsv", sep="\t", index=False)
    if origin is not None:
        roots = generator_roots(selection, selection.trkmcsim)
        pairs = roots.merge(gen, on=keys, suffixes=("_root", "_gen"), validate="many_to_many")
        pairs["pdg_compatible"] = pairs.pdg_root == pairs.pdg_gen
        pairs["position_compatible"] = True
        pairs["momentum_compatible"] = True
        for i, axis in enumerate("xyz"):
            pairs["position_compatible"] &= pairs["pos_" + axis] == (pairs[axis] - origin[i]).astype(np.float32)
            pairs["momentum_compatible"] &= pairs["mom_" + axis] == pairs["p" + axis].astype(np.float32)
        pairs["time_compatible"] = pairs.time == pairs.t.astype(np.float32)
        flags = ["pdg_compatible", "position_compatible", "momentum_compatible", "time_compatible"]
        pairs["state_compatible"] = pairs[flags].all(axis=1)
        pairs.to_csv(out / "event_key_state_compatibility.tsv", sep="\t", index=False)
        compatible = pairs.groupby(keys + ["gen_product_id", "idx"])["state_compatible"].any()
        candidates = gen.merge(overlap, on=keys, validate="many_to_one").merge(
            compatible, on=keys + ["gen_product_id", "idx"], how="left", validate="one_to_one")
        candidates["represented_state_compatible"] = candidates.state_compatible.eq(True)
        candidates = candidates.drop(columns="state_compatible")
        candidates.to_parquet(out / "selected_generator_candidates.parquet", index=False)
        summary["event_key_state_diagnostic"] = dict(candidate_pairs=len(pairs),
            compatible_pairs=int(pairs.state_compatible.sum()), detector_origin=list(origin),
            retained_generators_in_overlapping_events=len(candidates),
            generators_without_compatible_genealogy_state=int((~candidates.represented_state_compatible).sum()),
            limit="exact float32-stored PDG/start-state compatibility within same event keys; not particle identity")
    (out / "generator_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files")
    parser.add_argument("--generator-dump", help="independent retained-generator ROOT dump; census only, not fast inputs")
    parser.add_argument("--detector-origin", nargs=3, type=float, help="optional generator/start-state compatibility diagnostic")
    parser.add_argument("--truth", help="saved truth parquet, for local reporting")
    parser.add_argument("--card")
    parser.add_argument("--decisions")
    parser.add_argument("--support", help="optional saved transport-support per-track parquet")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--selection")
    parser.add_argument("--split-at", type=int, default=22)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--shards", nargs="+", type=int)

    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    paths = [line.strip() for line in Path(args.files).read_text().splitlines() if line.strip()] if args.files else []
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.inspect:
        if not paths:
            parser.error("--inspect requires --files")
        result = inspect_source(paths[0])
        (out / "branches.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    elif args.selection:
        selection = ak.from_parquet(args.selection)
        selection = selection[selection.shard < args.split_at]
        if args.shards is not None:
            selection = selection[np.isin(ak.to_numpy(selection.shard), args.shards)]
        if args.max_shards is not None:
            shards = np.unique(ak.to_numpy(selection.shard))[:args.max_shards]
            selection = selection[np.isin(ak.to_numpy(selection.shard), shards)]
        if args.generator_dump:
            retained_generators(args.generator_dump, selection, out, args.detector_origin)
        elif args.truth:
            if not args.card or not args.decisions:
                parser.error("--truth requires --card and --decisions")
            from . import VetoCard
            saved = json.loads(Path(args.card).read_text())
            values = dict(saved.get("card", saved))
            values.pop("sha", None)
            report(selection, ak.from_parquet(args.truth), VetoCard.from_dict(values), pd.read_parquet(args.decisions), out,
                   pd.read_parquet(args.support) if args.support else None)
        elif paths:
            collect(paths, selection, out)
        else:
            parser.error("collection requires --files")
    else:
        parser.error("specify --inspect or --selection")


if __name__ == "__main__":
    main()
