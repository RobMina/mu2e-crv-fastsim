"""Compare represented fast inputs with independently retained upstream generators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .paired import KEY, align
from .truth import generator_roots


def plan(selection, ancestors, out):
    """Select ancestor files by catalogue shard/subrun, without reading veto decisions."""
    events = pd.DataFrame({k: ak.to_numpy(selection[k]) for k in [*KEY, "shard"]})
    files = pd.DataFrame(ancestors)
    subruns = files.explode("subruns")
    subruns["run"] = subruns.subruns.astype(np.int64) // 1_000_000
    subruns["subrun"] = subruns.subruns.astype(np.int64) % 1_000_000
    mapped = events.merge(subruns[["shard", "run", "subrun", "fid", "name", "namespace", "retired"]],
                          on=["shard", "run", "subrun"], how="left", validate="many_to_one")
    if mapped.fid.isna().any() or mapped.retired.eq(True).any():
        raise ValueError("selected event lacks a unique non-retired upstream catalogue ancestor")
    chosen = files[files.fid.isin(mapped.fid)].drop_duplicates("fid").sort_values(["shard", "name"])
    out = Path(out)
    mapped.to_parquet(out / "event_sources.parquet", index=False)
    mapped.to_csv(out / "event_sources.tsv", sep="\t", index=False)
    chosen.drop(columns=["parents", "subruns"]).to_csv(out / "files.tsv", sep="\t", index=False)
    (out / "files.txt").write_text("\n".join(chosen.namespace + ":" + chosen.name) + "\n")
    summary = dict(events=len(events), source_shards=int(events.shard.nunique()),
                   upstream_files=len(chosen), catalog_input_bytes=int(chosen["size"].sum()),
                   scope="ancestor files selected by development event keys, independently of veto decisions")
    (out / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def collect(paths, selection):
    """Read each dump once and retain exact selected event keys, including zero-generator rows."""
    wanted = pd.MultiIndex.from_arrays([ak.to_numpy(selection[k]) for k in KEY], names=KEY)
    events, generators = [], []
    total_events = total_generators = 0
    for path in paths:
        with uproot.open(path) as root:
            census = pd.DataFrame(root["dump/events"].arrays(library="np"))
            gen = pd.DataFrame(root["dump/gen"].arrays(library="np"))
        observed = gen.groupby(list(KEY)).size().rename("observed")
        counts = census.merge(observed, on=list(KEY), how="left", validate="one_to_one")
        if not np.array_equal(counts.observed.fillna(0).to_numpy(), counts.n_gen.to_numpy()):
            raise ValueError("dump generator count disagrees with its event census")
        if len(gen.merge(census[list(KEY)], on=list(KEY), validate="many_to_one")) != len(gen):
            raise ValueError("generator row lacks an event census entry")
        if gen.duplicated([*KEY, "gen_product_id", "idx"]).any():
            raise ValueError("duplicate generator identity inside a dump")
        total_events += len(census)
        total_generators += len(gen)
        census = census[pd.MultiIndex.from_frame(census[list(KEY)]).isin(wanted)].copy()
        gen = gen[pd.MultiIndex.from_frame(gen[list(KEY)]).isin(wanted)].copy()
        census["dump_source"] = str(path)
        events.append(census)
        generators.append(gen)
    events, generators = pd.concat(events, ignore_index=True), pd.concat(generators, ignore_index=True)
    if events.duplicated(list(KEY)).any():
        raise ValueError("a selected event occurs in more than one upstream dump")
    events.attrs.update(dump_files=len(paths), dump_events=total_events, dump_generators=total_generators)
    return events, generators


def compare_inputs(selection, truth, events, gen, origin, supplied=None):
    """Measure retained-collection coverage without identifying particles by kinematic proximity."""
    from .primaries import extract
    truth = align(selection, truth)
    roots = generator_roots(selection, truth.all_trkmcsim)
    expected, _ = extract(selection, origin)
    if supplied is not None:
        expected = expected.sort_values([*KEY, "key"]).reset_index(drop=True)
        actual = supplied.merge(expected[[*KEY, "key"]], on=[*KEY, "key"], validate="one_to_one")
        actual = actual.sort_values([*KEY, "key"]).reset_index(drop=True)
        if len(actual) != len(expected) or len(supplied) != len(expected):
            raise ValueError("saved fast input identities differ from selected genealogy extraction")
        for field in [*KEY, "key", "pdg", "x", "y", "z", "px", "py", "pz", "t"]:
            if not np.array_equal(actual[field].to_numpy(), expected[field].to_numpy()):
                raise ValueError("saved fast input differs from selected genealogy extraction: " + field)
    fast_ids = pd.MultiIndex.from_frame(expected[[*KEY, "key"]])
    roots["in_fast"] = pd.MultiIndex.from_frame(roots[[*KEY, "id"]]).isin(fast_ids)
    pairs = gen.merge(roots, on=list(KEY), how="left", suffixes=("_gen", "_root"), validate="many_to_many")
    compatible = pairs.pdg_gen == pairs.pdg_root
    for i, axis in enumerate("xyz"):
        compatible &= pairs["pos_" + axis] == (pairs[axis] - origin[i]).astype(np.float32)
        compatible &= pairs["mom_" + axis] == pairs["p" + axis].astype(np.float32)
    compatible &= pairs.time == pairs.t.astype(np.float32)
    pairs["compatible"] = compatible
    pairs["compatible_fast"] = compatible & pairs.in_fast.eq(True)
    flags = pairs.groupby([*KEY, "gen_product_id", "idx"])[["compatible", "compatible_fast"]].sum()
    candidates = gen.merge(flags, on=[*KEY, "gen_product_id", "idx"], validate="one_to_one")
    candidates["absent_from_genealogy"] = candidates.compatible == 0
    candidates["absent_from_fast"] = candidates.compatible_fast == 0
    candidates["ambiguous_genealogy"] = candidates.compatible > 1
    candidates["missing_muon"] = candidates.absent_from_fast & (candidates.pdg.abs() == 13)
    candidates["excluded_nonmuon"] = candidates.absent_from_fast & (candidates.pdg.abs() != 13)
    state = ["pdg", "x", "y", "z", "px", "py", "pz", "e", "t"]
    candidates["duplicate_stored_state"] = candidates.duplicated([*KEY, *state], keep=False)
    root_flags = pairs.dropna(subset=["id"]).groupby([*KEY, "id"])["compatible"].sum().rename("upstream_compatible")
    roots = roots.merge(root_flags, on=[*KEY, "id"], how="left", validate="one_to_one")
    covered = pd.MultiIndex.from_frame(events[list(KEY)])
    roots["event_covered"] = pd.MultiIndex.from_frame(roots[list(KEY)]).isin(covered)
    roots["upstream_compatible"] = roots.upstream_compatible.fillna(0).astype(np.int64)
    counts = candidates.groupby(list(KEY)).agg(retained_generators=("idx", "size"),
        absent_from_genealogy=("absent_from_genealogy", "sum"), absent_from_fast=("absent_from_fast", "sum"),
        missing_muons=("missing_muon", "sum"), excluded_nonmuons=("excluded_nonmuon", "sum"),
        ambiguous_generators=("ambiguous_genealogy", "sum"), duplicate_stored_states=("duplicate_stored_state", "sum"))
    rows = pd.DataFrame({k: ak.to_numpy(selection[k]) for k in [*KEY, "shard"]})
    rows = rows.merge(events[[*KEY, "n_gen"]], on=list(KEY), how="left", validate="one_to_one")
    rows["event_covered"] = rows.n_gen.notna()
    rows = rows.merge(counts, on=list(KEY), how="left", validate="one_to_one")
    rows.loc[rows.event_covered, counts.columns] = rows.loc[rows.event_covered, counts.columns].fillna(0)
    fast_counts = expected.groupby(list(KEY)).size().rename("supplied_muons")
    rows = rows.merge(fast_counts, on=list(KEY), how="left", validate="one_to_one")
    rows["supplied_muons"] = rows.supplied_muons.fillna(0).astype(np.int64)
    roots["unresolved_fast_root"] = roots.in_fast & roots.event_covered & (roots.upstream_compatible != 1)
    unresolved = roots.groupby(list(KEY)).unresolved_fast_root.sum()
    rows = rows.merge(unresolved, on=list(KEY), how="left", validate="one_to_one")
    rows["unresolved_fast_root"] = rows.unresolved_fast_root.fillna(0).astype(np.int64)
    rows["retained_excess_count"] = rows.n_gen - rows.supplied_muons
    rows["absent_muon_and_nonmuon"] = (rows.missing_muons > 0) & (rows.excluded_nonmuons > 0)
    return rows, candidates, roots


def ancestry_coverage(ancestry, generators, origin):
    """Post-census native CRV primary signature compatibility; never construct generator inputs here."""
    keys = [*KEY, "track_index"]
    columns = [*keys, "decision", "crv_mc_primaryPdgId", *["crv_mc_primary.fCoordinates.f" + a for a in "XYZ"]]
    pairs = ancestry[columns].merge(generators, on=list(KEY), how="left", validate="many_to_many")
    match = pairs.crv_mc_primaryPdgId == pairs.pdg
    for i, axis in enumerate("xyz"):
        match &= pairs["crv_mc_primary.fCoordinates.f" + axis.upper()] == (pairs[axis] - origin[i]).astype(np.float32)
    pairs["primary_signature_in_retained"] = match
    pairs["primary_signature_in_missing"] = match & pairs.absent_from_fast.eq(True)
    flags = pairs.groupby(keys)[["primary_signature_in_retained", "primary_signature_in_missing"]].any()
    return ancestry.merge(flags, on=keys, how="left", validate="one_to_one")


def report(selection, truth, events, gen, origin, out, supplied=None, decisions=None, ancestry=None):
    rows, candidates, roots = compare_inputs(selection, truth, events, gen, origin, supplied)
    covered = rows[rows.event_covered]
    absent = candidates[candidates.absent_from_fast]
    summary = dict(extraction=dict(events.attrs),
        selected_events=len(rows), covered_events=len(covered), uncovered_events=int((~rows.event_covered).sum()),
        source_shards=int(rows.shard.nunique()), supplied_muons=int(rows.supplied_muons.sum()),
        events_without_supplied_muons=int((rows.supplied_muons == 0).sum()),
        covered_empty_fast_events_with_retained_muons=int(((covered.supplied_muons == 0) & (covered.missing_muons > 0)).sum()),
        covered_empty_fast_events_with_retained_nonmuons=int(((covered.supplied_muons == 0) & (covered.excluded_nonmuons > 0)).sum()),
        retained_generators=len(candidates), events_with_extra_generators=int((covered.absent_from_fast > 0).sum()),
        events_with_missing_muons=int((covered.missing_muons > 0).sum()),
        events_with_excluded_nonmuons=int((covered.excluded_nonmuons > 0).sum()),
        events_with_both_missing_types=int(covered.absent_muon_and_nonmuon.sum()),
        retained_species={str(int(k)): int(v) for k, v in candidates.pdg.value_counts().items()},
        generators_without_compatible_fast_input=len(absent),
        extra_species={str(int(k)): int(v) for k, v in absent.pdg.value_counts().items()},
        missing_muons=int(candidates.missing_muon.sum()), excluded_nonmuons=int(candidates.excluded_nonmuon.sum()),
        absent_from_all_genealogy=int(candidates.absent_from_genealogy.sum()),
        represented_nonmuons_excluded=int(((candidates.compatible > 0) & candidates.excluded_nonmuon).sum()),
        absent_nonmuons=int((candidates.absent_from_genealogy & candidates.excluded_nonmuon).sum()),
        events_with_more_retained_than_supplied=int((covered.retained_excess_count > 0).sum()),
        positive_retained_count_excess=int(covered.retained_excess_count.clip(lower=0).sum()),
        events_with_unresolved_fast_roots=int((covered.unresolved_fast_root > 0).sum()),
        ambiguous_generators=int(candidates.ambiguous_genealogy.sum()),
        duplicate_generator_states=int(candidates.duplicate_stored_state.sum()),
        supplied_roots_without_compatible_upstream_state=int((roots.in_fast & roots.event_covered & (roots.upstream_compatible == 0)).sum()),
        all_roots_without_compatible_upstream_state=int((roots.event_covered & (roots.upstream_compatible == 0)).sum()),
        saved_fast_inputs_checked=supplied is not None,
        scope="development population; retained upstream generators, not complete original CRY stream",
        comparison="exact event keys and float32-stored state compatibility; no particle-identity or nearest-neighbour match")
    out = Path(out)
    rows.to_parquet(out / "event_coverage.parquet", index=False)
    candidates.to_parquet(out / "generator_coverage.parquet", index=False)
    roots.to_parquet(out / "genealogy_coverage.parquet", index=False)
    absent.to_csv(out / "missing_generators.tsv", sep="\t", index=False)
    unresolved_keys = pd.MultiIndex.from_frame(roots.loc[roots.unresolved_fast_root, list(KEY)])
    unresolved = gen[pd.MultiIndex.from_frame(gen[list(KEY)]).isin(unresolved_keys)].merge(
        roots[roots.unresolved_fast_root], on=list(KEY), suffixes=("_gen", "_root"), validate="many_to_many")
    if len(unresolved):
        unresolved["dt"] = unresolved.t.astype(np.float32) - unresolved.time
        for i, axis in enumerate("xyz"):
            unresolved["dpos_" + axis] = (unresolved[axis] - origin[i]).astype(np.float32) - unresolved["pos_" + axis]
            unresolved["dmom_" + axis] = unresolved["p" + axis].astype(np.float32) - unresolved["mom_" + axis]
    unresolved.to_csv(out / "unresolved_state_pairs.tsv", sep="\t", index=False)
    if decisions is not None:
        tracks = decisions.merge(rows, on=list(KEY), suffixes=("", "_inputs"), how="left", validate="many_to_one")
        tracks.to_parquet(out / "track_coverage.parquet", index=False)
        summary["decisions"] = []
        missing_tracks = candidates[candidates.absent_from_fast].merge(
            decisions[[*KEY, "track_index", "decision"]], on=list(KEY), validate="many_to_many")
        missing_tracks.to_csv(out / "missing_generator_decisions.tsv", sep="\t", index=False)
        summary["missing_species_by_decision"] = missing_tracks.groupby(["decision", "pdg"]).size().rename(
            "generator_track_pairs").reset_index().to_dict("records")
        for name, sub in tracks.groupby("decision"):
            summary["decisions"].append(dict(decision=name, tracks=len(sub), covered=int(sub.event_covered.sum()),
                with_missing_muon=int((sub.missing_muons > 0).sum()), with_excluded_nonmuon=int((sub.excluded_nonmuons > 0).sum()),
                with_any_missing_generator=int((sub.absent_from_fast > 0).sum()),
                with_both_missing_types=int(sub.absent_muon_and_nonmuon.sum()),
                with_no_supplied_muon=int((sub.supplied_muons == 0).sum())))
    if ancestry is not None:
        attribution = ancestry_coverage(ancestry, candidates, origin)
        attribution.to_parquet(out / "ancestry_coverage.parquet", index=False)
        losses = attribution[attribution.decision == "stock_only"]
        summary["stock_only_native_ancestry"] = dict(tracks=len(losses),
            primary_signature_in_retained=int(losses.primary_signature_in_retained.sum()),
            primary_signature_in_missing=int(losses.primary_signature_in_missing.sum()),
            limit="native cluster summary compared after outcome-independent input census; signatures are not identities")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--ancestors")
    parser.add_argument("--dumps", nargs="+", help="existing independent generator dumps")
    parser.add_argument("--dump-list", help="completed_dumps.txt from the sequential extraction")
    parser.add_argument("--truth", help="saved all-track genealogy truth parquet")
    parser.add_argument("--primaries", help="actual supplied fast-primary parquet for exact extraction check")
    parser.add_argument("--decisions", help="optional saved decisions, joined only after input census")
    parser.add_argument("--ancestry", help="optional tracks_truth.parquet, compared only after input census")
    parser.add_argument("--detector-origin", nargs=3, type=float)
    parser.add_argument("--split-at", type=int, default=22)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    selection = ak.from_parquet(args.selection)
    selection = selection[selection.shard < args.split_at]
    if args.max_shards is not None:
        selection = selection[np.isin(ak.to_numpy(selection.shard), np.unique(ak.to_numpy(selection.shard))[:args.max_shards])]
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.ancestors:
        plan(selection, json.loads(Path(args.ancestors).read_text()), out)
    elif args.dumps or args.dump_list:
        if not args.truth or args.detector_origin is None:
            parser.error("generator comparison requires --truth and --detector-origin")
        paths = args.dumps or []
        if args.dump_list:
            paths += [p for p in Path(args.dump_list).read_text().splitlines() if p]
        events, gen = collect(paths, selection)
        truth = ak.from_parquet(args.truth)
        wanted = pd.MultiIndex.from_arrays([ak.to_numpy(selection[k]) for k in KEY])
        truth = truth[pd.MultiIndex.from_arrays([ak.to_numpy(truth[k]) for k in KEY]).isin(wanted)]
        supplied = pd.read_parquet(args.primaries) if args.primaries else None
        if supplied is not None:
            supplied = supplied[pd.MultiIndex.from_frame(supplied[list(KEY)]).isin(wanted)]
        decisions = pd.read_parquet(args.decisions) if args.decisions else None
        if decisions is not None:
            decisions = decisions[pd.MultiIndex.from_frame(decisions[list(KEY)]).isin(wanted)]
        ancestry = pd.read_parquet(args.ancestry) if args.ancestry else None
        if ancestry is not None:
            ancestry = ancestry[pd.MultiIndex.from_frame(ancestry[list(KEY)]).isin(wanted)]
        report(selection, truth, events, gen, args.detector_origin, out, supplied, decisions, ancestry)
        events.to_parquet(out / "selected_upstream_events.parquet", index=False)
        gen.to_parquet(out / "selected_upstream_generators.parquet", index=False)
    else:
        parser.error("specify --ancestors or generator dumps")


if __name__ == "__main__":
    main()
