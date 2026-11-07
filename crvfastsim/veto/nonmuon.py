"""Inspect C4 non-muon coverage with optional fixed-seed transport/library replay; no tuning."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from .paired import KEY


def load_sampler(path=None):
    """Use an explicit runtime snapshot when supplied, otherwise the installed sampler."""
    if path is None:
        from .. import tier1b
        return tier1b
    spec = importlib.util.spec_from_file_location("crvfastsim._c4_nonmuon_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_library(path):
    """Read only the reach, shower and shower-step tables at their highest ROOT cycles."""
    with uproot.open(path) as root:
        names = {key.split(";")[0].rsplit("/", 1)[-1]: key.split(";")[0]
                 for key in root.keys(recursive=True)}
        return {name: pd.DataFrame(root[names[name]].arrays(library="np"))
                for name in ("libgen", "libshower", "libshowersteps")}


def coverage(gen, lib, sampler):
    """Apply the sampler's species/KE join without drawing or consulting veto outcomes."""
    cells = ["apdg", "kebin"]
    reach, showers, steps = (lib[name] for name in ("libgen", "libshower", "libshowersteps"))
    if reach.duplicated(cells).any() or showers.shid.duplicated().any():
        raise ValueError("duplicate reach cell or shower identity")
    rows = gen[gen.pdg.abs() != 13].copy()
    rows["apdg"] = rows.pdg.abs()
    rows["ke"] = rows.e - rows.apdg.map(sampler.MASS).fillna(0.)
    rows["kebin"] = sampler._ke_bin(rows.ke.to_numpy())
    rows["known_mass"] = rows.apdg.isin(sampler.MASS)
    donors = showers.groupby(cells).agg(templates=("shid", "size"), donor_ke_min=("ke", "min"),
                                        donor_ke_max=("ke", "max")).reset_index()
    bins = reach.merge(donors, on=cells, how="outer", validate="one_to_one", indicator="coverage")
    bins["reach_cell"] = bins.coverage != "right_only"
    bins = bins.drop(columns="coverage").fillna({"ngen": 0, "nshower": 0, "templates": 0})
    bins["reach_probability"] = np.where(bins.ngen > 0, bins.nshower / np.maximum(bins.ngen, 1), 0.)
    if not bins.reach_probability.between(0, 1).all():
        raise ValueError("reach probability outside [0, 1]")
    if ((bins.reach_probability > 0) & (bins.templates == 0)).any():
        raise ValueError("positive reach probability without a shower template")
    rows = rows.merge(bins, on=cells, how="left", validate="many_to_one")
    rows = rows.fillna({"ngen": 0, "nshower": 0, "templates": 0, "reach_probability": 0})
    rows["reach_cell"] = rows.reach_cell.eq(True)
    rows["status"] = np.select([~rows.reach_cell, rows.ngen == 0, rows.nshower == 0],
                                ["missing_cell", "zero_denominator", "zero_shower_cell"], default="supported")
    rows["outside_donor_ke_range"] = (rows.templates > 0) & ((rows.ke < rows.donor_ke_min) | (rows.ke > rows.donor_ke_max))
    counts = rows.groupby(cells).agg(inputs=("pdg", "size"), input_ke_min=("ke", "min"),
                                    input_ke_max=("ke", "max"), expected_showers=("reach_probability", "sum")).reset_index()
    bins = bins.merge(counts, on=cells, how="outer", validate="one_to_one")
    bins = bins.fillna({"inputs": 0, "expected_showers": 0, "ngen": 0, "nshower": 0,
                        "templates": 0, "reach_probability": 0})
    bins["reach_cell"] = bins.reach_cell.eq(True)
    bins["ke_bin_low"] = sampler.KE_EDGES[bins.kebin.to_numpy(np.int64)]
    bins["ke_bin_high"] = sampler.KE_EDGES[bins.kebin.to_numpy(np.int64) + 1]
    p = rows.reach_probability.to_numpy()
    summary = dict(nonmuon_inputs=len(rows), input_events=len(rows[list(KEY)].drop_duplicates()),
                   library_shower_templates=len(showers), library_shower_steps=len(steps),
                   status_counts=rows.status.value_counts().to_dict(), unknown_mass=int((~rows.known_mass).sum()),
                   expected_showers=float(p.sum()), shower_count_variance=float(np.sum(p * (1 - p))),
                   probability_zero_showers=float(np.prod(1 - p)),
                   reach_count_template_mismatches=int((bins.nshower.fillna(0) != bins.templates.fillna(0)).sum()),
                   outside_donor_ke_range=int(rows.outside_donor_ke_range.sum()),
                   limitations=["Reach probabilities are fixed library estimates, not a selected-track closure model.",
                                "Zero-shower probability assumes independent sampler Bernoulli draws; no library uncertainty included.",
                                "Input position/direction and charge sign do not condition the non-muon donor draw.",
                                "Shower count is not coincidence count or veto count."])
    return rows, bins, summary


class DrawRecorder:
    """Record the sampler's existing draws without advancing or replacing its RNG stream."""

    def __init__(self, rng):
        self.rng, self.draws = rng, []

    def random(self, size):
        draw = self.rng.random(size)
        self.draws.append(draw.copy())
        return draw


def replay(selection, generators, gdml, directory, seed, out):
    """Repeat only transport/library at one fixed seed, observing the actual shower draws."""
    from .. import runconfig, tier1b, tier1b_assets, transport
    from .primary_chain import rekey
    gen = rekey(selection, generators, "retained")
    rc = runconfig.get("run1a_v01")
    geom = rc.geometry()
    world, _ = transport.load_world(gdml, banded=True, y_plane=transport.msc_plane(geom))
    below = transport.load_below(gdml, geom, margin=rc.below_margin)
    directory = Path(directory)
    options = dict(context=True, contain=True, point=True, topology=True, unvisited=True,
                   nearmiss=True, chord=True, anchor=True, per_primary=True)
    spec = tier1b_assets.Tier1bSpec(None, str(directory / "library.root"), str(directory / "topology.npz"),
                                  str(directory / "templates_v3.npz"), options=options)
    lib = tier1b_assets.load(spec, geom)
    rng = np.random.default_rng(seed)
    steps, parts = transport.transport(gen, geom, rng, stage="C", world=world, below=below,
        tagger_bar=rc.tagger_bar(geom), bar_cut=rc.bar_cut(geom), range_out=True, below_msc=rc.below_msc)
    observed = {}
    original = tier1b._showers_for_gen

    def observe(inputs, library, stream):
        recorder = DrawRecorder(stream)
        result = original(inputs, library, recorder)
        rows, _, _ = coverage(inputs, library, tier1b)
        rows["uniform_draw"] = recorder.draws[0]
        rows["accepted"] = rows.uniform_draw < rows.reach_probability
        rows.to_csv(Path(out) / "draws.tsv", sep="\t", index=False)
        observed.update(accepted=int(rows.accepted.sum()), shower_steps=len(result),
                        reach_draws=len(recorder.draws[0]), donor_draws=len(recorder.draws[1]))
        return result

    tier1b._showers_for_gen = observe
    try:
        steps, parts, info = tier1b.sample(steps, parts, gen, lib, geom, rng, showers=True, **options)
    finally:
        tier1b._showers_for_gen = original
    return dict(seed=seed, steps=len(steps), library_sampling=info, observed=observed,
                sampler_source=str(Path(tier1b.__file__).resolve()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("generators", "library", "outdir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--sampler-source", help="exact runtime tier1b.py snapshot, imported read-only")
    parser.add_argument("--decisions", help="optional post-coverage join to saved development track decisions")
    parser.add_argument("--replay-selection", help="optional fixed-seed transport/library replay, never digitisation")
    parser.add_argument("--gdml")
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args(argv)
    gen, sampler = pd.read_parquet(args.generators), load_sampler(args.sampler_source)
    if "shard" not in gen or (gen.shard >= 22).any():
        raise ValueError("this diagnostic requires development inputs with shard <22")
    rows, bins, summary = coverage(gen, read_library(args.library), sampler)
    summary["sampler_source"] = str(Path(sampler.__file__).resolve())
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "primaries.parquet", index=False)
    rows.to_csv(out / "primaries.tsv", sep="\t", index=False)
    bins.to_csv(out / "bins.tsv", sep="\t", index=False)
    species = rows.groupby("pdg").agg(inputs=("pdg", "size"), supported=("status", lambda x: (x == "supported").sum()),
                                     ke_min=("ke", "min"), ke_max=("ke", "max"),
                                     expected_showers=("reach_probability", "sum")).reset_index()
    species.to_csv(out / "species.tsv", sep="\t", index=False)
    summary["species"] = species.to_dict(orient="records")
    if args.decisions:
        decisions = pd.read_parquet(args.decisions)
        joined = rows.merge(decisions[[*KEY, "track_index", "decision"]], on=list(KEY), how="left", validate="many_to_many")
        joined.to_csv(out / "decisions.tsv", sep="\t", index=False)
        summary["post_coverage_decisions"] = joined.groupby("decision").agg(
            primary_track_pairs=("pdg", "size"), expected_showers=("reach_probability", "sum")).reset_index().to_dict(orient="records")
    if args.replay_selection:
        import awkward as ak
        selection = ak.from_parquet(args.replay_selection)
        if not args.gdml or args.sampler_source or ak.any(selection.shard >= 22):
            parser.error("replay requires --gdml, development selection, and the actual installed runtime sampler")
        summary["replay"] = replay(selection, gen, args.gdml, Path(args.library).parent, args.seed, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
