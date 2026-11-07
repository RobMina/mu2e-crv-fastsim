"""Replay only C4 transport and optional library to locate missing response upstream of digitisation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from .paired import KEY
from .primaries import extract
from .primary_chain import rekey


def step_counts(steps, selection, geom):
    """Return complete event rows with per-physical-sector step/layer counts and energy."""
    rows = pd.DataFrame({k: ak.to_numpy(selection[k]) for k in KEY})
    rows["dense"] = np.arange(1, len(rows) + 1)
    rows["steps"] = rows.dense.map(steps.groupby("event").size()).fillna(0).astype(np.int64)
    layers = steps.assign(layer=geom.layer[steps.bar.to_numpy(np.int64)]).groupby("event").layer.nunique()
    rows["layers_any"] = rows.dense.map(layers).fillna(0).astype(np.int64)
    for sector, name in enumerate(geom.sector_names):
        part = steps[geom.sector[steps.bar.to_numpy(np.int64)] == sector].copy()
        part["layer"] = geom.layer[part.bar.to_numpy(np.int64)]
        for label, value in (("layers", part.groupby("event").layer.nunique()),
                             ("edep", part.groupby("event").edep.sum())):
            rows[name + "_" + label] = rows.dense.map(value).fillna(0)
    return rows.drop(columns="dense")


def summarise(counts, decisions, sector_names):
    """Classify event-level transport support for each saved track decision without changing it."""
    rows = decisions.merge(counts, on=list(KEY), validate="many_to_one")
    rows["step_support"] = np.select([rows.steps == 0, rows.layers_any < 3],
                                     ["no_steps", "under_three_layers"], default="at_least_three_layers")
    rows["has_represented_muon"] = rows.represented_primaries > 0
    summary = rows.groupby(["decision", "has_represented_muon", "step_support"]).size().rename("tracks").reset_index().to_dict(orient="records")
    return rows, summary


def discordant_layers(steps, selection, geom, decisions):
    """Layer energy and true-bunch folded step times for the saved discordant tracks."""
    event = steps.event.to_numpy(np.int64) - 1
    rows = steps.copy()
    for key in KEY:
        rows[key] = ak.to_numpy(selection[key])[event]
    rows["sector"] = np.asarray(geom.sector_names)[geom.sector[steps.bar.to_numpy(np.int64)]]
    rows["layer"] = geom.layer[steps.bar.to_numpy(np.int64)]
    rows["folded_t0"] = np.mod(steps.t0.to_numpy() + ak.to_numpy(selection.pbtime_mc)[event], 1695.)
    rows = rows.merge(decisions[decisions.decision.isin(["stock_only", "fast_only"])][[*KEY, "track_index", "decision"]],
                      on=list(KEY), how="inner", validate="many_to_many")
    return rows.groupby([*KEY, "track_index", "decision", "sector", "layer"]).agg(
        steps=("edep", "size"), edep=("edep", "sum"), t_min=("folded_t0", "min"), t_max=("folded_t0", "max")).reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("selection", "decisions", "outdir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--gdml")
    parser.add_argument("--detector-origin", nargs=3, type=float)
    parser.add_argument("--saved-steps", nargs="+", help="summarise saved step parquets without replaying transport")
    parser.add_argument("--library-dir")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261105])
    args = parser.parse_args(argv)
    from .. import runconfig, transport
    selection = ak.from_parquet(args.selection)
    decisions = pd.read_parquet(args.decisions)
    rc = runconfig.get("run1a_v01")
    geom = rc.geometry()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.saved_steps:
        summary = {}
        for path in args.saved_steps:
            steps = pd.read_parquet(path)
            counts = step_counts(steps, selection, geom)
            discordant_layers(steps, selection, geom, decisions).to_csv(
                out / (Path(path).stem + "_layers.tsv"), sep="\t", index=False)
            rows, categories = summarise(counts, decisions, geom.sector_names)
            rows.to_parquet(out / (Path(path).stem + "_tracks.parquet"), index=False)
            rows[rows.decision.isin(["stock_only", "fast_only"])].to_csv(
                out / (Path(path).stem + "_discordant.tsv"), sep="\t", index=False)
            summary[Path(path).stem] = categories
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return
    if args.gdml is None or args.detector_origin is None:
        parser.error("transport replay requires --gdml and --detector-origin")
    primaries, _ = extract(selection, args.detector_origin)
    gen = rekey(selection, primaries)
    world, _ = transport.load_world(args.gdml, banded=True, y_plane=transport.msc_plane(geom))
    below = transport.load_below(args.gdml, geom, margin=rc.below_margin)
    lib = spec = None
    if args.library_dir:
        from .. import tier1b, tier1b_assets
        directory = Path(args.library_dir)
        options = dict(context=True, contain=True, point=True, topology=True, unvisited=True,
                       nearmiss=True, chord=True, anchor=True, per_primary=True)
        spec = tier1b_assets.Tier1bSpec(None, str(directory / "library.root"), str(directory / "topology.npz"),
                                      str(directory / "templates_v3.npz"), options=options)
        lib = tier1b_assets.load(spec, geom)
    summary = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        steps, parts = transport.transport(gen, geom, rng, stage="C", world=world, below=below,
            tagger_bar=rc.tagger_bar(geom), bar_cut=rc.bar_cut(geom), range_out=True, below_msc=rc.below_msc)
        stages = {"transport": step_counts(steps, selection, geom)}
        if lib is not None:
            steps, parts, _ = tier1b.sample(steps, parts, gen, lib, geom, rng, showers=True, **spec.options)
            stages["library"] = step_counts(steps, selection, geom)
        steps.to_parquet(out / f"steps_{seed}.parquet", index=False)
        summary[str(seed)] = {}
        for stage, counts in stages.items():
            rows, categories = summarise(counts, decisions, geom.sector_names)
            rows.to_parquet(out / f"{stage}_{seed}.parquet", index=False)
            rows[rows.decision.isin(["stock_only", "fast_only"])].to_csv(out / f"{stage}_{seed}.tsv", sep="\t", index=False)
            summary[str(seed)][stage] = dict(total_steps=int(counts.steps.sum()), categories=categories)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
