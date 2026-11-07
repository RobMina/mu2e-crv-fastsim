"""Prepare retained generator inputs using event identity and an independent event census."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from .paired import KEY

STATE = ("x", "y", "z", "px", "py", "pz", "e", "t")
IDENTITY = (*KEY, "gen_product_id", "idx")


def prepare(selection, generators, census):
    """Keep every retained species and source identity; never derive inputs from veto outcomes."""
    selected = pd.DataFrame({k: np.asarray(selection[k]) for k in (*KEY, "shard")})
    if selected.duplicated(list(KEY)).any() or census.duplicated(list(KEY)).any():
        raise ValueError("duplicate selected or upstream event identity")
    expected = selected.merge(census[[*KEY, "n_gen"]], on=list(KEY), how="left", validate="one_to_one")
    if expected.n_gen.isna().any():
        raise ValueError("generator census does not cover every selected event")
    gen = generators.drop(columns="shard", errors="ignore").merge(
        selected, on=list(KEY), how="inner", validate="many_to_one", sort=False)
    if gen.duplicated(list(IDENTITY)).any():
        raise ValueError("duplicate retained generator identity")
    counts = gen.groupby(list(KEY)).size().rename("observed")
    expected = expected.merge(counts, on=list(KEY), how="left", validate="one_to_one")
    if not np.array_equal(expected.n_gen.to_numpy(), expected.observed.fillna(0).to_numpy()):
        raise ValueError("retained generator rows disagree with the selected event census")
    if not np.isfinite(gen[list(STATE)].to_numpy()).all() or (gen.e < 0).any():
        raise ValueError("nonfinite state or negative generator total energy")
    p = np.sqrt(np.square(gen[["px", "py", "pz"]]).sum(axis=1))
    if (gen.e < p - 1e-9 * np.maximum(p, 1.)).any():
        raise ValueError("generator total energy is less than momentum magnitude")
    for field in ("pdg", "genid", "gen_product_id", "idx"):
        values = gen[field].to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("nonintegral generator identity or species: " + field)
    if (gen.idx < 0).any() or (gen.gen_product_id <= 0).any():
        raise ValueError("invalid retained generator product/index")
    gen = gen.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    report = dict(input_kind="retained", selected_events=len(selected), primary_particles=len(gen),
        primary_muons=int((gen.pdg.abs() == 13).sum()), nonmuon_primaries=int((gen.pdg.abs() != 13).sum()),
        empty_generator_events=int(expected.n_gen.eq(0).sum()),
        species={str(int(k)): int(v) for k, v in gen.pdg.value_counts().items()},
        scope="complete selected retained collection; not a complete original generated stream",
        coordinates="upstream Mu2e frame and global time; no fitted offsets")
    return gen, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "generators", "events", "outdir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split-at", type=int, default=22)
    args = parser.parse_args(argv)
    selection = ak.from_parquet(args.selection)
    selection = selection[selection.shard < args.split_at]
    census = pd.read_parquet(args.events)
    gen, summary = prepare(selection, pd.read_parquet(args.generators), census)
    events = pd.DataFrame({k: np.asarray(selection[k]) for k in KEY}).merge(
        census, on=list(KEY), validate="one_to_one")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(selection, out / "selection.parquet")
    gen.to_parquet(out / "primaries.parquet", index=False)
    events.to_parquet(out / "generator_events.parquet", index=False)
    (out / "inputs.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
