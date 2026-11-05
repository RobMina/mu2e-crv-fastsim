"""Truth-primary-conditioned native secondary diagnostic on disjoint reference shards, not E2E closure."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import genassoc, runconfig, tables, tier1b, tier1b_assets
from crvfastsim.analyses.census import classify


def moments(reference, candidate, universe, geom):
    """Paired full-generator-event energy/count moments per actual native sector and origin."""
    result = {}
    for sector, name in enumerate(geom.sector_names):
        for origin in ("mu_primary", "mu_descendant", "nonmu_primary", "nonmu_descendant"):
            frames = []
            for data in (reference, candidate):
                chosen = data[(geom.sector[data.bar.to_numpy()] == sector) & (data.cls == origin)]
                sums = chosen.groupby(tables.KEY).edep.agg(energy="sum", steps="size").reset_index()
                frames.append(universe.merge(sums, on=tables.KEY, how="left", validate="one_to_one").fillna(0))
            a, b = frames
            group = {}
            for key in ("energy", "steps"):
                x, y = a[key].to_numpy(float), b[key].to_numpy(float)
                delta = y - x
                se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None
                group[key] = dict(reference_sum=float(x.sum()), candidate_sum=float(y.sum()),
                                  reference_per_generated=float(x.mean()), candidate_per_generated=float(y.mean()),
                                  paired_delta_per_generated=float(delta.mean()), paired_se=se)
            result[name + "/" + origin] = group
    return result


def run(directory, reference, generator, config, seed, out):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    rc = runconfig.get(config)
    geom = rc.geometry()
    spec = tier1b_assets.native_bundle(directory, rc)
    lib = tier1b_assets.load(spec, geom)
    before = {str(Path(p).resolve()): tier1b_assets.sha256(p) for p in (reference, generator)}
    donor_report = genassoc.validate_donor(reference)
    gen_report = genassoc.validate_generator_stream(generator)
    if gen_report.get("schema_version") != 2 or gen_report.get("stream_role") != "full generator stream":
        raise ValueError("need checked full-generator census")
    universe = genassoc.generator_event_keys(generator)
    train = pd.DataFrame(spec.native["training_shards"])
    if len(universe.merge(train, on=["run", "subrun"], how="inner", validate="many_to_one")):
        raise ValueError("validation shards overlap native training shards")
    steps, parts, gen = tables.read(reference, "steps"), tables.read(reference, "particles"), tables.read(generator, "gen")
    for frame in (steps, parts):
        if len(frame.merge(universe, on=tables.KEY, how="inner")) != len(frame):
            raise ValueError("reference rows outside full generator census")
    primary = parts[(parts.isprim != 0) & (parts.pdg.abs() == 13)]
    selected = steps.merge(primary[tables.KEY + ["key"]].rename(columns={"key": "sp"}),
                           on=tables.KEY + ["sp"], how="inner", validate="many_to_one")
    st, pa, info = tier1b.sample(selected, primary, gen, lib, geom, np.random.default_rng(seed), **spec.options)
    groups = getattr(geom, "lib_group", None)
    a, b = classify(steps, parts, geom.sector, groups), classify(st, pa, geom.sector, groups)
    result = {"scope": "truth_primary_conditioned_secondary_diagnostic_not_transport_or_E2E_validation",
              "run_config": config, "seed": seed, "generated": len(universe), "tier1b": spec.describe(),
              "reference": donor_report, "generator": gen_report, "sample_counts": {k: info[k] for k in ("desc_steps", "desc_dropped", "desc_clamped", "showers", "shower_steps", "nonmu_primaries")},
              "paired": moments(a, b, universe, geom), "sources_sha256": before,
              "limitations": ["Uses true Geant4 primary-muon steps, not a measured response correction model.",
                              "Finite-library and systematic uncertainties are not estimated; no acceptance gate.",
                              "Primary-conditioned diagnostics do not establish independent end-to-end closure."]}
    if {str(Path(p).resolve()): tier1b_assets.sha256(p) for p in (reference, generator)} != before:
        raise ValueError("reference changed during diagnostic")
    tier1b_assets._check_native(spec, geom)
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"generated={len(universe)} candidate_steps={len(st)} output={out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.directory, args.reference, args.generator, args.run_config, args.seed, args.out)


if __name__ == "__main__":
    main()
