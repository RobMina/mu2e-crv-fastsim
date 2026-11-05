"""Named per-sector component oracles on identical saved steps; not stock-art or E2E certification."""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import genassoc, named_conditions, readout, runconfig, tables, tier1b_assets
from crvfastsim.response import lookup, photons, sipm, stockphotons, stocksipm
from crvfastsim.validation.v4photons import THRESH, cdf_shift, time_check


def compare(reference, candidate, universe, sector_of_channel, sector_names, active_channels):
    """Paired generator-event PE errors and cluster-robust time checks, including zero-response events."""
    result = {}
    n = len(universe)
    if n < 2:
        raise ValueError("V4 needs at least two generated events")
    for sector, name in enumerate(sector_names):
        arms = []
        for data in (reference, candidate):
            mask = sector_of_channel[data["channel"]] == sector
            event, weight = data["event"][mask], data["pe"][mask]
            arms.append((np.bincount(event, weights=weight, minlength=n), data["delay"][mask], event))
        (a, ta, ea), (b, tb, eb) = arms
        channels = int(np.sum(active_channels & (sector_of_channel == sector)))
        entry = dict(generated_events=n, active_channels=channels, reference_pe=float(a.sum()), candidate_pe=float(b.sum()))
        if channels == 0 or a.sum() <= 0 or b.sum() <= 0 or not len(ta) or not len(tb):
            result[name] = dict(**entry, status="unsupported", metrics_pass=False)
            continue
        ratio = b.sum() / a.sum()
        se = np.sqrt(n * np.var(b - ratio * a, ddof=1)) / a.sum()
        shift, error = cdf_shift(tb, ta, eb, ea)
        checks = dict(pe_per_sipm=bool(max(abs(ratio - 1) - 2 * se, 0) <= THRESH["pe_per_sipm_rel"]),
                      time_quantiles=time_check(shift, error))
        result[name] = dict(**entry, status="pass" if all(checks.values()) else "fail", metrics_pass=all(checks.values()),
                            mean_pe_reference=float(a.sum() / (n * channels)), mean_pe_candidate=float(b.sum() / (n * channels)),
                            pe_relative=float(ratio - 1), pe_relative_se=float(se),
                            time_shift_ns=shift.tolist(), time_shift_se_ns=error.tolist(), checks=checks)
    return result


def output_arrays(step, channel, time, weight, steps, universe, active):
    step, channel, time, weight = (np.asarray(value) for value in (step, channel, time, weight))
    if step.dtype.kind not in "iu" or channel.dtype.kind not in "iu":
        raise ValueError("component oracle requires integer step/channel indices")
    if (np.any(step < 0) or np.any(step >= len(steps)) or np.any(channel < 0) or np.any(channel >= len(active))
            or not np.isfinite(time).all() or not np.isfinite(weight).all() or np.any(weight < 0)):
        raise ValueError("invalid component oracle output")
    if not np.array_equal(channel // 4, steps.bar.to_numpy()[step]):
        raise ValueError("component oracle photon points to the wrong bar")
    ids = steps[tables.KEY].merge(universe.assign(_row=np.arange(len(universe))), on=tables.KEY,
                                  how="left", validate="many_to_one")
    if ids._row.isna().any():
        raise ValueError("steps outside full generator census")
    keep = active[channel]
    return dict(event=ids._row.to_numpy(np.int64)[step[keep]], channel=channel[keep], pe=weight[keep],
                delay=time[keep] - steps.t0.to_numpy()[step[keep]])


def identical_arrays(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def run(reference, generator, manifest, config, stage, seed, max_steps, out, no_stock=False,
        prepare_oracle=None, oracle_result=None):
    if no_stock and (prepare_oracle or oracle_result) or prepare_oracle and oracle_result:
        raise ValueError("oracle preparation/result and no-stock are mutually exclusive")
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    rc = runconfig.get(config)
    geom = rc.geometry()
    cond, evidence = named_conditions.load_conditions(manifest, rc, geom)
    cond = readout.apply(cond, geom, rc.counters_cfg())
    active = (readout.effective_status(cond) & 0b101) == 0
    import crvfastsim
    import mu2efastsim
    files = [reference, generator, manifest, __file__, rc.counters_path()]
    for group in ('inputs', 'assets'):
        files.extend(Path(manifest).resolve().parent / item['path'] for item in evidence[group].values())
    for package in (crvfastsim, mu2efastsim):
        files.extend(Path(package.__file__).parent.rglob('*.py'))
    files.extend(Path(lookup.lookup_dir()) / ("LookupTable_" + name) for name in sorted({sc.table for sc in rc.sectors.values()}))
    files.append(Path(lookup.lookup_dir()) / "photonMap.root")
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in files}
    donor, gen = genassoc.validate_donor(reference), genassoc.validate_generator_stream(generator)
    if gen.get("schema_version") != 2 or gen.get("stream_role") != "full generator stream":
        raise ValueError("V4 requires a strict full-generator stream")
    universe = genassoc.generator_event_keys(generator)
    steps = tables.read(reference, "steps")
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max-steps must be positive")
        steps = steps.iloc[:max_steps].reset_index(drop=True)
    if not len(steps):
        raise ValueError("empty V4 step input")
    parts = tables.read(reference, "particles", tables.KEY + ["key", "pdg"])
    pdg = steps.merge(parts.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], how="left", validate="many_to_one").pdg
    if pdg.isna().any():
        raise ValueError("missing step-particle association")
    cfg, scfg = rc.response(), sipm.SiPMConfig()
    s_loc, e_loc, beta, charge, known = photons.step_inputs(steps, pdg.to_numpy(), geom)
    if not known.all():
        raise ValueError("unsupported PDG in V4 steps")
    tabs = photons.load_tables(cfg)
    fast_ph = photons.make_photons(steps, pdg.to_numpy(), geom, cfg, tabs, cond.deviation, np.random.default_rng(seed))
    common = dict(stage=stage, seed=seed, response=dataclasses.asdict(cfg), sipm=dataclasses.asdict(scfg),
                  lookup_dir=lookup.lookup_dir(), sources_sha256=before)
    if stage == "photons":
        request = dict(s_loc=s_loc, e_loc=e_loc, t0=steps.t0.to_numpy(), t1=steps.t1.to_numpy(), beta=beta, charge=charge,
                       edep=steps.edep.to_numpy(), length=steps.len.to_numpy(), bar=steps.bar.to_numpy(),
                       sector_names=np.asarray(geom.sector_names)[geom.sector[steps.bar.to_numpy()]], deviation=cond.deviation)
    else:
        s, c, t = fast_ph
        keep = active[c]
        s, c, t = s[keep], c[keep], t[keep]
        event = steps[tables.KEY].merge(universe.assign(_row=np.arange(len(universe))), on=tables.KEY,
                                       how="left", validate="many_to_one")._row.to_numpy(np.int64)
        if (event < 0).any():
            raise ValueError("invalid oracle event index")
        request = dict(chan=event[s] * (geom.nbars * 4) + c, time=t, photon=s)
    if prepare_oracle:
        request_path = Path(prepare_oracle)
        if request_path.exists() or Path(str(request_path) + ".json").exists():
            raise FileExistsError(request_path)
        with request_path.open("xb") as stream:
            np.savez_compressed(stream, **request)
        common["request_sha256"] = tier1b_assets.sha256(request_path)
        Path(str(request_path) + ".json").write_text(json.dumps(common, indent=2) + "\n")
        print(f"Prepared {stage} oracle request {request_path}", flush=True)
        return
    external = None
    if oracle_result:
        result_path = Path(oracle_result)
        receipt = json.loads(Path(str(result_path) + ".json").read_text())
        normalized = json.loads(json.dumps(common))
        if receipt["output_sha256"] != tier1b_assets.sha256(result_path) or any(
                receipt["request"][k] != v for k, v in normalized.items() if k != "sources_sha256"):
            raise ValueError("oracle result/input identity mismatch")
        for path, digest in receipt["request"]["sources_sha256"].items():
            if tier1b_assets.sha256(path) != digest:
                raise ValueError("oracle original input/source drift")
        before.update(receipt["request"]["sources_sha256"])
        request_path = Path(receipt["request_path"])
        if tier1b_assets.sha256(request_path) != receipt["request"]["request_sha256"]:
            raise ValueError("oracle request changed")
        with np.load(request_path, allow_pickle=False) as frozen:
            different = [k for k, value in request.items() if k not in frozen or not identical_arrays(frozen[k], value)]
            if set(frozen.files) != set(request) or different:
                raise ValueError(f"oracle numerical request differs: {different}")
        with np.load(result_path, allow_pickle=False) as frozen:
            external = dict(frozen)
        before[str(result_path.resolve())] = tier1b_assets.sha256(result_path)
        before[str(Path(str(result_path) + ".json").resolve())] = tier1b_assets.sha256(str(result_path) + ".json")
    if stage == "photons":
        if external is not None:
            stock_ph = tuple(external[k] for k in ("step", "channel", "time"))
        elif no_stock:
            stock_ph = photons.make_photons(steps, pdg.to_numpy(), geom, cfg, tabs, cond.deviation, np.random.default_rng(seed + 1))
        else:
            oracle = stockphotons.StockPhotons(cfg, lookup.lookup_dir(), seed=seed)
            stock_ph = oracle.run(s_loc, e_loc, steps.t0.to_numpy(), steps.t1.to_numpy(), beta, charge,
                                   steps.edep.to_numpy(), steps.len.to_numpy(), steps.bar.to_numpy(),
                                   np.asarray(geom.sector_names)[geom.sector[steps.bar.to_numpy()]], cond.deviation)
        candidate = output_arrays(*fast_ph, np.ones(len(fast_ph[0])), steps, universe, active)
        reference_out = output_arrays(*stock_ph, np.ones(len(stock_ph[0])), steps, universe, active)
    elif stage == "sipm":
        s, c, t = fast_ph
        keep = active[c]
        s, c, t = s[keep], c[keep], t[keep]
        event = steps[tables.KEY].merge(universe.assign(_row=np.arange(len(universe))), on=tables.KEY,
                                       how="left", validate="many_to_one")._row.to_numpy(np.int64)
        labels = event[s] * (geom.nbars * 4) + c
        pmap = sipm.photon_map()
        f = sipm.simulate(labels, t, s, pmap, scfg, np.random.default_rng(seed + 7))
        r = external if external is not None else (
             sipm.simulate(labels, t, s, pmap, scfg, np.random.default_rng(seed + 8)) if no_stock else
             stocksipm.run(labels, t, s, scfg, stocksipm.photon_map_path(), seed=seed))
        def converted(data):
            return output_arrays(data["photon"], data["chan"] % (geom.nbars * 4), data["time"], data["pe"], steps, universe, active)
        candidate, reference_out = converted(f), converted(r)
    else:
        raise ValueError("stage must be photons or sipm")
    groups = compare(reference_out, candidate, universe, np.repeat(geom.sector, 4), geom.sector_names, active)
    result = dict(scope="named_component_oracle_not_stock_art_or_E2E_certification", stage=stage, seed=seed,
                  run_config=dataclasses.asdict(rc), conditions=evidence, generated=len(universe), steps=len(steps),
                  reference_arm="numpy_independent_seed_null" if no_stock else "Offline_per_sector_class",
                  thresholds=THRESH, groups=groups, metrics_pass=all(g["metrics_pass"] for g in groups.values()),
                  physics_admitted=False, sources_sha256=before, donor=donor, generator=gen,
                  limitations=["Per-sector MakeCrvPhotons objects, not historical art shared-lookup yield ownership.",
                               "Physical/status mask applied equally; SiPM uses fixed absolute 0..100000 ns, not event pbtime.",
                               "Optional step truncation is conditional; full generator census still supplies zeros.",
                               "No finite-library, stock-art execution or end-to-end certification."])
    if {path: tier1b_assets.sha256(path) for path in before} != before:
        raise ValueError("V4 input/implementation drift")
    named_conditions.load_conditions(manifest, rc, geom)
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(stage=stage, steps=len(steps), generated=len(universe), groups=groups)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--conditions-manifest", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--stage", choices=("photons", "sipm"), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-stock", action="store_true")
    parser.add_argument("--prepare-oracle", help="new numerical NPZ request for a separate pinned Offline runtime")
    parser.add_argument("--oracle-result", help="hash-bound saved stock-class NPZ response")
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    run(a.reference, a.generator, a.conditions_manifest, a.run_config, a.stage, a.seed, a.max_steps, a.out,
        a.no_stock, a.prepare_oracle, a.oracle_result)


if __name__ == "__main__":
    main()
