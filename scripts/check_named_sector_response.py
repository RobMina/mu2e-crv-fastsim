"""Saved-only per-sector pulse attribution on identical stock events, not full V-R4 or native physics admission."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import geometry, runconfig, tables, tier1b_assets
from crvfastsim.validation import reco_detail, v2compare
from scripts.check_native_response import canonical_keys


def pulse_summary(pulses, universe, geom):
    bar = pulses.bar.to_numpy()
    if bar.dtype.kind not in 'iu' or np.any((bar < 0) | (bar >= geom.nbars)) or not np.isfinite(pulses.pe).all():
        raise ValueError('invalid named pulse bar or PE')
    if len(pulses.merge(universe[tables.KEY], on=tables.KEY, how='inner', validate='many_to_one')) != len(pulses):
        raise ValueError('pulse outside saved event census')
    result = {}
    for sector, name in enumerate(geom.sector_names):
        selected = pulses[(geom.sector[bar] == sector) & (pulses.seq == 0)]
        energy = selected.groupby(tables.KEY).pe.sum().rename('_pe').reset_index()
        event_pe = universe[tables.KEY].merge(energy, on=tables.KEY, how='left', validate='one_to_one')._pe.fillna(0).to_numpy()
        result[name] = dict(pulses=int(len(selected)), spectrum=np.histogram(selected.pe, v2compare.PE_EDGES)[0],
                            underflow=int((selected.pe < v2compare.PE_EDGES[0]).sum()),
                            overflow=int((selected.pe > v2compare.PE_EDGES[-1]).sum()), event_pe=event_pe)
    return result


def compare_sector(reference, candidate):
    a, b = np.asarray(reference['event_pe'], dtype=np.float64), np.asarray(candidate['event_pe'], dtype=np.float64)
    if a.shape != b.shape or len(a) < 2:
        raise ValueError('paired event PE needs identical supported census')
    ratio = float(b.sum() / a.sum()) if a.sum() else None
    error = float(np.std(b - ratio * a, ddof=1) / (np.sqrt(len(a)) * a.mean())) if ratio is not None else None
    mr, er = v2compare.hist_median(reference['spectrum'], v2compare.PE_EDGES)
    mf, ef = v2compare.hist_median(candidate['spectrum'], v2compare.PE_EDGES)
    median = float(mf / mr - 1) if np.isfinite([mr, mf]).all() and mr else None
    median_error = float(abs(mf / mr) * np.hypot(ef / mf, er / mr)) if median is not None and mf else None
    band = v2compare.ratio_band(candidate['spectrum'], reference['spectrum'], v2compare.PE_EDGES, *v2compare.SPEC_RANGE)
    deviation = band['max_abs_dev_minus_2sigma']
    band['nonfinite_statistics'] = [key for key, value in band.items() if isinstance(value, float) and not np.isfinite(value)]
    for key in band['nonfinite_statistics']:
        band[key] = None
    return dict(reference_pulses=reference['pulses'], candidate_pulses=candidate['pulses'],
                event_pe_ratio=ratio, event_pe_ratio_se=error,
                pulse_pe_median_relative=median, pulse_pe_median_relative_independent_hist_se=median_error,
                pulse_pe_median_reference=float(mr) if np.isfinite(mr) else None,
                pulse_pe_median_candidate=float(mf) if np.isfinite(mf) else None,
                spectrum=band, diagnostic_rules=dict(
                    pe_median_2percent=bool(max(abs(median) - 2 * median_error, 0) <= .02)
                    if median is not None and median_error is not None else None,
                    spectrum_3percent=bool(deviation <= .03) if deviation is not None else None),
                reference_flows=dict(underflow=reference['underflow'], overflow=reference['overflow']),
                candidate_flows=dict(underflow=candidate['underflow'], overflow=candidate['overflow']))


def run(config, out):
    out, config = Path(out), Path(config).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    digest = tier1b_assets.sha256(config)
    spec = json.loads(config.read_text())
    rc = runconfig.get(spec['run_config'])
    paths = [config, __file__, canonical_keys.__code__.co_filename, reco_detail.__file__, v2compare.__file__,
             runconfig.__file__, geometry.__file__, geometry.find_counters(rc.counters)]
    paths += spec['reference'] + [path for values in spec['candidates'].values() for path in values]
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in paths}
    before[str(config)] = digest
    geom = rc.geometry()
    def read(paths):
        frames = [reco_detail._read(path)[0] for path in paths]
        events = pd.concat([frame['events'][tables.KEY] for frame in frames], ignore_index=True)
        canonical_keys(events)
        events = events.sort_values(tables.KEY).reset_index(drop=True)
        pulses = pd.concat([frame['pulses'] for frame in frames], ignore_index=True)
        return events, pulse_summary(pulses, events, geom)
    universe, reference = read(spec['reference'])
    results = {}
    for label, paths in spec['candidates'].items():
        events, candidate = read(paths)
        if not np.array_equal(canonical_keys(events), canonical_keys(universe)):
            raise ValueError('candidate event population differs from saved stock reference')
        results[label] = {sector: compare_sector(reference[sector], candidate[sector]) for sector in reference}
    if any(tier1b_assets.sha256(path) != value for path, value in before.items()):
        raise ValueError('saved sector response input drift')
    result = dict(scope='conditional_saved_event_per_sector_pulse_attribution_not_full_VR4', events=len(universe),
                  run_config=rc.name, results=results, sources_sha256=before, physics_admitted=False,
                  limitations=['Exact accepted-event conditioning; not inclusive transport, library or C4 closure.',
                               'Historical diagnostic arms are explicit controls, not defaults or corrected physics.',
                               'Per-event PE ratio SE excludes finite-library and model uncertainty.',
                               'Histogram median and spectrum errors use independent-pulse approximations, not event-clustered gates.',
                               'Three-percent spectrum rule is diagnostic; layer efficiency, objective and timing remain untested.',
                               'Shared sector-type clusters cannot be assigned uniquely to T1 or T2 by sector type alone.'])
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with out.open('x') as stream:
        stream.write(payload)
    print(json.dumps(result['results'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.config, args.out)
