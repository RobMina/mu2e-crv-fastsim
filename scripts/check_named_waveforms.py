"""Named saved-charge waveform control with exact full-event hardware grouping, not full V-R1 admission."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from crvfastsim import named_conditions, runconfig, tables, tier1b_assets
from crvfastsim.reco import compare, waveform
from crvfastsim.validation import event_shapes


def phases(events, channels, tdc0, targets, target_channels, cond):
    """Match FEB phase by full triples plus ROC/FEB, excluding channels with no hardware mapping."""
    src = events[tables.KEY].copy()
    src['roc'], src['feb'], src['phase'] = cond.roc[channels], cond.feb[channels], tdc0
    mapped = (src.roc >= 0) & (src.feb >= 0)
    src = src[mapped]
    keys = tables.KEY + ['roc', 'feb']
    spread = src.groupby(keys).phase.agg(['min', 'max']).reset_index()
    if (spread['max'] - spread['min'] > 1e-8).any():
        raise ValueError('stock FEB phases disagree within exact event/ROC/FEB')
    target = targets[tables.KEY].copy()
    target['roc'], target['feb'] = cond.roc[target_channels], cond.feb[target_channels]
    result = target.merge(spread[keys + ['min']], on=keys, how='left', validate='many_to_one')['min'].to_numpy(copy=True)
    result[(target.roc < 0) | (target.feb < 0)] = np.nan
    return result


def run(reference, manifest, config, out, noise=False, seed=1):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    rc = runconfig.get(config)
    geom = rc.geometry()
    cond, evidence = named_conditions.load_conditions(manifest, rc, geom)
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in (reference, manifest, __file__, waveform.__file__)}
    with uproot.open(reference) as root:
        ev, evmeta = event_shapes._highest(root, 'crvdump/events')
        q, qmeta = event_shapes._highest(root, 'crvdump/charges')
        m, mmeta = event_shapes._highest(root, 'crvdump/digimc')
        events = pd.DataFrame(ev.arrays(tables.KEY, library='np'))
        charges = q.arrays(tables.KEY + ['bar', 'sipm', 'time', 'charge'], library='ak')
        mc = m.arrays(tables.KEY + ['bar', 'sipm', 'start', 'tdc0', 'volt'], library='ak')
    if events.duplicated(tables.KEY).any():
        raise ValueError('duplicate event census')
    def frame(a):
        return pd.DataFrame({key: ak.to_numpy(a[key]) for key in tables.KEY})
    def indices(a):
        x = frame(a).merge(events.assign(_event=np.arange(len(events))), on=tables.KEY, how='left', validate='many_to_one')
        if x._event.isna().any():
            raise ValueError('response row outside full event census')
        return x._event.to_numpy(np.int64)
    qe, me = indices(charges), indices(mc)
    qc = ak.to_numpy(charges.bar).astype(np.int64) * 4 + ak.to_numpy(charges.sipm)
    mcchan = ak.to_numpy(mc.bar).astype(np.int64) * 4 + ak.to_numpy(mc.sipm)
    phase = phases(frame(mc), mcchan, ak.to_numpy(mc.tdc0), frame(charges), qc, cond)
    nq = ak.to_numpy(ak.num(charges.time))
    rows = np.repeat(np.arange(len(nq)), nq)
    times, values = ak.to_numpy(ak.flatten(charges.time)), ak.to_numpy(ak.flatten(charges.charge))
    cfg = waveform.WaveformConfig()
    output = waveform.synthesize(rows, times, values, np.nan_to_num(phase), cfg,
                                 rng=np.random.default_rng(seed) if noise else None)
    fk = np.column_stack([qe[output['chan']], qc[output['chan']], output['starttdc']]).astype(np.int64)
    rv = ak.to_numpy(ak.fill_none(ak.pad_none(mc.volt, cfg.samples_zs, clip=True), 0.))
    radc = np.clip((rv * cfg.adc_conversion + cfg.pedestal + .5).astype(np.int16), cfg.min_adc, cfg.max_adc)
    rstart = np.rint((ak.to_numpy(mc.start) - ak.to_numpy(mc.tdc0)) / 12.5).astype(np.int64)
    rk = np.column_stack([me, mcchan, rstart])
    finding = compare.vr1(dict(key=fk, adc=output['adc']), dict(key=rk, adc=radc))
    ia, ib = compare._join(fk, rk)
    known = np.isfinite(phase[output['chan'][ia]])
    finding['matched_known_phase'] = int(known.sum())
    finding['known_phase_sample_identical'] = bool(np.array_equal(output['adc'][ia[known]], radc[ib[known]])) if known.any() else None
    missing_ref = np.setdiff1d(np.arange(len(rk)), ib)
    extra_fast = np.setdiff1d(np.arange(len(fk)), ia)
    extra_known = np.isfinite(phase[output['chan'][extra_fast]])
    finding['missing_stock_digis'] = int(len(missing_ref))
    finding['extra_fast_known_phase'] = int(extra_known.sum())
    finding['extra_fast_unknown_phase'] = int((~extra_known).sum())
    finding['conditional_noise_free_pass'] = bool(not noise and not len(missing_ref) and not extra_known.any()
                                                 and known.any() and finding['known_phase_sample_identical'])
    finding['unknown_phase_charge_rows'] = int(np.isnan(phase).sum())
    result = dict(scope='conditional_named_saved_charges_not_full_VR1_or_E2E', events=len(events), noise=noise, seed=seed,
                  conditions=evidence, finding=finding, cycles=dict(events=evmeta, charges=qmeta, digimc=mmeta),
                  sources_sha256=before, physics_admitted=False,
                  limitations=['Noise-on samples are not expected to be sample-identical; full per-channel noise test pending.',
                               'No reference phase on an event/ROC/FEB uses zero only for the disclosed diagnostic.',
                               'Fixed offspill synthesis, not event-relative pbtime timing certification.'])
    if any(tier1b_assets.sha256(path) != digest for path, digest in before.items()):
        raise ValueError('waveform inputs changed')
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(events=len(events), finding=finding)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--conditions-manifest', required=True)
    parser.add_argument('--run-config', required=True)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', required=True)
    a = parser.parse_args()
    run(a.reference, a.conditions_manifest, a.run_config, a.out, a.noise, a.seed)
