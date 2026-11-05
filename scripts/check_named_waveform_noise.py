"""Saved-charge named waveform noise diagnostics with full event/ROC/FEB identity, not full V-R1 admission."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from crvfastsim import geometry, named_conditions, runconfig, tables, tier1b_assets
from crvfastsim.reco import compare, waveform
from crvfastsim.validation import event_shapes
from scripts.check_named_waveforms import phases
from scripts.check_native_response import canonical_keys
from mu2efastsim import match


MIN_DIGIS = 200
KS_P = 0.01


def unique_keys(keys):
    if len(keys) != len(np.unique(keys, axis=0)):
        raise ValueError('duplicate waveform identity')


def clock_ticks(start, phase):
    offset = (np.asarray(start) - np.asarray(phase)) / 12.5
    if not np.isfinite(offset).all() or np.any(np.abs(offset - np.rint(offset)) > 1e-8):
        raise ValueError('stock waveform clock is not on FEB sample grid')
    return np.rint(offset).astype(np.int64)


def read_reference(path, cond, cfg):
    """Read the highest cycles and map every response row into the exact saved event census."""
    with uproot.open(path) as root:
        trees = {name: event_shapes._highest(root, 'crvdump/' + name) for name in ('events', 'charges', 'digimc')}
        events = pd.DataFrame(trees['events'][0].arrays(tables.KEY + ['ndigi'], library='np'))
        q = trees['charges'][0].arrays(tables.KEY + ['bar', 'sipm', 'time', 'charge'], library='ak')
        m = trees['digimc'][0].arrays(tables.KEY + ['bar', 'sipm', 'start', 'tdc0', 'volt'], library='ak')
    canonical_keys(events)
    def frame(a):
        return pd.DataFrame({key: ak.to_numpy(a[key]) for key in tables.KEY})
    def channel(a):
        bar, sipm = ak.to_numpy(a.bar), ak.to_numpy(a.sipm)
        if bar.dtype.kind not in 'iu' or sipm.dtype.kind not in 'iu' or np.any(bar < 0) or np.any((sipm < 0) | (sipm > 3)):
            raise ValueError('invalid waveform channel')
        ch = bar.astype(np.int64) * 4 + sipm
        if np.any(ch >= len(cond.roc)):
            raise ValueError('waveform channel outside geometry')
        return ch
    def event_index(a):
        joined = frame(a).merge(events[tables.KEY].assign(_row=np.arange(len(events))), on=tables.KEY,
                                how='left', validate='many_to_one')
        if joined._row.isna().any():
            raise ValueError('waveform row outside event census')
        return joined._row.to_numpy(np.int64)
    qc, mc = channel(q), channel(m)
    qe, me = event_index(q), event_index(m)
    if not np.array_equal(np.bincount(me, minlength=len(events)), events.ndigi.to_numpy()):
        raise ValueError('stock digi count differs from full event census')
    qkeys = np.column_stack([frame(q).to_numpy(np.int64), qc])
    unique_keys(qkeys)
    phase = phases(frame(m), mc, ak.to_numpy(m.tdc0), frame(q), qc, cond)
    n = ak.to_numpy(ak.num(q.time))
    if not np.array_equal(n, ak.to_numpy(ak.num(q.charge))):
        raise ValueError('charge time/value lengths differ')
    times, charges = ak.to_numpy(ak.flatten(q.time)), ak.to_numpy(ak.flatten(q.charge))
    if np.any(ak.to_numpy(ak.num(m.volt)) != cfg.samples_zs):
        raise ValueError('stock waveform sample count differs from zero-suppressed configuration')
    volt = ak.to_numpy(m.volt)
    if not np.isfinite(times).all() or not np.isfinite(charges).all() or not np.isfinite(volt).all():
        raise ValueError('nonfinite waveform input')
    adc = np.clip((volt * cfg.adc_conversion + cfg.pedestal + .5).astype(np.int16), cfg.min_adc, cfg.max_adc)
    if not np.isfinite(ak.to_numpy(m.start)).all() or not np.isfinite(ak.to_numpy(m.tdc0)).all():
        raise ValueError('nonfinite waveform clock')
    start = clock_ticks(ak.to_numpy(m.start), ak.to_numpy(m.tdc0))
    keys = np.column_stack([me, mc, start])
    unique_keys(keys)
    return dict(events=events, qe=qe, qc=qc, qkeys=qkeys, phase=phase, n=n, time=times, charge=charges,
                key=keys, adc=adc, channel=mc, mapped=(cond.roc[mc] >= 0) & (cond.feb[mc] >= 0),
                cycles={name: metadata for name, (_, metadata) in trees.items()})


def residual_summary(fast_key, fast_adc, ref_key, ref_adc, base_key, base_adc, cfg):
    """Residuals use only unique digis shared by all three arms and nonsaturated baseline samples."""
    for keys in (fast_key, ref_key, base_key):
        unique_keys(keys)
    ir, rr = compare._join(base_key, ref_key) if len(base_key) and len(ref_key) else ([], [])
    iff, ff = compare._join(base_key, fast_key) if len(base_key) and len(fast_key) else ([], [])
    r, f = np.full(len(base_key), -1), np.full(len(base_key), -1)
    r[ir], f[iff] = rr, ff
    both = np.flatnonzero((r >= 0) & (f >= 0))
    base = base_adc[both].astype(np.int64)
    a, b = fast_adc[f[both]].astype(np.int64) - base, ref_adc[r[both]].astype(np.int64) - base
    ok = (base > cfg.min_adc + 5) & (base < cfg.max_adc - 5)
    if not ok.any():
        return dict(status='unsupported_no_common_samples', digis_in_all_three=len(both), samples=0, ks_nonrejection=None)
    x, y = a[ok].astype(float), b[ok].astype(float)
    d, p = compare.ks_2samp(x, y)
    pair = ok[:, :-1] & ok[:, 1:]
    def corr(values):
        left, right = values[:, :-1][pair], values[:, 1:][pair]
        return float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 and left.std() and right.std() else None
    return dict(status='diagnostic', digis_in_all_three=int(len(both)), samples=int(ok.sum()),
                rms_adc=dict(fast=float(x.std()), ref=float(y.std())), mean_adc=dict(fast=float(x.mean()), ref=float(y.mean())),
                expected_rms_adc=float(np.sqrt((cfg.noise * cfg.adc_conversion) ** 2 + 1 / 12)),
                ks=dict(d=d, p=p), neighbour_corr=dict(fast=corr(a), ref=corr(b)),
                ks_nonrejection=bool(p > KS_P), physics_pass=None,
                decision='descriptive_only_no_independent_support_or_noise_agreement_gate')


def channel_summary(fch, fadc, rch, radc, nchannels):
    """Retain the declared per-channel threshold; an empty eligible population is unsupported, never a pass."""
    nf, nr = np.bincount(fch, minlength=nchannels), np.bincount(rch, minlength=nchannels)
    chans = np.flatnonzero((nf >= MIN_DIGIS) & (nr >= MIN_DIGIS))
    of, orr = np.argsort(fch, kind='stable'), np.argsort(rch, kind='stable')
    sf, sr = np.r_[0, np.cumsum(nf)], np.r_[0, np.cumsum(nr)]
    rows = []
    for channel in chans:
        d, p = compare.ks_2samp(fadc[of[sf[channel]:sf[channel + 1]]].ravel(),
                               radc[orr[sr[channel]:sr[channel + 1]]].ravel())
        rows.append(dict(channel=int(channel), fast=int(nf[channel]), ref=int(nr[channel]), d=d, p=p))
    failed = sum(row['p'] <= KS_P for row in rows)
    return dict(status='diagnostic' if len(chans) else 'unsupported_no_eligible_channels',
                min_digis_per_channel=MIN_DIGIS, threshold_p=KS_P, channels_tested=int(len(chans)),
                total_channels=nchannels, channels_below_threshold=failed,
                min_p=min((row['p'] for row in rows), default=None),
                all_channels_pass=bool(not failed) if len(chans) else None,
                digi_counts=dict(fast=int(len(fch)), ref=int(len(rch))), per_channel=rows)


def require_same_charge_population(reference, baseline):
    for key in ('events',):
        if not np.array_equal(canonical_keys(reference[key]), canonical_keys(baseline[key])):
            raise ValueError('noise-free event population differs')
    if not np.array_equal(reference['events'][tables.KEY].to_numpy(), baseline['events'][tables.KEY].to_numpy()):
        raise ValueError('noise-free event order differs')
    for key in ('qkeys', 'n', 'time', 'charge'):
        a, b = reference[key], baseline[key]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise ValueError('noise-free saved charges differ')
    known = np.isfinite(reference['phase']) & np.isfinite(baseline['phase'])
    if np.any(np.abs(reference['phase'][known] - baseline['phase'][known]) > 1e-8):
        raise ValueError('noise-free FEB phases differ')


def validate_nonoise(base, cfg):
    rows = np.repeat(np.arange(len(base['n'])), base['n'])
    fast = waveform.synthesize(rows, base['time'], base['charge'], np.nan_to_num(base['phase']), cfg, rng=None)
    known = np.isfinite(base['phase'][fast['chan']])
    keys = np.column_stack([base['qe'][fast['chan']], base['qc'][fast['chan']], fast['starttdc']]).astype(np.int64)
    stock_key, stock_adc = base['key'][base['mapped']], base['adc'][base['mapped']]
    a, b = compare._join(keys[known], stock_key) if known.any() and len(stock_key) else ([], [])
    if not len(a) or len(a) != len(stock_key) or len(a) != int(known.sum()) or not np.array_equal(fast['adc'][known][a], stock_adc[b]):
        raise ValueError('noise-free baseline does not match deterministic saved-charge synthesis')
    return dict(known_phase_digis=int(len(a)), sample_identical=True, scope='conditional_noiseless_baseline')


def run(config, manifest, run_config, out, seed=1):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    config_hash = tier1b_assets.sha256(config)
    spec = json.loads(Path(config).read_text())
    if tier1b_assets.sha256(config) != config_hash:
        raise ValueError('waveform config changed during read')
    pairs = spec['pairs']
    if not pairs:
        raise ValueError('need saved waveform references')
    rc, cfg = runconfig.get(run_config), waveform.WaveformConfig()
    paths = [config, manifest, __file__, phases.__code__.co_filename, canonical_keys.__code__.co_filename,
             waveform.__file__, compare.__file__, match.__file__, named_conditions.__file__,
             runconfig.__file__, geometry.__file__, event_shapes.__file__, tables.__file__, tier1b_assets.__file__,
             geometry.find_counters(rc.counters), geometry.find_counters(cfg.single_pe_file)]
    for pair in pairs:
        paths += [pair['reference']] + ([pair['nonoise_reference']] if pair.get('nonoise_reference') else [])
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in paths}
    before[str(Path(config).resolve())] = config_hash
    cond, evidence = named_conditions.load_conditions(manifest, rc, rc.geometry())
    for group in ('inputs', 'assets'):
        for item in evidence[group].values():
            path = str((Path(manifest).resolve().parent / item['path']).resolve())
            before[path] = item['sha256']
    rng = np.random.default_rng(seed)
    censuses, fch, fadc, rch, radc, controls = [], [], [], [], [], []
    for pair in pairs:
        ref = read_reference(pair['reference'], cond, cfg)
        censuses.append(ref['events'][tables.KEY])
        canonical_keys(pd.concat(censuses, ignore_index=True))
        rows = np.repeat(np.arange(len(ref['n'])), ref['n'])
        fast = waveform.synthesize(rows, ref['time'], ref['charge'], np.nan_to_num(ref['phase']), cfg, rng=rng)
        keys = np.column_stack([ref['qe'][fast['chan']], ref['qc'][fast['chan']], fast['starttdc']]).astype(np.int64)
        unique_keys(keys)
        known = np.isfinite(ref['phase'][fast['chan']])
        fch.append(ref['qc'][fast['chan'][known]])
        fadc.append(fast['adc'][known])
        mapped = ref['mapped']
        rch.append(ref['channel'][mapped])
        radc.append(ref['adc'][mapped])
        control = dict(reference=pair['reference'], events=len(ref['events']), cycles=ref['cycles'],
                       unknown_phase_charge_rows=int(np.isnan(ref['phase']).sum()),
                       excluded_unknown_phase_fast_digis=int((~known).sum()),
                       excluded_unmapped_stock_digis=int((~mapped).sum()))
        if pair.get('nonoise_reference'):
            if before[str(Path(pair['reference']).resolve())] == before[str(Path(pair['nonoise_reference']).resolve())]:
                raise ValueError('noise-free reference aliases noisy input')
            base = read_reference(pair['nonoise_reference'], cond, cfg)
            require_same_charge_population(ref, base)
            control['nonoise_identity'] = validate_nonoise(base, cfg)
            control['noise_residual'] = residual_summary(keys[known], fast['adc'][known], ref['key'][mapped], ref['adc'][mapped],
                                                       base['key'][base['mapped']], base['adc'][base['mapped']], cfg)
        controls.append(control)
    summary = channel_summary(np.concatenate(fch), np.concatenate(fadc), np.concatenate(rch), np.concatenate(radc), len(cond.roc))
    if any(tier1b_assets.sha256(path) != digest for path, digest in before.items()):
        raise ValueError('waveform noise input drift')
    result = dict(scope='conditional_named_saved_charges_noise_not_full_VR1_or_E2E', events=sum(map(len, censuses)),
                  seed=seed, configuration=run_config, conditions=evidence, per_channel_adc=summary,
                  controls=controls, sources_sha256=before, physics_admitted=False,
                  limitations=['Saved accepted-event population; no generated-exposure or native transport admission.',
                               'Per-channel ADC KS is signal dominated; adjacent ADC samples are not independent trials.',
                               'Noise residuals condition on digis shared by all three arms and exclude saturation.',
                               'Unknown FEB phases are disclosed, not inferred from unrelated events or ROCs.',
                               'Fixed absolute offspill window; stock event-relative timing is not certified.',
                               'No full V-R1 or GPU/art certification from this host saved-charge diagnostic.'])
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(events=result['events'], channels=summary['channels_tested'], min_p=summary['min_p'],
                          below_threshold=summary['channels_below_threshold'], controls=controls)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--conditions-manifest', required=True)
    parser.add_argument('--run-config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=1)
    a = parser.parse_args()
    run(a.config, a.conditions_manifest, a.run_config, a.out, a.seed)
