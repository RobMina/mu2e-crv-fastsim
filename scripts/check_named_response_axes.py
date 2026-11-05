"""Saved-pulse response/finder factorial diagnostic; no response redraw, stock producer or physics admission."""
import argparse
import dataclasses
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from crvfastsim import geometry, readout, response_receipt, runconfig, tables, tier1b_assets
from crvfastsim.named_conditions import load_conditions
from crvfastsim.reco import coinc, coinc_jax
from crvfastsim.response import shared_yield
from crvfastsim.validation import event_shapes, reco_detail
from scripts.check_named_coinc import compare
from scripts.check_named_sector_response import compare_sector, pulse_summary
from scripts.check_native_response import canonical_keys
from scripts.compare_native_histograms import EDGES, histogram


def normalized(value):
    return json.loads(json.dumps(value))


def require_metadata(metadata, rc, evidence, response, attachment):
    expected = dataclasses.replace(rc.chain_config(), window=runconfig.OFFSPILL_WINDOW,
                                   response=response, finder=dataclasses.replace(
                                       rc.finder(), add_opposite_side_pulses=attachment))
    if (metadata.get('schema') != 'crv-fasttables-named-v1' or metadata.get('run_config') != rc.name
            or metadata.get('stage') != 'chain' or type(metadata.get('seed')) is not int
            or metadata.get('seed') != 1 or metadata.get('jax') is not False or metadata.get('noise') is not True
            or metadata.get('conditions') != normalized(evidence)
            or metadata.get('chain_config') != normalized(dataclasses.asdict(expected))
            or metadata.get('finder') != normalized(dataclasses.asdict(expected.finder))
            or metadata.get('configured_window_ns') != list(rc.window)
            or metadata.get('effective_window_ns') != list(runconfig.OFFSPILL_WINDOW)
            or metadata.get('window_selection') != 'offspill' or metadata.get('window_applied') is not True
            or metadata.get('timing_policy') != 'fixed_absolute_crop_no_pbtime_rebase'
            or metadata.get('onspill_equivalence') is not False):
        raise ValueError('saved producer metadata differs from declared response/finder axis')


def validate_pulses(pulses, nbars):
    for name, maximum in (('bar', nbars), ('sipm', 4)):
        values = pulses[name].to_numpy()
        if values.dtype.kind not in 'iu' or np.any((values < 0) | (values >= maximum)):
            raise ValueError('invalid saved pulse ' + name)
    if not np.isfinite(pulses[['pe', 'time']].to_numpy()).all():
        raise ValueError('nonfinite saved pulse PE/time')


def saved_members(path):
    with uproot.open(path) as root:
        tree, _ = event_shapes._highest(root, 'crvdump/clusters')
        rows = tree.arrays(tables.KEY + ['idx', 'pulses'], library='ak')
        order = np.lexsort(tuple(ak.to_numpy(rows[key]) for key in ('idx', 'event', 'subrun', 'run')))
        return rows.pulses[order]


def cluster_frame(events, output):
    universe = canonical_keys(events)
    if (len(np.unique(universe[:, 0])) != 1 or np.any(universe[:, 1:] >= 2**20)):
        raise ValueError('finder axis requires one run and bounded packed event labels')
    labels = (universe[:, 1] << 20) | universe[:, 2]
    keys = np.asarray(output['cl_event'])
    if keys.ndim != 1 or (len(keys) and keys.dtype.kind not in 'iu'):
        raise ValueError('invalid finder event labels')
    positions = np.searchsorted(labels, keys)
    if np.any(positions >= len(labels)) or not np.array_equal(labels[positions], keys):
        raise ValueError('finder event outside saved census')
    result = pd.DataFrame(universe[positions], columns=tables.KEY)
    for name, source in [('pe', 'cl_pe'), ('start', 'cl_start'), ('end', 'cl_end')]:
        result[name] = np.asarray(output[source], dtype=np.float64)
    pos = np.asarray(output['cl_pos'], dtype=np.float64)
    if pos.shape != (len(result), 3):
        raise ValueError('invalid finder positions')
    for index, axis in enumerate(('x', 'y', 'z')):
        result[axis] = pos[:, index]
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('nonfinite finder cluster output')
    return result


def summarize(events, clusters):
    universe = events[tables.KEY].sort_values(tables.KEY).reset_index(drop=True)
    canonical_keys(universe)
    if len(clusters.merge(universe, on=tables.KEY, how='inner', validate='many_to_one')) != len(clusters):
        raise ValueError('cluster outside saved census')
    values = clusters.assign(pe=clusters.pe.to_numpy(dtype=np.float64)).groupby(tables.KEY).agg(
        count=('pe', 'size'), pe=('pe', 'sum')).reset_index()
    joined = universe.merge(values, on=tables.KEY, how='left', validate='one_to_one').fillna({'count': 0, 'pe': 0})
    features = {'cluster_pe': clusters.pe, 'cluster_duration': clusters.end - clusters.start,
                'cluster_x': clusters.x, 'cluster_y': clusters.y, 'cluster_z': clusters.z,
                'event_clusters': joined['count']}
    return dict(events=len(universe), clusters=len(clusters), cluster_events=int((joined['count'] > 0).sum()),
                total_cluster_pe=float(clusters.pe.to_numpy(dtype=np.float64).sum()), event_count=joined['count'].to_numpy(),
                event_pe=joined.pe.to_numpy(dtype=np.float64),
                histograms={name: histogram(value, EDGES[name]).tolist() for name, value in features.items()})


def contrast(reference, candidate):
    if reference['events'] != candidate['events']:
        raise ValueError('unpaired finder summaries')
    result = {}
    for name in ('event_count', 'event_pe'):
        delta = candidate[name] - reference[name]
        result[name] = dict(mean_difference=float(delta.mean()),
                            paired_event_se=float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None,
                            changed_events=int(np.count_nonzero(delta)),
                            reference_total=float(reference[name].sum()), candidate_total=float(candidate[name].sum()))
    a, b = reference['event_count'] > 0, candidate['event_count'] > 0
    result['cluster_presence'] = dict(both=int((a & b).sum()), reference_only=int((a & ~b).sum()),
                                      candidate_only=int((~a & b).sum()), neither=int((~a & ~b).sum()))
    result['histogram_count_differences'] = {
        name: (np.asarray(candidate['histograms'][name]) - values).tolist()
        for name, values in reference['histograms'].items()}
    return result


def public(summary):
    return {key: value for key, value in summary.items() if key not in ('event_count', 'event_pe')}


def compact_identity(result):
    result = dict(result)
    for key in ('stock_only_cluster_signatures', 'fast_only_cluster_signatures'):
        values = result.pop(key)
        result[key + '_count'] = len(values)
        result[key + '_first10'] = values[:10]
    return result


def run(config, out):
    config, out = Path(config).resolve(), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    first_hash = tier1b_assets.sha256(config)
    spec = json.loads(config.read_text())
    if spec['run_config'] != 'run1a_v01':
        raise ValueError('fixed axis diagnostic is scoped to run1a_v01')
    paths = [spec[key] for key in ('reference', 'default', 'historical')]
    if len({str(Path(path).resolve()) for path in paths}) != 3:
        raise ValueError('response axis inputs must be distinct files')
    from mu2efastsim import precision, determinism
    if precision.NAME != 'float64':
        raise ValueError('finder axis diagnostic requires explicit MU2EFS_DEVICE_FLOAT=float64')
    rc = runconfig.get(spec['run_config'])
    sources = [config, *paths, *(path + '.metadata.json' for path in paths[1:]),
               spec['conditions_manifest'], spec['resolved_fcl'], __file__, geometry.__file__,
               geometry.find_counters(rc.counters), readout.__file__, runconfig.__file__,
               coinc.__file__, coinc_jax.__file__, reco_detail.__file__, event_shapes.__file__,
               precision.__file__, determinism.__file__, load_conditions.__code__.co_filename,
               shared_yield.__file__, compare.__code__.co_filename, pulse_summary.__code__.co_filename,
               canonical_keys.__code__.co_filename, histogram.__code__.co_filename]
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in sources}
    before[str(config)] = first_hash
    geom = rc.geometry()
    cond, evidence = load_conditions(spec['conditions_manifest'], rc, geom)
    cond = readout.apply(cond, geom, rc.counters_cfg())
    historical_response, historical_evidence = shared_yield.emulate(rc.response(), spec['resolved_fcl'])
    metas = {label: json.loads(Path(spec[label] + '.metadata.json').read_text()) for label in ('default', 'historical')}
    require_metadata(metas['default'], rc, evidence, rc.response(), True)
    require_metadata(metas['historical'], rc, evidence, historical_response, False)
    if ('diagnostic_override' in metas['default'] or metas['historical'].get('diagnostic_override') != historical_evidence):
        raise ValueError('response axis diagnostic declaration differs')
    frames, members, inventories = {}, {}, {}
    for label in ('reference', 'default', 'historical'):
        frames[label], _, inventories[label] = reco_detail._read(spec[label])
        members[label] = saved_members(spec[label])
    universe = canonical_keys(frames['reference']['events'])
    if len(universe) < 2:
        raise ValueError('need at least two saved events')
    for frame in frames.values():
        if not np.array_equal(canonical_keys(frame['events']), universe):
            raise ValueError('response axis event populations differ')
    group = coinc.sector_geom(geom, rc.counters_cfg())
    status = readout.effective_status(cond)
    summaries, results = {}, {}
    stock = summarize(frames['reference']['events'], frames['reference']['clusters'])
    ref_pulses = pulse_summary(frames['reference']['pulses'], frames['reference']['events'], geom)
    for label, frame in frames.items():
        pulses = frame['pulses']
        validate_pulses(pulses, geom.nbars)
        pulse_values = pulse_summary(pulses, frame['events'], geom)
        ev = (pulses.subrun.to_numpy(dtype=np.int64) << 20) | pulses.event.to_numpy(dtype=np.int64)
        results[label] = dict(pulses=len(pulses), actual_sector_pulses={
            sector: compare_sector(ref_pulses[sector], pulse_values[sector]) for sector in ref_pulses})
        summaries[label] = {}
        for attachment in (False, True):
            mode = 'attachment_on' if attachment else 'attachment_off'
            cfg = dataclasses.replace(rc.finder(), add_opposite_side_pulses=attachment)
            output = coinc_jax.finish(coinc_jax.find_async(
                ev, *[pulses[key].to_numpy() for key in ('idx', 'bar', 'sipm', 'time', 'pe', 'seq')],
                group, status, cfg, with_pos=True))
            summary = summarize(frame['events'], cluster_frame(frame['events'], output))
            summaries[label][mode] = summary
            results[label][mode] = dict(summary=public(summary), versus_saved_stock=contrast(stock, summary),
                own_saved_membership=compact_identity(compare(frame['events'], frame['clusters'], members[label], output)),
                fallback_events=len(output['fallback_events']))
        results[label]['attachment_on_minus_off'] = contrast(
            summaries[label]['attachment_off'], summaries[label]['attachment_on'])
    response_contrasts = {mode: contrast(summaries['default'][mode], summaries['historical'][mode])
                          for mode in ('attachment_off', 'attachment_on')}
    if any(tier1b_assets.sha256(path) != value for path, value in before.items()):
        raise ValueError('response axis input/source drift')
    import jax
    result = dict(scope='saved_pulse_response_finder_factorial_not_stock_art_or_physics_admission',
                  events=len(universe), event_identity=response_receipt.event_identity(spec['reference']),
                  run_config=rc.name, backend=jax.default_backend(), precision=precision.provenance(), stock=public(stock), results=results,
                  historical_minus_default_response=response_contrasts, edges={name: EDGES[name].tolist() for name in stock['histograms']},
                  historical_response_evidence=historical_evidence, highest_cycles=inventories,
                  sources_sha256=before, physics_admitted=False, limitations=[
                      'Response populations are existing frozen draws, not response reruns or exact common random variates.',
                      'Only finder attachment varies within each saved pulse population; pulses are never modified.',
                      'Reference events are G4-retained, not inclusive generated exposure; no transport/library attribution.',
                      'Raw histogram differences and conditional paired-event SE are descriptive, not calibrated shape gates.',
                      'No finite-library/model uncertainty, stock-art producer certification, V-R4/V-R5 or C4 admission.',
                      'Historical shared-yield and no-attachment controls never change named defaults.'])
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with out.open('x') as stream:
        stream.write(payload)
    print(json.dumps({label: {mode: result['results'][label][mode]['summary']['clusters']
                              for mode in ('attachment_off', 'attachment_on')} for label in frames}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.config, args.out)
