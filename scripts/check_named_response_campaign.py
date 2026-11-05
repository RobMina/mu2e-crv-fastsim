"""Saved-only paired response campaign; no redraw, defaults, transport or full V-R4 admission.

Config: run_config, seed, conditions_manifest, generated_per_pair (default 50000),
pairs [{generator, reference: [ROOT paths], candidate: [ROOT paths]}]. A pair may
override seed/conditions_manifest to retain its actual historical identity. Candidate
ROOT.metadata.json must match that declaration. Producer/source lineage and identical
G4 steps require external evidence; receipts are not retroactively manufactured.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import genassoc, geometry, named_conditions, runconfig, tables, tier1b_assets
from crvfastsim.validation import event_shapes, paired_shards, reco_detail, v2compare
from scripts import compare_native_histograms as hist
from scripts.check_named_sector_response import pulse_summary, compare_sector
from scripts.check_native_response import canonical_keys


def read_arm(paths, geom):
    if not isinstance(paths, list) or not paths:
        raise ValueError('response arm needs a nonempty ROOT file list')
    frames, cycles = [], []
    for path in paths:
        frame, _, metadata = reco_detail._read(path)
        frames.append(frame)
        cycles.append(dict(path=path, highest_cycles=metadata))
    merged = {key: pd.concat([frame[key] for frame in frames], ignore_index=True)
              for key in ('events', 'pulses', 'clusters')}
    canonical_keys(merged['events'])  # also rejects overlapping split files
    merged['events'] = merged['events'].sort_values(tables.KEY).reset_index(drop=True)
    p, c = merged['pulses'], merged['clusters']
    bar = p.bar.to_numpy()
    if bar.dtype.kind not in 'iu' or np.any((bar < 0) | (bar >= geom.nbars)):
        raise ValueError('invalid saved pulse bar')
    if (not np.isfinite(p[['pe', 'time']].to_numpy()).all() or (p.pe < 0).any() or
            not np.isfinite(c[['pe', 'start', 'end', 'x', 'y', 'z']].to_numpy()).all() or
            (c.pe < 0).any() or (c.end < c.start).any()):
        raise ValueError('invalid saved pulse/cluster PE, time or position')
    return merged, cycles


def admit_pair(reference, candidate, universe, seen):
    """Exact integer triples, including response-empty events; no packed-key joins."""
    generated = canonical_keys(universe)
    a, b = canonical_keys(reference), canonical_keys(candidate)
    if not np.array_equal(a, b):
        raise ValueError('paired saved event population differs')
    index = lambda values: pd.MultiIndex.from_arrays(values.T, names=tables.KEY)
    generated_index = index(generated)
    if not index(a).isin(generated_index).all():
        raise ValueError('accepted response outside generator census')
    if seen and generated_index.isin(index(np.concatenate(seen))).any():
        raise ValueError('overlapping generator shards')
    seen.append(generated)


def response_features(frame, universe, geom):
    # Reuse the declared flow convention; step histograms are deliberately excluded.
    steps = pd.DataFrame({key: pd.Series(dtype='int64') for key in tables.KEY + ['bar']})
    steps['edep'], steps['len'] = pd.Series(dtype=float), pd.Series(dtype=float)
    return {name: value for name, value in hist.features(
        frame['pulses'], frame['clusters'], steps, universe[tables.KEY], geom).items()
        if '/step_' not in name}


def require_metadata(metadata, rc, seed, conditions):
    from dataclasses import asdict, replace
    cfg = replace(rc.chain_config(jax=False, noise=True), window=runconfig.OFFSPILL_WINDOW)
    if (metadata.get('schema') != 'crv-fasttables-named-v1' or
            metadata.get('stage') != 'chain' or metadata.get('run_config') != rc.name or
            type(seed) is not int or seed < 0 or type(metadata.get('seed')) is not int or
            metadata['seed'] != seed or metadata.get('conditions') != conditions or
            metadata.get('jax') is not False or metadata.get('noise') is not True or
            metadata.get('chain_config') != json.loads(json.dumps(asdict(cfg))) or
            metadata.get('effective_window_ns') != list(runconfig.OFFSPILL_WINDOW) or
            metadata.get('timing_policy') != 'fixed_absolute_crop_no_pbtime_rebase' or
            'diagnostic_override' in metadata):
        raise ValueError('candidate named run/seed/conditions metadata mismatch')


def run(config, out):
    config, out = Path(config).resolve(), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    before = {}
    def watch(path):
        path = str(Path(path).resolve())
        digest = tier1b_assets.sha256(path)
        if path in before and before[path] != digest:
            raise ValueError('saved response input drift')
        before[path] = digest
        return path
    watch(config)
    spec = json.loads(config.read_text())
    pairs = spec['pairs']
    expected = spec.get('generated_per_pair', 50000)
    if len(pairs) < 4 or type(expected) is not int or expected <= 0:
        raise ValueError('need at least four pairs and positive declared generated exposure')
    rc = runconfig.get(spec['run_config'])
    if rc.name != 'run1a_v01':
        raise ValueError('fixed histogram edges require run1a_v01')
    for path in (__file__, canonical_keys.__code__.co_filename, pulse_summary.__code__.co_filename,
                 hist.__file__, event_shapes.__file__, paired_shards.__file__, reco_detail.__file__, v2compare.__file__,
                 genassoc.__file__, tables.__file__, tier1b_assets.__file__, runconfig.__file__,
                 geometry.__file__, named_conditions.__file__, geometry.find_counters(rc.counters)):
        watch(path)
    geom = rc.geometry()
    arms, overlays = [{}, {}], [{}, {}]
    manifests, summaries, seen, totals = [], [[], []], [], [[], []]
    for pair in pairs:
        generator = watch(pair['generator'])
        manifest = Path(watch(pair.get('conditions_manifest', spec.get('conditions_manifest'))))
        declared = json.loads(manifest.read_text())
        for group in ('inputs', 'assets'):
            for item in declared[group].values():
                watch(manifest.parent / item['path'])
        _, conditions = named_conditions.load_conditions(manifest, rc, geom)
        seed = pair.get('seed', spec.get('seed'))
        pair = dict(pair, reference=[str(Path(path).resolve()) for path in pair['reference']],
                    candidate=[str(Path(path).resolve()) for path in pair['candidate']])
        files = pair['reference'] + pair['candidate']
        if len(files) != len(set(files)):
            raise ValueError('saved response input aliases reuse a reference or candidate')
        for path in files:
            watch(path)
        for path in pair['candidate']:
            meta = json.loads(Path(watch(str(path) + '.metadata.json')).read_text())
            require_metadata(meta, rc, seed, conditions)
        report = genassoc.validate_generator_stream(generator)
        if report.get('schema_version') != 2 or report.get('stream_role') != 'full generator stream':
            raise ValueError('strict schema-v2 full generator required')
        universe = genassoc.generator_event_keys(generator)
        if len(universe) != expected:
            raise ValueError('generator exposure differs from declared generated_per_pair')
        a, ac = read_arm(pair['reference'], geom)
        b, bc = read_arm(pair['candidate'], geom)
        admit_pair(a['events'], b['events'], universe, seen)
        if len(a['events']) < 2:
            raise ValueError('paired event summaries need at least two accepted events')
        for index, frame in enumerate((a, b)):
            summaries[index].append(pulse_summary(frame['pulses'], frame['events'], geom))
            totals[index].append(len(frame['clusters']))
            for target, population in ((arms[index], frame['events']), (overlays[index], universe)):
                for name, values in response_features(frame, population, geom).items():
                    target.setdefault(name, []).append(values)
        manifests.append(dict(generator=generator, generator_report=report, generated=len(universe),
                              accepted=len(a['events']), reference_cycles=ac, candidate_cycles=bc,
                              seed=seed, conditions_manifest=str(manifest)))
    stacked = [{name: np.stack(values) for name, values in arm.items()} for arm in arms]
    comparison = paired_shards.compare(*stacked)
    a, b = np.asarray(totals[0], float), np.asarray(totals[1], float)
    ratio = float(b.sum() / a.sum()) if a.sum() else None
    error = float(np.sqrt(paired_shards.batch_variance((b - ratio * a)[:, None])[0]) / a.sum()) if ratio is not None else None
    sectors = {}
    for name in geom.sector_names:
        combined = []
        for arm in summaries:
            combined.append({key: np.concatenate([s[name][key] for s in arm]) if key == 'event_pe'
                             else sum(s[name][key] for s in arm) for key in arm[0][name]})
        sectors[name] = compare_sector(*combined)
    result = dict(scope='same_step_saved_response_not_fullVR4', physics_admitted=False,
                  run_config=rc.name, shards=len(pairs), pairs=manifests,
                  arm_order=['reference', 'candidate'], underflow_overflow=True,
                  histogram_normalization='raw_counts_per_equal_full_generated_exposure_shard_conditional_on_accepted_events',
                  generated_per_pair=expected, comparison=comparison, per_actual_sector=sectors,
                  global_cluster_count=dict(reference_per_pair=totals[0], candidate_per_pair=totals[1],
                                            ratio=ratio, paired_shard_ratio_se=error),
                  raw_features=[{k: v.tolist() for k, v in arm.items()} for arm in stacked],
                  full_generator_zero_overlay_diagnostic=[{k: [v.tolist() for v in values] for k, values in arm.items()} for arm in overlays],
                  edges={k: v.tolist() for k, v in hist.EDGES.items() if not k.startswith('step_')},
                  producer_identity_status='external_evidence_not_attested',
                  producer_identity_external_evidence=spec.get('producer_identity_external_evidence'),
                  sources_sha256=before, limitations=[
                      'Same G4-step producer lineage is external evidence, not inferred from saved response tables or receipts.',
                      'Accepted populations may differ across pairs; matched within each pair. No acceptance or inclusive-response claim.',
                      'Full-generator zero overlays assume missing rejected response is zero; diagnostic only, never inclusive physics.',
                      'Pulse median/spectrum errors are independent-pulse approximations; global cluster SE uses paired shards.',
                      'No finite-library uncertainty, transport residual closure, full V-R4 or default promotion.'])
    if any(tier1b_assets.sha256(path) != digest for path, digest in before.items()):
        raise ValueError('saved response input drift')
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.config, args.out)
