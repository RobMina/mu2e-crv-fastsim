"""Paired per-shard native diagnostic histograms with exact population joins and separate sampling-seed arms."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import genassoc, native_record, runconfig, tables, tier1b_assets
from crvfastsim.validation import paired_shards, reco_detail
from scripts.check_native_response import arrays, canonical_keys


EDGES = {'pulse_pe': np.arange(0., 2020., 20.), 'pulse_time': np.linspace(0., 1e5, 51),
         'pulse_bar': np.arange(0., 1281., 16.), 'cluster_pe': np.arange(0., 5100., 100.),
         'cluster_duration': np.arange(0., 202., 2.), 'cluster_x': np.arange(-8000., 8001., 200.),
         'cluster_y': np.arange(-2000., 16001., 200.), 'cluster_z': np.arange(-12000., 22001., 400.),
         'step_energy': np.arange(0., 30.5, .5), 'step_length': np.arange(0., 102., 2.),
         'event_clusters': np.arange(-.5, 10.5, 1.)}


def histogram(values, edges):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('nonfinite/nonscalar histogram input')
    return np.bincount(np.searchsorted(edges, values, side='right'), minlength=len(edges) + 1)


def features(pulses, clusters, steps, universe, geom):
    """Fixed edges with under/overflow, all sectors, and zero-cluster generated events."""
    result = {}
    for sector, name in enumerate(geom.sector_names):
        p = pulses.loc[geom.sector[pulses.bar.to_numpy(dtype=np.int64)] == sector]
        s = steps.loc[geom.sector[steps.bar.to_numpy(dtype=np.int64)] == sector]
        for label, values in [('pulse_pe', p.pe), ('pulse_time', p.time), ('pulse_bar', p.bar),
                              ('step_energy', s.edep), ('step_length', s.len)]:
            result[name + '/' + label] = histogram(values, EDGES[label])
    for label, values in [('cluster_pe', clusters.pe), ('cluster_duration', clusters.end - clusters.start),
                          ('cluster_x', clusters.x), ('cluster_y', clusters.y), ('cluster_z', clusters.z)]:
        result['all/' + label] = histogram(values, EDGES[label])
    sums = clusters.groupby(tables.KEY).size().rename('count').reset_index()
    counts = universe.merge(sums, on=tables.KEY, how='left', validate='one_to_one')['count'].fillna(0)
    result['all/event_clusters'] = histogram(counts, EDGES['event_clusters'])
    return result


def native_frames(data):
    keys = pd.DataFrame(data['generated_keys'], columns=tables.KEY)
    pulses, clusters = [], []
    for chunk in range(int(data['chunk_count'])):
        prefix = f'chunk_{chunk:04d}__'
        p = pd.DataFrame(data[prefix + 'pulse_keys'], columns=tables.KEY)
        for name in ('bar', 'pe', 'time'):
            p[name] = data[prefix + 'pulses__' + name]
        pulses.append(p)
        c = pd.DataFrame(data[prefix + 'cluster_keys'], columns=tables.KEY)
        for name, source in [('pe', 'cl_pe'), ('start', 'cl_start'), ('end', 'cl_end')]:
            c[name] = data[prefix + source]
        for axis, name in enumerate(('x', 'y', 'z')):
            c[name] = data[prefix + 'cl_pos'][:, axis]
        clusters.append(c)
    st = pd.DataFrame({name: data['steps__' + name] for name in tables.KEY + ['bar', 'edep', 'len']})
    return pd.concat(pulses, ignore_index=True), pd.concat(clusters, ignore_index=True), st, keys


def require_subset(subset, universe):
    if len(subset[tables.KEY].merge(universe[tables.KEY], on=tables.KEY, how='inner', validate='many_to_one')) != len(subset):
        raise ValueError('accepted stock events outside configured generator census')


def arm_seed(label, meta, path, used, declaration=None):
    digest = meta['output_sha256']
    if digest in used:
        raise ValueError('candidate aliases reuse the same output record')
    used.add(digest)
    expected = int(label.removeprefix('seed'))
    actual = meta.get('seed')
    if actual is not None:
        if type(actual) is not int or actual != expected:
            raise ValueError('candidate sampling seed differs from arm label')
        return 'recorded_seed'
    if declaration is None or declaration['sha256'] != tier1b_assets.sha256(declaration['path']):
        raise ValueError('legacy record requires hash-bound external sampling-seed declaration')
    text = Path(declaration['path']).read_text()
    if 'for seed in 1 2' not in text or expected not in (1, 2) or f'_seed{expected}.npz' not in str(path):
        raise ValueError('legacy sampling seed not bound by declared producer script')
    return 'external_producer_script_declaration_not_recorded_seed'


def common_assets(meta):
    def varying(path):
        return '/generators/' in path or path.endswith('.py')
    return {path: value for path, value in meta['inputs_sha256'].items() if not varying(path)}


def run(config_path, out):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    pairs = config['pairs']
    if len(pairs) < 4:
        raise ValueError('need at least four independent generator shards for paired variance')
    rc = runconfig.get(config['run_config'])
    if rc.name != 'run1a_v01':
        raise ValueError('fixed histogram edges are scoped to run1a_v01')
    geom = rc.geometry()
    before = {str(config_path): tier1b_assets.sha256(config_path), __file__: tier1b_assets.sha256(__file__),
              paired_shards.__file__: tier1b_assets.sha256(paired_shards.__file__)}
    from crvfastsim.analyses.e2e_bench import rows
    arms, exposures, manifests = {}, [], []
    seen = np.empty(0, dtype=np.dtype([(f'k{i}', '<i8') for i in range(3)]))
    native_identity, asset_identity = None, None
    seed_evidence = []
    declaration = config.get('legacy_producer_script')
    if declaration:
        for item in [declaration, *declaration.get('additional_scripts', [])]:
            actual_hash = tier1b_assets.sha256(item['path'])
            if actual_hash != item['sha256']:
                raise ValueError('legacy producer script changed')
            before[item['path']] = actual_hash
    for pair in pairs:
        generator, donor = pair['generator'], pair['donor']
        for path in [generator, donor, *pair['stock'], *pair['candidates'].values()]:
            before[str(Path(path).resolve())] = tier1b_assets.sha256(path)
        report = genassoc.validate_generator_stream(generator)
        genassoc.validate_donor(donor)
        if report.get('schema_version') != 2 or report.get('stream_role') != 'full generator stream':
            raise ValueError('strict full-generator exposure required')
        universe = genassoc.generator_event_keys(generator)
        current = rows(universe[tables.KEY].to_numpy())
        if np.intersect1d(seen, current).size:
            raise ValueError('overlapping generator shards')
        seen = np.concatenate([seen, current])
        exposures.append(len(universe))
        accepted, stock_tables = [], []
        for path in pair['stock']:
            frames, _, metadata = reco_detail._read(path)
            accepted.append(frames['events'])
            stock_tables.append(frames)
            manifests.append(dict(path=path, highest_cycles=metadata))
        if not np.array_equal(canonical_keys(pd.concat(accepted)), canonical_keys(tables.read(donor, 'events', tables.KEY))):
            raise ValueError('stock response/accepted donor population mismatch')
        require_subset(pd.concat(accepted), universe)
        stock_p = pd.concat([v['pulses'] for v in stock_tables], ignore_index=True)
        stock_c = pd.concat([v['clusters'] for v in stock_tables], ignore_index=True)
        stock_steps = tables.read(donor, 'steps')
        family = features(stock_p, stock_c, stock_steps, universe, geom)
        for name, values in family.items():
            arms.setdefault('stock', {}).setdefault(name, []).append(values)
        accepted_keys = pd.concat(accepted)[tables.KEY]
        accepted_family = features(stock_p, stock_c, stock_steps, accepted_keys, geom)
        for name, values in accepted_family.items():
            arms.setdefault('selected_stock', {}).setdefault(name, []).append(values)
        used = set()
        for seed, path in pair['candidates'].items():
            data, meta = arrays(path)
            seed_evidence.append(arm_seed(seed, meta, path, used, declaration))
            if meta['run_config']['name'] != rc.name:
                raise ValueError('candidate run configuration differs from histogram geometry')
            assets = common_assets(meta)
            if asset_identity is not None and assets != asset_identity:
                raise ValueError('candidate conditions/GDML/optical asset identity changed')
            asset_identity = assets
            sidecar = str(Path(str(path) + '.json').resolve())
            before[sidecar] = tier1b_assets.sha256(sidecar)
            if meta.get('effective_transport') is None and declaration is None:
                raise ValueError('legacy effective transport requires external producer declaration')
            identity = (meta['tier1b'], meta['run_config'], meta['chain_config'], meta.get('effective_transport'))
            if native_identity is not None and identity != native_identity:
                raise ValueError('native bundle/configuration changed between arms')
            native_identity = identity
            p, c, s, keys = native_frames(data)
            if not np.array_equal(canonical_keys(keys), canonical_keys(universe)):
                raise ValueError('candidate/full generator population mismatch')
            if meta['counts']['generated_events'] != len(universe):
                raise ValueError('candidate denominator mismatch')
            family = features(p, c, s, universe, geom)
            for name, values in family.items():
                arms.setdefault(seed, {}).setdefault(name, []).append(values)
            selected = [frame.merge(accepted_keys, on=tables.KEY, how='inner', validate='many_to_one') for frame in (p, c, s)]
            selected_family = features(*selected, accepted_keys, geom)
            for name, values in selected_family.items():
                arms.setdefault('selected_' + seed, {}).setdefault(name, []).append(values)
    if len(set(exposures)) != 1:
        raise ValueError('paired shard rule requires equal generated exposure')
    for arm in arms:
        for name in arms[arm]:
            if len(arms[arm][name]) != len(pairs):
                raise ValueError('missing arm/shard histogram')
            arms[arm][name] = np.stack(arms[arm][name])
    seeds = sorted(name for name in arms if name not in ('stock', 'selected_stock') and not name.startswith('selected_'))
    results = {seed: paired_shards.compare(arms['stock'], arms[seed]) for seed in seeds}
    selected_results = {seed: paired_shards.compare(arms['selected_stock'], arms['selected_' + seed]) for seed in seeds}
    null = paired_shards.compare(arms[seeds[0]], arms[seeds[1]]) if len(seeds) == 2 else None
    result = dict(scope='native_paired_histogram_diagnostic_not_stock_configuration_or_physics_certification',
                  run_config=rc.name, generated_per_seed=sum(exposures), shards=len(pairs), results=results,
                  selected_event_diagnostic=selected_results, sampling_seed_contrast=null,
                  seed_evidence=sorted(set(seed_evidence)),
                  seed_contrast_independence='declared_by_external_script' if declaration else 'recorded_seeds',
                  edges={k: v.tolist() for k, v in EDGES.items()},
                  underflow_overflow=True, stock_cycles=manifests, sources_sha256=before,
                  limitations=['Rejected G4 events are treated as zero response; selection audit remains required.',
                               'Stock response/finder semantics require external producer evidence; legacy references may differ from named defaults.',
                               'Frozen legacy records lack effective transport flags; saved producer scripts supply declarations only.',
                               'Ten shards provide limited null-tail calibration; seeds are not added exposure.',
                               'No finite-library/systematic uncertainty, MC-data guardrail or E2E/C4 certification.'],
                  physics_admitted=False)
    if any(tier1b_assets.sha256(path) != value for path, value in before.items()):
        raise ValueError('input drift during paired histogram comparison')
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'paired histograms shards={len(pairs)} generated_per_seed={sum(exposures)} output={out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    a = parser.parse_args()
    run(a.config, a.out)
