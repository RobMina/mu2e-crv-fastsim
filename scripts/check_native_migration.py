"""Saved-only generated-event migration of native clusters against retained stock selection; no tuning or closure."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import genassoc, tables, tier1b_assets
from crvfastsim.validation import reco_detail
from scripts.check_native_response import arrays, canonical_keys
from scripts.compare_native_histograms import arm_seed, require_subset


def counts(events, universe):
    canonical_keys(universe)
    require_subset(events, universe)
    sums = events.groupby(tables.KEY).size().rename('count').reset_index()
    return universe.merge(sums, on=tables.KEY, how='left', validate='one_to_one')['count'].fillna(0).to_numpy(np.int64)


def run(config, out):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    source = Path(config).resolve()
    before = {str(source): tier1b_assets.sha256(source)}
    spec = json.loads(source.read_text())
    pairs = spec['pairs']
    if not pairs:
        raise ValueError('need migration pairs')
    declaration = spec.get('legacy_producer_script')
    paths = [__file__, arrays.__code__.co_filename, arm_seed.__code__.co_filename, genassoc.__file__, reco_detail.__file__]
    if declaration:
        for item in [declaration, *declaration.get('additional_scripts', [])]:
            if tier1b_assets.sha256(item['path']) != item['sha256']:
                raise ValueError('legacy producer script changed')
            paths.append(item['path'])
    for pair in pairs:
        paths += [pair['generator'], pair['donor'], *pair['stock']]
        for path in pair['candidates'].values():
            paths += [path, str(path) + '.json']
    for path in paths:
        resolved = str(Path(path).resolve())
        if resolved not in before:
            before[resolved] = tier1b_assets.sha256(resolved)
    results, seen, seed_set = {}, [], set(pairs[0]['candidates'])
    if not seed_set:
        raise ValueError('need migration seed arms')
    for pair in pairs:
        if set(pair['candidates']) != seed_set:
            raise ValueError('migration seed arms differ between shards')
        report = genassoc.validate_generator_stream(pair['generator'])
        genassoc.validate_donor(pair['donor'])
        if report.get('schema_version') != 2 or report.get('stream_role') != 'full generator stream':
            raise ValueError('migration requires strict full-generator census')
        universe = genassoc.generator_event_keys(pair['generator'])
        canonical_keys(universe)
        seen.append(universe)
        canonical_keys(pd.concat(seen, ignore_index=True))
        frames = [reco_detail._read(path)[0] for path in pair['stock']]
        accepted = pd.concat([f['events'][tables.KEY] for f in frames], ignore_index=True)
        stock = pd.concat([f['clusters'][tables.KEY] for f in frames], ignore_index=True)
        require_subset(accepted, universe)
        if not np.array_equal(canonical_keys(accepted), canonical_keys(tables.read(pair['donor'], 'events', tables.KEY))):
            raise ValueError('migration stock/donor population mismatch')
        selected = universe.merge(accepted.assign(_accepted=True), on=tables.KEY, how='left', validate='one_to_one')._accepted.eq(True).to_numpy()
        used = set()
        for seed, path in pair['candidates'].items():
            data, meta = arrays(path)
            arm_seed(seed, meta, path, used, declaration)
            keys = pd.DataFrame(data['generated_keys'], columns=tables.KEY)
            if not np.array_equal(canonical_keys(keys), canonical_keys(universe)):
                raise ValueError('migration candidate/full generator population mismatch')
            candidate = pd.concat([pd.DataFrame(data[f'chunk_{chunk:04d}__cluster_keys'], columns=tables.KEY)
                                   for chunk in range(int(data['chunk_count']))], ignore_index=True)
            a, b = counts(stock, universe), counts(candidate, universe)
            row = results.setdefault(seed, {})
            for label, mask in [('accepted', selected), ('not_retained', ~selected)]:
                values = row.setdefault(label, dict(events=0, stock_clusters=0, native_clusters=0,
                                                    both_cluster_events=0, stock_only_events=0, native_only_events=0))
                values['events'] += int(mask.sum())
                values['stock_clusters'] += int(a[mask].sum())
                values['native_clusters'] += int(b[mask].sum())
                values['both_cluster_events'] += int(((a > 0) & (b > 0) & mask).sum())
                values['stock_only_events'] += int(((a > 0) & (b == 0) & mask).sum())
                values['native_only_events'] += int(((a == 0) & (b > 0) & mask).sum())
    result = dict(scope='saved_event_migration_diagnostic_not_native_C4_or_physics_closure', seeds=results,
                  sources_sha256=before, physics_admitted=False,
                  limitations=['Stock not-retained response is unavailable, not a measured zero.',
                               'Event migration includes independent transport/scattering/response/library draws.',
                               'Seed arms reuse the same exposure and donor library.',
                               'Legacy seed identity is an external producer declaration, not execution attestation.'])
    if any(tier1b_assets.sha256(path) != value for path, value in before.items()):
        raise ValueError('migration input drift')
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with out.open('x') as stream:
        stream.write(payload)
    print(json.dumps(result['seeds'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    a = parser.parse_args()
    run(a.config, a.out)
