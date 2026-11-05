"""Measured training support, not bootstrap errors, support closure or physics certification.

API: run(config, out), config={bundle: directory, run_config: name,
 native_records?: [{path: NPZ, seed: integer}]}. Paths are explicit (relative to cwd).
A bare NPZ path is allowed only when its metadata records a seed. Legacy object
seeds are external declarations, never execution attestations. Current bundles
require rebuilt training-census receipts; historical records need not bind them.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import runconfig, tier1b, tier1b_assets as T


def integer_ids(values, size, label):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a >= size):
        raise ValueError('invalid ' + label)
    return a.astype(np.int64)


def rows(frame):
    # Pandas handles null ratios without emitting nonstandard JSON NaN tokens.
    return json.loads(frame.to_json(orient='records', double_precision=15))


def reach(frame, numerator, denominator):
    r = frame.copy()
    for field in (numerator, denominator):
        if not np.isfinite(r[field]).all() or (r[field] < 0).any():
            raise ValueError('invalid reach counts')
    r['num_gt_den'] = r[numerator] > r[denominator]
    r['num_positive_den_zero'] = (r[numerator] > 0) & (r[denominator] == 0)
    r['raw_ratio'] = r[numerator] / r[denominator].replace(0, np.nan)
    return rows(r)


def support(lib, topology, manifest):
    """Vectorized payload audit; loops are over files/fields/bins, never events."""
    lx, ld = lib['libxing'], lib['libdesc']
    donors = topology['donors']
    if [d['path'] for d in donors] != manifest['inputs'] or len(set(manifest['inputs'])) != len(donors):
        raise ValueError('ordered topology donors differ from declared source files')
    counts = [d['crossings'] for d in donors]
    if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != len(lx):
        raise ValueError('topology crossing total mismatch')
    xid = integer_ids(lx.xid, len(lx), 'crossing xids')
    if not np.array_equal(xid, np.arange(len(lx))):
        raise ValueError('crossing xids must be ordered contiguous ids')
    dx = integer_ids(ld.xid, len(lx), 'descendant xids')
    nbins = len(tier1b.GROUPS) * tier1b.N_PBINS * tier1b.N_LBINS
    bins = integer_ids(lx.bin, nbins, 'crossing bins')
    mu = np.asarray(lib['xgeo']['mu'])
    if len(mu) != len(lx) or mu.dtype.kind not in 'iu' or np.any(mu < 0):
        raise ValueError('invalid crossing muons')
    ax = integer_ids(lib['anchor']['axid'], len(lx), 'anchor xids')
    if len(ax) != len(ld):
        raise ValueError('anchor/descendant row mismatch')
    source = np.repeat(np.arange(len(donors)), counts)
    x = pd.DataFrame(dict(bin=bins, donor=source, mu=mu))
    d = pd.DataFrame(dict(bin=bins[dx], edep=ld.edep.to_numpy()))
    if not np.isfinite(d.edep).all() or (d.edep < 0).any():
        raise ValueError('invalid descendant energy')
    perbin = pd.DataFrame(index=pd.Index(np.arange(nbins), name='bin'))
    perbin['crossings'] = x.groupby('bin').size()
    perbin['source_files'] = x.groupby('bin').donor.nunique()
    perbin['unique_muons'] = x.groupby('bin').mu.nunique()
    perbin['desc_rows'] = d.groupby('bin').size()
    perbin['desc_edep'] = d.groupby('bin').edep.sum()
    anchors = pd.DataFrame(dict(bin=bins[ax], axid=ax))
    perbin['current_anchor_crossings'] = anchors.groupby('bin').axid.nunique()
    perbin = perbin.fillna(0).reset_index()
    for field in perbin.columns.difference(['desc_edep']):
        perbin[field] = perbin[field].astype(np.int64)
    coverage = x.groupby(['bin', 'donor']).size().rename('crossings').reset_index()
    nm = lib['nmreach']
    if (not np.isfinite(nm[['ncensus', 'prob']]).all().all()
            or (nm.ncensus < 0).any() or ((nm.prob < 0) | (nm.prob > 1)).any()):
        raise ValueError('invalid near-miss census/probability')
    keys = ['apdg', 'kebin']
    lg = lib['libgen']
    if lg.duplicated(keys).any():
        raise ValueError('duplicate libgen keys')
    actual = lib['libshower'].groupby(keys).size().rename('actual_nshower').reset_index()
    shower = lg.merge(actual, on=keys, how='outer', indicator=True).fillna(
        {'ngen': 0, 'nshower': 0, 'actual_nshower': 0})
    shower['absent_from_libgen'] = shower['_merge'].eq('right_only')
    shower['saved_numerator_mismatch'] = shower.nshower != shower.actual_nshower
    shower['saved_nshower_gt_ngen'] = shower.nshower > shower.ngen
    return dict(tables={name: len(lib[name]) for name in tier1b.LIB_TREES},
                declared_source_files=manifest['inputs'], topology_donors=donors,
                training_census=manifest['training_census'], bins=rows(perbin),
                donor_coverage=rows(coverage), donor_coverage_omitted_pairs_are_zero=True,
                unique_muons=int(np.unique(mu).size), current_anchor_crossings=int(np.unique(ax).size),
                current_anchor={k: lib['anchor'][k] for k in ('n_roots', 'n_multi', 'forward_edep_share')},
                nearmiss=reach(nm, 'ntemp', 'nden'),
                showers=reach(shower.drop(columns='_merge'), 'actual_nshower', 'ngen'),
                shower_missing_denominator='Not recoverable from libgen for numerator-only absent keys; zero is a diagnostic placeholder.')


def records(entries, library_hash, rc_name):
    from scripts.check_native_response import arrays
    reports, seeds, seen, populations = [], {}, set(), {}
    for entry in entries:
        item = entry if isinstance(entry, dict) else dict(path=entry)
        data, meta = arrays(item['path'])
        digest, recorded = meta['output_sha256'], meta.get('seed')
        seed = item.get('seed', recorded)
        if type(seed) is not int or (recorded is not None and (type(recorded) is not int or recorded != seed)):
            raise ValueError('recorded/external seed mismatch or missing integer seed')
        if digest in seen:
            raise ValueError('duplicate native record hash')
        seen.add(digest)
        if meta['run_config']['name'] != rc_name:
            raise ValueError('native record run configuration mismatch')
        keys = data['generated_keys']
        if (keys.ndim != 2 or keys.shape[1] != 3 or keys.dtype.kind not in 'iu'
                or len(np.unique(keys, axis=0)) != len(keys) or len(keys) != meta['counts']['generated_events']):
            raise ValueError('native record generated census/count mismatch')
        counts = {k: meta['sample_counts'][k] for k in ('desc_steps', 'desc_clamped', 'desc_dropped')}
        if any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError('invalid native sample counts')
        for table, field in (('steps', 'steps__edep'), ('particles', 'particles__key'), ('digis', 'digis__chan')):
            if len(data[field]) != meta['counts'][table]:
                raise ValueError('native record table/count mismatch: ' + table)
        for table, suffix in (('clusters', 'cluster_keys'), ('pulses', 'pulse_keys'), ('members', 'member_keys')):
            if sum(len(a) for k, a in data.items() if k.endswith('__' + suffix)) != meta['counts'][table]:
                raise ValueError('native record chunk/count mismatch: ' + table)
        declared = meta['tier1b'].get('library_sha256')
        assets = meta['tier1b'].get('native', {}).get('assets_sha256', {})
        snapshots = meta.get('inputs_sha256', {})
        asset_paths = meta['tier1b'].get('paths', [])
        if (not isinstance(declared, str) or len(declared) != 64 or any(c not in '0123456789abcdef' for c in declared)
                or assets.get('library.root') != declared or not asset_paths
                or any(assets.get(Path(p).name) != snapshots.get(p) or p not in snapshots for p in asset_paths)):
            raise ValueError('record tier1b asset hash declaration mismatch')
        report = dict(path=str(Path(item['path']).resolve()), sha256=digest, seed=seed,
                      seed_evidence='recorded_not_attested' if recorded is not None else 'externally_declared_legacy_not_attested',
                      declared_library_sha256=declared, support_bound_to_current_library=declared == library_hash,
                      generated_events=len(keys), sample_counts=counts,
                      clamp_fraction=counts['desc_clamped'] / counts['desc_steps'] if counts['desc_steps'] else None,
                      drop_expected_zero=counts['desc_dropped'] == 0)
        reports.append(report)
        group = seeds.setdefault(str(seed), dict(records=0, generated_events=0, desc_steps=0, desc_clamped=0, desc_dropped=0))
        # Do not sum overlapping populations within a seed; across seeds exposure is never summed.
        previous = populations.get(seed, np.empty((0, 3), dtype=keys.dtype))
        joined = np.concatenate([keys, previous])
        if len(np.unique(joined, axis=0)) != len(joined):
            raise ValueError('overlapping generated exposure within seed')
        populations[seed] = joined
        group['records'] += 1
        group['generated_events'] += len(keys)
        for field in counts:
            group[field] += counts[field]
    for group in seeds.values():
        group['clamp_fraction'] = group['desc_clamped'] / group['desc_steps'] if group['desc_steps'] else None
    return dict(records=reports, per_seed=seeds, exposure_across_seeds='not_added')


def run(config, out):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if os.environ.get('CRVFS_PREPARED_LIBRARY_CACHE'):
        raise ValueError('unset CRVFS_PREPARED_LIBRARY_CACHE')
    rc = runconfig.get(config['run_config'])
    spec = T.native_bundle(config['bundle'], rc)
    from scripts import check_native_response
    from crvfastsim import genassoc, geometry
    from crvfastsim.analyses import census, context_audit
    paths = [*spec.paths(), rc.counters_path(), __file__, T.__file__, tier1b.__file__, runconfig.__file__,
             check_native_response.__file__, genassoc.__file__, geometry.__file__, census.__file__, context_audit.__file__]
    for entry in config.get('native_records', []):
        path = entry['path'] if isinstance(entry, dict) else entry
        paths.extend([path, str(path) + '.json'])
    before = {str(Path(p).absolute()): T.sha256(p) for p in paths}
    lib = T.load(spec, rc.geometry())
    manifest = json.loads(Path(spec.library).with_suffix('.json').read_text())
    topology = json.loads(Path(spec.topology + '.json').read_text())
    result = dict(scope='measured_training_donor_support_not_bootstrap_or_physics_certification',
                  config=config, run_config=rc.name, tier1b=spec.describe(), support=support(lib, topology, manifest),
                  native=records(config.get('native_records', []), spec.library_sha256, rc.name),
                  physics_admitted=False, support_closed=False, finite_library_uncertainty='open',
                  limitations=['Empty bins may reflect sparse geometry, not automatic failure.',
                               'Counts measure frozen training support, not independent physics closure or bootstrap uncertainty.',
                               'Cross-stream generator pointer correspondence remains unestablished.',
                               'Historical record library hashes may differ after widening; their sample counts do not bind current support.'],
                  inputs_sha256=before)
    if {p: T.sha256(p) for p in before} != before:
        raise ValueError('support audit input drift')
    result['report_payload_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run(config, args.out)
