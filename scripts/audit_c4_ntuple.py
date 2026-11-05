"""Read-only cached C4 source inventory and optional bounded original-selection replay, not native C4 admission."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from crvfastsim import tables, tier1b_assets


def canonical_keys(frame):
    values = frame[tables.KEY].to_numpy()
    if values.dtype.kind not in 'iu' or len(np.unique(values, axis=0)) != len(values):
        raise ValueError('C4 event census requires unique integer triples')
    return values[np.lexsort(values[:, ::-1].T)]


def event_fields(branches):
    """Recognize only complete explicit event-record schemas, never an unrelated run/event suffix."""
    for prefix in ('evtinfo/', 'evtinfo.', 'evt/', 'evt.', ''):
        fields = {key: prefix + key for key in tables.KEY}
        if all(value in branches for value in fields.values()):
            return fields
    return None


def compare_cached_records(candidate, cached):
    left = pd.DataFrame({key: ak.to_numpy(candidate[key]) for key in tables.KEY})
    right = pd.DataFrame({key: ak.to_numpy(cached[key]) for key in tables.KEY})
    canonical_keys(left)
    canonical_keys(right)
    order = left.merge(right.assign(_cached=np.arange(len(right))), on=tables.KEY, how='left', validate='one_to_one')
    if order._cached.isna().any() or len(left) != len(right):
        raise ValueError('bounded selection replay event identities differ from cached source subset')
    cached = cached[order._cached.to_numpy(np.int64)]
    def same(a, b):
        if not np.array_equal(ak.to_numpy(ak.num(a)), ak.to_numpy(ak.num(b))):
            return False
        a, b = ak.to_numpy(ak.flatten(a)), ak.to_numpy(ak.flatten(b))
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    checks = {key: same(candidate[key], cached[key]) for key in ('p_mid', 't_mid')}
    if set(candidate.coincs.fields) != set(cached.coincs.fields):
        raise ValueError('bounded selection replay coincidence schema differs')
    checks.update({'coincs/' + key: same(candidate.coincs[key], cached.coincs[key]) for key in candidate.coincs.fields})
    if not all(checks.values()):
        raise ValueError('bounded selection replay track momentum/time or coincidences differ from cache')
    return dict(events=len(left), tracks=int(ak.sum(ak.num(candidate.p_mid))), exact_fields=checks,
                selected_track_identity='original_analysis_mask_and_exact_cached_momentum_time_not_truth_ancestry')


def publish_lineage(replay, path):
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix='.c4-lineage-', dir=path.parent) as temp:
        staged = Path(temp) / 'selected.parquet'
        ak.to_parquet(replay, staged)
        os.link(staged, path)
    return dict(path=str(path.resolve()), sha256=tier1b_assets.sha256(path),
                scope='selected_original_track_indices_and_genealogy_not_generator_association')


def run(source, selection, out, mca=None, lineage_output=None):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if lineage_output and (not mca or Path(lineage_output).exists() or Path(lineage_output).is_symlink()):
        raise ValueError('lineage output requires original analysis replay and a new path')
    paths = [selection, __file__]
    if mca:
        from crvfastsim.veto import cosmic_ana
        paths += [cosmic_ana.__file__, str(Path(mca) / 'config/common/cuts.yaml')]
        paths += [str(path) for sub in ('src/core', 'src/utils') for path in (Path(mca) / sub).glob('*.py')]
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in paths}
    replay_cached = None
    selected = ak.from_parquet(selection)
    cached = pd.DataFrame({key: ak.to_numpy(selected[key]) for key in tables.KEY})
    canonical_keys(cached)
    result = dict(scope='read_only_cached_C4_source_identity_inventory_not_native_C4_or_ancestry_certification',
                  input=source, cached_events=len(cached), trees={}, sources_sha256=before, physics_admitted=False,
                  limitations=['Source catalogue/producer lineage is external evidence, not authenticated by branch names.',
                               'Event overlap does not establish selected-track index, SimParticle or generator ancestry.',
                               'No stock selection rerun, trigger replay, timing alignment or native response comparison.',
                               'Remote ROOT source is not payload-hashed; exact catalogue checksum must be recorded separately.'])
    with uproot.open(source) as root:
        for name, kind in root.classnames(recursive=True, cycle=False).items():
            if kind != 'TTree':
                continue
            tree = root[name]
            branches = tree.keys()
            fields = event_fields(branches)
            item = dict(entries=int(tree.num_entries), event_fields=fields,
                        truth_branches=[key for key in branches if any(token in key.lower() for token in
                                        ('trkmc', 'simid', 'simkey', 'genid', 'genkey', 'parent', 'ancestor'))])
            if fields:
                arrays = tree.arrays(list(fields.values()), library='np')
                events = pd.DataFrame({key: arrays[value] for key, value in fields.items()})
                canonical_keys(events)
                overlap = cached.assign(_row=np.arange(len(cached))).merge(events, on=tables.KEY, how='inner', validate='one_to_one')
                if mca:
                    if replay_cached is not None:
                        raise ValueError('ambiguous C4 event tree for bounded selection replay')
                    replay_cached = selected[overlap._row.to_numpy(np.int64)]
                item.update(shared_cached_events=len(overlap), source_runs=sorted(map(int, events.run.unique())),
                            source_subruns=sorted(map(int, events.subrun.unique())),
                            event_min=int(events.event.min()) if len(events) else None,
                            event_max=int(events.event.max()) if len(events) else None)
            result['trees'][name] = item
    if mca:
        if replay_cached is None:
            raise ValueError('no C4 source event schema for bounded selection replay')
        Analyse, CutManager = cosmic_ana._import_mca(mca)
        analysis = Analyse(on_spill=True, cut_config_path=str(Path(mca) / 'config/common/cuts.yaml'),
                           cutset_name='dev', verbosity=0)
        replay = cosmic_ana.select_file(source, analysis, CutManager, include_lineage=True)
        result['bounded_selection_replay'] = compare_cached_records(replay, replay_cached)
        lineage = replay.trkmcsim
        result['selected_track_lineage'] = dict(
            tracks=int(ak.sum(ak.num(replay.track_index))),
            tracks_with_genealogy=int(ak.sum(ak.num(lineage, axis=2) > 0)),
            genealogy_records=int(ak.sum(ak.num(lineage, axis=2))),
            fields=lineage.fields,
            scope='original_track_index_aligned_genealogy_not_generator_ProductID_association')
        if {'id', 'pdg', 'gen', 'nhits'}.issubset(lineage.fields):
            for field in ('pdg', 'gen'):
                values, counts = np.unique(ak.to_numpy(ak.flatten(lineage[field], axis=None)), return_counts=True)
                result['selected_track_lineage'][field + '_counts'] = {str(int(v)): int(n) for v, n in zip(values, counts)}
            result['selected_track_lineage']['negative_sim_ids'] = int(ak.sum(lineage.id < 0))
        else:
            result['selected_track_lineage']['identity_status'] = 'unsupported_missing_explicit_sim_or_generator_fields'
        result['limitations'][2] = 'One bounded original-analysis selection replay only; no trigger replay, native response or full44file validation.'
    if any(tier1b_assets.sha256(path) != digest for path, digest in before.items()):
        raise ValueError('C4 inventory input drift')
    if lineage_output:
        result['lineage_output'] = publish_lineage(replay, lineage_output)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with out.open('x') as stream:
        stream.write(payload)
    print(payload, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mca', help='optional original mu2e_cosmic_ana checkout for one bounded dev selection replay')
    parser.add_argument('--lineage-output', help='optional new parquet retaining exact original track indices and MC genealogy')
    a = parser.parse_args()
    run(a.input, a.selection, a.out, a.mca, a.lineage_output)
