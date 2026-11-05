"""Native detail records retain the full census without claiming benchmark or physics admission."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import native_record as N, runconfig
from crvfastsim.analyses.e2e_bench import fingerprint
from crvfastsim.analyses.runchain import run
from test_e2e_bench import detail


def test_full_census_raw_output_and_exclusive_publication(tmp_path):
    output = tmp_path / 'native.npz'
    source = tmp_path / 'input'
    source.write_text('frozen')
    before = {str(source): N.tier1b_assets.sha256(source)}
    universe = pd.DataFrame(dict(run=[1400, 1400], subrun=[2, 2], event=[3, 4]))
    steps = universe.iloc[:1].assign(edep=1.)
    rc = runconfig.get('run1a_v01')
    spec = SimpleNamespace(describe=lambda: {'native': True})
    digis = dict(chan=np.array([1]), starttdc=np.array([0]), adc=np.zeros((1, 12)))
    info = N.write(output, universe, steps, steps, digis, [detail()], before, rc, rc.chain_config(), spec, {})
    assert info['counts']['generated_events'] == 2 and info['counts']['events_with_clusters'] == 1
    with np.load(output, allow_pickle=False) as arrays:
        assert arrays['generated_keys'].shape == (2, 3)
        assert arrays['chunk_0000__cluster_keys'].tolist() == [[1400, 2, 3]]
        assert arrays['steps__edep'].tolist() == [1.]
        digest = fingerprint(dict(arrays))
    evidence = json.loads(N.paths(output)[1].read_text())
    assert evidence['array_fingerprint'] == digest and evidence['physics_pass'] is False
    assert evidence['scope'] == N.SCOPE
    with pytest.raises(FileExistsError):
        N.write(output, universe, steps, steps, digis, [detail()], before, rc, rc.chain_config(), spec, {})
    source.write_text('changed')
    with pytest.raises(ValueError, match='drift'):
        N.verify(before)


def test_native_output_admission_precedes_setup():
    with pytest.raises(ValueError, match='requires an explicit native'):
        run([], 'unused', native_output='unused.npz')


def test_cli_native_record_flag():
    from crvfastsim.cli import build_parser
    args = build_parser().parse_args(['runchain', 'gen.root', '--native-output', 'out.npz'])
    assert args.native_output == 'out.npz'


def test_response_census_sort_preserves_full_integer_triples():
    from scripts.check_native_response import canonical_keys
    frame = pd.DataFrame(dict(run=[1400, 1400, 1400], subrun=[104, 102, 102], event=[1, 3, 1]))
    assert canonical_keys(frame).tolist() == [[1400, 102, 1], [1400, 102, 3], [1400, 104, 1]]
    with pytest.raises(ValueError, match='unique integer'):
        canonical_keys(pd.concat([frame, frame.iloc[:1]]))


def test_c4_overlap_never_pairs_equal_event_numbers_across_runs():
    from scripts.audit_native_c4 import overlap
    a = pd.DataFrame(dict(run=[1, 1], subrun=[2, 2], event=[1, 2]))
    b = pd.DataFrame(dict(run=[3, 3], subrun=[2, 2], event=[1, 2]))
    result = overlap(a, b)
    assert result['shared_events'] == 0 and not result['exact_population']


def test_migration_counts_keep_generator_zeros():
    from scripts.check_native_migration import counts
    universe = pd.DataFrame(dict(run=[1400, 1400], subrun=[102, 102], event=[1, 2]))
    events = pd.concat([universe.iloc[:1]] * 2)
    assert counts(events, universe).tolist() == [2, 0]


def test_migration_rejects_outside_census_and_duplicate_exposure():
    from scripts.check_native_migration import counts
    universe = pd.DataFrame(dict(run=[1400], subrun=[102], event=[1]))
    with pytest.raises(ValueError, match='outside configured generator'):
        counts(universe.assign(run=1430), universe)
    with pytest.raises(ValueError, match='unique integer triples'):
        counts(universe, pd.concat([universe] * 2))


def test_c4_source_inventory_requires_complete_explicit_event_record():
    from scripts.audit_c4_ntuple import event_fields
    assert event_fields(['evtinfo/run', 'evtinfo/subrun', 'evtinfo/event'])['run'] == 'evtinfo/run'
    assert event_fields(['trkmc/run', 'evtinfo/subrun', 'evtinfo/event']) is None


def test_c4_bounded_source_replay_requires_exact_cached_track_and_coincidence_values():
    import awkward as ak
    from scripts.audit_c4_ntuple import compare_cached_records
    cached = ak.Array([dict(run=1430, subrun=0, event=1, p_mid=[100.], t_mid=[900.], coincs=dict(time=[901.]))])
    assert compare_cached_records(cached, cached)['tracks'] == 1
    changed = ak.with_field(cached, ak.Array([[901.]]), 't_mid')
    with pytest.raises(ValueError, match='momentum/time'):
        compare_cached_records(changed, cached)


def test_c4_lineage_keeps_original_track_identity_not_nearest_time():
    import awkward as ak
    from crvfastsim.veto.cosmic_ana import selected_lineage
    genealogy = ak.Array([[[dict(id=11)], [dict(id=22), dict(id=33)]]])
    mask, keep = ak.Array([[False, True]]), ak.Array([True])
    result = selected_lineage(dict(trkmc=dict(trkmcsim=genealogy)), mask, keep)
    assert ak.to_numpy(ak.flatten(result['track_index'])).tolist() == [1]
    assert ak.to_numpy(ak.flatten(result['trkmcsim'].id, axis=None)).tolist() == [22, 33]
    with pytest.raises(ValueError, match='not aligned'):
        selected_lineage(dict(trkmc=dict(trkmcsim=genealogy[:, :1])), mask, keep)


def test_c4_lineage_publication_is_exclusive_and_readable(tmp_path):
    import awkward as ak
    from scripts.audit_c4_ntuple import publish_lineage
    output = tmp_path / 'lineage.parquet'
    rows = ak.Array([dict(run=1430, track_index=[1], trkmcsim=[[dict(id=22)]])])
    evidence = publish_lineage(rows, output)
    assert evidence['sha256'] and len(ak.from_parquet(output)) == 1
    with pytest.raises(FileExistsError):
        publish_lineage(rows, output)


def test_snapshot_preserves_asset_symlink_binding(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(inputs={}, assets={})))
    old, new, link = tmp_path / 'old', tmp_path / 'new', tmp_path / 'lookup'
    old.write_text('original asset')
    new.write_text('different asset')
    link.symlink_to(old)
    from crvfastsim import geometry
    from crvfastsim.response import lookup
    monkeypatch.setattr(lookup, 'lookup_dir', lambda: str(tmp_path))
    monkeypatch.setattr(geometry, 'find_counters', lambda *a: str(old))
    for name in ('LookupTable_6000_0', 'photonMap.root'):
        (tmp_path / name).symlink_to(old)
    spec = SimpleNamespace(paths=lambda: [link])
    rc = SimpleNamespace(counters_path=lambda: str(old), sectors={})
    before = N.snapshot([old], spec, rc, old, manifest)
    assert str(link) in before and str(old) in before
    link.unlink()
    link.symlink_to(new)
    with pytest.raises(ValueError, match='drift'):
        N.verify(before)
