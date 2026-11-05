"""Small metadata and synthetic-library controls; no production physics claims."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crvfastsim import native_record, tier1b, tier1b_assets as T
from crvfastsim.analyses.e2e_bench import fingerprint
from scripts import check_native_support as D
from test_native_bundle import bundle  # strict rebuilt-census metadata fixture


@pytest.fixture
def model():
    return dict(libbins=pd.DataFrame(dict(edge=[1.])),
                libxing=pd.DataFrame(dict(xid=[0, 1, 2], bin=[0, 0, 1])),
                libdesc=pd.DataFrame(dict(xid=[0, 1, 2], edep=[1., 2., 4.])),
                libshower=pd.DataFrame(dict(apdg=[22, 22, 11], kebin=[0, 0, 2])),
                libshowersteps=pd.DataFrame(dict(shid=[0, 1, 2])),
                libgen=pd.DataFrame(dict(apdg=[22, 2112], kebin=[0, 3], ngen=[1, 5], nshower=[2, 0])),
                xgeo=dict(mu=np.array([0, 0, 1])),
                anchor=dict(axid=np.array([0, 0, 2]), n_roots=2, n_multi=1, forward_edep_share=.2),
                nmreach=pd.DataFrame(dict(cell=[0, 1, 2], ntemp=[3., 1., 0.], ncensus=[2., 0., 0.],
                                         nden=[2., 0., 0.], prob=[1., 0., 0.])))


def sources():
    return dict(donors=[dict(path='a', crossings=2), dict(path='b', crossings=1)])


def test_support_counts_empty_bins_and_raw_reach(model):
    r = D.support(model, sources(), dict(inputs=['a', 'b'], training_census=dict(total_exposure=17)))
    assert r['tables'] == {name: len(model[name]) for name in tier1b.LIB_TREES}
    b = r['bins'][0]
    assert (b['crossings'], b['source_files'], b['unique_muons'], b['desc_rows'], b['desc_edep']) == (2, 1, 1, 2, 3.)
    assert b['current_anchor_crossings'] == 1
    assert r['bins'][2]['crossings'] == 0  # Sparse bins are reported, never failed.
    assert r['donor_coverage'] == [dict(bin=0, donor=0, crossings=2), dict(bin=1, donor=1, crossings=1)]
    assert r['nearmiss'][0]['raw_ratio'] == 1.5  # Do not substitute clipped prob=1.
    assert r['nearmiss'][1]['num_positive_den_zero'] and r['nearmiss'][1]['raw_ratio'] is None
    assert not r['nearmiss'][2]['num_gt_den']
    missing = next(x for x in r['showers'] if x['apdg'] == 11)
    assert missing['absent_from_libgen'] and missing['num_positive_den_zero']
    assert next(x for x in r['showers'] if x['apdg'] == 22)['num_gt_den']
    assert r['training_census']['total_exposure'] == 17
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('change,match', [('total', 'crossing total'), ('order', 'ordered topology'),
    ('xid', 'contiguous'), ('desc', 'descendant xids'), ('anchor', 'anchor xids'), ('bin', 'crossing bins')])
def test_reconstruction_rejects_ambiguous_model(model, change, match):
    top = sources()
    if change == 'total': top['donors'][0]['crossings'] = 1
    if change == 'order': top['donors'].reverse()
    if change == 'xid': model['libxing']['xid'] = [1, 0, 2]
    if change == 'desc': model['libdesc']['xid'] = [0, 1, 4]
    if change == 'anchor': model['anchor']['axid'] = np.array([0, -1, 2])
    if change == 'bin': model['libxing']['bin'] = [0., 0., 1.]
    with pytest.raises(ValueError, match=match):
        D.support(model, top, dict(inputs=['a', 'b'], training_census={}))


@pytest.mark.parametrize('num,den', [(np.nan, 1), (1, np.inf), (-1, 1), (1, -1)])
def test_reach_rejects_invalid_counts(num, den):
    with pytest.raises(ValueError, match='reach counts'):
        D.reach(pd.DataFrame(dict(n=[num], d=[den])), 'n', 'd')


def test_strict_load_metadata_hash_and_exclusive_output(bundle, model, monkeypatch, tmp_path):
    directory, rc, _ = bundle
    monkeypatch.delenv('CRVFS_PREPARED_LIBRARY_CACHE', raising=False)
    p = directory / 'topology.npz.json'
    top = json.loads(p.read_text())
    top['donors'][0]['crossings'] = 3
    p.write_text(json.dumps(top))
    calls = []
    monkeypatch.setattr(T, 'load', lambda spec, geom: calls.append(spec) or model)
    cfg = dict(bundle=str(directory), run_config=rc.name)
    out = tmp_path / 'new.json'
    r = D.run(cfg, out)
    assert calls[0].native['training_census']['count'] == 17
    assert not r['physics_admitted'] and not r['support_closed']
    assert r['finite_library_uncertainty'] == 'open'
    assert r['native']['per_seed'] == {}
    saved = json.loads(out.read_text())
    digest = saved.pop('report_payload_sha256')
    assert digest == hashlib.sha256(json.dumps(saved, sort_keys=True, allow_nan=False).encode()).hexdigest()
    with pytest.raises(FileExistsError): D.run(cfg, out)
    for name in ('library.json', 'build_check.json'):
        p = directory / name
        meta = json.loads(p.read_text())
        meta.pop('training_census')
        p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='rebuild the library'):
        D.run(cfg, tmp_path / 'old.json')


def record(tmp_path, name='r', seed=1, event=1, library='a' * 64, clamped=2, desc=4):
    path = tmp_path / (name + '.npz')
    data = dict(generated_keys=np.array([[1400, 102, event]], dtype=np.int64),
                steps__edep=np.array([1.]), particles__key=np.array([1]), digis__chan=np.array([], dtype=int))
    np.savez(path, **data)
    asset = '/historical/library.root'
    meta = dict(schema=native_record.SCHEMA, scope=native_record.SCOPE, output_sha256=T.sha256(path),
                array_fingerprint=fingerprint(data), seed=seed, run_config=dict(name='run1a_v01'),
                counts=dict(generated_events=1, steps=1, particles=1, digis=0, clusters=0, pulses=0, members=0),
                sample_counts=dict(desc_steps=desc, desc_clamped=clamped, desc_dropped=0),
                tier1b=dict(library_sha256=library, paths=[asset], native=dict(assets_sha256={'library.root': library})),
                inputs_sha256={asset: library})
    Path(str(path) + '.json').write_text(json.dumps(meta))
    return str(path)


def test_records_historical_hash_seed_arms_and_zero_fraction(tmp_path):
    a = record(tmp_path)
    b = record(tmp_path, name='b', seed=2)
    # Make record bytes distinct without changing generated population; seed is metadata, not NPZ bytes.
    with np.load(b) as arrays: data = dict(arrays)
    data['steps__edep'] = np.array([2.])
    np.savez(b, **data)
    p = Path(b + '.json'); m = json.loads(p.read_text())
    m.update(output_sha256=T.sha256(b), array_fingerprint=fingerprint(data))
    p.write_text(json.dumps(m))
    r = D.records([a, b], 'b' * 64, 'run1a_v01')
    assert all(not x['support_bound_to_current_library'] for x in r['records'])
    assert [x['generated_events'] for x in r['per_seed'].values()] == [1, 1]
    assert r['exposure_across_seeds'] == 'not_added' and r['records'][0]['clamp_fraction'] == .5
    c = record(tmp_path, name='legacy', seed=None, event=2, desc=0, clamped=0)
    legacy = D.records([dict(path=c, seed=7)], 'a' * 64, 'run1a_v01')['records'][0]
    assert legacy['clamp_fraction'] is None and legacy['support_bound_to_current_library']
    assert legacy['seed_evidence'] == 'externally_declared_legacy_not_attested'
    with pytest.raises(ValueError, match='seed mismatch'): D.records([c], 'a' * 64, 'run1a_v01')
    with pytest.raises(ValueError, match='seed mismatch'): D.records([dict(path=a, seed=3)], 'a' * 64, 'run1a_v01')
    with pytest.raises(ValueError, match='duplicate native record hash'): D.records([a, a], 'a' * 64, 'run1a_v01')


def test_records_hashbinding_and_within_seed_exposure(tmp_path):
    a, b = record(tmp_path), record(tmp_path, name='b', desc=1)
    p = Path(b + '.json'); m = json.loads(p.read_text())
    m['inputs_sha256']['/historical/library.root'] = '0' * 64
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='asset hash declaration'): D.records([b], 'a' * 64, 'run1a_v01')
    m['inputs_sha256']['/historical/library.root'] = 'a' * 64
    p.write_text(json.dumps(m))
    with np.load(b) as arrays: data = dict(arrays)
    data['steps__edep'] = np.array([2.])
    np.savez(b, **data)
    m.update(output_sha256=T.sha256(b), array_fingerprint=fingerprint(data))
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='overlapping generated exposure'): D.records([a, b], 'a' * 64, 'run1a_v01')
    Path(a).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'): D.records([a], 'a' * 64, 'run1a_v01')
