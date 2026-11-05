"""Tiny real saved ROOT tables; strict generator census and paired-shard response controls."""
import json
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim import genassoc, runconfig
from crvfastsim.validation import paired_shards
from scripts import check_named_response_campaign as campaign


def keys(subrun=1, events=(1, 2), run=1400):
    return pd.DataFrame(dict(run=np.full(len(events), run, dtype=np.uint32),
                             subrun=np.full(len(events), subrun, dtype=np.uint32),
                             event=np.asarray(events, dtype=np.uint32)))


def write_tree(root, name, columns):
    types = {key: ('var * int64' if isinstance(value, ak.Array) else np.asarray(value).dtype)
             for key, value in columns.items()}
    root.mktree(name, types)
    if len(next(iter(columns.values()))):
        root[name].extend(columns)


def saved(path, events, *, bar=0, pe=20., time=10.):
    # Second accepted event has genuinely saved zero response.
    counts = np.zeros(len(events), dtype=np.int64)
    counts[0] = 1
    ids = {key: events[key].to_numpy() for key in campaign.tables.KEY}
    first = {key: value[:1] for key, value in ids.items()}
    base = dict(idx=np.array([0], dtype=np.int64))
    channel = dict(bar=np.array([bar]), sipm=np.array([0], dtype=np.int64))
    with uproot.recreate(path) as root:
        write_tree(root, 'crvdump/events', dict(ids, ndigi=counts, npulse=counts, ncluster=counts))
        write_tree(root, 'crvdump/digis', dict(first, **base, **channel))
        write_tree(root, 'crvdump/pulses', dict(first, **base, **channel, seq=np.array([0]),
                   flags=np.array([0]), pe=np.array([pe]), time=np.array([time]), wfidx=ak.Array([[0]])))
        write_tree(root, 'crvdump/clusters', dict(first, **base, sectortype=np.array([1]),
                   poscalc=np.array([0]), pe=np.array([pe]), start=np.array([0.]), end=np.array([1.]),
                   x=np.array([0.]), y=np.array([0.]), z=np.array([0.]), pulses=ak.Array([[0]])))
    return str(path)


def generator(path, events, *, schema=2):
    frame = events.assign(schema_version=np.uint32(schema), associations_checked=np.uint32(0),
                          full_generator_stream=np.uint32(1), gen_product_id=np.uint32(10),
                          sim_product_id=np.uint32(0), n_gen=np.int64(0),
                          n_particles=np.int64(0), n_steps=np.int64(0))
    empty = {key: np.array([], dtype=np.uint32) for key in campaign.tables.KEY + ['gen_product_id']}
    empty.update({key: np.array([], dtype=np.int64) for key in ('idx', 'pdg')})
    empty.update({key: np.array([], dtype=float) for key in ('e', 't')})
    with uproot.recreate(path) as root:
        write_tree(root, 'dump/events', {key: frame[key].to_numpy() for key in frame})
        write_tree(root, 'dump/gen', empty)
    return str(path)


@pytest.fixture
def geom():
    return SimpleNamespace(nbars=2, sector=np.array([0, 1]), sector_names=['T1', 'T2'])


def test_permuted_events_keep_full_integer_pairing_and_zeros(tmp_path, geom):
    ev = keys(events=(2**32 - 1, 1))
    a, _ = campaign.read_arm([saved(tmp_path / 'a.root', ev)], geom)
    b, _ = campaign.read_arm([saved(tmp_path / 'b.root', ev.iloc[::-1])], geom)
    campaign.admit_pair(a['events'], b['events'], ev, [])
    family = campaign.response_features(a, a['events'], geom)
    assert len(family) == 12
    assert family['all/event_clusters'].tolist()[:3] == [0, 1, 1]
    assert family['T1/pulse_pe'].sum() == 1
    assert family['T2/pulse_pe'].sum() == 0
    assert all('/step_' not in name for name in family)


@pytest.mark.parametrize('change,match', [
    ('wrong_run', 'population differs'), ('duplicate', 'unique integer'),
    ('outside_generator', 'outside generator'), ('shard_overlap', 'overlapping generator')])
def test_pair_population_admission(change, match):
    a, b, universe, seen = keys(), keys(), keys(), []
    if change == 'wrong_run':
        b = b.assign(run=np.uint32(1430))
    if change == 'duplicate':
        b = pd.concat([b, b.iloc[:1]])
    if change == 'outside_generator':
        universe = universe.assign(run=np.uint32(1430))
    if change == 'shard_overlap':
        seen.append(np.array([[1400, 1, 2]], dtype=np.int64))
    with pytest.raises(ValueError, match=match):
        campaign.admit_pair(a, b, universe, seen)


@pytest.mark.parametrize('kwargs,match', [({'bar': 2}, 'pulse bar'), ({'bar': -1}, 'integer'),
    ({'pe': float('nan')}, 'PE'), ({'pe': -1.}, 'PE'), ({'time': float('inf')}, 'time')])
def test_invalid_saved_numeric_inputs(tmp_path, geom, kwargs, match):
    with pytest.raises(ValueError, match=match):
        campaign.read_arm([saved(tmp_path / 'bad.root', keys(), **kwargs)], geom)


def test_overlapping_split_files_rejected(tmp_path, geom):
    path = saved(tmp_path / 'a.root', keys())
    with pytest.raises(ValueError, match='unique integer'):
        campaign.read_arm([path, path], geom)


def metadata(rc, conditions):
    from dataclasses import asdict, replace
    cfg = replace(rc.chain_config(jax=False, noise=True), window=runconfig.OFFSPILL_WINDOW)
    return dict(schema='crv-fasttables-named-v1', stage='chain', run_config=rc.name,
                seed=1, conditions=conditions, jax=False, noise=True,
                chain_config=json.loads(json.dumps(asdict(cfg))), effective_window_ns=[0., 100000.],
                timing_policy='fixed_absolute_crop_no_pbtime_rebase')


def test_metadata_requires_declared_actual_named_identity():
    rc, conditions = runconfig.get('run1a_v01'), {'old_or_new_strict_identity': 'explicit'}
    meta = metadata(rc, conditions)
    campaign.require_metadata(meta, rc, 1, conditions)
    for update in ({'run_config': 'extracted'}, {'seed': 2}, {'seed': True}, {'conditions': {}},
                   {'jax': True}, {'noise': False}, {'chain_config': {}}, {'diagnostic_override': {}},
                   {'effective_window_ns': [200., 1500.]}, {'timing_policy': 'onspill'}):
        with pytest.raises(ValueError, match='metadata mismatch'):
            campaign.require_metadata(dict(meta, **update), rc, 1, conditions)


def test_synthetic_paired_kernel_uses_shards_not_independent_pulse_errors():
    a = np.arange(10.) + 20
    b = a * 1.2 + np.array([0, 1, -1, 2, -2] * 2)
    result = paired_shards.compare({'counts': a[:, None]}, {'counts': b[:, None]}, nflip=500)
    ratio = b.sum() / a.sum()
    expected = np.std(b - ratio * a, ddof=1) / (np.sqrt(10) * a.mean())
    assert result['per_histogram']['counts']['yield_ratio_se'] == pytest.approx(expected)
    assert result['family']['shards'] == 10
    assert result['physics_admitted'] is False


def test_strict_generator_rejects_legacy_schema(tmp_path):
    with pytest.raises(ValueError, match='schema_version=2'):
        genassoc.validate_generator_stream(generator(tmp_path / 'bad.root', keys(), schema=1))


def test_campaign_ten_shards_differing_accepted_exposure(tmp_path, monkeypatch, geom):
    real_rc = runconfig.get('run1a_v01')
    counters = tmp_path / 'synthetic-counters.txt'
    counters.write_text('synthetic geometry supplied by fixture')
    rc = SimpleNamespace(name=real_rc.name, counters=str(counters), geometry=lambda: geom,
                         chain_config=real_rc.chain_config)
    monkeypatch.setattr(campaign.runconfig, 'get', lambda _: rc)
    cond = dict(inputs={}, assets={})
    manifest = tmp_path / 'conditions.json'
    manifest.write_text(json.dumps(cond))
    monkeypatch.setattr(campaign.named_conditions, 'load_conditions', lambda *args: (None, cond))
    pairs = []
    for index in range(10):
        universe = keys(subrun=index + 1, events=(1, 2, 3, 4))
        accepted = universe.iloc[:2 + index % 2]
        a = saved(tmp_path / f'a{index}.root', accepted)
        b = saved(tmp_path / f'b{index}.root', accepted.iloc[::-1], bar=1)
        (tmp_path / f'b{index}.root.metadata.json').write_text(json.dumps(metadata(rc, cond)))
        pairs.append(dict(generator=generator(tmp_path / f'g{index}.root', universe),
                          reference=[a], candidate=[b]))
    spec = dict(run_config=rc.name, seed=1, conditions_manifest=str(manifest),
                generated_per_pair=4, pairs=pairs)
    config, out = tmp_path / 'config.json', tmp_path / 'report.json'
    config.write_text(json.dumps(spec))
    result = campaign.run(config, out)
    assert json.loads(out.read_text())['physics_admitted'] is False
    assert result['shards'] == 10 and result['scope'] == 'same_step_saved_response_not_fullVR4'
    assert {pair['accepted'] for pair in result['pairs']} == {2, 3}
    assert result['global_cluster_count']['ratio'] == 1
    assert result['global_cluster_count']['paired_shard_ratio_se'] == 0
    assert result['producer_identity_status'] == 'external_evidence_not_attested'
    assert len(result['raw_features'][0]['all/event_clusters']) == 10
    assert sum(result['raw_features'][0]['all/event_clusters'][0]) == 2
    assert sum(result['full_generator_zero_overlay_diagnostic'][0]['all/event_clusters'][0]) == 4
    assert all('/step_' not in name for name in result['raw_features'][0])
    with pytest.raises(FileExistsError):
        campaign.run(config, out)
    spec['generated_per_pair'] = 5
    config.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='generator exposure'):
        campaign.run(config, tmp_path / 'wrong.json')
