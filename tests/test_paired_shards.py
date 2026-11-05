"""Paired shard differences, independent-arm overlays and shared-family null controls."""
import numpy as np
import pytest

from crvfastsim.validation.paired_shards import batch_variance, compare, statistic


def test_batch_variance_and_student_bias_match_formula():
    x = np.arange(40.).reshape(10, 4) % 7
    np.testing.assert_allclose(batch_variance(x), 10 * np.var(x, axis=0, ddof=1))
    assert statistic(x) == pytest.approx(7/9 * np.mean(x.sum(0)**2 / batch_variance(x)))


def test_identical_and_empty_statuses_do_not_fake_coverage():
    a = np.ones((10, 4))
    result = compare({'same': a, 'empty': a * 0, 'missing': a * 0},
                     {'same': a, 'empty': a * 0, 'missing': a}, nflip=500)
    assert result['per_histogram']['same']['status'] == 'identical'
    assert result['per_histogram']['empty']['status'] == 'both_empty'
    assert result['per_histogram']['missing']['flag'] is None
    assert result['per_histogram']['missing']['descriptive_one_empty']
    assert result['physics_admitted'] is False


def test_shared_sign_family_null_and_large_offset():
    rng = np.random.default_rng(4)
    a, b = rng.poisson(300, (30, 12)), rng.poisson(300, (30, 12))
    result = compare({'null': a, 'offset': a}, {'null': b, 'offset': b + 100}, nflip=3000)
    assert not result['per_histogram']['null']['flag']
    assert result['per_histogram']['offset']['flag']
    assert result['per_histogram']['offset']['p_family'] < .01
    assert result['family']['fitted_histograms'] == 2


def test_mixed_shard_counts_rejected():
    with pytest.raises(ValueError, match='shard count'):
        compare({'a': np.ones((10, 2)), 'b': np.ones((11, 2))},
                {'a': np.ones((10, 2)), 'b': np.ones((11, 2))}, nflip=100)


def test_native_histograms_keep_flows_and_zero_cluster_events():
    import pandas as pd
    from types import SimpleNamespace
    from scripts.compare_native_histograms import features, histogram
    universe = pd.DataFrame(dict(run=[1400, 1400], subrun=[102, 102], event=[1, 2]))
    p = universe.iloc[:1].assign(bar=0, pe=20., time=10.)
    c = universe.iloc[:1].assign(pe=10., start=10., end=11., x=0., y=0., z=0.)
    s = universe.iloc[:1].assign(bar=0, edep=1., len=10.)
    family = features(p, c, s, universe, SimpleNamespace(sector=np.array([0]), sector_names=['T1']))
    assert family['all/event_clusters'].sum() == 2
    assert family['T1/pulse_pe'].sum() == 1
    assert histogram(np.array([-1., 0., 1., 2.]), np.array([0., 1., 2.])).tolist() == [1, 1, 1, 1]


def test_sparse_one_empty_is_not_a_calibrated_rejection():
    a, b = np.zeros((10, 1)), np.zeros((10, 1))
    b[0] = 1
    r = compare({'rare': a}, {'rare': b}, nflip=500)['per_histogram']['rare']
    assert r['status'] == 'one_empty' and r['flag'] is None and r['descriptive_one_empty']


def test_histogram_population_and_seed_admission(tmp_path):
    import pandas as pd
    from scripts.compare_native_histograms import require_subset, arm_seed, common_assets
    universe = pd.DataFrame(dict(run=[1400], subrun=[102], event=[1]))
    with pytest.raises(ValueError, match='outside configured'):
        require_subset(universe.assign(run=1430), universe)
    used = set()
    assert arm_seed('seed1', {'seed': 1, 'output_sha256': 'a'}, 'one.npz', used) == 'recorded_seed'
    with pytest.raises(ValueError, match='aliases'):
        arm_seed('seed2', {'seed': 2, 'output_sha256': 'a'}, 'two.npz', used)
    with pytest.raises(ValueError, match='differs from arm'):
        arm_seed('seed2', {'seed': 1, 'output_sha256': 'b'}, 'two.npz', set())
    a = common_assets({'inputs_sha256': {'/conditions/calib.csv': '1', '/generators/gen.root': 'a'}})
    b = common_assets({'inputs_sha256': {'/conditions/calib.csv': '2', '/generators/gen.root': 'b'}})
    assert a != b and '/generators/gen.root' not in a
