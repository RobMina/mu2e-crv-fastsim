"""Paired diagnostic moments retain zero-step events and native sector names."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.check_native_secondaries import moments


def test_summary_repeated_seed_is_not_added_exposure():
    import copy
    import pytest
    from scripts.summarize_native_secondaries import summarize
    base = dict(scope='truth_primary_conditioned_secondary_diagnostic_not_transport_or_E2E_validation',
                run_config='run1a_v01', tier1b={'id': 'same'}, generated=2, sample_counts={'desc_clamped': 1},
                generator={'path': 'shard102'}, seed=1,
                paired={'T1/mu_descendant': {name: dict(reference_sum=2., candidate_sum=4., paired_se=1.)
                                             for name in ('energy', 'steps')}})
    other = copy.deepcopy(base)
    other['seed'] = 2
    result = summarize([base, other])
    assert all(row['generated'] == 2 for row in result['seeds'].values())
    with pytest.raises(ValueError, match='duplicate generator'):
        summarize([base, copy.deepcopy(base)])


def test_full_generator_census_includes_zero_events():
    universe = pd.DataFrame(dict(run=[1400, 1400], subrun=[102, 102], event=[1, 2]))
    a = universe.iloc[:1].assign(bar=0, cls='mu_descendant', edep=2.)
    b = universe.iloc[:1].assign(bar=0, cls='mu_descendant', edep=4.)
    geom = SimpleNamespace(sector=np.array([0]), sector_names=['T1'])
    result = moments(a, b, universe, geom)
    energy = result['T1/mu_descendant']['energy']
    assert energy['reference_per_generated'] == 1.
    assert energy['candidate_per_generated'] == 2.
    assert energy['paired_delta_per_generated'] == 1. and energy['paired_se'] == 1.
    assert result['T1/nonmu_primary']['steps']['reference_sum'] == 0
