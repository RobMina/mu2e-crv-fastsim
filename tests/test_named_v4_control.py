"""Named component statistics keep per-sector and full-event zero support explicit."""
import numpy as np
import pandas as pd
import pytest

from scripts.check_named_v4 import compare, output_arrays


def test_zero_response_event_and_absent_sector_not_pass():
    universe = pd.DataFrame(dict(run=[1400, 1400], subrun=[102, 102], event=[1, 2]))
    ref = dict(event=np.repeat(0, 30), channel=np.zeros(30, np.int64), pe=np.ones(30), delay=np.arange(30.))
    cand = {k: v.copy() for k, v in ref.items()}
    result = compare(ref, cand, universe, np.array([0, 1]), ['T1', 'T2'], np.ones(2, bool))
    assert result['T1']['mean_pe_reference'] == 15.
    assert result['T1']['pe_relative_se'] == 0. and result['T1']['metrics_pass']
    assert result['T2']['status'] == 'unsupported' and not result['T2']['metrics_pass']


def test_component_indices_full_triples_and_physical_mask():
    universe = pd.DataFrame(dict(run=[1400, 1401], subrun=[102, 102], event=[1, 1]))
    steps = universe.assign(bar=[0, 1], t0=[10., 20.])
    result = output_arrays(np.array([0, 1]), np.array([0, 4]), np.array([11., 22.]), np.ones(2),
                           steps, universe, np.array([True] * 4 + [False] * 4))
    assert result['event'].tolist() == [0] and result['delay'].tolist() == [1.]
    with pytest.raises(ValueError, match='wrong bar'):
        output_arrays(np.array([0]), np.array([4]), np.array([11.]), np.ones(1), steps, universe, np.ones(8, bool))


def test_request_identity_is_bitwise_including_nan_payload():
    from scripts.check_named_v4 import identical_arrays
    a = np.array([np.nan, 1.])
    assert identical_arrays(a, a.copy())
    assert not identical_arrays(a, np.array([np.nan, 2.]))
    assert not identical_arrays(np.array([0.]), np.array([-0.]))


def test_component_failure_serializes_without_numpy_boolean():
    import json
    universe = pd.DataFrame(dict(run=[1400, 1400], subrun=[102, 102], event=[1, 2]))
    ref = dict(event=np.repeat(0, 30), channel=np.zeros(30, np.int64), pe=np.ones(30), delay=np.arange(30.))
    cand = {**ref, 'pe': np.ones(30) * 2}
    result = compare(ref, cand, universe, np.array([0]), ['T1'], np.ones(1, bool))
    assert not result['T1']['metrics_pass']
    json.dumps(result, allow_nan=False)
