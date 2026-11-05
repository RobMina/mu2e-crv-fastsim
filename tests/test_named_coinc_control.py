"""Exact membership comparison retains empty events and integer identities."""
import awkward as ak
import numpy as np
import pandas as pd
import pytest

from scripts.check_named_coinc import compare, signatures


def case():
    events = pd.DataFrame(dict(run=[1400, 1400], subrun=[102, 102], event=[1, 2]))
    clusters = events.iloc[:1].assign(sectortype=1)
    output = dict(cl_event=np.array([(102 << 20) | 1]), cl_type=np.array([1]),
                  mem_cl=np.array([0, 0]), mem_idx=np.array([7, 3]))
    return events, clusters, ak.Array([[3, 7]]), output


def test_membership_order_independent_and_zero_event_retained():
    result = compare(*case())
    assert result['events'] == 2 and result['identical_cluster_lists'] and result['matched'] == 1


def test_wrong_member_rejected():
    events, clusters, ids, output = case()
    output['mem_idx'][0] = 8
    assert not compare(events, clusters, ids, output)['identical_cluster_lists']


def test_event_outside_census_rejected():
    events, clusters, ids, output = case()
    output['cl_event'][0] = (103 << 20) | 1
    with pytest.raises(ValueError, match='outside stock'):
        compare(events, clusters, ids, output)


def test_empty_clusters_preserve_the_event_universe():
    events, clusters, ids, output = case()
    clusters = clusters.iloc[:0]
    output = {key: np.array([], dtype=np.int64) for key in output}
    result = compare(events, clusters, ak.Array([[]])[:0], output)
    assert result['events'] == 2 and result['identical_cluster_lists'] and result['matched'] == 0


def test_duplicate_cluster_rejected():
    with pytest.raises(ValueError, match='duplicate cluster'):
        signatures(np.array([[1, 2, 3], [1, 2, 3]]), np.ones(2), ak.Array([[7], [7]]), 1)
