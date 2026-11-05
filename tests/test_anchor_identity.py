"""Anchor grouping uses complete particle identities rather than a bounded arithmetic packing."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import tier1b as T


def tiny(roots):
    geom = SimpleNamespace(nbars=4, sector=np.zeros(4, dtype=np.int64), module=np.zeros(4, dtype=np.int64),
                           layer=np.array([0, 1, 0, 1]), counter=np.zeros(4, dtype=np.int64),
                           local_to_world=np.tile([1, 2, 0], (4, 1)), center=np.array([[0., y, 0.] for y in [1, 0, 1, 0]]))
    lib = dict(libxing=pd.DataFrame(dict(xid=np.arange(4), bin=0, ctx=0)),
               libdesc=pd.DataFrame(dict(xid=[0, 1, 2, 3], root=roots, edep=1.)),
               xgeo=dict(mu=np.array([0, 0, 1, 1]), nxt=np.array([1, -1, 3, -1]), bar=np.arange(4)))
    return lib, geom


@pytest.mark.parametrize('scope', ['module', 'muon'])
@pytest.mark.parametrize('large', [2**31 + 7, 2**60 + 7, np.iinfo(np.int64).max])
def test_large_roots_remain_separate_across_muons(scope, large):
    small = large - 2**31
    lib, geom = tiny(np.array([large, large, small, small], dtype=np.int64))
    anchor = T.attach_anchors(lib, geom, scope)['anchor']
    assert anchor['n_roots'] == 2 and anchor['n_multi'] == 2
    np.testing.assert_array_equal(anchor['axid'], [0, 0, 2, 2])
    np.testing.assert_array_equal(anchor['dk'], [0, 1, 0, 1])


@pytest.mark.parametrize('scope', ['module', 'muon'])
def test_anchor_identity_relabeling_does_not_change_placement(scope):
    small, geom = tiny(np.array([7, 7, 8, 8], dtype=np.int64))
    large, _ = tiny(np.array([2**60 + 7, 2**60 + 7, 2**60 + 8, 2**60 + 8], dtype=np.int64))
    a, b = T.attach_anchors(small, geom, scope)['anchor'], T.attach_anchors(large, geom, scope)['anchor']
    for name in a:
        np.testing.assert_equal(a[name], b[name])


@pytest.mark.parametrize('roots', [np.array([1., 1., 2., 2.]), np.array([-1, -1, 2, 2]),
                                    np.array([2**63, 2**63, 2, 2], dtype=np.uint64)])
def test_invalid_root_identity_is_rejected(roots):
    lib, geom = tiny(roots)
    with pytest.raises(ValueError, match='anchor roots'):
        T.attach_anchors(lib, geom)
