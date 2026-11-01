"""The deck flag rule on stephist accumulators: variance choice per key, clustering factor, floor threshold."""
import numpy as np
import pytest

from valhist import deck_rule as R
from valhist import stephist as H


def _acc(n, w2=None):
    n = np.asarray(n, float)
    return {"all": (n, n.copy() if w2 is None else np.asarray(w2, float))}


def _nstep(counts_at_centres):
    nb = H.BOOK["NStep"][0] + 2
    v = np.zeros(nb)
    for centre, count in counts_at_centres.items():
        v[1 + int(np.floor((centre + 9.5) / 10.0))] = count
    return v


def test_clustering_factor_uses_bin_centres_above_zero():
    v = _nstep({-4.5: 1000, 5.5: 3, 15.5: 1})
    assert R.clustering_factor(v) == pytest.approx((3 * 5.5 ** 2 + 15.5 ** 2) / (3 * 5.5 + 15.5))


def test_variance_kind_per_key():
    assert [R.variance_kind(k) for k in ("t", "t2", "lt", "posx", "EX/y", "bar", "E")] == \
        ["clustered_k", "clustered_k", "clustered_k", "exact", "exact", "poisson", "poisson"]


def test_compare_applies_k_exact_and_floor():
    nb = 102
    a, b = np.zeros(nb), np.zeros(nb)
    a[1:4], b[1:4] = [100, 100, 100], [130, 70, 100]
    ns = _nstep({5.5: 10, 25.5: 10})
    k = R.clustering_factor(ns)
    ref = {"t": _acc(a), "bar": _acc(a), "posx": _acc(a, 4 * a), "NStep": _acc(ns), "_total/all": _acc([1.0])}
    cand = {"t": _acc(b), "bar": _acc(b), "posx": _acc(b, 4 * b), "NStep": _acc(ns), "_total/all": _acc([1.0])}
    poisson = (900 / 230 + 900 / 170) / 3
    res = R.compare(ref, cand)
    assert "_total/all" not in res
    assert res["bar"]["chi2ndf"] == pytest.approx(poisson) and res["bar"]["flag"]
    assert res["t"]["chi2ndf"] == pytest.approx(poisson / k) and res["t"]["k"] == [k, k]
    assert res["posx"]["chi2ndf"] == pytest.approx(poisson / 4)
    floor = R.compare(ref, cand, ref, {**cand, "bar": _acc(a + np.r_[0, 40, -40, 0, np.zeros(nb - 4)])})
    assert floor["bar"]["floor_chi2ndf"] == pytest.approx((1600 / 240 + 1600 / 160) / 3)
    assert floor["bar"]["threshold"] == pytest.approx(3 * floor["bar"]["floor_chi2ndf"])
    assert not floor["bar"]["flag"]
