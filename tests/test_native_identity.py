"""Tiny stage-local controls; these helpers do not admit multi-muon construction."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim import native_identity as ni, tier1b
from crvfastsim.tables import KEY


def _steps(rows):
    s = pd.DataFrame(rows, columns=["event", "sp", "bar", "t0", "x0", "edep"])
    for c in ("event", "sp", "bar"):
        s[c] = s[c].astype(np.int64)
    return s.assign(run=np.int64(1400), subrun=np.int64(101), y0=0., z0=0.,
                    x1=s.x0 + 2., y1=1., z1=2., px0=3., py0=4., pz0=0., len=2.)


def _classified(s, prim_key):
    d = s.copy(deep=True)
    d["prim_key"] = np.asarray(prim_key, dtype=np.int64)
    d["cls"] = "mu_descendant"
    return d


def _historical_scores(d, x):
    d = d.reset_index(drop=True).assign(did=np.arange(len(d), dtype=np.int64))
    x = x.assign(xid=np.arange(len(x), dtype=np.int64))
    m = d[KEY + ["did", "bar", "x0", "y0", "z0", "x1", "y1", "z1"]].merge(
        x[KEY + ["xid", "bar", "x0", "y0", "z0"]].rename(columns={
            "bar": "xbar", "x0": "ex", "y0": "ey", "z0": "ez"}), on=KEY)
    mid = .5 * (m[["x0", "y0", "z0"]].to_numpy() + m[["x1", "y1", "z1"]].to_numpy())
    dist = np.linalg.norm(mid - m[["ex", "ey", "ez"]].to_numpy(), axis=1)
    m["score"] = np.where(m.bar.to_numpy() == m.xbar.to_numpy(), -1., dist)
    return m.loc[m.groupby("did").score.idxmin(), ["did", "xid", "score"]].reset_index(drop=True)


def test_samebar_two_muons_separate_sums_entry_and_exit():
    s = _steps([(1, 7, 0, 5., 50., 2.), (1, 9, 0, 1., 100., 3.),
                (1, 7, 0, 2., 20., 4.), (1, 9, 0, 9., 900., 5.)])
    s["len"] = [3., 5., 7., 11.]
    s["px0"] = [30., 6., 3., 60.]
    s["y0"], s["z0"] = [5., 10., 2., 90.], [50., 100., 20., 900.]
    s["y1"], s["z1"] = [6., 11., 3., 91.], [51., 101., 21., 901.]
    before = s.copy(deep=True)
    x = ni.crossings(s)
    assert x.sp.to_numpy().tolist() == [9, 7]
    assert x.prim_key.equals(x.sp)
    np.testing.assert_array_equal(x["len"], [16., 10.])
    np.testing.assert_array_equal(x.edep, [8., 6.])
    np.testing.assert_array_equal(x.t0, [1., 2.])
    np.testing.assert_array_equal(x.x0, [100., 20.])
    np.testing.assert_array_equal(x.x1, [902., 52.])
    np.testing.assert_array_equal(x.y0, [10., 2.])
    np.testing.assert_array_equal(x.z0, [100., 20.])
    np.testing.assert_array_equal(x.y1, [91., 6.])
    np.testing.assert_array_equal(x.z1, [901., 51.])
    np.testing.assert_allclose(x.p, [np.sqrt(52.), 5.])
    np.testing.assert_array_equal(x.xid, [0, 1])
    pd.testing.assert_frame_equal(s, before)


def test_equal_time_entry_and_exit_follow_input_order():
    s = _steps([(1, 7, 0, 1., 20., 1.), (1, 7, 0, 1., 10., 2.)])
    x = ni.crossings(s)
    assert x.x0.iloc[0] == 20.
    assert x.x1.iloc[0] == 12.


def test_pick_nearest_wrong_primary_trap_and_wrong_samebar_trap():
    x = ni.crossings(_steps([(1, 7, 0, 10., 0., 2.), (1, 9, 1, 1., 100., 3.)]))
    d = _classified(_steps([(1, 8, 2, 11., 100., .2), (1, 10, 3, 12., 0., .3),
                             (1, 11, 1, 13., 0., .7)]), [7, 9, 7])
    before_d, before_x = d.copy(deep=True), x.copy(deep=True)
    attached, unattached = ni.attach_descendants(d, x)
    assert unattached.empty
    np.testing.assert_array_equal(attached.selected_primary, [7, 9, 7])
    np.testing.assert_array_equal(attached.xbar, [0, 1, 0])
    assert (attached.score > 0).all()
    pd.testing.assert_frame_equal(d, before_d)
    pd.testing.assert_frame_equal(x, before_x)


def test_no_own_crossing_stays_unattached_even_in_crossed_event():
    x = ni.crossings(_steps([(1, 7, 0, 1., 0., 1.)]))
    d = _classified(_steps([(1, 8, 0, 2., 0., .2), (2, 8, 0, 2., 0., .3),
                             (1, 10, 0, 2., 0., .4)]), [9, 7, -1])
    attached, unattached = ni.attach_descendants(d, x)
    assert attached.empty
    np.testing.assert_array_equal(unattached.sp, d.sp)
    assert unattached.edep.sum() == pytest.approx(.9)
    assert unattached.prim_key.dtype == np.int64


def test_samebar_priority_within_ancestor_and_deterministic_distance_ties():
    x = ni.crossings(_steps([(1, 7, 3, 1., 100., 1.), (1, 7, 1, 1., -10., 1.),
                             (1, 7, 2, 1., 10., 1.)]))
    x["xid"] = np.array([99, 1, 0], dtype=np.int64)
    d = _classified(_steps([(1, 8, 3, 2., -1., .2), (1, 8, 0, 2., -1., .3)]), [7, 7])
    d[["y0", "y1", "z0", "z1"]] = 0.
    attached, _ = ni.attach_descendants(d, x)
    np.testing.assert_array_equal(attached.xbar, [3, 1])
    np.testing.assert_array_equal(attached.score, [-1., 10.])
    np.testing.assert_array_equal(attached.xid, [0, 99])


def test_large_adjacent_int64_primary_and_complete_event_keys():
    b = 2**60
    s = _steps([(1, b + 7, 0, 1., 0., 1.), (1, b + 8, 0, 2., 100., 2.),
                (1, b + 7, 0, 1., 200., 3.), (1, b + 7, 0, 1., 300., 4.),
                (2, b + 7, 0, 1., 400., 5.)])
    s.loc[2, "subrun"] = 102
    s.loc[3, "run"] = 1401
    x = ni.crossings(s)
    d = _classified(s.assign(sp=np.int64(b + 9)), [b + 8, b + 7, b + 7, b + 7, b + 7])
    attached, unattached = ni.attach_descendants(d, x)
    assert unattached.empty and len(x) == 5
    np.testing.assert_array_equal(attached.selected_primary, d.prim_key)
    selected = attached[KEY + ["xid"]].merge(x[KEY + ["xid", "x0"]], on=KEY + ["xid"])
    np.testing.assert_array_equal(selected.x0, [100., 0., 200., 300., 400.])
    assert attached.selected_primary.dtype == np.int64
    assert attached.prim_key.dtype == np.int64
    assert x.sp.dtype == np.int64


def test_single_muon_matches_historical_crossings_and_score():
    s = _steps([(2, 7, 2, 4., 12., 2.), (1, 7, 1, 3., 30., 3.),
                (1, 7, 0, 1., 0., 4.), (1, 7, 0, 2., 2., 5.)])
    old = tier1b._crossings(s)
    x = ni.crossings(s)
    pd.testing.assert_frame_equal(x[old.columns], old)
    d = _classified(_steps([(1, 8, 1, 10., 1., .2), (1, 9, 2, 11., 5., .3),
                             (2, 8, 3, 12., 15., .4)]), [7, 7, 7])
    attached, unattached = ni.attach_descendants(d, x)
    assert unattached.empty
    pd.testing.assert_frame_equal(attached[["did", "xid", "score"]], _historical_scores(d, old))


@pytest.mark.parametrize("which", ["both", "desc", "crossings", "unclassified_empty"])
def test_empty_inputs(which):
    s = _steps([(1, 7, 0, 1., 0., 1.)])
    x = ni.crossings(s.iloc[:0] if which in ("both", "crossings") else s)
    d = _classified(s, [7])
    if which in ("both", "desc"):
        d = d.iloc[:0]
    if which == "unclassified_empty":
        d = pd.DataFrame()
    a, u = ni.attach_descendants(d, x)
    assert a.empty
    assert len(u) == (1 if which == "crossings" else 0)
    assert a.xid.dtype == np.int64
    assert a.score.dtype == np.float64
    assert ni.crossings(pd.DataFrame()).sp.dtype == np.int64


def test_classified_non_descendants_are_not_attached_or_unattached():
    s = _steps([(1, 7, 0, 1., 0., 1.)])
    a, u = ni.attach_descendants(_classified(s, [7]).assign(cls="mu_primary"), ni.crossings(s))
    assert a.empty and u.empty


@pytest.mark.parametrize("column,value", [("sp", 7.), ("event", "1"), ("bar", True),
                                         ("sp", np.uint64(2**63))])
def test_crossings_reject_invalid_integers(column, value):
    s = _steps([(1, 7, 0, 1., 0., 1.)])
    s[column] = value
    with pytest.raises(ValueError, match="integer|int64"):
        ni.crossings(s)


@pytest.mark.parametrize("bad", ["missing", "float", "null", "mismatch"])
def test_ancestry_is_required_and_never_inferred(bad):
    s = _steps([(1, 7, 0, 1., 0., 1.)])
    if bad == "mismatch":
        with pytest.raises(ValueError, match="prim_key must equal sp"):
            ni.crossings(s.assign(prim_key=np.int64(9)))
        return
    d = _classified(s, [7])
    if bad == "missing":
        d = d.drop(columns="prim_key")
    elif bad == "float":
        d["prim_key"] = 7.
    else:
        d["prim_key"] = pd.array([None], dtype="Int64")
    with pytest.raises(ValueError, match="ancestry|integer"):
        ni.attach_descendants(d, ni.crossings(s))


def test_nullable_integer_ancestry_without_missing_values_is_exact():
    s = _steps([(1, 2**60 + 7, 0, 1., 0., 1.)])
    d = _classified(s, [2**60 + 7])
    d["prim_key"] = d.prim_key.astype("Int64")
    a, _ = ni.attach_descendants(d, ni.crossings(s))
    assert a.selected_primary.iloc[0] == 2**60 + 7


def test_duplicate_crossing_ids_refuse_even_if_other_primaries():
    s = _steps([(1, 7, 0, 1., 0., 1.), (1, 9, 1, 2., 10., 1.)])
    x = ni.crossings(s).assign(xid=np.int64(0))
    with pytest.raises(ValueError, match="duplicate crossing ids"):
        ni.attach_descendants(_classified(s, [7, 9]), x)


@pytest.mark.parametrize("bad", ["not-a-number", np.nan, np.inf])
@pytest.mark.parametrize("side", ["descendant", "crossing", "primary"])
def test_nonnumeric_or_nonfinite_score_data_refuses(side, bad):
    s = _steps([(1, 7, 0, 1., 0., 1.)])
    d, x = _classified(s, [7]), ni.crossings(s)
    if side == "primary":
        with pytest.raises(ValueError, match="numeric"):
            ni.crossings(s.assign(x0=bad))
    else:
        if side == "descendant":
            d["x0"] = bad
        else:
            x["x0"] = bad
        with pytest.raises(ValueError, match="numeric"):
            ni.attach_descendants(d, x)


def test_full_nearmiss_bridge_explicitly_refuses_without_fake_receipt():
    with pytest.raises(NotImplementedError, match="authoritative source mapping") as error:
        ni.require_full_nearmiss_bridge()
    assert "index joins" in str(error.value)
    assert "proximity" in str(error.value)
    assert "guard remains" in str(error.value)


def test_two_muons_samebar_attach_only_own_ancestor_and_keep_duplicate_input_rows():
    s = _steps([(1, 7, 0, 1., 0., 1.), (1, 9, 0, 2., 100., 2.)])
    x = ni.crossings(s)
    d = _classified(_steps([(1, 8, 0, 3., 100., .2), (1, 10, 0, 3., 0., .3),
                             (1, 8, 0, 3., 100., .2)]), [7, 9, 7])
    d.index = [5, 5, 5]
    a, u = ni.attach_descendants(d, x)
    assert u.empty and len(a) == 3
    np.testing.assert_array_equal(a.selected_primary, [7, 9, 7])
    np.testing.assert_array_equal(a.xid, [0, 1, 0])
    np.testing.assert_array_equal(a.score, [-1., -1., -1.])
    np.testing.assert_array_equal(a.did, [0, 1, 2])


@pytest.mark.parametrize("side,column", [("descendant", "event"), ("descendant", "prim_key"),
                                         ("crossing", "xid"), ("crossing", "sp")])
def test_attachment_rejects_float_identity_even_when_integral(side, column):
    s = _steps([(1, 7, 0, 1., 0., 1.)])
    d, x = _classified(s, [7]), ni.crossings(s)
    frame = d if side == "descendant" else x
    frame[column] = frame[column].astype(np.float64)
    with pytest.raises(ValueError, match="exact integer"):
        ni.attach_descendants(d, x)


def test_empty_crossings_have_typed_complete_output_schema():
    x = ni.crossings(pd.DataFrame())
    for c in KEY + ["sp", "prim_key", "bar", "xid"]:
        assert x[c].dtype == np.int64
    for c in ["t0", "x0", "y0", "z0", "x1", "y1", "z1", "p", "len", "edep"]:
        assert x[c].dtype == np.float64
