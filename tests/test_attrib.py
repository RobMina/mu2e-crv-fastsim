"""valhist.attrib: arm parsing, hybrid tables and the effect decomposition, on synthetic tables."""
import numpy as np
import pandas as pd
import pytest

from valhist import attrib as A


def table(cls, t0, root=None):
    n = len(cls)
    sp = np.arange(n) + 1
    d = {"run": np.full(n, 1400, np.uint32), "subrun": np.zeros(n, np.uint32), "event": np.ones(n, np.uint32),
         "nstep": np.full(n, n), "bar": np.arange(n), "sp": np.array(sp), "t0": np.array(t0, float), "edep": np.ones(n),
         "cls": np.array(cls, np.int8), "root": np.array(root if root is not None else np.ones(n), np.int64),
         "pdg": np.full(n, 13, np.int32)}
    return pd.DataFrame(d)


def test_parse_arm():
    assert A.parse_arm("gfg@2") == ("g", "f", "g", 2)
    assert A.parse_arm("g--") == ("g", "-", "-", 0)
    for bad in ("gg", "rgg", "ggr", "xgg"):
        with pytest.raises(ValueError):
            A.parse_arm(bad)


def test_shard_list():
    assert A.shard_list("22-26") == [22, 24, 26]
    assert A.shard_list("22,30") == [22, 30]


def test_pure_arm_keeps_row_order_and_keys():
    g = table([1, 0, 2], [5.0, 1.0, 3.0], [1, 1, 0])
    s, parts = A.build_arm("ggg", {"g": g, "f": g.iloc[:0]})
    assert s.t0.tolist() == [5.0, 1.0, 3.0]
    assert s.sp.tolist() == g.sp.tolist()
    assert len(parts) == 3


def test_hybrid_takes_each_class_from_its_source_with_distinct_keys():
    g = table([0, 1, 2], [1.0, 2.0, 3.0])
    f = table([0, 1, 2], [1.5, 2.5, 3.5])
    s, parts = A.build_arm("gfg", {"g": g, "f": f})
    assert s.t0.tolist() == [1.0, 2.5, 3.0]
    assert s.sp.iloc[1] == 2 + A.SOURCE_OFFSET["f"]
    assert s.nstep.tolist() == [3, 3, 3]
    assert not parts.duplicated(["event", "key"]).any()


def test_dropped_class():
    g = table([0, 1, 2], [1.0, 2.0, 3.0])
    s, _ = A.build_arm("g--", {"g": g, "f": g.iloc[:0]})
    assert s.cls.tolist() == [0]


def test_effects_sum_to_total():
    rng = np.random.default_rng(1)
    arms = ("ggg", "fff", "fgg", "gfg", "ggf", "ffg", "fgf", "gff")
    h = {a: rng.poisson(100, (5, 8)) for a in arms}
    eff, total = A.decompose(h)
    assert np.array_equal(sum(eff.values()), total)


def test_signal_subtracts_noise():
    rng = np.random.default_rng(2)
    null = rng.normal(0, 3, (400, 50))
    wt = np.ones(50)
    assert abs(A.signal(null, wt)) < 3 * 400 * 9 * np.sqrt(2 * 50)
    shifted = null + 1.0
    assert A.signal(shifted, wt) == pytest.approx((400 * 1.0) ** 2 * 50, rel=0.15)


def test_histogram_overflow_last():
    h = A.histogram([1.0, 25.0, 1999.0, 2000.0, 5000.0])
    assert h[0] == 1 and h[1] == 1 and h[99] == 1 and h[-1] == 2 and h.sum() == 5


def test_flip_null_centres_on_one_and_sees_a_shift():
    rng = np.random.default_rng(3)
    d = rng.normal(0, 3, (50, 30))
    obs, ndf, mean, sd, p = A.flip_null(d, nflip=4000)
    assert ndf == 30 and abs(mean - 1.0) < 0.05 and p > 1e-3
    assert A.flip_null(d + 3.0, nflip=4000)[4] < 1e-2


def test_parse_new_letters():
    assert A.parse_arm("gkg") == ("g", "k", "g", 0) and A.parse_arm("sgf") == ("s", "g", "f", 0)
    assert A.draw_key(A.parse_arm("fkf")) == ("k", "f") and A.draw_key(A.parse_arm("stg")) == ("t", "g")
    for bad in ("ftf", "-kg", "gsg", "kgg"):
        with pytest.raises(ValueError):
            A.parse_arm(bad)


def crossings(p, t0, sp=1, bar=None):
    n = len(p)
    return pd.DataFrame({"run": np.full(n, 1400, np.uint32), "subrun": np.zeros(n, np.uint32),
                         "event": np.ones(n, np.uint32), "sp": np.full(n, sp), "t0": np.array(t0, float),
                         "px0": np.zeros(n), "py0": -np.array(p, float), "pz0": np.zeros(n),
                         "bar": np.arange(n) if bar is None else np.array(bar),
                         "x0": np.zeros(n), "y0": -10.0 * np.arange(n), "z0": np.zeros(n)})


def test_loss_to_next_in_time_order():
    x = crossings([900.0, 1000.0, 950.0], [2.0, 0.0, 1.0])
    v = A.loss_to_next(x)
    assert v[0] == A.LAST_CROSSING
    assert v[1] == pytest.approx(A.tkin(1000.0) - A.tkin(950.0))
    assert v[2] == pytest.approx(A.tkin(950.0) - A.tkin(900.0))


def test_attached_energy_same_bar_first_then_nearest_entry():
    x = crossings([1000.0, 1000.0], [0.0, 1.0], bar=[5, 6])
    d = pd.DataFrame({"run": np.full(3, 1400, np.uint32), "subrun": np.zeros(3, np.uint32),
                      "event": np.ones(3, np.uint32), "bar": [6, 9, 9], "edep": [1.0, 2.0, 4.0],
                      "x0": [0.0, 0.0, 0.0], "x1": [0.0, 0.0, 0.0], "z0": [0.0, 0.0, 0.0], "z1": [0.0, 0.0, 0.0],
                      "y0": [-50.0, -1.0, -9.0], "y1": [-50.0, -1.0, -9.0]})
    assert np.allclose(A.attached_energy(x, d), [2.0, 5.0])


def test_first_draw_returns_given_numbers_once():
    rng = A._FirstDraw(np.random.default_rng(1), np.array([0.25, 0.75]))
    assert np.array_equal(rng.random(2), [0.25, 0.75])
    assert rng.random(3).shape == (3,) and rng.integers(5) < 5
    with pytest.raises(ValueError):
        A._FirstDraw(np.random.default_rng(1), np.array([0.5])).random(2)


def test_write_table_round_trip(tmp_path):
    pytest.importorskip("uproot")
    from crvfastsim import tables
    t = table([A.P, A.D, A.S], [1.0, 2.0, 3.0])
    for c in A.STEP_COLS:
        if c not in t.columns:
            t[c] = 0.5
    steps, parts = A.build_arm("ggg", {"g": t})
    A.write_table(tmp_path / "ggg.root", steps, parts)
    s = tables.read(str(tmp_path / "ggg.root"), "steps")
    q = tables.read(str(tmp_path / "ggg.root"), "particles")
    assert len(s) == 3 and sorted(q.key) == [1, 2, 3] and q.set_index("key").isprim.to_dict() == {1: 1, 2: 0, 3: 0}
