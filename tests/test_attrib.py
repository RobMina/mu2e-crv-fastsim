"""valhist.attrib: arm parsing, hybrid tables and the effect decomposition, on synthetic tables."""
import numpy as np
import pandas as pd
import pytest

from valhist import attrib as A


def table(cls, sp, t0, root=None):
    n = len(cls)
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
    assert s.sp.iloc[1] == 1 + A.SOURCE_OFFSET["f"]
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
    assert abs(A.signal(null, wt)) < 0.15 * 400 * 9 * 50
    shifted = null + 1.0
    assert A.signal(shifted, wt) == pytest.approx((400 * 1.0) ** 2 * 50, rel=0.15)


def test_histogram_overflow_last():
    h = A.histogram([1.0, 25.0, 1999.0, 2000.0, 5000.0])
    assert h[0] == 1 and h[1] == 1 and h[99] == 1 and h[-1] == 2 and h.sum() == 5
