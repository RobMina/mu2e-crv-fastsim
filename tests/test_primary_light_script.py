"""scripts/primary_light.py on synthetic inputs: crossing sums, histogram statistics, cluster-PE report and share."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("primary_light", SCRIPTS / "primary_light.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)


def _steps(rows):
    cols = pl.K + ["bar", "sp", "edep", "len", "px0", "py0", "pz0", "p1", "t0", "t1", "x0", "y0", "z0", "x1", "y1", "z1"]
    key = {k: 0 for k in pl.K}
    return pd.DataFrame([{**key, **r} for r in rows])[cols]


def test_crossings_sum_steps_per_muon_and_bar():
    base = dict(px0=0.0, py0=0.0, pz0=1000.0, p1=990.0, x0=0.0, y0=0.0, x1=0.0, y1=0.0)
    s = _steps([dict(base, bar=5, sp=1, edep=1.0, len=8.0, t0=1.0, t1=2.0, z0=0.0, z1=8.0),
                dict(base, bar=5, sp=1, edep=2.0, len=12.0, t0=2.0, t1=3.0, z0=8.0, z1=20.0, p1=980.0),
                dict(base, bar=6, sp=1, edep=4.0, len=20.0, t0=5.0, t1=6.0, z0=50.0, z1=70.0)])
    c = pl.crossings(s).set_index("bar")
    assert len(c) == 2
    assert c.loc[5, "edep"] == pytest.approx(3.0) and c.loc[5, "len"] == pytest.approx(20.0)
    assert c.loc[5, "nst"] == 2 and c.loc[5, "edmax"] == pytest.approx(2.0)
    assert c.loc[5, "chord"] == pytest.approx(20.0) and c.loc[5, "p1"] == pytest.approx(980.0)
    assert c.loc[5, "pin"] == pytest.approx(1000.0) and c.loc[6, "nst"] == 1


def test_stats_on_a_landau_like_histogram():
    rng = np.random.default_rng(3)
    v = 1.6 + 0.12 * rng.gumbel(size=400000)
    h = np.histogram(v, pl.VBINS)[0]
    mean, med, mpv, width, fwhm = pl.stats(h, len(v), v.sum())
    assert mean == pytest.approx(v.mean())
    assert med == pytest.approx(np.median(v), abs=0.01)
    assert mpv == pytest.approx(1.6, abs=0.02)                   # Gumbel mode
    assert 0.1 < width < 0.2 and 0.2 < fwhm < 0.4


def _pe_shards(d, arm, scale, n=6, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        cpe = rng.gamma(9.0, 90.0, 4000)
        np.savez(d / f"{arm}.001400_{22 + 2 * i:08d}.npz", ev=np.arange(len(cpe)), cpe=cpe * scale, sec=np.zeros(len(cpe)),
                 pev=np.zeros(3), bar=np.zeros(3), sipm=np.zeros(3), ppe=np.ones(3) * scale, ptime=np.zeros(3),
                 pseq=np.zeros(3))


def test_pe_report_measures_a_scale_shift(tmp_path):
    _pe_shards(tmp_path, "ref", 1.0)
    _pe_shards(tmp_path, "arm", 1.01)
    pl.pe_report(str(tmp_path / "r.json"), "ref", ["arm"], str(tmp_path))
    r = json.load(open(tmp_path / "r.json"))
    assert r["shards"] == 6
    assert r["arm"]["pulse_pct"] == pytest.approx(1.0, abs=1e-9)
    assert r["arm"]["count_pct"] == pytest.approx(0.0, abs=1e-12)
    assert 0.5 < r["arm"]["mean_pct"] < 1.5
    assert r["arm"]["count_by_sectortype"] == {"0": [0.0, 0.0]}


def test_pe_report_needs_common_shards(tmp_path):
    _pe_shards(tmp_path, "ref", 1.0, n=2)
    _pe_shards(tmp_path, "arm", 1.0, n=2)
    with pytest.raises(SystemExit):
        pl.pe_report(str(tmp_path / "r.json"), "ref", ["arm"], str(tmp_path))


def test_share_is_primary_fraction_times_primary_shift_over_full_shift(tmp_path):
    p, f = tmp_path / "p", tmp_path / "f"
    p.mkdir()
    f.mkdir()
    _pe_shards(p, "gp", 0.25, seed=1)
    _pe_shards(p, "fp", 0.25 * 1.004, seed=1)
    _pe_shards(f, "gf", 1.0, seed=1)
    _pe_shards(f, "ff", 1.01, seed=1)
    pl.share(str(tmp_path / "s.json"), str(p), "gp", "fp", str(f), "gf", "ff")
    s = json.load(open(tmp_path / "s.json"))
    assert s["shards"] == 6
    r = s["0-2000 PE"]
    assert 0.2 < r["w"][0] < 0.3                                  # primary clusters sit lower in PE
    assert r["dp"][0] == pytest.approx(0.004, abs=5e-4) and r["dfull"][0] == pytest.approx(0.01, abs=1e-3)
    assert r["share"][0] == pytest.approx(r["w"][0] * r["dp"][0] / r["dfull"][0])
    assert r["count_primary"][0] == pytest.approx(0.0, abs=1e-3)
