"""Native secondary-library geometry with absent extracted groups."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import tier1b as T
from crvfastsim.analyses import tier1b as analysis


def one_group():
    return SimpleNamespace(nbars=1, sector=np.array([0]), lib_group=np.array(["EX"]),
                           center=np.array([[0., 0., 0.]]), halflen=np.array([[1., 2., 3.]]))


def test_sparse_group_bin_edges(monkeypatch):
    key = dict(run=[1, 1], subrun=[1, 1], event=[1, 2])
    parts = pd.DataFrame(dict(**key, key=[1, 1], pdg=[13, -13], isprim=[1, 1]))
    steps = pd.DataFrame(dict(**key, bar=[0, 0], sp=[1, 1], t0=0., edep=1., len=[2., 3.],
                              x0=0., y0=0., z0=0., px0=0., py0=[-1000., -2000.], pz0=0.))
    monkeypatch.setattr(T.tables, "read", lambda path, tree, columns: {"steps": steps, "particles": parts}[tree][columns])
    bins = T._bin_edges(["synthetic"], one_group())
    assert set(bins.group) == {0}
    assert len(bins) == len(T.P_QUANTILES) + len(T.L_QUANTILES)
    expected = np.concatenate([np.quantile([1000., 2000.], T.P_QUANTILES), np.quantile([2., 3.], T.L_QUANTILES)])
    np.testing.assert_array_equal(bins.edge, expected)
    assert np.all(T._bin(bins, np.array([0, 0]), np.array([1000., 2000.]), np.array([2., 3.])) >= 0)


def test_bin_edges_rejects_unknown_library_group(monkeypatch):
    key = dict(run=[1], subrun=[1], event=[1])
    parts = pd.DataFrame(dict(**key, key=[1], pdg=[13], isprim=[1]))
    steps = pd.DataFrame(dict(**key, bar=[0], sp=[1], t0=0., edep=1., len=2.,
                              x0=0., y0=0., z0=0., px0=0., py0=-1000., pz0=0.))
    monkeypatch.setattr(T.tables, "read", lambda path, tree, columns: {"steps": steps, "particles": parts}[tree][columns])
    geom = one_group()
    geom.lib_group = np.array(["unsupported"])
    with pytest.raises(ValueError, match="supported library groups"):
        T._bin_edges(["synthetic"], geom)


def test_closest_approach_ignores_absent_groups():
    geom = one_group()
    lo, hi = T.group_boxes(geom)
    assert np.all(lo[1:] > hi[1:])
    g, distance, time = T.closest_approach(np.array([[0., 10., 0.], [5., 10., 0.]]),
                                          np.array([[0., -1., 0.], [0., -1., 0.]]), geom)
    np.testing.assert_array_equal(g, [0, 0])
    np.testing.assert_allclose(distance, [0., 4.])
    assert np.isfinite(time).all()


def test_closest_approach_rejects_geometry_without_library_group():
    geom = one_group()
    geom.lib_group = np.array(["unsupported"])
    with pytest.raises(ValueError, match="no populated library group"):
        T.closest_approach(np.array([[0., 0., 0.]]), np.array([[0., -1., 0.]]), geom)


def test_native_build_geometry_uses_config_mapping(monkeypatch):
    from crvfastsim import runconfig
    geom = one_group()
    seen = []
    class Config:
        def geometry(self):
            seen.append(True)
            return geom
    monkeypatch.setattr(runconfig, "get", lambda name: Config())
    assert analysis._geom(None, "run1a_v01") is geom
    assert seen == [True]


def test_native_counters_receipt_uses_native_asset(monkeypatch):
    from crvfastsim import runconfig
    rc = SimpleNamespace(counters_path=lambda: "native-counters")
    monkeypatch.setattr(runconfig, "get", lambda name: rc)
    assert analysis._counters(None, "run1a_v01") == "native-counters"


def test_native_builder_persists_build_time_geometry(tmp_path, monkeypatch):
    import json
    from crvfastsim.analyses.context_audit import content_fingerprint
    geom = one_group()
    counters = tmp_path / "counters.txt"
    counters.write_text("synthetic geometry")
    monkeypatch.setattr(analysis, "_geom", lambda *a: geom)
    monkeypatch.setattr(analysis, "_counters", lambda *a: str(counters))
    monkeypatch.setattr(T, "build", lambda *a, **kw: ({}, {"generated_scaled_to": 17}))
    monkeypatch.setattr(T, "write_library", lambda *a: None)
    from crvfastsim import native_training
    donor, generator = tmp_path / "donor", tmp_path / "generator"
    donor.write_text("synthetic donor")
    generator.write_text("synthetic generator")
    training = dict(total_exposure=17, pairs=[dict(steps=dict(path=str(donor), sha256=analysis._sha256(donor)),
                        generator=dict(path=str(generator), sha256=analysis._sha256(generator)))])
    monkeypatch.setattr(native_training, "validate", lambda *a: training)
    monkeypatch.setattr("crvfastsim.native_construction.require_supported", lambda *a: {"supported": True})
    out = tmp_path / "library.root"
    analysis.build([str(donor)], str(tmp_path), gen=[str(generator)], out=str(out), run_config="run1a_v01")
    manifest = json.loads(out.with_suffix(".json").read_text())
    assert manifest["geometry_fingerprint"] == content_fingerprint(geom, workers=1)
    assert manifest["counters_sha256"] == analysis._sha256(counters)


def test_cli_accepts_native_library_config():
    from crvfastsim.cli import build_parser
    args = build_parser().parse_args(["tier1b-build", "d.root", "--gen", "g.root", "--out", "l.root",
                                      "--roots", "--run-config", "run1a_v01"])
    assert args.run_config == "run1a_v01" and args.roots
