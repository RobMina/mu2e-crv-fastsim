"""Saved decision diagnostics preserve the stock population and do not match clusters by proximity."""
import json

import awkward as ak
import numpy as np
import pytest

from crvfastsim.veto import VetoCard
from crvfastsim.veto.diagnose import arm_columns, diagnose, main, summarise


def sample():
    time = ak.Array([[900.], [900.], [], [900.], [900.], [900.]])
    co = ak.zip(dict(time=time, timeStart=time - 1, timeEnd=time + 1, PEs=ak.ones_like(time) * 50,
                     nLayers=ak.ones_like(time) * 3, sectorType=ak.ones_like(time, dtype=np.int64)))
    vec = dict(fCoordinates=dict(fX=1., fY=2., fZ=3.))
    sim = dict(valid=True, gen=38, pdg=13, id=9, time=800., pos=vec, mom=vec, prirel={"_rel": -1})
    return ak.zip(dict(run=[1] * 6, subrun=[1] * 6, event=np.arange(6), shard=[0, 0, 0, 1, 1, 1],
                       track_index=ak.Array([[2], [3], [0], [0], [1], [1]]),
                       p_mid=ak.Array([[100.]] * 6), t_mid=ak.Array([[1000.]] * 6), coincs=co,
                       trkmcsim=ak.Array([[[]], [[sim]], [[sim]], [[sim]], [[sim]], [[sim]]])), depth_limit=1)


def fast(data):
    t = ak.Array([[], [1001.], [900.], [300.], [850.], [1000.]])
    co = ak.zip(dict(time=t, timeStart=t - 1, timeEnd=t + 1, PEs=ak.ones_like(t) * 50,
                     nLayers=ak.ones_like(t) * 3, sectorType=ak.ones_like(t, dtype=np.int64)))
    return ak.with_field(data, co, "coincs")


def test_empty_and_timing_categories_retain_population():
    data = sample()
    result = diagnose(data, fast(data)[[5, 4, 3, 2, 1, 0]], VetoCard(gate_start_lo=420))
    assert len(result) == 6 and result.stock_veto.sum() == 5 and result.fast_veto.sum() == 1
    assert result.mechanism[0] == "no_coincidence" and result.represented_primaries[0] == 0
    assert np.isnan(result.fast_dt[0])
    np.testing.assert_array_equal(result.fast_margin[1:], [-1, 50, -550, 0, 0])
    assert not result.fast_gate[3] and result.stock_veto[3]
    summary = summarise(result)
    assert summary["stock_only"] == 5 and summary["fast_only"] == 1
    assert summary["categories"]["stock_only"]["nonveto_near_boundary"]["1.0"] == 3
    assert summary["categories"]["stock_only"]["no_represented_primary"] == 1
    assert result.track_index[0] == 2


def test_representative_is_eligible_and_max_margin():
    data = sample()
    co = ak.concatenate([data.coincs[:1], data.coincs[:1]], axis=1)
    co = ak.with_field(co, ak.Array([[1000., 900.]]), "time")
    co = ak.with_field(co, ak.Array([[1000., 2.]]), "PEs")
    result = arm_columns(ak.Array([[1000., 1100.]]), co, VetoCard(min_pe=5))
    np.testing.assert_array_equal(ak.flatten(result["veto"]), [False, True])
    np.testing.assert_array_equal(ak.flatten(result["dt"]), [0., 100.])
    assert ak.all(result["PEs"] == 1000.)


def test_all_ineligible_and_no_coincidence_are_distinct():
    data = sample()
    cand = fast(data)
    cand = ak.with_field(cand, ak.with_field(cand.coincs, cand.coincs.PEs * 0, "PEs"), "coincs")
    result = diagnose(data, cand, VetoCard(min_pe=5))
    assert result.mechanism[0] == "no_coincidence"
    assert result.mechanism[1] == "no_eligible_coincidence"
    assert np.isnan(result.fast_margin[1])


def test_multitrack_primary_counts_are_deduplicated():
    data = sample()[1:2]
    data = ak.with_field(data, ak.concatenate([data.trkmcsim, data.trkmcsim], axis=1), "trkmcsim")
    for field in ("p_mid", "t_mid", "track_index"):
        data = ak.with_field(data, ak.concatenate([data[field], data[field]], axis=1), field)
    result = diagnose(data, data, VetoCard(), primary=data)
    assert len(result) == 2 and np.all(result.represented_primaries == 1)
    assert summarise(result)["primary_vs_library"]["gained"] == 0


def test_invalid_event_set_rejected():
    data = sample()
    with pytest.raises(ValueError, match="every cached"):
        diagnose(data, data[:-1], VetoCard())


def test_cli_saved_outputs(tmp_path):
    data = sample()
    ak.to_parquet(data, tmp_path / "selection.parquet")
    ak.to_parquet(fast(data), tmp_path / "fast.parquet")
    (tmp_path / "card.json").write_text(json.dumps({"card": {"sha": "informational"}}))
    main(["--selection", str(tmp_path / "selection.parquet"), "--fast", str(tmp_path / "fast.parquet"),
          "--primary", str(tmp_path / "selection.parquet"), "--card", str(tmp_path / "card.json"),
          "--outdir", str(tmp_path / "out")])
    result = json.loads((tmp_path / "out/summary.json").read_text())
    assert result["selected"] == 6 and result["primary_vs_library"]["lost"] == 5
    assert (tmp_path / "out/tracks.parquet").exists() and (tmp_path / "out/shards.tsv").exists()


def test_nullable_parquet_boolean_stays_boolean(tmp_path):
    data = sample()
    data = ak.with_field(data, ak.mask(data.t_mid, ak.ones_like(data.t_mid, dtype=bool)), "t_mid")
    ak.to_parquet(data, tmp_path / "sel.parquet")
    data = ak.from_parquet(tmp_path / "sel.parquet")
    result = diagnose(data, fast(data), VetoCard())
    assert result.stock_veto.dtype == bool and result.fast_veto.dtype == bool


def test_truth_ray_diagnostic_and_nonmuon_roots():
    from types import SimpleNamespace
    from crvfastsim.veto.diagnose import geometry_diagnostic, lineage_rows
    data = sample()[1:2]
    result = diagnose(data, data, VetoCard())
    rows = lineage_rows(data, result)
    geom = SimpleNamespace(nbars=1, center=np.array([[10., 20., 30.]]), halflen=np.array([[1., 1., 1.]]),
                           sector=np.array([0]), layer=np.array([0]), sector_names=["T1"])
    roots, tracks, summary = geometry_diagnostic(rows, result, geom, [0, 0, 0])
    assert roots.ray_bars.iloc[0] == 1 and tracks.ray_max_layers.iloc[0] == 1
    rows["pdg"] = 2212
    roots, tracks, summary = geometry_diagnostic(rows, result, geom, [0, 0, 0])
    assert roots.ray_bars.iloc[0] == 1 and np.isnan(tracks.ray_max_layers.iloc[0])
    assert summary["generator_species"][0]["pdg"] == 2212


def test_step_support_counts_empty_events_and_all_sectors():
    from types import SimpleNamespace
    import pandas as pd
    from crvfastsim.veto.transport_diagnose import step_counts, summarise as steps_summary
    data = sample()
    steps = pd.DataFrame(dict(event=[1, 1, 1, 2], bar=[0, 1, 2, 3], edep=[1., 2., 3., 4.]))
    geom = SimpleNamespace(sector=np.array([0, 0, 0, 1]), layer=np.array([0, 1, 2, 0]))
    geom.sector_names = ["T1", "T2"]
    counts = step_counts(steps, data, geom)
    assert counts.steps.sum() == 4 and counts.T1_layers.iloc[0] == 3 and counts.T2_edep.iloc[1] == 4
    result = diagnose(data, fast(data), VetoCard())
    rows, summary = steps_summary(counts, result, geom.sector_names)
    assert rows.step_support.iloc[0] == "at_least_three_layers" and rows.step_support.iloc[2] == "no_steps"
    assert sum(row["tracks"] for row in summary) == len(result)


def test_layers_can_span_physical_sectors_with_common_type():
    from types import SimpleNamespace
    import pandas as pd
    from crvfastsim.veto.transport_diagnose import step_counts, summarise as steps_summary, discordant_layers
    data = ak.with_field(sample(), np.full(6, -200.), "pbtime_mc")
    steps = pd.DataFrame(dict(event=[1, 1, 1], bar=[0, 1, 2], edep=[1., 2., 3.], t0=[1200., 1201., 1202.]))
    geom = SimpleNamespace(sector=np.array([0, 0, 1]), layer=np.array([0, 1, 2]), sector_names=["T1", "T2"])
    counts = step_counts(steps, data, geom)
    result = diagnose(data, fast(data), VetoCard())
    rows, summary = steps_summary(counts, result, geom.sector_names)
    assert counts.T1_layers.iloc[0] == 2 and counts.T2_layers.iloc[0] == 1
    assert rows.step_support.iloc[0] == "at_least_three_layers"
    layers = discordant_layers(steps, data, geom, result)
    np.testing.assert_allclose(layers.t_min, [1000., 1001., 1002.])
    assert layers.edep.sum() == 6
