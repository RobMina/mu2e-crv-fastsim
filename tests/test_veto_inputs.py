"""Outcome-independent ancestor selection and retained-generator completeness diagnostics."""
import json

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.veto.inputs import collect, compare_inputs, plan, report
from crvfastsim.veto.primaries import extract


def fixture():
    vector = dict(fCoordinates=dict(fX=1., fY=2., fZ=3.))
    mu = dict(valid=True, id=4, gen=38, pdg=13, time=100., pos=vector, mom=vector, prirel=dict(_rel=-1))
    proton = dict(mu, id=5, pdg=2212)
    selection = ak.Array([dict(run=1, subrun=2, event=3, shard=0, trkmcsim=[[mu]]),
                          dict(run=1, subrun=2, event=4, shard=0, trkmcsim=[[proton]]),
                          dict(run=1, subrun=7, event=8, shard=1, trkmcsim=[[]])])
    truth = ak.zip({**{k: selection[k] for k in ("run", "subrun", "event")},
                    "all_trkmcsim": selection.trkmcsim}, depth_limit=1)
    events = pd.DataFrame(dict(run=[1, 1, 1], subrun=[2, 2, 7], event=[3, 4, 8], n_gen=[2, 1, 0]))
    gen = pd.DataFrame(dict(run=[1, 1, 1], subrun=[2, 2, 2], event=[3, 3, 4], gen_product_id=[7, 7, 7],
                            idx=[0, 1, 0], pdg=[13, -13, 2212], genid=[38, 38, 38],
                            x=[11., 21., 11.], y=[22., 22., 22.], z=[33., 33., 33.],
                            px=[1., 2., 1.], py=[2., 3., 2.], pz=[3., 4., 3.], e=[106., 107., 939.],
                            t=[100., 100., 100.]))
    return selection, truth, events, gen


def test_catalog_plan_uses_only_event_keys(tmp_path):
    selection, _, _, _ = fixture()
    records = [dict(shard=0, fid="10", name="a.art", namespace="test", size=10, retired=False,
                    subruns=[1000002], events=2, parents=[]),
               dict(shard=1, fid="11", name="b.art", namespace="test", size=20, retired=False,
                    subruns=[1000007], events=1, parents=[])]
    plan(selection, records, tmp_path)
    result = json.loads((tmp_path / "plan.json").read_text())
    assert result["events"] == 3 and result["upstream_files"] == 2
    assert list(pd.read_parquet(tmp_path / "event_sources.parquet").fid) == ["10", "10", "11"]
    with pytest.raises(ValueError, match="ancestor"):
        plan(selection, records[:1], tmp_path)


def test_missing_muon_and_excluded_nonmuon_are_separate():
    selection, truth, events, gen = fixture()
    supplied, _ = extract(selection, [10., 20., 30.])
    rows, candidates, roots = compare_inputs(selection, truth, events, gen, [10., 20., 30.], supplied)
    assert list(rows.missing_muons) == [1, 0, 0]
    assert list(rows.excluded_nonmuons) == [0, 1, 0]
    assert list(rows.supplied_muons) == [1, 0, 0]
    assert candidates.absent_from_genealogy.sum() == 1
    assert candidates.absent_from_fast.sum() == 2
    assert roots.upstream_compatible.eq(1).all()
    assert rows.event_covered.all()


def test_uncovered_event_is_not_silently_treated_as_empty():
    selection, truth, events, gen = fixture()
    rows, _, roots = compare_inputs(selection, truth, events.iloc[:1], gen.iloc[:2], [10., 20., 30.])
    assert list(rows.event_covered) == [True, False, False]
    assert rows.missing_muons.iloc[1:].isna().all()
    assert not roots.event_covered.iloc[1]


def test_saved_fast_input_validation_and_float32_precision():
    selection, truth, events, gen = fixture()
    supplied, _ = extract(selection, [10., 20., 30.])
    gen.loc[0, "px"] += 1e-9
    _, candidates, _ = compare_inputs(selection, truth, events, gen, [10., 20., 30.], supplied)
    assert not candidates.absent_from_fast.iloc[0]
    supplied.loc[0, "px"] += 1.
    with pytest.raises(ValueError, match="saved fast input differs"):
        compare_inputs(selection, truth, events, gen, [10., 20., 30.], supplied)


def test_duplicate_stored_states_are_explicit_not_deduplicated():
    selection, truth, events, gen = fixture()
    twin = gen.iloc[[0]].copy()
    twin["idx"] = 2
    gen = pd.concat([gen, twin], ignore_index=True)
    events.loc[0, "n_gen"] = 3
    _, candidates, roots = compare_inputs(selection, truth, events, gen, [10., 20., 30.])
    assert candidates.duplicate_stored_state.sum() == 2
    assert roots.upstream_compatible.iloc[0] == 2


def test_collection_keeps_empty_events_and_rejects_duplicate_sources(tmp_path):
    selection, _, events, gen = fixture()
    path = tmp_path / "dump.root"
    with uproot.recreate(path) as root:
        root["dump/events"] = {k: events[k].to_numpy() for k in events}
        root["dump/gen"] = {k: gen[k].to_numpy() for k in gen}
    census, actual = collect([path], selection)
    assert len(census) == 3 and len(actual) == 3 and census.n_gen.iloc[-1] == 0
    with pytest.raises(ValueError, match="more than one"):
        collect([path, path], selection)


def test_report_joins_decisions_after_census(tmp_path):
    selection, truth, events, gen = fixture()
    decisions = pd.DataFrame(dict(run=[1, 1, 1], subrun=[2, 2, 7], event=[3, 4, 8], track_index=[0, 0, 0],
                                  decision=["both_veto", "neither_veto", "stock_only"]))
    summary = report(selection, truth, events, gen, [10., 20., 30.], tmp_path, decisions=decisions)
    assert summary["covered_events"] == 3 and summary["events_with_extra_generators"] == 2
    assert summary["missing_muons"] == summary["excluded_nonmuons"] == 1
    assert summary["all_roots_without_compatible_upstream_state"] == 0
    assert len(pd.read_parquet(tmp_path / "track_coverage.parquet")) == 3


def test_native_ancestry_is_post_census_and_not_identity(tmp_path):
    from crvfastsim.veto.inputs import ancestry_coverage
    selection, truth, events, gen = fixture()
    _, candidates, _ = compare_inputs(selection, truth, events, gen, [10., 20., 30.])
    ancestry = pd.DataFrame(dict(run=[1], subrun=[2], event=[3], track_index=[0], decision=["stock_only"],
                                 crv_mc_primaryPdgId=[-13], **{"crv_mc_primary.fCoordinates.fX": [11.],
                                 "crv_mc_primary.fCoordinates.fY": [2.], "crv_mc_primary.fCoordinates.fZ": [3.]}))
    result = ancestry_coverage(ancestry, candidates, [10., 20., 30.])
    assert result.primary_signature_in_retained.all() and result.primary_signature_in_missing.all()
    assert len(candidates) == 3
