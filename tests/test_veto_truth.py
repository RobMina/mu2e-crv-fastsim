"""Native CRV cluster ancestry alignment and represented-root coverage diagnostics."""
import awkward as ak
import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import VetoCard
from crvfastsim.veto.truth import ancestry_table, generator_roots


def fixture():
    vec = dict(fCoordinates=dict(fX=1., fY=2., fZ=3.))
    mu = dict(valid=True, id=4, gen=38, pdg=13, time=100., pos=vec, mom=vec)
    proton = dict(mu, id=5, pdg=2212)
    selection = ak.Array([dict(run=1, subrun=2, event=3, t_mid=[1000., 1050.], track_index=[0, 2], trkmcsim=[[mu], [mu]]),
                          dict(run=1, subrun=2, event=4, t_mid=[1000.], track_index=[1], trkmcsim=[[proton]])])
    co = ak.Array([[{"time": 900., "PEs": 50., "PEsPerLayer[4]": [10., 10., 10., 20.]},
                    {"time": 950., "PEs": 100., "PEsPerLayer[4]": [20., 20., 20., 40.]}], []])
    mc = ak.Array([[{"valid": True, "pdgId": 13, "primaryPdgId": 13, "primary.fCoordinates.fX": 1.,
                     "primary.fCoordinates.fY": 2., "primary.fCoordinates.fZ": 3.},
                    {"valid": True, "pdgId": 2212, "primaryPdgId": 2212, "primary.fCoordinates.fX": 1.,
                     "primary.fCoordinates.fY": 2., "primary.fCoordinates.fZ": 3.}], []])
    truth = ak.zip(dict(run=[1, 1], subrun=[2, 2], event=[3, 4], crvcoincs=co, crvcoincsmc=mc,
                        crvsummarymc=ak.Array([{"nHitCounters": 4}, {"nHitCounters": 0}]),
                        all_trkmcsim=ak.Array([[[mu, proton], [mu]], [[proton]]])), depth_limit=1)
    decisions = pd.DataFrame(dict(run=[1, 1, 1], subrun=[2, 2, 2], event=[3, 3, 4], track_index=[0, 2, 1]))
    return selection, truth, decisions


def test_root_ids_deduplicate_across_tracks_not_events():
    selection, truth, _ = fixture()
    roots = generator_roots(selection, truth.all_trkmcsim)
    assert len(roots) == 3 and list(roots.pdg) == [13, 2212, 2212]
    roots = generator_roots(selection, selection.trkmcsim)
    assert len(roots) == 2


def test_native_cluster_index_and_excluded_species():
    selection, truth, decisions = fixture()
    rows, roots = ancestry_table(selection, truth[[1, 0]], VetoCard(), decisions)
    assert list(rows.cluster_index[:2]) == [1., 1.]
    assert list(rows.crv_mc_primaryPdgId[:2]) == [2212., 2212.]
    assert np.isnan(rows.cluster_index.iloc[2])
    assert rows.start_state_compatible.iloc[0] and not rows.fast_start_state_compatible.iloc[0]
    assert roots.in_fast_input.sum() == 1 and (~roots.in_selected_genealogy).sum() == 1
    assert rows.stock_PE_layer3.iloc[0] == 40


def test_start_state_disagreement_is_not_an_identity_match():
    selection, truth, decisions = fixture()
    mc = ak.with_field(truth.crvcoincsmc, truth.crvcoincsmc["primary.fCoordinates.fX"] + 1000., "primary.fCoordinates.fX")
    truth = ak.with_field(truth, mc, "crvcoincsmc")
    rows, _ = ancestry_table(selection, truth, VetoCard(), decisions)
    assert rows.primary_species_represented.iloc[0] and not rows.start_state_compatible.iloc[0]


def test_inconsistent_root_states_raise():
    selection, truth, _ = fixture()
    moms = truth.all_trkmcsim.mom
    coords = ak.with_field(moms.fCoordinates, ak.Array([[[1., 1.], [2.]], [[1.]]]), "fX")
    sims = ak.with_field(truth.all_trkmcsim, ak.with_field(moms, coords, "fCoordinates"), "mom")
    with pytest.raises(ValueError, match="inconsistent states"):
        generator_roots(selection, sims)


def test_report_partitions_losses_without_changing_decisions(tmp_path):
    import json
    from crvfastsim.veto.truth import report
    selection, truth, decisions = fixture()
    decisions["decision"] = ["stock_only", "stock_only", "neither_veto"]
    decisions["stock_veto"] = [True, True, False]
    support = decisions[["run", "subrun", "event", "track_index"]].copy()
    support["step_support"] = "no_steps"
    report(selection, truth, VetoCard(), decisions, tmp_path, support)
    summary = json.loads((tmp_path / "ancestry_summary.json").read_text())
    assert summary["decisions"]["stock_only"]["n"] == 2
    assert summary["stock_only_coverage"] == [dict(coverage_diagnostic="represented_nonmuon_excluded",
                                                   step_support="no_steps", tracks=2)]
    rows = pd.read_parquet(tmp_path / "tracks_truth.parquet")
    assert rows.stock_veto.sum() == 2 and rows.attributed_stock_PEs.iloc[0] == 100


def test_source_reader_preserves_native_alignment(monkeypatch):
    from crvfastsim.veto import truth as module
    selection, data, _ = fixture()
    co = data.crvcoincs
    for field, value in (("timeStart", co.time - 1), ("timeEnd", co.time + 1),
                         ("nLayers", ak.ones_like(co.time) * 4), ("sectorType", ak.ones_like(co.time))):
        co = ak.with_field(co, value, field)
    selection = ak.with_field(selection, co, "coincs")
    data = ak.with_field(data, co, "crvcoincs")
    branches = {"evtinfo/" + k: data[k] for k in ("run", "subrun", "event")}
    for name in ("crvcoincs", "crvcoincsmc", "crvsummarymc"):
        branches.update({name + "/" + name + "." + k: data[name][k] for k in data[name].fields})
    branches["crvcoincsmcplane/crvcoincsmcplane.pdgId"] = co.time * 0 + 13
    class Branch:
        def keys(self):
            return []
        def array(self, library):
            return data.all_trkmcsim
    class Tree:
        def keys(self):
            return branches.keys()
        def __getitem__(self, key):
            return Branch()
        def arrays(self, keys, library, how=None):
            return {k: ak.to_numpy(branches[k]) if library == "np" else branches[k] for k in keys}
    class Root:
        def __enter__(self):
            return {"EventNtuple/ntuple": Tree()}
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(module.uproot, "open", lambda path: Root())
    rows = module.source_rows("source.root", selection[[1, 0]])
    assert list(ak.to_numpy(rows.event)) == [4, 3]
    assert ak.all(ak.num(rows.crvcoincsmc.valid) == ak.num(rows.crvcoincs.time))
    changed = ak.with_field(selection, ak.with_field(co, co.PEs + 1, "PEs"), "coincs")
    with pytest.raises(ValueError, match="values differ"):
        module.source_rows("source.root", changed)


def test_retained_generator_census_keeps_empty_events(tmp_path):
    import uproot
    from crvfastsim.veto.truth import retained_generators
    path = tmp_path / "gen.root"
    with uproot.recreate(path) as root:
        root["dump/events"] = dict(run=np.array([1, 1]), subrun=np.array([2, 2]), event=np.array([3, 4]),
                                   n_gen=np.array([1, 0]), full_generator_stream=np.array([0, 0]),
                                   gen_tag=ak.Array(["compressed", "compressed"]))
        root["dump/gen"] = dict(run=np.array([1]), subrun=np.array([2]), event=np.array([3]),
                                gen_product_id=np.array([7]), idx=np.array([0]), pdg=np.array([2212]), genid=np.array([38]),
                                x=np.array([1.]), y=np.array([2.]), z=np.array([3.]),
                                px=np.array([1.]), py=np.array([2.]), pz=np.array([3.]), t=np.array([100.]))
    selection, _, _ = fixture()
    result = retained_generators(path, selection, tmp_path, origin=[0., 0., 0.])
    assert result["events"] == 2 and result["empty_events"] == 1 and result["generators"] == 1
    assert result["exact_event_key_overlap"] == 2 and not result["full_generator_stream"]
    assert "no upstream/downstream" in result["mapping_limit"]
    assert len(pd.read_parquet(tmp_path / "upstream_events.parquet")) == 2
    assert result["event_key_state_diagnostic"]["generators_without_compatible_genealogy_state"] == 1
    assert not pd.read_parquet(tmp_path / "selected_generator_candidates.parquet").represented_state_compatible.any()


def test_retained_generator_census_rejects_mismatched_counts(tmp_path):
    import uproot
    from crvfastsim.veto.truth import retained_generators
    path = tmp_path / "gen.root"
    with uproot.recreate(path) as root:
        root["dump/events"] = dict(run=np.array([1]), subrun=np.array([2]), event=np.array([3]), n_gen=np.array([2]))
        root["dump/gen"] = dict(run=np.array([1]), subrun=np.array([2]), event=np.array([3]),
                                gen_product_id=np.array([7]), idx=np.array([0]))
    selection, _, _ = fixture()
    with pytest.raises(ValueError, match="event census"):
        retained_generators(path, selection, tmp_path)
