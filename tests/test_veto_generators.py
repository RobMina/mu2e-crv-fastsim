"""Retained generator ingestion preserves all species, source keys and total energies."""
import awkward as ak
import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto.generators import prepare
from crvfastsim.veto.primary_chain import rekey, run


def fixture():
    selection = ak.Array(dict(run=[2, 1, 1], subrun=[4, 4, 4], event=[5, 5, 6], shard=[0, 0, 0]))
    census = pd.DataFrame(dict(run=[1, 1, 2], subrun=[4, 4, 4], event=[5, 6, 5], n_gen=[2, 0, 1]))
    gen = pd.DataFrame(dict(run=[1, 1, 2], subrun=[4, 4, 4], event=[5, 5, 5],
        gen_product_id=[7, 7, 9], idx=[3, 8, 3], pdg=[13, -11, 2112], genid=[38, 38, 38],
        x=[1.123456789012, 2., 3.], y=[0., 0., 0.], z=[4., 5., 6.],
        px=[1., 2., 3.], py=[0., 0., 0.], pz=[0., 0., 0.], e=[106., 2.064, 940.], t=[7.123456789012, 8., 9.]))
    return selection, gen, census


def test_all_species_empty_events_and_precision():
    selection, raw, census = fixture()
    gen, report = prepare(selection, raw, census)
    assert report["primary_particles"] == 3 and report["primary_muons"] == 1
    assert report["nonmuon_primaries"] == 2 and report["empty_generator_events"] == 1
    pd.testing.assert_frame_equal(gen.drop(columns="shard"), raw)
    again, _ = prepare(selection, gen, census)
    pd.testing.assert_frame_equal(gen, again)
    mapped = rekey(selection, gen, "retained")
    assert list(mapped.event) == [1, 2, 2] and list(mapped.source_idx) == [3, 3, 8]
    assert list(mapped.source_run) == [2, 1, 1] and list(mapped.gen_product_id) == [9, 7, 7]
    assert list(mapped.e) == [940., 106., 2.064]
    assert mapped.x.iloc[1] == raw.x.iloc[0] and mapped.t.iloc[1] == raw.t.iloc[0]
    assert list(mapped.idx) == [0, 0, 1]
    assert mapped.index.is_unique


@pytest.mark.parametrize("failure", ["missing_event", "missing_generator", "duplicate", "nan", "bad_energy"])
def test_bad_or_incomplete_collection_rejected(failure):
    selection, gen, census = fixture()
    if failure == "missing_event":
        census = census.iloc[:2]
    elif failure == "missing_generator":
        gen = gen.iloc[:2]
    elif failure == "duplicate":
        gen = pd.concat([gen, gen.iloc[:1]])
    elif failure == "nan":
        gen.loc[0, "t"] = np.nan
    else:
        gen.loc[0, "e"] = 0.
    with pytest.raises(ValueError):
        prepare(selection, gen, census)


def test_represented_route_does_not_accept_nonmuons():
    selection, gen, _ = fixture()
    with pytest.raises(ValueError, match="muons only"):
        rekey(selection, gen)
    mu = gen[gen.pdg == 13]
    old = rekey(selection, mu)
    assert old.e.iloc[0] != 106.
    assert "source_idx" not in old


def test_nonmuons_cannot_be_silently_ignored_without_library():
    selection, gen, census = fixture()
    with pytest.raises(ValueError, match="require the shower library"):
        run(selection, gen, "unused", "unused", "unused", input_kind="retained", census=census)
    with pytest.raises(ValueError, match="independent event census"):
        run(selection, gen, "unused", "unused", "unused", input_kind="retained")


def test_development_only_preparation_does_not_import_other_events():
    selection, gen, census = fixture()
    selected, report = prepare(selection[:1], gen, census)
    assert len(selected) == 1 and selected.pdg.iloc[0] == 2112
    assert report["selected_events"] == 1


def test_preparation_cli_keeps_development_population_and_census(tmp_path):
    import json
    from crvfastsim.veto.generators import main
    selection, gen, census = fixture()
    selection = ak.with_field(selection, [0, 22, 0], "shard")
    ak.to_parquet(selection, tmp_path / "selection.parquet")
    gen.to_parquet(tmp_path / "generators.parquet", index=False)
    census.to_parquet(tmp_path / "events.parquet", index=False)
    main(["--selection", str(tmp_path / "selection.parquet"), "--generators", str(tmp_path / "generators.parquet"),
          "--events", str(tmp_path / "events.parquet"), "--outdir", str(tmp_path / "prepared")])
    result = json.loads((tmp_path / "prepared/inputs.json").read_text())
    assert result["selected_events"] == 2 and result["primary_particles"] == 1
    assert result["empty_generator_events"] == 1 and result["nonmuon_primaries"] == 1
    assert len(pd.read_parquet(tmp_path / "prepared/generator_events.parquet")) == 2
