"""Shower support diagnostics distinguish absent cells, zero counts and stochastic rejection."""
import json

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim import tier1b
from crvfastsim.veto.nonmuon import DrawRecorder, coverage, load_sampler, main, read_library


def fixture():
    gen = pd.DataFrame(dict(run=[1] * 4, subrun=[0] * 4, event=[1, 2, 3, 4], idx=[0] * 4,
                           pdg=[-11, 2112, 22, 13], e=[100.511, 1039.565, 100., 205.658],
                           t=[10.] * 4, shard=[0] * 4))
    kebin = int(tier1b._ke_bin(np.array([100.]))[0])
    reach = pd.DataFrame(dict(apdg=[11, 2112], kebin=[kebin, kebin], ngen=[10, 2], nshower=[2, 0]))
    showers = pd.DataFrame(dict(shid=[0, 1], apdg=[11, 11], kebin=[kebin, kebin], ke=[110., 120.]))
    steps = pd.DataFrame(dict(shid=[0, 1], sp=[1, 1], dt0=[2., 3.], dt1=[3., 4.], edep=[1., 2.]))
    return gen, dict(libgen=reach, libshower=showers, libshowersteps=steps)


def test_support_and_probability_are_outcome_independent():
    gen, lib = fixture()
    rows, bins, summary = coverage(gen, lib, tier1b)
    assert list(rows.status) == ["supported", "zero_shower_cell", "missing_cell"]
    assert list(rows.pdg) == [-11, 2112, 22]
    assert rows.outside_donor_ke_range.sum() == 1
    assert summary["expected_showers"] == pytest.approx(.2)
    assert summary["probability_zero_showers"] == pytest.approx(.8)
    assert summary["reach_count_template_mismatches"] == 0
    assert bins.loc[bins.apdg == 22, "reach_probability"].iloc[0] == 0
    assert not bins.loc[bins.apdg == 22, "reach_cell"].iloc[0]
    gen["decision"] = "stock_only"
    assert coverage(gen, lib, tier1b)[2] == summary


def test_signed_species_and_ke_clipping_follow_sampler():
    gen, lib = fixture()
    gen.loc[0, "e"] = .511
    gen.loc[1, "e"] = 1e8
    rows, _, _ = coverage(gen, lib, tier1b)
    assert list(rows.kebin[:2]) == [0, len(tier1b.KE_EDGES) - 2]
    assert rows.apdg.iloc[0] == 11


def test_invalid_reach_and_duplicate_cells_rejected():
    gen, lib = fixture()
    lib["libgen"].loc[0, "nshower"] = 11
    with pytest.raises(ValueError, match="probability"):
        coverage(gen, lib, tier1b)
    gen, lib = fixture()
    lib["libgen"] = pd.concat([lib["libgen"], lib["libgen"].iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        coverage(gen, lib, tier1b)


def test_positive_reach_requires_template():
    gen, lib = fixture()
    lib["libshower"] = lib["libshower"].iloc[:0]
    with pytest.raises(ValueError, match="without a shower"):
        coverage(gen, lib, tier1b)


def test_empty_nonmuon_population_has_unit_zero_probability():
    gen, lib = fixture()
    rows, _, summary = coverage(gen[gen.pdg == 13], lib, tier1b)
    assert len(rows) == 0 and summary["expected_showers"] == 0
    assert summary["probability_zero_showers"] == 1


def test_recording_sampler_draws_leaves_output_and_rng_unchanged():
    gen, lib = fixture()
    lib["libgen"].loc[0, "ngen"] = 2
    plain, observed = np.random.default_rng(7), DrawRecorder(np.random.default_rng(7))
    expected = tier1b._showers_for_gen(gen, lib, plain)
    actual = tier1b._showers_for_gen(gen, lib, observed)
    pd.testing.assert_frame_equal(actual, expected)
    assert len(observed.draws[0]) == 3 and len(observed.draws[1]) == 1
    assert len(actual) == 1 and actual.t0.iloc[0] in (12., 13.)
    np.testing.assert_array_equal(plain.random(5), observed.rng.random(5))


def test_snapshot_loader_and_highest_root_cycle(tmp_path):
    gen, lib = fixture()
    snapshot = load_sampler(tier1b.__file__)
    assert coverage(gen, lib, snapshot)[2] == coverage(gen, lib, tier1b)[2]
    path = tmp_path / "library.root"
    with uproot.recreate(path) as root:
        for key, frame in lib.items():
            root["dump/" + key] = {c: frame[c].to_numpy() for c in frame}
    lib["libgen"].loc[0, "ngen"] = 20
    with uproot.update(path) as root:
        root["dump/libgen"] = {c: lib["libgen"][c].to_numpy() for c in lib["libgen"]}
    pd.testing.assert_frame_equal(read_library(path)["libgen"], lib["libgen"])


def test_cli_refuses_heldout_and_writes_empty_support(tmp_path):
    gen, lib = fixture()
    path = tmp_path / "library.root"
    with uproot.recreate(path) as root:
        for key, frame in lib.items():
            root[key] = {c: frame[c].to_numpy() for c in frame}
    args = ["--generators", str(tmp_path / "gen.parquet"), "--library", str(path), "--outdir", str(tmp_path / "out")]
    gen.to_parquet(tmp_path / "gen.parquet", index=False)
    main(args)
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["nonmuon_inputs"] == 3
    gen["shard"] = 22
    gen.to_parquet(tmp_path / "gen.parquet", index=False)
    with pytest.raises(ValueError, match="development"):
        main(args)
