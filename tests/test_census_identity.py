"""Exact, event-local ancestry and explicit geometry groups for the census."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim import tables
from crvfastsim.analyses.census import CLASSES, SECTOR_GROUP, classify, primary_of


def particles(keys, parents, pdgs, run=1, subrun=2, event=3):
    return pd.DataFrame(dict(run=run, subrun=subrun, event=event,
                             key=np.asarray(keys, dtype=np.int64),
                             parent=np.asarray(parents, dtype=np.int64),
                             pdg=np.asarray(pdgs, dtype=np.int32)))


def steps_for(parts):
    return parts[tables.KEY + ["key"]].rename(columns={"key": "sp"}).assign(bar=np.int64(0), edep=1.)


def test_deep_chain_has_no_arbitrary_depth_limit():
    n = 8195
    parts = particles(np.arange(n), np.arange(n) - 1, np.r_[-13, np.full(n - 1, 11)])
    parts = parts.sample(frac=1, random_state=19)
    p = primary_of(parts)
    np.testing.assert_array_equal(p.prim_key, np.zeros(n, dtype=np.int64))
    np.testing.assert_array_equal(p.prim_pdg, np.full(n, -13))
    np.testing.assert_array_equal(p.uid, np.arange(n))
    assert p.index.equals(parts.index)
    s = classify(steps_for(parts), parts, np.array([0]))
    assert s.loc[s.sp == 0, "cls"].tolist() == ["mu_primary"]
    assert (s.loc[s.sp != 0, "cls"] == "mu_descendant").all()


def test_large_int64_keys_and_missing_association_are_exact():
    a = 2**60 + 7
    b = np.iinfo(np.int64).max
    parts = particles([a, a + 1, b], [-1, a, -1], [-13, 11, 22])
    steps = steps_for(parts)
    steps = pd.concat([steps.iloc[[1]], steps.iloc[[0]], steps.iloc[[2]],
                       steps.iloc[[0]].assign(sp=a + 2)], ignore_index=True)
    s = classify(steps, parts, np.array([0]))
    assert s.cls.tolist() == ["mu_descendant", "mu_primary", "nonmu_primary", "unknown"]
    assert s.prim_key.iloc[:3].tolist() == [a, a, b]
    assert pd.isna(s.prim_key.iloc[3]) and pd.isna(s.prim_pdg.iloc[3])
    assert s.prim_key.dtype == pd.Int64Dtype()
    np.testing.assert_array_equal(primary_of(parts).prim_key, [a, a, b])
    pd.testing.assert_frame_equal(steps, s[steps.columns])


def test_complete_event_tuple_separates_reused_keys():
    a = 2**60 + 7
    frames = [particles([a, a + 1], [-1, a], [pdg, 11], run=r, subrun=sr, event=e)
              for r, sr, e, pdg in [(1, 1, 7, -13), (2, 1, 7, 22), (1, 2, 7, 13),
                                    (1, 1, 7 + 2**20, 211), (2**32 - 1, 2**31 + 1, 7, 0)]]
    parts = pd.concat(frames, ignore_index=True)
    orphan = particles([a + 2], [a], [11], run=3, subrun=1, event=7)
    parts = pd.concat([parts, orphan], ignore_index=True)
    s = classify(steps_for(parts), parts, np.array([0]))
    assert s.cls.tolist() == ["mu_primary", "mu_descendant", "nonmu_primary", "nonmu_descendant",
                              "mu_primary", "mu_descendant", "nonmu_primary", "nonmu_descendant",
                              "nonmu_primary", "nonmu_descendant", "unknown"]
    assert s.prim_pdg.tolist() == [-13, -13, 22, 22, 13, 13, 211, 211, 0, 0, 0]
    assert s.prim_key.iloc[-1] == -1


def test_broken_and_cyclic_chains_never_invent_roots():
    parts = particles([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                      [-1, 0, 999, 2, 4, 6, 5, 8, 9, 7, 7, -2, 11],
                      [-13, 11, 13, 11, 13, 13, 22, 13, 11, 22, 11, 13, 11])
    p = primary_of(parts)
    assert p.prim_key.tolist() == [0, 0] + [-1] * 11
    assert p.prim_pdg.tolist() == [-13, -13] + [0] * 11
    s = classify(steps_for(parts), parts, np.array([0]))
    assert s.cls.tolist() == ["mu_primary", "mu_descendant"] + ["unknown"] * 11


def test_extracted_defaults_preserve_small_input_classes_and_outputs():
    parts = particles([10, 11, 20, 21, 30, 31], [-1, 10, -1, 20, -1, 30], [-13, 22, 211, 11, 0, 11])
    steps = steps_for(parts).assign(bar=[0, 1, 2, 3, 9, 10], edep=np.arange(1., 7.))
    sectors = np.arange(11)
    default = classify(steps, parts, sectors)
    explicit = classify(steps, parts, sectors, group_of_sector=SECTOR_GROUP)
    pd.testing.assert_frame_equal(default, explicit)
    pd.testing.assert_frame_equal(default[steps.columns], steps)
    assert default.cls.tolist() == list(CLASSES) + ["nonmu_primary", "nonmu_descendant"]
    assert default.prim_key.tolist() == [10, 10, 20, 20, 30, 30]
    assert default.prim_pdg.tolist() == [-13, -13, 211, 211, 0, 0]
    assert default.group.tolist() == ["EX", "T1", "T2", "M", "M", "M"]


def test_explicit_23_sector_mapping_does_not_reuse_extracted_groups():
    parts = particles([1], [-1], [13])
    sectors = np.arange(22, -1, -1, dtype=np.int32)
    groups = np.array(["EX", "T1", "T2", "M", "native"] * 5, dtype=object)[:23]
    steps = pd.concat([steps_for(parts)] * 23, ignore_index=True).assign(bar=np.arange(23))
    s = classify(steps, parts, sectors, group_of_sector=groups)
    np.testing.assert_array_equal(s.group, groups[sectors])
    assert (s.cls == "mu_primary").all()
    with pytest.raises(ValueError, match="sector.*bounds"):
        classify(steps, parts, sectors)


@pytest.mark.parametrize("bars,sectors,groups,match", [
    ([-1], [0], None, "bar.*bounds"),
    ([1], [0], None, "bar.*bounds"),
    ([0.0], [0], None, "steps.bar.*integer"),
    ([np.nan], [0], None, "steps.bar.*integer"),
    ([0], [-1], None, "sector.*bounds"),
    ([0], [11], None, "sector.*bounds"),
    ([0], [0.0], None, "sector_of_bar.*integer"),
    ([0], [1], ["EX"], "sector.*bounds"),
    ([0], [[0]], None, "sector_of_bar.*one-dimensional"),
    ([0], [0], [["EX"]], "group_of_sector.*one-dimensional"),
    ([2**64 - 1], [0], None, "steps.bar.*signed int64"),
])
def test_geometry_indices_reject_noninteger_negative_or_out_of_range(bars, sectors, groups, match):
    parts = particles([1], [-1], [13])
    steps = steps_for(parts).assign(bar=np.asarray(bars))
    with pytest.raises(ValueError, match=match):
        classify(steps, parts, sectors, group_of_sector=groups)


@pytest.mark.parametrize("column", tables.KEY + ["key", "parent"])
def test_float_particle_identity_is_rejected(column):
    parts = particles([1], [-1], [13])
    parts[column] = parts[column].astype(float)
    with pytest.raises(ValueError, match=f"particles.{column}.*integer"):
        primary_of(parts)


@pytest.mark.parametrize("column", tables.KEY + ["sp"])
def test_float_step_identity_is_rejected(column):
    parts = particles([1], [-1], [13])
    steps = steps_for(parts)
    steps[column] = steps[column].astype(float)
    with pytest.raises(ValueError, match=f"steps.{column}.*integer"):
        classify(steps, parts, [0])


def test_duplicate_particle_identity_is_rejected():
    parts = particles([1, 1], [-1, -1], [13, 22])
    with pytest.raises(ValueError, match="duplicate"):
        primary_of(parts)


def test_empty_typed_inputs_and_unmatched_steps():
    parts = particles([1], [-1], [13])
    empty_parts = parts.iloc[:0]
    p = primary_of(empty_parts)
    assert p.empty and p.prim_key.dtype == np.int64
    empty_steps = steps_for(parts).iloc[:0]
    assert classify(empty_steps, parts, [0]).empty
    assert classify(empty_steps, empty_parts, np.array([], dtype=np.int64)).empty
    s = classify(steps_for(parts), empty_parts, [0])
    assert s.cls.tolist() == ["unknown"]
    assert s.prim_key.isna().all() and s.prim_pdg.isna().all()


def test_unknown_pdg_at_known_root_preserves_nonmu_semantics():
    parts = particles([1, 2], [-1, 1], [0, 11])
    parts["pdg"] = [np.nan, 11.]
    s = classify(steps_for(parts), parts, [0])
    assert s.cls.tolist() == ["nonmu_primary", "nonmu_descendant"]
    assert s.prim_pdg.isna().all()
