"""Native pair admission on tiny real schema-v2 TTrees, with stage-specific associations and drift controls."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim import genassoc as G, native_training as N


def payload(run=1400, subrun=101, offset=0):
    events = pd.DataFrame(dict(run=np.full(4, run, dtype=np.uint32),
        subrun=np.full(4, subrun, dtype=np.uint32), event=np.arange(1 + offset, 5 + offset, dtype=np.uint32),
        schema_version=np.uint32(2), associations_checked=np.uint32(0), full_generator_stream=np.uint32(1),
        gen_product_id=np.uint32(99), sim_product_id=np.uint32(0), n_gen=np.array([1, 0, 1, 0], dtype=np.int64),
        n_particles=np.int64(0), n_steps=np.int64(0)))
    gen = events.iloc[[0, 2]][G.KEY + ["gen_product_id"]].copy().reset_index(drop=True)
    gen = gen.assign(idx=np.int64(0), pdg=np.int32(-13), genid=np.int32(1),
                     x=1., y=2., z=3., px=4., py=5., pz=6., e=200., t=0.)
    full = dict(events=events, gen=gen)
    donor_events = events.iloc[:2].copy().assign(associations_checked=np.uint32(1),
        full_generator_stream=np.uint32(0), sim_product_id=np.uint32(202))
    donor_events["n_particles"] = np.array([1, 0], dtype=np.int64)
    donor_events["n_steps"] = np.array([1, 0], dtype=np.int64)
    part = donor_events.iloc[:1][G.KEY + ["gen_product_id", "sim_product_id"]].copy()
    part = part.assign(key=np.int64(2**60 + 7), parent=np.int64(-1), parent_product_id=np.uint32(0),
        genidx=np.int64(0), isprim=np.int32(1), pdg=np.int32(-13), se=200., st=0.)
    step = part[G.KEY + ["key", "sim_product_id"]].rename(columns={"key": "sp", "sim_product_id": "sp_product_id"})
    donor = dict(events=donor_events, gen=gen.iloc[:1].copy(), particles=part, steps=step)
    return donor, full


def write(path, frames, update=False):
    with (uproot.update(path) if update else uproot.recreate(path)) as root:
        for name, frame in frames.items():
            cols = {column: frame[column].to_numpy() for column in frame}
            root.mktree("dump/" + name, {column: values.dtype for column, values in cols.items()})
            if len(frame):
                root["dump/" + name].extend(cols)
    return path


def pair(tmp_path, name="a", **kw):
    donor, full = payload(**kw)
    return write(tmp_path / (name + "-steps.root"), donor), write(tmp_path / (name + "-gen.root"), full)


def admit(paths, **kw):
    return N.validate([paths[0]], [paths[1]], **kw)


def test_receipt_full_exposure_hashes_actual_shards_json_and_strict_reports(tmp_path, monkeypatch):
    paths = pair(tmp_path)
    called = []
    donor_gate, full_gate = G.validate_donor, G.validate_generator_stream
    def donor(path):
        called.append(("donor", path))
        return donor_gate(path)
    def full(path):
        called.append(("full", path))
        return full_gate(path)
    monkeypatch.setattr(G, "validate_donor", donor)
    monkeypatch.setattr(G, "validate_generator_stream", full)
    result = admit(paths, declared_shards=[np.int64(101)])
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["schema"] == "crvfastsim.native_training_census.v1"
    assert result["receipt_version"] == 1
    assert result["census_admitted"] is True
    assert result["per_stage_associations_validated"] is True
    assert result["cross_stream_generator_pointer_join"] == "not_established"
    assert result["shared_generator_provenance"] == "not_established_by_census"
    assert result["physics_admitted"] is False
    assert result["total_exposure"] == result["full_generator_events"]["count"] == 4
    assert result["actual_shards"] == result["declared_shards"] == [{"run": 1400, "subrun": 101}]
    expected = np.array([[1400, 101, event] for event in range(1, 5)], dtype="<i8")
    assert result["full_generator_events"]["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    assert result["event_hash_encoding"] == N.EVENT_HASH_ENCODING
    receipt = result["pairs"][0]
    assert receipt["donor_events"]["count"] == 2 and receipt["generator_events"]["count"] == 4
    assert receipt["generator_rows"] == {"donor": 1, "full": 2}
    for label, path in zip(("steps", "generator"), paths):
        assert receipt[label]["path"] == str(path.resolve())
        assert receipt[label]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert receipt[label]["validation"]["schema_version"] == 2
    assert called == [("donor", paths[0]), ("full", paths[1])]


def test_shuffled_rows_and_columns_have_same_census_hash(tmp_path):
    a = admit(pair(tmp_path))
    donor, full = payload()
    for frames in (donor, full):
        for name, frame in frames.items():
            frames[name] = frame.iloc[::-1, ::-1]
    b = admit((write(tmp_path / "shuffled-d.root", donor), write(tmp_path / "shuffled-g.root", full)))
    assert a["full_generator_events"] == b["full_generator_events"]


def test_multiple_pairs_disjoint_events_even_in_same_actual_shard(tmp_path):
    a, b = pair(tmp_path), pair(tmp_path, "b", offset=4)
    result = N.validate([a[0], b[0]], [a[1], b[1]], declared_shards=[101])
    assert result["total_exposure"] == 8 and len(result["pairs"]) == 2
    reverse = N.validate([b[0], a[0]], [b[1], a[1]])
    assert result["full_generator_events"] == reverse["full_generator_events"]
    assert reverse["declared_shards"] is None


def test_complete_triples_not_packed_or_subrun_only(tmp_path):
    a = pair(tmp_path, run=2**32 - 1, subrun=2**31 + 1, offset=2**32 - 5)
    b = pair(tmp_path, "b", run=2**31 + 1, subrun=2**31 + 1, offset=2**32 - 5)
    declared = [{"run": 2**32 - 1, "subrun": 2**31 + 1}, {"run": 2**31 + 1, "subrun": 2**31 + 1}]
    result = N.validate([a[0], b[0]], [a[1], b[1]], declared)
    assert result["total_exposure"] == 8
    assert result["actual_shards"] == sorted(declared, key=lambda row: row["run"])
    with pytest.raises(ValueError, match="ambiguous for multiple runs"):
        N.validate([a[0], b[0]], [a[1], b[1]], [2**31 + 1])


@pytest.mark.parametrize("steps,gens", [([], []), (["d"], []), ([], ["g"]), (["d", "d"], ["g"])])
def test_equal_nonzero_counts(tmp_path, steps, gens):
    d, g = pair(tmp_path)
    lookup = {"d": d, "g": g}
    with pytest.raises(ValueError, match="equal nonzero"):
        N.validate([lookup[p] for p in steps], [lookup[p] for p in gens])


@pytest.mark.parametrize("alias", ["repeat", "symlink", "hardlink", "crossrole"])
def test_path_aliases_rejected_before_strict_reads(tmp_path, monkeypatch, alias):
    d, g = pair(tmp_path)
    monkeypatch.setattr(G, "validate_donor", lambda *a: pytest.fail("alias reached gate"))
    if alias == "crossrole":
        steps, gens = [d], [d]
    elif alias == "repeat":
        _, second_gen = pair(tmp_path, "second", subrun=102)
        steps, gens = [d, d], [g, second_gen]
    else:
        other = tmp_path / "alias.root"
        if alias == "symlink":
            other.symlink_to(d)
        else:
            other.hardlink_to(d)
        steps, gens = [d], [other]
    with pytest.raises(ValueError, match="distinct"):
        N.validate(steps, gens)


@pytest.mark.parametrize("declared,match", [
    ([], "cannot be empty"), ([101, 101], "distinct"), ([102], "do not match"),
    ([101, 102], "do not match"), ([True], "require integers"), ([101.], "require integers"),
    ([{"run": 1401, "subrun": 101}], "do not match"), ([{"run": 1400, "subrun": 102}], "do not match"),
    ([{"run": 1400, "subrun": 101.}], "must be integers"),
    ([{"subrun": 101}], "require integers"), ([{"run": 1400, "subrun": 101, "extra": 0}], "require integers"),
    ([-1], "uint32"), ([2**32], "uint32"), ("101", "must be a sequence"),
])
def test_declarations_match_actual_contents_not_filenames(tmp_path, declared, match):
    with pytest.raises(ValueError, match=match):
        admit(pair(tmp_path), declared_shards=declared)


def test_multiple_actual_shards_in_one_file(tmp_path):
    donor, full = payload()
    full["events"].loc[3, "subrun"] = np.uint32(102)
    result = admit((write(tmp_path / "d.root", donor), write(tmp_path / "g.root", full)), declared_shards=[102, 101])
    assert result["actual_shards"] == [{"run": 1400, "subrun": 101}, {"run": 1400, "subrun": 102}]


def test_wrong_pair_and_crossed_order(tmp_path):
    a, b = pair(tmp_path), pair(tmp_path, "b", subrun=102)
    with pytest.raises(ValueError, match="outside paired full-generator census"):
        N.validate([a[0], b[0]], [b[1], a[1]])


def test_full_censuses_overlap_only_in_zero_generator_non_donor_event(tmp_path):
    a = pair(tmp_path)
    donor, full = payload(offset=10)
    full["events"].loc[3, "event"] = np.uint32(4)
    b = write(tmp_path / "b-d.root", donor), write(tmp_path / "b-g.root", full)
    with pytest.raises(ValueError, match="overlapping full-generator event censuses"):
        N.validate([a[0], b[0]], [a[1], b[1]])


def test_actual_compressed_stage_different_productids_counts_reindexed_duplicates_valid(tmp_path):
    donor, full = payload()
    # SimParticle-order compression copies full idx=1 four times to idx=0..3.
    # No identity-preserving/deduplicating mapping is implied by the event census.
    extra = full["gen"].iloc[:1].copy()
    extra["idx"] = np.int64(1)
    extra["pdg"] = np.int32(13)
    full["gen"] = pd.concat([full["gen"], extra], ignore_index=True)
    full["events"].loc[0, "n_gen"] = np.int64(2)
    compressed = pd.concat([extra, extra, extra, extra], ignore_index=True)
    compressed["idx"] = np.arange(4, dtype=np.int64)
    compressed["gen_product_id"] = np.uint32(301)
    donor["gen"] = compressed
    donor["events"]["gen_product_id"] = np.uint32(301)
    donor["events"].loc[0, "n_gen"] = np.int64(4)
    donor["particles"]["gen_product_id"] = np.uint32(301)
    donor["particles"] = pd.concat([donor["particles"]] * 4, ignore_index=True)
    donor["particles"]["key"] = np.arange(4, dtype=np.int64) + np.int64(2**60 + 7)
    donor["particles"]["genidx"] = np.arange(4, dtype=np.int64)
    donor["particles"]["pdg"] = np.int32(13)
    donor["events"].loc[0, "n_particles"] = np.int64(4)
    receipt = admit((write(tmp_path / "d.root", donor), write(tmp_path / "g.root", full)))
    assert receipt["census_admitted"] is True
    assert receipt["pairs"][0]["generator_rows"] == {"donor": 4, "full": 3}
    assert receipt["total_exposure"] == 4
    assert receipt["pairs"][0]["donor_events"]["count"] == 2


def test_no_cross_stage_generator_payload_schema_or_content_assertion(tmp_path):
    donor, full = payload()
    full["gen"] = full["gen"].drop(columns=["x", "genid"])
    full["gen"]["extra"] = np.int64(7)
    full["gen"]["e"] = full["gen"].e.astype(np.float32) + np.float32(1.)
    full["gen"]["t"] = -0.
    full["gen"]["pdg"] = np.int32(13)
    receipt = admit((write(tmp_path / "d.root", donor), write(tmp_path / "g.root", full)))
    assert receipt["census_admitted"] is True
    assert receipt["physics_admitted"] is False


def test_local_zero_gen_event_need_not_be_zero_in_other_stage(tmp_path):
    donor, full = payload()
    full["events"].loc[1, "n_gen"] = np.int64(1)
    added = full["gen"].iloc[:1].copy()
    added["event"] = np.uint32(2)
    full["gen"] = pd.concat([full["gen"], added], ignore_index=True)
    receipt = admit((write(tmp_path / "d.root", donor), write(tmp_path / "g.root", full)))
    assert receipt["pairs"][0]["generator_rows"] == {"donor": 1, "full": 3}


@pytest.mark.parametrize("change,match", [("donor_role", "full_generator_stream"), ("wrong_full", "associations_checked"),
    ("legacy", "need schema-v2"), ("missing_gen", "missing dump/gen"),
    ("partial", "missing events census"), ("counts", "payload count mismatch"),
    ("ancestry", "missing referenced identity"), ("float_identity", "must be integer"),
    ("donor_genpid", "generator ProductID differs"), ("donor_genidx", "missing referenced identity"),
    ("donor_count", "payload count mismatch"), ("full_nonfinite", "nonfinite energy/time")])
def test_existing_strict_validators_remain_mandatory(tmp_path, change, match):
    donor, full = payload()
    if change == "donor_role":
        donor["events"]["full_generator_stream"] = np.uint32(1)
    elif change == "wrong_full":
        full = deepcopy(donor)
    elif change == "legacy":
        full.pop("events")
        full["gen"] = full["gen"].drop(columns="gen_product_id")
    elif change == "missing_gen":
        full.pop("gen")
    elif change == "partial":
        full.pop("events")
    elif change == "counts":
        full["events"].loc[3, "n_gen"] = 1
    elif change == "ancestry":
        donor["steps"]["sp"] = np.int64(2**60 + 8)
    elif change == "donor_genpid":
        donor["gen"]["gen_product_id"] = np.uint32(100)
    elif change == "donor_genidx":
        donor["particles"]["genidx"] = np.int64(1)
    elif change == "donor_count":
        donor["events"].loc[1, "n_gen"] = np.int64(1)
    elif change == "full_nonfinite":
        full["gen"].loc[0, "e"] = np.nan
    else:
        full["gen"]["idx"] = full["gen"].idx.astype(float)
    with pytest.raises(ValueError, match=match):
        admit((write(tmp_path / "d.root", donor), write(tmp_path / "g.root", full)))


def test_all_zero_generator_events_are_real_exposure(tmp_path):
    donor, full = payload()
    for frames in (donor, full):
        frames["events"][["n_gen", "n_particles", "n_steps"]] = np.int64(0)
        for name in frames.keys() - {"events"}:
            frames[name] = frames[name].iloc[:0]
    receipt = admit((write(tmp_path / "d.root", donor), write(tmp_path / "g.root", full)))
    assert receipt["total_exposure"] == 4
    assert receipt["pairs"][0]["generator_rows"] == {"donor": 0, "full": 0}


def test_highest_cycles_are_locally_validated(tmp_path):
    paths = pair(tmp_path)
    _, full = payload()
    full["gen"].loc[0, "gen_product_id"] = np.uint32(100)
    write(paths[1], {"gen": full["gen"]}, update=True)
    with pytest.raises(ValueError, match="generator ProductID differs from event census"):
        admit(paths)
    full["events"].loc[0, "gen_product_id"] = np.uint32(100)
    write(paths[1], {"events": full["events"]}, update=True)
    receipt = admit(paths)
    assert receipt["pairs"][0]["generator"]["validation"]["trees"]["gen"]["selected_cycle"] == 2


@pytest.mark.parametrize("target", [0, 1])
def test_drift_detected_before_return(tmp_path, monkeypatch, target):
    paths = pair(tmp_path)
    original = N._compare
    def compare(*args):
        original(*args)
        with paths[target].open("ab") as stream:
            stream.write(b"changed after reads")
    monkeypatch.setattr(N, "_compare", compare)
    with pytest.raises(ValueError, match="input drift"):
        admit(paths)


def test_mocked_int64_event_hash_precision():
    keys = pd.DataFrame({"run": np.array([2**60 + 7, 2**60 + 8], dtype=np.int64),
                         "subrun": np.array([1, 1], dtype=np.int64), "event": np.array([1, 1], dtype=np.int64)})
    assert N._summary(keys.iloc[:1])["sha256"] != N._summary(keys.iloc[1:])["sha256"]
    assert not N._index(keys).duplicated().any()
    with pytest.raises(ValueError, match="outside paired full-generator census"):
        N._compare(keys.iloc[:1], keys.iloc[1:])
