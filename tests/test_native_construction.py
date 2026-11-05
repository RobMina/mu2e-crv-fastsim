"""Construction restriction does not reject valid compressed-stage census admission."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from crvfastsim import native_construction as C, native_training as N
from test_native_training import payload, write


def files(tmp_path, multiple=None, compressed=False):
    donor, full = payload()
    if multiple == "donor":
        extra = donor["particles"].copy()
        extra["key"] += 1
        donor["particles"] = pd.concat([donor["particles"], extra], ignore_index=True)
        donor["events"].loc[0, "n_particles"] += 1
    if multiple == "full":
        extra = full["gen"].iloc[:1].copy().assign(idx=np.int64(1))
        full["gen"] = pd.concat([full["gen"], extra], ignore_index=True)
        full["events"].loc[0, "n_gen"] += 1
    if compressed:
        extra = donor["gen"].copy().assign(idx=np.int64(1))
        donor["gen"] = pd.concat([donor["gen"], extra], ignore_index=True)
        donor["events"].loc[0, "n_gen"] += 1
    return write(tmp_path / "donor.root", donor), write(tmp_path / "full.root", full)


def test_supported_receipt_and_compressed_generator_duplicates(tmp_path):
    d, g = files(tmp_path, compressed=True)
    census = N.validate([d], [g], [101])
    receipt = C.require_supported(census)
    assert receipt["supported"] is True and receipt["events_filtered"] == 0
    assert receipt["pairs"][0]["steps_multiplicity"]["primary_muons"] == 1
    assert receipt["pairs"][0]["generator_multiplicity"]["primary_muons"] == 2
    manifest = dict(training_census=census, construction_support=receipt)
    assert C.check_receipt(manifest, dict(construction_support=deepcopy(receipt))) == receipt


@pytest.mark.parametrize("role", ["donor", "full"])
def test_valid_multi_muon_census_refuses_construction_without_filtering(tmp_path, role):
    d, g = files(tmp_path, multiple=role)
    census = N.validate([d], [g])
    assert census["census_admitted"] is True and census["total_exposure"] == 4
    with pytest.raises(ValueError, match="multiple primary muons.*no events were filtered"):
        C.require_supported(census)


@pytest.mark.parametrize("when", ["before", "during"])
def test_source_drift_refuses_support_receipt(tmp_path, monkeypatch, when):
    d, g = files(tmp_path)
    census = N.validate([d], [g])
    if when == "before":
        with d.open("ab") as stream:
            stream.write(b"drift")
    else:
        original = C.multiplicity
        def drift(*args, **kw):
            result = original(*args, **kw)
            with d.open("ab") as stream:
                stream.write(b"drift")
            return result
        monkeypatch.setattr(C, "multiplicity", drift)
    with pytest.raises(ValueError, match="input drift"):
        C.require_supported(census)


@pytest.mark.parametrize("change", ["missing", "source", "scope", "multi", "float", "pair"])
def test_build_bound_receipt_rejects_incomplete_or_unsupported_evidence(tmp_path, change):
    d, g = files(tmp_path)
    census = N.validate([d], [g])
    receipt = C.require_supported(census)
    manifest = dict(training_census=census, construction_support=deepcopy(receipt))
    check = dict(construction_support=deepcopy(receipt))
    for item in (manifest, check):
        evidence = item["construction_support"]
        if change == "missing":
            item.pop("construction_support")
        elif change == "source":
            evidence["input_sha256"][str(d)] = "0" * 64
        elif change == "scope":
            evidence["physics_admitted"] = True
        elif change == "pair":
            evidence["pairs"] = []
        else:
            evidence["pairs"][0]["steps_multiplicity"]["multiple_muon_events"] = 1 if change == "multi" else 0.
    with pytest.raises(ValueError, match="construction support|multiplicity"):
        C.check_receipt(manifest, check)


def test_full_integer_primary_keys_are_not_generator_row_counts():
    frame = pd.DataFrame(dict(run=[1, 1], subrun=[1, 1], event=[1, 1],
        key=np.array([2**60, 2**60 + 1], dtype=np.int64), pdg=[13, 13], isprim=[1, 1]))
    assert C.multiplicity(frame, primary=True)["multiple_muon_events"] == 1
    assert C.multiplicity(frame.iloc[:0], primary=True)["max_muons_per_event"] == 0
