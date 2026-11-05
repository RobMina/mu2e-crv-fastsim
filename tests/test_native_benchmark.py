"""Synthetic ordinary-call timing contracts; no detector assets or physics admission."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import native_record as N, runconfig
from scripts import bench_native_chain as B
from test_e2e_bench import detail


@pytest.fixture
def harness(tmp_path, monkeypatch):
    source, asset, generator = [tmp_path / name for name in ("source.py", "asset", "gen.root")]
    for path in (source, asset, generator):
        path.write_text("synthetic immutable bytes")
    rc = runconfig.get("run1a_v01")
    cfg = rc.chain_config()
    spec = SimpleNamespace(describe=lambda: {"native": True})
    universe = pd.DataFrame(dict(run=[1400, 1400], subrun=[2, 2], event=[3, 4]), dtype=np.int64)
    before = {str(p.resolve()): N.tier1b_assets.sha256(p) for p in (source, asset, generator)}
    state = dict(calls=[], mutate=None, universe=universe, source=source, asset=asset, generator=generator)
    clock = iter(range(1000))
    monkeypatch.setattr(B, "time", SimpleNamespace(perf_counter=lambda: float(next(clock))))
    monkeypatch.setattr(B, "source_paths", lambda: [source])
    monkeypatch.setattr(B, "software_hardware", lambda: {"synthetic": True})
    monkeypatch.setattr(B.precision, "provenance", lambda: {"device_float": "float32", "host": "float64"})
    monkeypatch.setattr(B.R, "_resolve", lambda *a, **kw: (rc, cfg, str(asset), {}, spec))
    monkeypatch.setattr(B.R, "_native_generator_input", lambda *a: (universe.iloc[:1], universe))
    monkeypatch.setattr(N, "snapshot", lambda *a: dict(before))

    def ordinary(inputs, outdir, **kw):
        assert inputs == [str(generator)]
        assert not {"benchmark", "prepared", "jax", "max_events", "compare", "emit"} & kw.keys()
        state["calls"].append((outdir, kw))
        steps = universe.iloc[:1].assign(edep=1.)
        output = detail()
        if state["mutate"]:
            state["mutate"](len(state["calls"]), output)
        info = N.write(kw["native_output"], state.get("record_universe", universe), steps, steps,
                       dict(chan=np.array([1]), starttdc=np.array([0]), adc=np.zeros((1, 12))),
                       [output], before, rc, cfg, spec, {}, seed=kw["seed"])
        result = dict(generated_events=state.get("generated_events", len(universe)),
                      wall_s=state.get("wall_s", .25), timers={"outer_s": 4., "nested_s": 3.},
                      native_record=info, conditions={"synthetic": True})
        (Path(outdir) / "runchain_stats.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(B.R, "run", ordinary)
    config = dict(generator=str(generator), run_config="run1a_v01", native_library_dir=str(tmp_path),
                  conditions_manifest=str(asset), gdml=str(asset), seed=19, window="offspill", repeat=2)
    state.update(config=config, outdir=tmp_path / "new")
    return state


def invoke(state):
    return B.run(state["config"], state["outdir"])


def incomplete(state):
    assert not (state["outdir"] / "native_benchmark.json").exists()


def test_ordinary_reload_full_census_and_independent_timer_closure(harness):
    report = invoke(harness)
    assert report["scope"] == B.SCOPE and report["physics_admitted"] is False
    assert report["context_policy"] == "reload_each_pass"
    assert report["repeat_fingerprints_identical"] is True
    assert report["generated_events"] == 2  # event 4 has no generator/response rows
    assert [p["stage"] for p in report["passes"]] == ["first", "warm"]
    assert len({p["array_fingerprint"] for p in report["passes"]}) == 1
    for index, p in enumerate(report["passes"]):
        assert p["elapsed_s"] == 1.
        assert p["generated_events_per_s"] == 2.
        assert p["runchain"]["wall_s"] == .25
        assert p["runchain"]["timers"] == {"outer_s": 4., "nested_s": 3.}  # not additive
        assert p["counts"]["steps"] == p["counts"]["pulses"] == p["counts"]["clusters"] == 1
        with np.load(harness["outdir"] / f"pass_{index:03d}" / "native.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["generated_keys"], harness["universe"])
    calls = harness["calls"]
    assert calls[0][0] != calls[1][0]
    assert calls[0][1]["seed"] == calls[1][1]["seed"] == 19
    saved = json.loads((harness["outdir"] / "native_benchmark.json").read_text())
    assert saved["status"] == "complete" and saved["inputs_sha256"] and saved["output_sha256"]
    assert saved["invocation_elapsed_before_report_s"] >= saved["passes_elapsed_s"]


def test_repeat_array_identity_drift(harness):
    harness["mutate"] = lambda index, out: out["cl_pe"].__setitem__(0, 18. if index == 2 else 17.)
    with pytest.raises(ValueError, match="repeat fingerprint"):
        invoke(harness)
    incomplete(harness)


@pytest.mark.parametrize("which", ["source", "asset", "generator"])
def test_input_source_asset_drift(harness, which):
    def change(index, out):
        if index == 2:
            harness[which].write_text("changed bytes")
    harness["mutate"] = change
    with pytest.raises(ValueError, match="drift"):
        invoke(harness)
    incomplete(harness)


@pytest.mark.parametrize("repeat", [0, 1, -1, True, 2.5])
def test_repeat_admission(harness, repeat):
    harness["config"]["repeat"] = repeat
    with pytest.raises(ValueError, match="repeat"):
        invoke(harness)
    assert not harness["calls"] and not harness["outdir"].exists()


@pytest.mark.parametrize("kind", ["directory", "file", "dangling_symlink"])
def test_destination_no_clobber(harness, kind):
    path = harness["outdir"]
    if kind == "directory":
        path.mkdir()
    elif kind == "file":
        path.write_text("keep")
    else:
        path.symlink_to(path.parent / "nonexistent")
    with pytest.raises(FileExistsError):
        invoke(harness)
    assert not harness["calls"]


def test_full_exposure_cannot_be_truncated_or_rematched(harness):
    harness["record_universe"] = harness["universe"].iloc[:1]
    with pytest.raises(ValueError, match="census/exposure"):
        invoke(harness)
    incomplete(harness)


def test_runchain_denominator_must_match_census(harness):
    harness["generated_events"] = 1
    with pytest.raises(ValueError, match="census/exposure"):
        invoke(harness)
    incomplete(harness)


@pytest.mark.parametrize("wall", [-1., 2., float("nan")])
def test_wall_timer_must_close_inside_measured_call(harness, wall):
    harness["wall_s"] = wall
    with pytest.raises(ValueError, match="timer closure"):
        invoke(harness)
    incomplete(harness)


@pytest.mark.parametrize("option", ["prepared", "benchmark", "max_events", "jax", "emit", "compare"])
def test_no_benchmark_or_untracked_extra_options(harness, option):
    harness["config"][option] = None
    with pytest.raises(ValueError, match="config"):
        invoke(harness)
    assert not harness["calls"]


def test_optional_reference_is_explicit_unmatched_and_not_a_multiplier(harness):
    harness["config"]["stock_rate"] = dict(events_per_s=21.2, label="historical illustrative stock observation")
    report = invoke(harness)
    ref = report["optional_reference_basis"]
    assert ref["label"] == harness["config"]["stock_rate"]["label"]
    assert ref["matched_Run1A"] is False and ref["events_per_s"] == 21.2
    assert "multiplier" not in json.dumps(report)


def test_stock_reference_requires_explicit_label(harness):
    harness["config"]["stock_rate"] = dict(events_per_s=21.2)
    with pytest.raises(ValueError, match="stock_rate"):
        invoke(harness)


def test_nonfinite_report_never_published(harness, monkeypatch):
    monkeypatch.setattr(B, "software_hardware", lambda: {"bad": float("nan")})
    with pytest.raises(ValueError, match="JSON"):
        invoke(harness)
    incomplete(harness)
