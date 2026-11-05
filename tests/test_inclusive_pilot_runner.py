"""Execution-scope and generated-shell tests without an art runtime or allocation."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_inclusive_run1a_pilot.sh"


def environment(tmp_path):
    input_file = tmp_path / "input.art"
    input_file.write_bytes(b"test input")
    fake = tmp_path / "fake_runner.sh"
    fake.write_text('bash -n "$OFFLINE_CHAIN"\nprintf "runtime intentionally unavailable\\n"\nexit 77\n')
    return {**os.environ, "SLURM_JOB_ID": "test", "INCLUSIVE_INPUT": str(input_file),
            "INCLUSIVE_OUT": str(tmp_path / "result"), "INCLUSIVE_CODE": str(ROOT),
            "INCLUSIVE_RUNTIME": str(tmp_path / "runtime"), "INCLUSIVE_RUNNER": str(fake),
            "INCLUSIVE_SEED": "202611041", "CVMFSEXEC_DIST": str(tmp_path / "cvmfs"),
            "OFFLINE_IMAGE": str(tmp_path / "image"), "OFFLINE_BINDS": str(tmp_path),
            "INCLUSIVE_EXECUTE": "0", "INCLUSIVE_EVENTS": "preflight"}


def invoke(env):
    return subprocess.run(["bash", str(RUNNER)], env=env, text=True, capture_output=True)


def test_execution_disabled_before_output_creation(tmp_path):
    env = environment(tmp_path)
    result = invoke(env)
    assert result.returncode == 2
    assert "Execution disabled" in result.stdout
    assert not Path(env["INCLUSIVE_OUT"]).exists()


@pytest.mark.parametrize("events", ["0", "2", "11", "1000", "-1", "1; false"])
def test_only_explicit_bounded_modes(tmp_path, events):
    env = environment(tmp_path)
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_EVENTS=events)
    assert invoke(env).returncode == 2
    assert not Path(env["INCLUSIVE_OUT"]).exists()


@pytest.mark.parametrize("events", ["preflight", "1", "10"])
def test_generated_chain_syntax_and_scope(tmp_path, events):
    env = environment(tmp_path)
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_EVENTS=events)
    result = invoke(env)
    assert result.returncode == 77, result.stderr
    chain = (Path(env["INCLUSIVE_OUT"]) / "chain.sh").read_text()
    assert 'backing=${executable%/build/*}' in chain
    assert "--debug-config validated.fcl" in chain
    assert 'Offline/Print/src/StatusG4Printer.cc' in chain
    if events == "preflight":
        assert "transport.log" not in chain
        assert "inclusive-run1a.art" not in chain
    else:
        assert f"-n {events} > transport.log" in chain
        assert chain.index("status.log") < chain.index(" > dump.log")
        assert f"-n {events} > input-dump.log" in chain


def test_exclusive_output_and_integer_seed(tmp_path):
    env = environment(tmp_path)
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_SEED="1.5")
    assert invoke(env).returncode == 2
    output = Path(env["INCLUSIVE_OUT"])
    assert not output.exists()
    env["INCLUSIVE_SEED"] = "123"
    output.mkdir()
    (output / "keep.txt").write_text("existing")
    assert invoke(env).returncode == 2
    assert (output / "keep.txt").read_text() == "existing"


@pytest.mark.parametrize("skip,expected,events", [("1", "", "1"), ("-1", "1:2:3", "1"),
    ("01", "1:2:3", "1"), ("1;exit", "1:2:3", "1"), ("999999999", "1:2:3", "1"),
    ("1", "1:2:3", "10"), ("1", "1:2", "1")])
def test_edge_skip_requires_bounded_expected_event(tmp_path, skip, expected, events):
    env = environment(tmp_path)
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_EVENTS=events,
               INCLUSIVE_SKIP=skip, INCLUSIVE_EXPECT_EVENT=expected)
    assert invoke(env).returncode == 2
    assert not Path(env["INCLUSIVE_OUT"]).exists()


def test_edge_chain_checks_original_event_before_transport_and_status_before_steps(tmp_path):
    env = environment(tmp_path)
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_EVENTS="1", INCLUSIVE_SKIP="16482",
               INCLUSIVE_EXPECT_EVENT="1400:101:16483")
    assert invoke(env).returncode == 77
    out = Path(env["INCLUSIVE_OUT"])
    assert "source.skipEvents : 16482" in (out / "pilot.fcl").read_text()
    assert "source.skipEvents : 16482" in (out / "dump_input.fcl").read_text()
    chain = (out / "chain.sh").read_text()
    assert chain.index("input-dump.log") < chain.index("EXPECTED_EVENT.txt") < chain.index("transport.log")
    assert "n==1" in chain
    assert chain.index("status.log") < chain.index("n==expected") < chain.index(" > dump.log")
