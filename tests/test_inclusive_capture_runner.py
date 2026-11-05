"""Raw-first configuration and disabled bounded runner checks without art execution."""
import os
from pathlib import Path
import subprocess

import pytest

from test_inclusive_pilot_runner import environment

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_inclusive_run1a_capture.sh"


def invoke(env):
    return subprocess.run(["bash", str(RUNNER)], env=env, text=True, capture_output=True)


def test_capture_and_conversion_are_separate_jobs():
    capture = (ROOT / "grid/fcl/ref_inclusive_capture_run1a.fcl").read_text()
    conversion = (ROOT / "grid/fcl/convert_inclusive_capture_run1a.fcl").read_text()
    dump = (ROOT / "art/CrvFastSim/fcl/dump_inclusive_capture_steps.fcl").read_text()
    assert '#include "ref_inclusive_run1a.fcl"' in capture
    assert "physics.PrimaryPath : [ g4run ]" in capture
    assert "SelectEvents" not in capture
    assert "keep mu2e::CrvSteps" not in capture
    for product in ("SimParticlemv_g4run__R3Inclusive", "StatusG4_g4run__R3Inclusive",
                    "StepPointMCs_g4run_CRV_R3Inclusive", "GenParticles_cosmicTimeOffset__Primary"):
        assert product in capture and product in conversion
    assert "physics.PrimaryPath : [ CrvSteps ]" in conversion
    assert "process_name : R3InclusiveSteps" in conversion
    assert "CrvSteps_CrvSteps__R3InclusiveSteps" in conversion
    assert '"CrvSteps::R3InclusiveSteps"' in dump
    assert "maxSimParticleCollectionSize" not in capture + conversion
    assert all(ord(c) < 128 for c in capture + conversion + dump)


def test_disabled_before_output(tmp_path):
    env = environment(tmp_path)
    result = invoke(env)
    assert result.returncode == 2
    assert "Execution disabled" in result.stdout
    assert not Path(env["INCLUSIVE_OUT"]).exists()


@pytest.mark.parametrize("mode", ["0", "2", "11", "50000", "-1", "1;false"])
def test_bounded_modes_only(tmp_path, mode):
    env = environment(tmp_path)
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_EVENTS=mode)
    assert invoke(env).returncode == 2
    assert not Path(env["INCLUSIVE_OUT"]).exists()


@pytest.mark.parametrize("mode", ["preflight", "1", "10"])
def test_generated_chain_never_runs_conversion_or_step_dumper(tmp_path, mode):
    env = environment(tmp_path)
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_EVENTS=mode)
    result = invoke(env)
    assert result.returncode == 77, result.stderr
    out = Path(env["INCLUSIVE_OUT"])
    chain = (out / "chain.sh").read_text()
    assert 'backing=${executable%/build/*}' in chain
    assert "--debug-config conversion-validated.fcl" in chain
    assert " > dump.log" not in chain
    assert "dump_inclusive_capture_steps" not in chain
    assert "n==expected" not in chain
    assert "RAW_CAPTURE_READBACK_EXIT=0" not in result.stdout
    assert (out / "CAPTURE_WRAPPER_EXIT.txt").read_text().strip() == "77"
    assert "chain.sh" in (out / "OUTPUTS.sha256").read_text()
    if mode == "preflight":
        assert "transport.log" not in chain
        assert " > status.log" not in chain
    else:
        assert f"-n {mode} > transport.log" in chain
        assert chain.index("TRANSPORT_EXIT.txt") < chain.index(" > status.log") < chain.index("raw-gen-dump.log")
        assert "inclusive-raw-run1a.art" in chain
        assert "|| exit" in chain


@pytest.mark.parametrize("skip,expected,mode", [("1", "", "1"), ("-1", "1:2:3", "1"),
    ("01", "1:2:3", "1"), ("1;false", "1:2:3", "1"), ("999999999", "1:2:3", "1"),
    ("1", "1:2:3", "10"), ("1", "1:2", "1")])
def test_capture_edge_scope(tmp_path, skip, expected, mode):
    env = environment(tmp_path)
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_EVENTS=mode,
               INCLUSIVE_SKIP=skip, INCLUSIVE_EXPECT_EVENT=expected)
    assert invoke(env).returncode == 2
    assert not Path(env["INCLUSIVE_OUT"]).exists()


def test_capture_selected_input_gate_precedes_g4(tmp_path):
    env = environment(tmp_path)
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_EVENTS="1", INCLUSIVE_SKIP="16482",
               INCLUSIVE_EXPECT_EVENT="1400:101:16483")
    assert invoke(env).returncode == 77
    out = Path(env["INCLUSIVE_OUT"])
    assert "source.skipEvents : 16482" in (out / "capture.fcl").read_text()
    assert "source.skipEvents : 16482" in (out / "dump_input.fcl").read_text()
    assert "source.skipEvents" not in (out / "dump_raw_gen.fcl").read_text()
    chain = (out / "chain.sh").read_text()
    assert chain.index("input-dump.log") < chain.index("EXPECTED_EVENT.txt") < chain.index("transport.log")


@pytest.mark.parametrize("rc", [0, 42])
def test_wrapper_retains_and_hashes_raw_on_success_or_readback_failure(tmp_path, rc):
    env = environment(tmp_path)
    fake = Path(env["INCLUSIVE_RUNNER"])
    fake.write_text('bash -n "$OFFLINE_CHAIN"\nprintf raw-sentinel > inclusive-raw-run1a.art\n'
                    'printf "status: 10 overflowSimP: 1\\n" > status.log\n'
                    f'exit {rc}\n')
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_EVENTS="1")
    result = invoke(env)
    assert result.returncode == rc, result.stderr
    out = Path(env["INCLUSIVE_OUT"])
    assert (out / "inclusive-raw-run1a.art").read_text() == "raw-sentinel"
    assert "status: 10" in (out / "status.log").read_text()
    assert "inclusive-raw-run1a.art" in (out / "OUTPUTS.sha256").read_text()
    check = subprocess.run(["shasum", "-a", "256", "-c", "OUTPUTS.sha256"], cwd=out,
                           text=True, capture_output=True)
    assert check.returncode == 0, check.stderr
    assert ("CONVERSION_AUTHORIZED=false" in result.stdout) == (rc == 0)


def test_source_drift_refuses_but_keeps_diagnostic_art(tmp_path):
    env = environment(tmp_path)
    fake = Path(env["INCLUSIVE_RUNNER"])
    fake.write_text('printf raw-sentinel > inclusive-raw-run1a.art\nprintf drift >> "$0"\n')
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_EVENTS="1")
    assert invoke(env).returncode != 0
    out = Path(env["INCLUSIVE_OUT"])
    assert (out / "inclusive-raw-run1a.art").exists()
    assert "FAILED" in (out / "source-hash-check.log").read_text()


def test_exclusive_output_integer_seed_and_symlink(tmp_path):
    env = environment(tmp_path)
    env.update(INCLUSIVE_EXECUTE="1", INCLUSIVE_SEED="1.5")
    assert invoke(env).returncode == 2
    out = Path(env["INCLUSIVE_OUT"])
    assert not out.exists()
    env["INCLUSIVE_SEED"] = "123"
    out.symlink_to(tmp_path / "absent")
    assert invoke(env).returncode == 2
    assert out.is_symlink()
