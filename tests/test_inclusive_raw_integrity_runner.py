"""Source and mocked shell checks only; no art invocation, build or saved-input mutation."""
from pathlib import Path
import subprocess

from test_inclusive_pilot_runner import environment

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_inclusive_run1a_raw_integrity.sh"


def _env(tmp_path):
    env = environment(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "inclusive-raw-run1a.art").write_bytes(b"opaque saved raw input")
    (raw / "transport.log").write_text("capture log\n")
    env.update(INCLUSIVE_RAW_DIRECTORY=str(raw), INCLUSIVE_EVENTS="1", INCLUSIVE_READBACK_EXECUTE="0")
    return env


def _run(env):
    return subprocess.run(["bash", str(RUNNER)], env=env, text=True, capture_output=True)


def test_disabled_and_unbounded_modes_do_not_create_output(tmp_path):
    env = _env(tmp_path)
    assert _run(env).returncode == 2
    assert not Path(env["INCLUSIVE_OUT"]).exists()
    env.update(INCLUSIVE_READBACK_EXECUTE="1", INCLUSIVE_EVENTS="50000")
    assert _run(env).returncode == 2
    assert not Path(env["INCLUSIVE_OUT"]).exists()


def test_chain_reads_saved_input_only_and_requires_built_readers(tmp_path):
    env = _env(tmp_path)
    env["INCLUSIVE_READBACK_EXECUTE"] = "1"
    result = _run(env)
    assert result.returncode == 77, result.stderr
    out = Path(env["INCLUSIVE_OUT"])
    chain = (out / "chain.sh").read_text()
    assert "CrvRawIntegrityDumper CrvRecordingConfigDumper" in chain
    assert "[[ -f $lib ]] || exit 2" in chain
    assert "al9-prof-e29-p107" in chain and "*/Offline/v13_39_00" in chain
    assert "--debug-config readback-validated.fcl" in chain
    assert " > exporter.log" in chain
    assert env["INCLUSIVE_RAW_DIRECTORY"] + "/inclusive-raw-run1a.art" in chain
    assert " -n " not in chain and "skipEvents" not in chain
    assert "transport.log" not in chain
    assert "convert_inclusive" not in chain and "mu2e -c capture" not in chain
    assert "muse build" not in chain and "sbatch" not in chain
    assert "RAW_EXPORT_READBACK_EXIT=0" not in result.stdout
    assert "chain.sh" in (out / "OUTPUTS.sha256").read_text()
    assert (out / "READBACK_WRAPPER_EXIT.txt").read_text().strip() == "77"
    assert (Path(env["INCLUSIVE_RAW_DIRECTORY"]) / "inclusive-raw-run1a.art").read_bytes() == b"opaque saved raw input"


def test_input_drift_refuses_without_deleting_output(tmp_path):
    env = _env(tmp_path)
    fake = Path(env["INCLUSIVE_RUNNER"])
    fake.write_text('printf export > inclusive_raw_integrity.root\nprintf drift >> "$INCLUSIVE_RAW_DIRECTORY/transport.log"\n')
    env["INCLUSIVE_READBACK_EXECUTE"] = "1"
    result = _run(env)
    assert result.returncode != 0
    out = Path(env["INCLUSIVE_OUT"])
    assert (out / "inclusive_raw_integrity.root").read_bytes() == b"export"
    assert "FAILED" in (out / "input-check.log").read_text()


def test_no_event_producer_filter_or_root_output_in_export_fcls():
    source = (ROOT / "art/CrvFastSim/src/CrvRawIntegrityDumper_module.cc").read_text()
    recording = (ROOT / "art/CrvFastSim/src/CrvRecordingConfigDumper_module.cc").read_text()
    fcl = (ROOT / "art/CrvFastSim/fcl/dump_inclusive_raw_integrity.fcl").read_text()
    extra = (ROOT / "art/CrvFastSim/fcl/dump_inclusive_recording_config.fcl").read_text()
    assert "class CrvRawIntegrityDumper : public art::EDAnalyzer" in source
    assert "class CrvRecordingConfigDumper : public art::EDAnalyzer" in recording
    assert 'scalar(tree, prefix + "_key", key, \'l\')' in source
    assert "result.member && result.available" in source
    assert 'config.getInt("g4.stepsSizeLimit", 0)' in recording
    assert "config.inputFileHash()" in recording
    assert "mu2e_ConfigTools" in (ROOT / "art/CrvFastSim/src/SConscript").read_text()
    assert "module_type : RootInput" in fcl
    assert "rawdump : {" in fcl
    assert 'rawSteps : "g4run:CRV:R3Inclusive"' in fcl
    assert "physics.integrity : [ rawdump, recording ]" in extra
    for text in (fcl, extra):
        assert "producers" not in text and "filters" not in text and "RootOutput" not in text
        assert "SelectEvents" not in text
        assert all(ord(c) < 128 for c in text)
