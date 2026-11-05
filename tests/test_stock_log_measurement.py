"""Tiny local ROOT censuses and stock-log fixtures; no jobs or remote access."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import uproot

from scripts import measure_stock_logs as M


def logtext(n=2, cpu="3.0", real="4.0", elapsed="54:34.38"):
    return (f"Config=run1a\nTrigReport Events total = {n} passed = 1 failed = {n - 1}\n"
            f"TimeReport CPU = {cpu} Real = {real}\n"
            "Art has completed and will exit with status 0.\n"
            f'\tCommand being timed: "mu2e -c fcl/shard.fcl -n {n}"\n'
            f"\tElapsed (wall clock) time (h:mm:ss or m:ss): {elapsed}\n"
            "\tExit status: 0\n")


def generator(path, n, run, subrun, legacy=False, donor=False, event_duplicate=False):
    u32 = lambda x: np.full(n, x, dtype=np.uint32)
    i64 = lambda x: np.full(n, x, dtype=np.int64)
    keys = dict(run=u32(run), subrun=u32(subrun), event=np.arange(n, dtype=np.uint32))
    if event_duplicate:
        keys["event"][:] = 0
    with uproot.recreate(path) as root:
        if not legacy:
            root.mktree("dump/events", dict(keys, schema_version=u32(2), associations_checked=u32(int(donor)),
                        full_generator_stream=u32(int(not donor)), gen_product_id=u32(101),
                        sim_product_id=u32(0), n_gen=i64(1), n_particles=i64(0), n_steps=i64(0)))
        payload = dict(keys, idx=i64(0), pdg=i64(13), e=np.full(n, 200.), t=np.zeros(n))
        if not legacy:
            payload["gen_product_id"] = u32(101)
        root.mktree("dump/gen", payload)
    return path


@pytest.fixture
def records(tmp_path):
    result = []
    for i in range(3):
        n = i + 2
        log = tmp_path / f"worker_{i}.log"
        log.write_text(logtext(n, cpu=str(2 * n), real=str(n + 1)))
        table = generator(tmp_path / f"gen_{i}.root", n, 1400, i + 1)
        result.append(dict(path=str(log), generator_table=str(table), generated=n, run=1400,
                           subrun=i + 1, configuration=M.CONFIGURATION, stage=M.STAGE))
    return result


def test_measured_aggregate_uses_sum_events_over_sum_real_not_mean_rate(records):
    report = M.measure(records)
    assert report["jobs"] == 3 and report["generated"] == 9
    assert report["summary"]["real_s"]["sum_s"] == 12
    assert report["summary"]["real_s"]["generated_per_s"] == 9 / 12
    assert report["summary"]["cpu_s"]["generated_per_s"] == 9 / 18
    assert report["summary"]["real_s"]["per_job_generated_per_s"] == [2 / 3, 3 / 4, 4 / 5]
    assert not report["physics_admitted"] and not report["matched_to_fast_boundary"]
    assert report["full_chain_speedup"] == "not established"
    assert len(report["input_sha256"]) == 6
    assert report["records"][0]["gnu_elapsed_s"] == 3274.38
    assert report["records"][0]["generator_validation"]["counts"]["events"] == 2


@pytest.mark.parametrize("old,new", [
    ("total = 2", "total = -2"), ("passed = 1", "passed = +1"),
    ("passed = 1", "passed = 3"), ("failed = 1", "failed = 0"),
    ("failed = 1", "failed = -1"), ("total = 2", "total = 3"),
    ("total = 2", "total = 9223372036854775808"),
    ("status 0.", "status 1."), ("Exit status: 0", "Exit status: 1"),
    ("Exit status: 0", "Exit status: -1"),
    ("CPU = 3.0", "CPU = nan"), ("Real = 4.0", "Real = inf"),
    ("Real = 4.0", "Real = 0"), ("CPU = 3.0", "CPU = -1"),
    ("54:34.38", "54:60"), ("54:34.38", "1:60:00"),
    ("54:34.38", "-1:30"), ("54:34.38", "nan"),
    ("Config=run1a", "Config=run2"), ("Config=run1a", ""),
    ("-n 2", "-n 3"),
    ("TimeReport CPU", "TimeReport BAD"),
])
def test_malformed_failed_signed_bounds_nonfinite(old, new):
    with pytest.raises(ValueError):
        M.parse_log(logtext().replace(old, new), 2)


@pytest.mark.parametrize("extra", [
    "TimeReport CPU = 3 Real = 4\n", "TrigReport Events total = 2 passed = 1 failed = 1\n",
    "Art has completed and will exit with status 0.\n", "Exit status: 0\n",
    'Command being timed: "mu2e -c fcl/shard.fcl -n 2"\n', "Config=run2\n",
])
def test_multiple_or_conflicting_blocks_fail(extra):
    with pytest.raises(ValueError):
        M.parse_log(logtext() + extra, 2)


def test_gnu_optional_and_hours_and_configuration_path():
    text = logtext().split('\tCommand being timed:')[0].replace("Config=run1a", '#include "ref_sim_run1a.fcl"')
    report = M.parse_log(text, 2)
    assert "gnu_elapsed_s" not in report
    assert M.parse_log(logtext(elapsed="1:02:03.25"), 2)["gnu_elapsed_s"] == 3723.25


def test_partial_gnu_block_rejected():
    with pytest.raises(ValueError):
        M.parse_log(logtext().replace("\tExit status: 0\n", ""), 2)


@pytest.mark.parametrize("key,value", [("run", -1), ("subrun", 2**32), ("generated", True),
    ("generated", 2.0), ("generated", 0), ("run", "1400"), ("configuration", "run2"),
    ("stage", "full_chain")])
def test_declared_metadata_strict(records, key, value):
    records[0][key] = value
    with pytest.raises(ValueError):
        M.measure(records)


def test_census_binding_required_not_log_metadata(records):
    records[0]["subrun"] = 10
    with pytest.raises(ValueError, match="actual full-generator census"):
        M.measure(records)


@pytest.mark.parametrize("legacy,donor,duplicate", [(True, False, False), (False, True, False), (False, False, True)])
def test_strict_actual_generator_stream_required(records, legacy, donor, duplicate):
    generator(records[0]["generator_table"], 2, 1400, 1, legacy, donor, duplicate)
    with pytest.raises(ValueError):
        M.measure(records)


def test_census_generated_count_mismatch(records):
    generator(records[0]["generator_table"], 3, 1400, 1)
    with pytest.raises(ValueError, match="actual full-generator census"):
        M.measure(records)


def test_duplicate_actual_identity(records):
    generator(records[1]["generator_table"], 3, 1400, 1)
    records[1]["subrun"] = 1
    with pytest.raises(ValueError, match="duplicate actual"):
        M.measure(records)


def test_duplicate_paths_and_content(records):
    records[1]["path"] = records[0]["path"]
    with pytest.raises(ValueError, match="duplicate"):
        M.measure(records)


def test_duplicate_content_in_distinct_files(records):
    Path(records[1]["path"]).write_text(Path(records[0]["path"]).read_text())
    with pytest.raises(ValueError, match="duplicate log content"):
        M.measure(records)


def test_sample_size(records):
    with pytest.raises(ValueError, match="3 or 10"):
        M.measure(records[:2])


def test_gzip_local_artifact(records):
    path = Path(records[0]["path"])
    compressed = path.with_suffix(".log.gz")
    with gzip.open(compressed, "wt") as stream:
        stream.write(path.read_text())
    records[0]["path"] = str(compressed)
    assert M.measure(records)["generated"] == 9


def test_input_drift_rejected(records, monkeypatch):
    original = M.genassoc.validate_generator_stream
    def change_after_validation(path):
        result = original(path)
        with open(records[0]["path"], "a") as stream:
            stream.write("changed after parsing\n")
        return result
    monkeypatch.setattr(M.genassoc, "validate_generator_stream", change_after_validation)
    with pytest.raises(ValueError, match="input hash drift"):
        M.measure(records)


def test_cli_exclusive_artifact_and_manifest_hash(records, tmp_path):
    spec, output = tmp_path / "records.json", tmp_path / "measurement.json"
    spec.write_text(json.dumps(records))
    M.main([str(spec), "--out", str(output)])
    report = json.loads(output.read_text())
    assert report["input_sha256"][str(spec)] == M.sha256(spec)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        M.main([str(spec), "--out", str(output)])
    assert output.read_bytes() == before


def test_ten_disjoint_actual_jobs(records, tmp_path):
    for i in range(3, 10):
        log = tmp_path / f"worker_{i}.log"
        log.write_text(logtext(n=i + 2))
        table = generator(tmp_path / f"gen_{i}.root", i + 2, 1400, i + 1)
        records.append(dict(path=str(log), generator_table=str(table), generated=i + 2,
                            run=1400, subrun=i + 1, configuration=M.CONFIGURATION, stage=M.STAGE))
    report = M.measure(records)
    assert report["jobs"] == 10 and report["generated"] == 65


def test_nonfinite_aggregate_rate_rejected(records):
    Path(records[0]["path"]).write_text(logtext(n=2, real="5e-324"))
    with pytest.raises(ValueError, match="nonfinite aggregate"):
        M.measure(records)


def test_log_size_limit(records, monkeypatch):
    monkeypatch.setattr(M, "MAX_LOG_BYTES", 1)
    with pytest.raises(ValueError, match="small-file"):
        M.measure(records)


def summarytext(n=50000, subrun=102, shard=0, elapsed="49:50.69", user="2937.24"):
    return ("Sun Oct 25 00:00:00 UTC 2026 BEGIN EXECUTION grid_ref_sim.sh /storage-test/g4ref_run1a_v1\n"
            f"=== grid_ref_sim: shard {shard} subrun {subrun} seed 20264201 nevts {n} host worker.test Sun Oct 25\n"
            "/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39\n"
            f"TrigReport Events total = {n} passed = 1 failed = {n - 1}\n"
            "MemReport  VmHWM = 4300000 kB\n"
            f" User time (seconds): {user}\n"
            f" Elapsed (wall clock) time (h:mm:ss or m:ss): {elapsed}\n"
            f"=== grid_ref_sim: done shard {shard}\n"
            "Mon Oct 26 00:00:00 UTC 2026 grid_ref_sim.sh COMPLETED with exit status 0\n")


def parse_summary(text=None, **kwargs):
    return M.parse_log(summarytext() if text is None else text, 50000,
                       format="worker_summary", run=1400, subrun=102, **kwargs)


@pytest.fixture
def summary_records(records):
    for i, record in enumerate(records):
        record["format"] = "worker_summary"
        Path(record["path"]).write_text(summarytext(record["generated"], record["subrun"], i,
                                                   elapsed=f"0:{i + 3:02d}"))
    return records


def test_actual_summary_no_fake_art_or_total_cpu():
    report = parse_summary()
    assert report["format"] == "worker_summary"
    assert report["gnu_elapsed_s"] == 2990.69
    assert report["gnu_user_s"] == 2937.24
    assert report["worker_exit_status"] == 0 and report["shard"] == 0
    assert report["cpu_total"] == "UNAVAILABLE"
    assert not {"cpu_s", "real_s", "gnu_system_s", "gnu_exit_status", "art_exit_status"} & report.keys()
    assert report["configuration_evidence"]["kind"] == "worker_output_directory_label"
    assert report["configuration_evidence"]["expanded_configuration_verified"] is False
    assert M.parse_log(summarytext(elapsed="1:02:03.25"), 50000,
                       format="worker_summary", subrun=102)["gnu_elapsed_s"] == 3723.25


def test_summary_format_is_explicit_and_subrun_required():
    with pytest.raises(ValueError):
        M.parse_log(summarytext(), 50000)
    with pytest.raises(ValueError, match="subrun"):
        M.parse_log(summarytext(), 50000, format="worker_summary")
    with pytest.raises(ValueError, match="unknown log format"):
        M.parse_log(summarytext(), 50000, format="auto")
    with pytest.raises(ValueError):
        M.parse_log(logtext(), 2, format="worker_summary", subrun=102)


@pytest.mark.parametrize("old,new", [
    ("total = 50000", "total = 49999"), ("passed = 1", "passed = 50001"),
    ("failed = 49999", "failed = 49998"), ("total = 50000", "total = -50000"),
    ("subrun 102", "subrun 103"), ("subrun 102", "subrun -102"),
    ("nevts 50000", "nevts 2"), ("nevts 50000", "nevts +50000"),
    ("done shard 0", "done shard 1"), ("done shard 0", "done shard -1"),
    ("=== grid_ref_sim: shard 0", "=== grid_ref_sim: broken 0"),
    ("=== grid_ref_sim: done shard 0\n", ""),
    ("COMPLETED with exit status 0", "COMPLETED with exit status 1"),
    ("COMPLETED with exit status 0", "COMPLETED with exit status -1"),
    ("COMPLETED with exit status 0", "terminated"),
    ("g4ref_run1a_v1", "g4ref_run2_v1"),
    ("BEGIN EXECUTION", "BEGIN"),
    ("User time (seconds): 2937.24", "User time (seconds): nan"),
    ("User time (seconds): 2937.24", "User time (seconds): -1"),
    ("User time (seconds): 2937.24", "missing user time"),
    ("49:50.69", "49:60"), ("49:50.69", "1:60:00"),
    ("49:50.69", "nan"), ("49:50.69", "0:00"),
])
def test_summary_bad_counts_header_completion_and_timing(old, new):
    with pytest.raises(ValueError):
        parse_summary(summarytext().replace(old, new))


@pytest.mark.parametrize("extra", [
    "=== grid_ref_sim: shard 0 subrun 102 seed 1 nevts 50000 host test\n",
    "=== grid_ref_sim: done shard 0\n",
    "date grid_ref_sim.sh COMPLETED with exit status 0\n",
    "date BEGIN EXECUTION grid_ref_sim.sh /storage-test/g4ref_run1a_v1\n",
    "TrigReport Events total = 50000 passed = 1 failed = 49999\n",
    "User time (seconds): 2937.24\n",
    "Elapsed (wall clock) time (h:mm:ss or m:ss): 49:50.69\n",
    "MU2E FAILED rc=1\n", "ERROR: fcl\n", "COPY gen FAILED\n", "SKIP: outputs exist\n",
    "Config=run2\n", '#include "ref_sim_run2.fcl"\n',
    "TimeReport CPU = 3000 Real = 3001\n", "System time (seconds): 10\n",
])
def test_summary_ambiguous_failure_or_foreign_payload_rejected(extra):
    with pytest.raises(ValueError):
        parse_summary(summarytext() + extra)


def test_summary_aggregate_measured_wall_and_boundary(summary_records):
    report = M.measure(summary_records)
    assert report["format"] == "worker_summary" and report["jobs"] == 3
    assert report["generated"] == 9
    assert report["timing_denominator"] == "gnu_elapsed_s"
    assert set(report["summary"]) == {"gnu_elapsed_s"}
    wall = report["summary"]["gnu_elapsed_s"]
    assert wall["sum_s"] == 12 and wall["generated_per_s"] == 9 / 12
    assert wall["per_job_generated_per_s"] == [2 / 3, 3 / 4, 4 / 5]
    assert report["cpu_total"] == "UNAVAILABLE"
    assert all(row["cpu_total"] == "UNAVAILABLE" for row in report["records"])
    assert not report["physics_admitted"] and not report["matched_to_fast_boundary"]
    assert "excludes CRV response/reco" in report["boundary"]
    assert report["full_chain_speedup"] == "not established"
    assert report["records"][0]["generator_validation"]["counts"]["events"] == 2
    assert len(report["input_sha256"]) == 6


def test_summary_ten_disjoint_actual_jobs(summary_records, tmp_path):
    for i in range(3, 10):
        log = tmp_path / f"worker_{i}.out"
        log.write_text(summarytext(i + 2, i + 1, i, elapsed=f"0:{i + 3:02d}"))
        table = generator(tmp_path / f"gen_{i}.root", i + 2, 1400, i + 1)
        summary_records.append(dict(path=str(log), generator_table=str(table), generated=i + 2,
            run=1400, subrun=i + 1, format="worker_summary", configuration=M.CONFIGURATION, stage=M.STAGE))
    report = M.measure(summary_records)
    assert report["jobs"] == 10 and report["generated"] == 65
    assert report["summary"]["gnu_elapsed_s"]["sum_s"] == 75
    assert report["summary"]["gnu_elapsed_s"]["generated_per_s"] == 65 / 75


@pytest.mark.parametrize("format", ["full_payload", "auto"])
def test_no_mixed_or_unknown_aggregate(summary_records, format):
    summary_records[0]["format"] = format
    with pytest.raises(ValueError, match="mixed/unknown"):
        M.measure(summary_records)


@pytest.mark.parametrize("legacy,donor,duplicate", [(True, False, False), (False, True, False), (False, False, True)])
def test_summary_still_requires_strict_full_generator(summary_records, legacy, donor, duplicate):
    generator(summary_records[0]["generator_table"], 2, 1400, 1, legacy, donor, duplicate)
    with pytest.raises(ValueError):
        M.measure(summary_records)


@pytest.mark.parametrize("n,run,subrun", [(3, 1400, 1), (2, 1401, 1), (2, 1400, 2)])
def test_summary_still_requires_actual_census_identity(summary_records, n, run, subrun):
    generator(summary_records[0]["generator_table"], n, run, subrun)
    with pytest.raises(ValueError, match="actual full-generator census"):
        M.measure(summary_records)


def test_summary_actual_reported_filter_counts():
    text = summarytext().replace("passed = 1 failed = 49999", "passed = 8193 failed = 41807")
    report = parse_summary(text)
    assert (report["generated"], report["passed"], report["failed"]) == (50000, 8193, 41807)


def test_summary_explicit_local_gzip_artifact(summary_records):
    path = Path(summary_records[0]["path"])
    compressed = path.with_suffix(".log.gz")
    with gzip.open(compressed, "wt") as stream:
        stream.write(path.read_text())
    summary_records[0]["path"] = str(compressed)
    report = M.measure(summary_records)
    assert report["summary"]["gnu_elapsed_s"]["sum_s"] == 12
    assert report["records"][0]["format"] == "worker_summary"
