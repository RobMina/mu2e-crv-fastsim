"""Measure 3 or 10 explicitly listed existing stock worker logs; never launch jobs.

Usage: python scripts/measure_stock_logs.py records.json --out measurement.json
Records require path, generator_table, generated, run, subrun, configuration
(run1a_v01), stage (generation_geant4_filtered_steps). Paths are local, relative
to the invocation directory. A strict schema-v2 full-generator census is mandatory;
filenames, caller-supplied census summaries and log metadata are not denominators.
Optional format defaults to full_payload (art CPU/Real). worker_summary requires
run/subrun metadata and the known g4ref_run1a_v1 worker output-directory label;
its denominator is measured GNU payload elapsed, with user time diagnostic only.
Configuration labels are not verification of the expanded worker FHiCL.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from crvfastsim import genassoc

CONFIGURATION = "run1a_v01"
STAGE = "generation_geant4_filtered_steps"
FORMATS = ("full_payload", "worker_summary")
MAX_LOG_BYTES = 16 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value, name, maximum=2**63 - 1):
    require(type(value) is int and 0 <= value <= maximum, f"invalid integer {name}")
    return value


def one(text, marker, pattern):
    require(len(re.findall(marker, text, re.I)) == 1, f"missing/ambiguous {marker}")
    matches = re.findall(pattern, text, re.M | re.I)
    require(len(matches) == 1, f"malformed {marker}")
    return matches[0]


def positive(token):
    value = float(token)
    require(math.isfinite(value) and value > 0, "nonfinite/nonpositive timing")
    return value


def elapsed_seconds(elapsed):
    parts = elapsed.split(":")
    require(float(parts[-1]) < 60 and (len(parts) == 2 or int(parts[-2]) < 60), "invalid GNU elapsed bounds")
    wall = int(parts[0]) * (3600 if len(parts) == 3 else 60) + float(parts[-1])
    if len(parts) == 3:
        wall += int(parts[1]) * 60
    return positive(str(wall))


def parse_worker_summary(text, generated, *, run=None, subrun=None):
    """Parse the worker's grep output, not a reconstructed full mu2e.log."""
    integer(subrun, "subrun", 2**32 - 1)
    if run is not None:
        integer(run, "run", 2**32 - 1)  # Run is bound by the actual census, not this header.
    # TrigReport's failed-event accounting is not a worker failure diagnostic.
    diagnostics = re.sub(r"^\s*TrigReport\s+Events\s+total[^\n]*", "", text, flags=re.M | re.I)
    require(not re.search(r"\bERROR\b|\bFAILED\b|\bSKIP\s*:", diagnostics, re.I),
            "worker summary contains failure/error/skip")
    require(not re.search(r"TimeReport|art has completed|Command being timed:|System time \(seconds\)|^\s*Exit status:", text, re.I | re.M),
            "full payload evidence is not a worker_summary")
    outdir = one(text, r"BEGIN EXECUTION", r"^.*\bBEGIN EXECUTION\s+(?:\S*/)?grid_ref_sim\.sh\s+(\S+)\s*$")
    require(Path(outdir).name == "g4ref_run1a_v1", "missing/conflicting known worker output-directory configuration label")
    labels = re.findall(r"\bConfig\s*=\s*([\w-]+)|\bref_sim_([\w-]+)\.fcl\b", text, re.I)
    require(all((a or b) == "run1a" for a, b in labels), "conflicting run1a configuration label")
    header = one(text, r"===\s*grid_ref_sim:\s*shard\b",
                 r"^=== grid_ref_sim: shard (\d+) subrun (\d+) seed (\d+) nevts (\d+) host (\S+)(?:[^\n]*)$")
    shard, actual_subrun, seed, nevts = (integer(int(v), "worker header") for v in header[:4])
    require(actual_subrun == subrun and nevts == generated, "worker header subrun/nevts disagree with declared metadata")
    done = one(text, r"===\s*grid_ref_sim:\s*done\b", r"^=== grid_ref_sim: done shard (\d+)\s*$")
    require(int(done) == shard, "worker done shard disagrees with header")
    status = one(text, r"\bCOMPLETED\b", r"^.*\bgrid_ref_sim\.sh COMPLETED with exit status (\d+)\s*$")
    require(status == "0", "worker completion exit status must be 0")
    elapsed = one(text, r"Elapsed \(wall clock\)", r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*((?:\d+:)?\d+:\d+(?:\.\d+)?)\s*$")
    user = one(text, r"User time \(seconds\)", r"^\s*User time \(seconds\):\s*(\S+)\s*$")
    return dict(format="worker_summary", shard=shard, seed=seed, worker_subrun=actual_subrun,
                worker_exit_status=0, gnu_elapsed_s=elapsed_seconds(elapsed), gnu_elapsed_text=elapsed,
                gnu_user_s=positive(user), cpu_total="UNAVAILABLE",
                configuration_evidence=dict(kind="worker_output_directory_label", output_directory=outdir,
                    label="g4ref_run1a_v1", expanded_configuration_verified=False))


def parse_log(text, generated, *, format="full_payload", run=None, subrun=None):
    """Full payload: art Real/CPU. Worker summary: measured GNU elapsed only."""
    require(format in FORMATS, "unknown log format")
    integer(generated, "generated")
    counts = one(text, r"TrigReport\s+Events\s+total", r"^\s*TrigReport\s+Events\s+total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)\s*$")
    total, passed, failed = (integer(int(x), "TrigReport count") for x in counts)
    require(total == generated and total > 0 and passed <= total and failed == total - passed,
            "TrigReport counts disagree with generated or passed/failed accounting")
    if format == "worker_summary":
        result = parse_worker_summary(text, generated, run=run, subrun=subrun)
        result.update(generated=total, passed=passed, failed=failed)
        return result
    status = one(text, r"art has completed", r"^.*\bart has completed and will exit with status\s+(\d+)\.?\s*$")
    require(status == "0", "art exit status must be 0")
    timing = one(text, r"TimeReport\s+CPU", r"^\s*TimeReport\s+CPU\s*=\s*(\S+)\s+Real\s*=\s*(\S+)\s*$")
    labels = re.findall(r"\bConfig\s*=\s*([\w-]+)|\bref_sim_([\w-]+)\.fcl\b", text, re.I)
    require(labels and all((a or b) == "run1a" for a, b in labels), "missing/conflicting run1a configuration label")
    result = dict(format="full_payload", generated=total, passed=passed, failed=failed,
                  cpu_s=positive(timing[0]), real_s=positive(timing[1]), art_exit_status=0)
    if re.search(r"Command being timed:|Elapsed \(wall clock\)|Exit status:", text):
        command = one(text, r"Command being timed:", r'^\s*Command being timed:\s*"([^"\n]+)"\s*$')
        require(re.fullmatch(r"mu2e -c fcl/shard\.fcl -n " + str(total), command) is not None,
                "GNU time command does not match stock worker/generated count")
        exit_status = one(text, r"Exit status:", r"^\s*Exit status:\s*(\d+)\s*$")
        require(exit_status == "0", "GNU time exit status must be 0")
        elapsed = one(text, r"Elapsed \(wall clock\)", r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*((?:\d+:)?\d+:\d+(?:\.\d+)?)\s*$")
        result.update(gnu_elapsed_s=elapsed_seconds(elapsed), gnu_elapsed_text=elapsed, gnu_exit_status=0)
    return result


def measure(records):
    require(isinstance(records, list) and len(records) in (3, 10), "require exactly 3 or 10 records")
    formats = {record.get("format", "full_payload") for record in records}
    require(len(formats) == 1 and formats <= set(FORMATS), "mixed/unknown log formats cannot share an aggregate")
    format = formats.pop()
    rows, hashes, identities, log_hashes = [], {}, set(), set()
    for record in records:  # File loop only; all census checks are vectorized.
        require(record.get("configuration") == CONFIGURATION and record.get("stage") == STAGE,
                "configuration/stage must be run1a_v01/generation_geant4_filtered_steps")
        run, subrun = (integer(record[k], k, 2**32 - 1) for k in ("run", "subrun"))
        generated = integer(record["generated"], "generated")
        require(generated > 0, "generated must be positive")
        paths = [Path(record[k]).resolve(strict=True) for k in ("path", "generator_table")]
        for path in paths:
            require(path.is_file() and str(path) not in hashes, "duplicate/nonfile input path")
            hashes[str(path)] = sha256(path)
        log, generator = paths
        require(hashes[str(log)] not in log_hashes, "duplicate log content")
        log_hashes.add(hashes[str(log)])
        with (gzip.open(log, "rt") if log.suffix == ".gz" else log.open()) as stream:
            text = stream.read(MAX_LOG_BYTES + 1)
        require(len(text) <= MAX_LOG_BYTES, "log exceeds small-file size limit")
        row = parse_log(text, generated, format=format, run=run, subrun=subrun)
        validation = genassoc.validate_generator_stream(generator)
        require(validation["schema_version"] == 2 and validation["stream_role"] == "full generator stream",
                "strict full-generator census required, not legacy inferred events")
        keys = genassoc.generator_event_keys(generator)
        require(len(keys) == generated and bool(((keys.run == run) & (keys.subrun == subrun)).all()),
                "actual full-generator census does not match run/subrun/generated")
        require((run, subrun) not in identities, "duplicate actual run/subrun identity")
        identities.add((run, subrun))
        row.update(path=str(log), generator_table=str(generator), run=run, subrun=subrun,
                   configuration=CONFIGURATION, stage=STAGE, generator_validation=validation)
        rows.append(row)
    total = sum(r["generated"] for r in rows)
    summary = {}
    timing_fields = ("real_s", "cpu_s") if format == "full_payload" else ("gnu_elapsed_s",)
    for field in timing_fields:
        values = [r[field] for r in rows]
        rates = [r["generated"] / r[field] for r in rows]
        summed = math.fsum(values)
        require(math.isfinite(summed) and all(math.isfinite(r) for r in rates)
                and math.isfinite(total / summed), "nonfinite aggregate timing/rate")
        summary[field] = dict(sum_s=summed, generated_per_s=total / summed,
                              per_job_generated_per_s=rates, min_generated_per_s=min(rates),
                              median_generated_per_s=statistics.median(rates), max_generated_per_s=max(rates))
    require(all(sha256(p) == h for p, h in hashes.items()), "input hash drift during measurement")
    return dict(configuration=CONFIGURATION, stage=STAGE, format=format, jobs=len(rows), generated=total,
                timing_denominator="real_s" if format == "full_payload" else "gnu_elapsed_s",
                cpu_total="measured art CPU" if format == "full_payload" else "UNAVAILABLE",
                physics_admitted=False, matched_to_fast_boundary=False,
                boundary="stock generation + Geant4 + accepted-filter step output only; excludes CRV response/reco",
                precision="not inferred from logs", overhead="not measured or added",
                full_chain_speedup="not established", input_sha256=hashes, records=rows, summary=summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    spec = Path(args.records).resolve(strict=True)
    before = sha256(spec)
    result = measure(json.loads(spec.read_text()))
    require(sha256(spec) == before, "record manifest hash drift")
    result["input_sha256"][str(spec)] = before
    with open(args.out, "x") as stream:  # Never overwrite an existing artifact/input.
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
