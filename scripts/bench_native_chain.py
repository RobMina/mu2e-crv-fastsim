"""Host timing of repeated ordinary native-chain calls, not E2E/physics/GPU certification.

CLI: python scripts/bench_native_chain.py --config explicit.json --outdir NEW
Config requires generator (one local full-generator shard), run_config,
native_library_dir, conditions_manifest, gdml, seed, window and repeat >= 2.
Optional stock_rate = {"events_per_s": 21.2, "label": "EXPLICIT descriptive basis"}
is retained verbatim as an unmatched reference, never a stock speedup.
"""
import time
_IMPORT_START = time.perf_counter()
import argparse
import json
import math
from pathlib import Path

import numpy as np
from mu2efastsim import precision
from crvfastsim import native_record as N, tier1b_assets
from crvfastsim.analyses import runchain as R
from crvfastsim.analyses.e2e_bench import local_assets_only, software_hardware, source_paths
from scripts.check_native_response import arrays, canonical_keys

_IMPORT_ELAPSED_S = time.perf_counter() - _IMPORT_START
SCOPE = "measured_host_ordinary_native_invocation_NOTphysicsorGPUcert"


def run(config, outdir):
    """Reload setup on every pass; exclusively publish native_benchmark.json after closure."""
    started = time.perf_counter()
    config = json.loads(json.dumps(config, allow_nan=False))  # own immutable-by-caller arguments
    required = {"generator", "run_config", "native_library_dir", "conditions_manifest", "gdml", "seed", "window", "repeat"}
    if not required <= config.keys() or config.keys() - required - {"stock_rate"}:
        raise ValueError("need exactly the explicit native timing config (optional stock_rate)")
    if type(config["repeat"]) is not int or config["repeat"] < 2:
        raise ValueError("repeat must be an integer >= 2")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**64:
        raise ValueError("seed must be uint64")
    for key in ("generator", "run_config", "native_library_dir", "conditions_manifest", "gdml", "window"):
        if config[key] is None or config[key] == "":
            raise ValueError(f"explicit {key} required")
    if "stock_rate" in config:
        ref = config["stock_rate"]
        if (not isinstance(ref, dict) or set(ref) != {"events_per_s", "label"}
                or type(ref["events_per_s"]) not in (int, float) or not math.isfinite(ref["events_per_s"])
                or ref["events_per_s"] <= 0 or not isinstance(ref["label"], str) or not ref["label"].strip()):
            raise ValueError("stock_rate needs positive events_per_s and explicit descriptive label")
    if not outdir:
        raise ValueError("explicit new outdir required")
    destination = Path(outdir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    generator = str(Path(config["generator"]).resolve())
    if not Path(generator).is_file() or not Path(generator).stat().st_size:
        raise ValueError("need one nonempty local generator file, not a pattern/list")
    options = {k: config[k] for k in ("run_config", "native_library_dir", "conditions_manifest", "gdml", "seed", "window")}
    for key in ("native_library_dir", "conditions_manifest", "gdml"):
        options[key] = str(Path(options[key]).resolve())
    with local_assets_only():
        rc, cfg, gdml, window, spec = R._resolve(None, None, None, options["gdml"], False, None,
                                               options["run_config"], options["window"],
                                               conditions_manifest=options["conditions_manifest"],
                                               native_library_dir=options["native_library_dir"])
        before = N.snapshot([generator], spec, rc, gdml, options["conditions_manifest"])
        for path in [*source_paths(), __file__, Path(__file__).with_name("check_native_response.py")]:
            before[str(Path(path).resolve())] = tier1b_assets.sha256(path)
        _, universe = R._native_generator_input(generator, spec)  # no max-events; includes gen-empty events
        keys = canonical_keys(universe)
        hardware, precision_before = software_hardware(), precision.provenance()
        N.verify(before)
        report = dict(schema="ordinary_native_timing.v1", scope=SCOPE, physics_admitted=False,
                      context_policy="reload_each_pass", arguments=config, inputs_sha256=before,
                      hardware_software=hardware, precision_before=precision_before,
                      module_import_elapsed_s=_IMPORT_ELAPSED_S,
                      imports="outside invocation/pass timers; cached dependencies may already be imported",
                      first_definition="first wrapper pass, not necessarily first process call; preflight warms IO",
                      warm_definition="same process/seed, setup reloaded; no cache flush or JIT reset",
                      timing_policy="elapsed_s encloses whole R.run including output; wall_s and nested/overlapping timers are independent diagnostics, never summed",
                      preflight_elapsed_s=time.perf_counter() - started, passes=[])
        destination.mkdir(parents=True, exist_ok=False)
        outputs, identity = {}, None
        passes_start = time.perf_counter()
        for index in range(config["repeat"]):
            N.verify(before)
            folder = destination / f"pass_{index:03d}"
            folder.mkdir(exist_ok=False)
            output = folder / "native.npz"
            a = time.perf_counter()
            result = R.run([generator], str(folder), native_output=str(output), **options)
            elapsed = time.perf_counter() - a
            data, meta = arrays(output)
            counts = meta["counts"]
            raw = {"generated_events": len(keys), "steps": len(data["steps__run"]),
                   "particles": len(data["particles__run"]), "digis": len(data["digis__chan"])}
            for name, suffix in (("pulses", "__pulses__ev"), ("clusters", "__cl_event"), ("members", "__mem_cl")):
                raw[name] = sum(len(v) for k, v in data.items() if k.endswith(suffix))
            if (not np.array_equal(data["generated_keys"], keys) or result["generated_events"] != len(keys)
                    or any(counts[k] != v for k, v in raw.items()) or counts != result["native_record"]["counts"]):
                raise ValueError("exact census/exposure or raw counts mismatch")
            if meta.get("seed") != config["seed"] or any(before.get(p) != h for p, h in meta["inputs_sha256"].items()):
                raise ValueError("record seed/input identity drift")
            current = (meta["array_fingerprint"], counts, meta["run_config"], meta["chain_config"],
                       meta["tier1b"], meta.get("effective_transport"), result.get("conditions"))
            if identity is not None and current != identity:
                raise ValueError("repeat fingerprint/identity drift")
            identity = current
            wall = result["wall_s"]
            if not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(wall) or not 0 <= wall <= elapsed:
                raise ValueError("invalid measured elapsed/wall timer closure")
            report["passes"].append(dict(stage="first" if index == 0 else "warm", elapsed_s=elapsed,
                                          generated_events_per_s=len(keys) / elapsed, counts=counts,
                                          array_fingerprint=meta["array_fingerprint"], runchain=result))
            for path in (*N.paths(output), folder / "runchain_stats.json"):
                outputs[str(path.resolve())] = tier1b_assets.sha256(path)
            N.verify(before)
        report["passes_elapsed_s"] = time.perf_counter() - passes_start
        a = time.perf_counter()
        N.verify(before)
        N.verify(outputs)
        report.update(status="complete", repeat_fingerprints_identical=True, generated_events=len(keys),
                      output_sha256=outputs, precision_after=precision.provenance(),
                      final_verification_elapsed_s=time.perf_counter() - a)
        if "stock_rate" in config:
            report["optional_reference_basis"] = dict(**config["stock_rate"], matched_Run1A=False,
                                                       scope="descriptive_unmatched_reference_not_speedup")
        report["invocation_elapsed_before_report_s"] = time.perf_counter() - started
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with (destination / "native_benchmark.json").open("x") as stream:
            stream.write(payload)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    run(json.loads(Path(args.config).read_text()), args.outdir)


if __name__ == "__main__":
    main()
