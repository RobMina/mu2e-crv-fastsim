"""Hash-bound detailed native-chain diagnostics; no benchmark, execution or physics certification."""
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from . import tier1b_assets

SCHEMA = "crvfastsim.native_record.v1"
SCOPE = "ordinary_native_chain_diagnostic_not_benchmark_or_physics_admission"


def paths(output):
    output = Path(output)
    return output, Path(str(output) + ".json")


def require_new(output):
    for path in paths(output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)


def snapshot(inputs, spec, rc, gdml, manifest):
    """Bind native inputs and external assets before loading the chain."""
    from .response import lookup
    import mu2efastsim
    manifest = Path(os.path.abspath(manifest))
    evidence = json.loads(manifest.read_text())
    files = [*inputs, *spec.paths(), rc.counters_path(), gdml, manifest]
    for group in ("inputs", "assets"):
        files.extend(manifest.parent / item["path"] for item in evidence[group].values())
    files.extend(Path(lookup.lookup_dir()) / ("LookupTable_" + name)
                 for name in sorted({sector.table for sector in rc.sectors.values()}))
    files.append(Path(lookup.lookup_dir()) / "photonMap.root")
    from . import geometry
    from .reco.waveform import WaveformConfig
    files.append(geometry.find_counters(WaveformConfig().single_pe_file))
    for directory in (Path(__file__).parent, Path(mu2efastsim.__file__).parent):
        files.extend(directory.rglob("*.py"))
    lexical = {os.path.abspath(path) for path in files}
    paths = lexical | {str(Path(path).resolve()) for path in lexical}
    return {path: tier1b_assets.sha256(path) for path in sorted(paths)}


def verify(before):
    try:
        changed = any(tier1b_assets.sha256(path) != digest for path, digest in before.items())
    except OSError as exc:
        raise ValueError("native record input/implementation drift") from exc
    if changed:
        raise ValueError("native record input/implementation drift")


def write(output, universe, steps, particles, digis, outs, before, rc, cfg, spec, sample_counts, seed=None,
          effective_transport=None):
    """Retain full triples, raw numeric tables, pulses and memberships, including zero-response events."""
    from .analyses.e2e_bench import detail_arrays, fingerprint
    require_new(output)
    verify(before)
    arrays, counts, sectors = detail_arrays(universe.to_numpy(dtype=np.int64), outs)
    for label, data in (("steps", steps), ("particles", particles)):
        for name in data.columns:
            value = data[name].to_numpy()
            if value.dtype.kind not in "biuf":
                raise ValueError(f"native record nonnumeric {label}.{name}")
            arrays[label + "__" + name] = value
    for name in ("chan", "starttdc", "adc"):
        arrays["digis__" + name] = np.asarray(digis[name])
    counts.update(generated_events=len(universe), steps=len(steps), particles=len(particles), digis=len(digis["chan"]))
    metadata = {"schema": SCHEMA, "scope": SCOPE, "seed": seed, "counts": counts, "cluster_type_counts": sectors,
                "array_fingerprint": fingerprint(arrays), "inputs_sha256": before,
                "run_config": dataclasses.asdict(rc), "chain_config": dataclasses.asdict(cfg),
                "tier1b": spec.describe(), "sample_counts": sample_counts, "effective_transport": effective_transport,
                "physics_pass": False, "limitations": ["Input hashes are integrity evidence, not execution attestation.",
                "Fixed absolute response window, not event-relative stock timing.",
                "Finite-library uncertainty and physics closure are not established."]}
    output, sidecar = paths(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata["output_sha256"] = tier1b_assets.sha256(output)
    verify(before)
    with sidecar.open("x") as stream:
        stream.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return {"path": str(output.resolve()), "metadata": str(sidecar.resolve()),
            "sha256": metadata["output_sha256"], "counts": counts, "scope": SCOPE}
