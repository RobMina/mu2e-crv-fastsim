"""Build a native root-coherent library and sidecars from checked donor and full-generator pairs."""
import argparse
import json
from pathlib import Path

from crvfastsim import native_construction, native_training, runconfig, tier1b, tier1b_assets
from crvfastsim.analyses.tier1b import _geom, build, read_template_sidecar, read_topology_sidecar
from valhist import descendant_templates, descendant_topology


def run(donors, generators, outdir, run_config, shards):
    outdir = Path(outdir)
    if outdir.exists() or outdir.is_symlink():
        raise FileExistsError(outdir)
    outdir = outdir.resolve()
    donors, generators = str(Path(donors).resolve()), str(Path(generators).resolve())
    if (not shards or any(type(k) is not int for k in shards) or len(set(shards)) != len(shards)):
        raise ValueError("native builder requires distinct declared integer shards")
    steps = [str(Path(donors.format(s=f"001400_{k:08d}", k=k)).resolve()) for k in shards]
    gen = [str(Path(generators.format(s=f"001400_{k:08d}", k=k)).resolve()) for k in shards]
    training = native_training.validate(steps, gen, declared_shards=shards)
    construction = native_construction.require_supported(training)
    exposure = training["total_exposure"]
    outdir.mkdir(exist_ok=False, parents=True)
    library = outdir / "library.root"
    build(steps, str(outdir), gen=gen, out=str(library), generated=exposure, roots=True, run_config=run_config)
    topology, templates = outdir / "topology.npz", outdir / "templates_v3.npz"
    descendant_topology.sidecar(library.with_suffix(".json"), topology)
    descendant_templates.sidecar(library.with_suffix(".json"), [donors], [generators], [shards], templates)
    geom = _geom(None, run_config)
    lib = tier1b.read_library(library)
    lib = tier1b.attach_topology(lib, *read_topology_sidecar(topology, library), geom)
    lib = tier1b.attach_templates(lib, read_template_sidecar(templates, library), geom)
    lib = tier1b.attach_anchors(lib, geom)
    expected = {item[role]["path"]: item[role]["sha256"] for item in training["pairs"] for role in ("steps", "generator")}
    if any(tier1b_assets.sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("native training inputs changed during sidecar construction")
    result = {"run_config": run_config, "shards": shards, "generated": exposure, "training_census": training,
              "construction_support": construction, "library_sha256": tier1b_assets.sha256(library),
              "counts": {name: len(lib[name]) for name in tier1b.LIB_TREES},
              "anchor": {name: lib["anchor"][name] for name in ("n_roots", "n_multi", "forward_edep_share")},
              "status": "built and attachments checked; physics validation pending"}
    (outdir / "build_check.json").write_text(json.dumps(result, indent=2) + "\n")
    rc = runconfig.get(run_config)
    spec = tier1b_assets.native_bundle(outdir, rc)
    tier1b_assets.load(spec, rc.geometry())
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donors", required=True, help="donor pattern with {s} or {k}")
    parser.add_argument("--generators", required=True, help="full-generator pattern with {s} or {k}")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--shards", nargs="+", required=True, type=int)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run(args.donors, args.generators, args.outdir, args.run_config, args.shards)


if __name__ == "__main__":
    main()
