"""Tier-1b asset sets: the library, its placement sidecars and the sampling options that go with them.

The callers (tier1b-apply, runchain, e2e-bench, the endpoint kits) pick one set by name; crvfastsim.tier1b itself has
no default library. docs/tier1b-default.md has the provenance and the evidence behind the default.

  strict_v5  default, extracted_v04: the strict library rebuilt with step-tree roots (the same 20 donors, bins and
             showers) with crossing context, endpoint containment, point and topology placement, the never-visited-layer
             / near-miss templates (v3 sidecar), chord placement and root-coherent draws: a delta ray and its cascade
             stay together along the fast muon within a module (tier1b.attach_anchors, sample(anchor)). No hard-muon
             draws.
  strict_v4  the default before 2026-11-02, opt-in: the association-certified library (20 donors) with the same
             placement and templates, chord placement and whole-muon draws of hard library muons (summed descendant
             energy above 30 MeV, keyed by momentum class and the group and layer of each crossing; tier1b.attach_hard).
             Byte-identical per seed to the default it was.

Other option combinations run with a custom library and explicit placement flags (tier1b-apply --library ...
--crossing-context ...; docs/tier1b-default.md). Named run configurations have no default set: ordinary runchain
can explicitly select a checked native builder output with --native-library-dir; --library retains bare sampling.

    spec = resolve(run_config="extracted_v04")            # the configuration's default set under $CRVFS_DATA
    spec = resolve("strict_v4", directory="/dev/shm/x")   # the previous default, files staged elsewhere
    spec = resolve(library="lib.root")                     # a custom library, no placement options
    lib = load(spec, geom); tier1b.sample(..., lib, geom, rng, **spec.options)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT = "strict_v5"
EXTRACTED = "extracted_v04"


@dataclass(frozen=True)
class AssetSet:
    directory: str                     # under $CRVFS_DATA
    library: str
    topology: str | None = None        # valhist.descendant_topology sidecar (+ .json)
    templates: str | None = None       # valhist.descendant_templates sidecar (+ .json)
    library_sha256: str | None = None  # pinned content of the library the sidecars were made for
    options: dict = field(default_factory=dict)   # tier1b.sample keywords
    run_configs: tuple = (EXTRACTED,)

    def files(self):
        """Every file of the set, relative to its directory (sidecars with their metadata)."""
        side = [s for p in (self.topology, self.templates) if p for s in (p, p + ".json")]
        return [self.library] + side


_STRICT = dict(context=True, contain=True, point=True, topology=True, unvisited=True, nearmiss=True)
_STRICT_LIB = "ebb015a76d56edcf96c5bc533daa96729e3851ee3e59185abf0be5fd469257c7"
_ROOTS_LIB = "fc6e1518a90632217f948e43cefde7c0a4f8919e6e4ba0e8ffa2d1d939bcc5f4"

SETS = {
    "strict_v5": AssetSet(
        "tier1b_strict_v5", "library.root", "topology.npz", "templates_v3.npz", _ROOTS_LIB,
        dict(_STRICT, chord=True, anchor=True)),
    "strict_v4": AssetSet(
        "tier1b_strict_v2", "library.root", "topology_sidecar_strict_v2_v2.npz", "templates_v3.npz", _STRICT_LIB,
        dict(_STRICT, chord=True, hard=True)),
}


@dataclass(frozen=True)
class Tier1bSpec:
    """A resolved choice: name (None for a custom library), absolute paths and the sample options."""
    name: str | None
    library: str
    topology: str | None = None
    templates: str | None = None
    library_sha256: str | None = None
    options: dict = field(default_factory=dict)
    native: dict | None = None

    def paths(self):
        """Every file the choice reads (library, sidecars and their metadata)."""
        side = [s for p in (self.topology, self.templates) if p for s in (p, p + ".json")]
        extra = [str(Path(self.library).with_suffix(".json")), str(Path(self.library).parent / "build_check.json")]
        return [self.library] + side + (extra if self.native is not None else [])

    def describe(self):
        return {"set": self.name, "paths": self.paths(), "options": dict(self.options),
                "library_sha256": self.library_sha256, **({"native": self.native} if self.native is not None else {})}


def default_for(run_config=None):
    """The default set of a run configuration: strict_v5 for extracted_v04, None for the named ones (no library)."""
    return DEFAULT if (run_config or EXTRACTED) == EXTRACTED else None


def resolve(tier1b=None, library=None, directory=None, run_config=None, data=None):
    """Tier1bSpec from a set name or a custom library path (not both). Neither: the run configuration's default set.
    directory: where the set's files are (default $CRVFS_DATA/<set directory>)."""
    rc = run_config or EXTRACTED
    if library and (tier1b or directory):
        raise ValueError("pass either a custom --library or --tier1b [--tier1b-dir], not both")
    if library:
        return Tier1bSpec(None, str(library))
    name = tier1b or default_for(rc)
    if name is None:
        raise ValueError(f"run configuration {rc!r} has no tier-1b asset set (no library has been built on its "
                         f"geometry): pass a custom --library")
    if name not in SETS:
        raise ValueError(f"unknown tier-1b set {name!r}; known: {sorted(SETS)}")
    s = SETS[name]
    if rc not in s.run_configs:
        raise ValueError(f"tier-1b set {name!r} has no library for run configuration {rc!r}: pass a custom --library")
    if directory is None:
        data = data or os.environ.get("CRVFS_DATA")
        if not data:
            raise ValueError(f"tier-1b set {name!r}: set CRVFS_DATA (source site.env), pass --tier1b-dir or --library")
        directory = Path(data) / s.directory
    d = Path(directory)
    p = lambda f: str(d / f) if f else None
    return Tier1bSpec(name, p(s.library), p(s.topology), p(s.templates), s.library_sha256, dict(s.options))


def native_bundle(directory, rc):
    """Resolve an explicit native builder output; integrity checks do not certify physics closure."""
    from .analyses.context_audit import content_fingerprint
    if rc.name == EXTRACTED:
        raise ValueError("native bundles require an explicit named run configuration")
    d = Path(directory).resolve()
    library, topology, templates = (str(d / name) for name in ("library.root", "topology.npz", "templates_v3.npz"))
    spec = Tier1bSpec(None, library, topology, templates, options=dict(_STRICT, chord=True, anchor=True, showers=True), native={})
    before = {Path(p).name: sha256(p) for p in spec.paths()}
    manifest = json.loads(Path(library).with_suffix(".json").read_text())
    check = json.loads((d / "build_check.json").read_text())
    if manifest.get("run_config") != rc.name or check.get("run_config") != rc.name:
        raise ValueError("native library run configuration mismatch")
    if check.get("library_sha256") != before["library.root"]:
        raise ValueError("native build check does not bind this library")
    generated = check.get("generated")
    if type(generated) is not int or generated <= 0 or generated != manifest.get("generated_scaled_to"):
        raise ValueError("native library generated exposure mismatch")
    shards = check.get("shards")
    if not isinstance(shards, list) or not shards or any(type(k) is not int for k in shards) or len(set(shards)) != len(shards):
        raise ValueError("native library needs distinct integer donor shards")
    donors = manifest.get("generator_associations", [])
    generators = manifest.get("generator_streams", [])
    if (len(donors) != len(shards) or len(generators) != len(shards)
            or any(m.get("schema_version") != 2 or m.get("associations_checked") is not True for m in donors)
            or any(m.get("schema_version") != 2 for m in generators)):
        raise ValueError("native library requires strict schema-v2 donor and generator evidence")
    if ([m.get("path") for m in donors] != manifest.get("inputs")
            or [m.get("path") for m in generators] != manifest.get("gen")
            or any(m.get("stream_role") != "checked donor" for m in donors)
            or any(m.get("stream_role") != "full generator stream" for m in generators)
            or any(m.get("counts", {}).get("events", 0) <= 0 for m in generators)
            or sum(m.get("counts", {}).get("events", 0) for m in generators) != generated):
        raise ValueError("native strict reports do not bind the declared training population/exposure")
    training = _native_training_census(manifest, check, shards, generated)
    from . import native_construction
    construction = native_construction.check_receipt(manifest, check)
    counters = sha256(rc.counters_path())
    fingerprint = content_fingerprint(rc.geometry(), workers=1)
    if manifest.get("counters_sha256") != counters or manifest.get("geometry_fingerprint") != fingerprint:
        raise ValueError("native library lacks matching build-time geometry/group-map evidence; rebuild the library")
    for path in (topology, templates):
        meta = json.loads(Path(path + ".json").read_text())
        if meta.get("library_sha256") != before["library.root"] or meta.get("sidecar_sha256") != before[Path(path).name]:
            raise ValueError("native sidecar/library content mismatch")
        if meta.get("geometry_sha256") != counters:
            raise ValueError("native sidecar counters geometry mismatch")
        if [m.get("path") for m in meta.get("donors", [])] != manifest.get("inputs"):
            raise ValueError("native sidecar donor population mismatch")
        if [m.get("sha256") for m in meta["donors"]] != [m["steps"]["sha256"] for m in training["pairs"]]:
            raise ValueError("native sidecar donor hashes differ from build-time training inputs")
        if path == templates:
            census = meta.get("census", [])
            if ([m.get("steps") for m in census] != manifest.get("inputs")
                    or [m.get("gen") for m in census] != manifest.get("gen")
                    or meta.get("generated") != generated or meta.get("census_events") != generated):
                raise ValueError("native near-miss census must use the training population only")
            if ([m.get("steps_sha256") for m in census] != [m["steps"]["sha256"] for m in training["pairs"]]
                    or [m.get("gen_sha256") for m in census] != [m["generator"]["sha256"] for m in training["pairs"]]):
                raise ValueError("native near-miss census hashes differ from build-time training inputs")
    identity = {"run_config": rc.name, "counters_sha256": counters, "anchor_scope": "module",
                "geometry_fingerprint": content_fingerprint(rc.geometry(), workers=1), "assets_sha256": before,
                "training_census": training["full_generator_events"], "training_shards": training["actual_shards"],
                "construction_support": construction}
    spec = Tier1bSpec(None, library, topology, templates, before["library.root"], dict(spec.options), identity)
    _check_native(spec, rc.geometry())
    return spec


def _native_training_census(manifest, check, shards, generated):
    from . import native_training
    training, declared = manifest.get("training_census"), check.get("training_census")
    if not isinstance(training, dict) or not isinstance(declared, dict):
        raise ValueError("native library lacks build-time training census evidence; rebuild the library")
    common = ("schema", "receipt_version", "full_generator_events", "total_exposure", "actual_shards", "pairs")
    if (training.get("schema") != native_training.SCHEMA or training.get("receipt_version") != native_training.RECEIPT_VERSION
            or any(training.get(key) != declared.get(key) for key in common)
            or training.get("total_exposure") != generated
            or training.get("full_generator_events", {}).get("count") != generated):
        raise ValueError("native training census reports disagree with build population/exposure")
    for receipt in (training, declared):
        if (receipt.get("census_admitted") is not True or receipt.get("per_stage_associations_validated") is not True
                or receipt.get("cross_stream_generator_pointer_join") != "not_established"
                or receipt.get("shared_generator_provenance") != "not_established_by_census"
                or receipt.get("physics_admitted") is not False):
            raise ValueError("native training census overstates its admitted scope")
    actual = training.get("actual_shards", [])
    if (not actual or any(set(row) != {"run", "subrun"} or any(type(value) is not int or value < 0 or value >= 2**32
            for value in row.values()) for row in actual)
            or len({(row["run"], row["subrun"]) for row in actual}) != len(actual)
            or len({row["run"] for row in actual}) != 1
            or sorted(row["subrun"] for row in actual) != sorted(shards)
            or declared.get("declared_shards") != actual):
        raise ValueError("native declared shards do not bind the actual training run/subruns")
    pairs = training.get("pairs", [])
    if (len(pairs) != len(shards) or [row["steps"]["path"] for row in pairs] != manifest["inputs"]
            or [row["generator"]["path"] for row in pairs] != manifest["gen"]
            or sum(row["generator_events"]["count"] for row in pairs) != generated):
        raise ValueError("native training census does not bind the input paths/exposure")
    digests = [training["full_generator_events"].get("sha256")]
    digests += [row[role].get("sha256") for row in pairs for role in ("steps", "generator")]
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in digests):
        raise ValueError("native training census needs complete input/population hashes")
    return training


def _check_native(spec, geom):
    from .analyses.context_audit import content_fingerprint
    if {Path(p).name: sha256(p) for p in spec.paths()} != spec.native["assets_sha256"]:
        raise ValueError("native library bundle changed since resolution")
    if content_fingerprint(geom, workers=1) != spec.native["geometry_fingerprint"]:
        raise ValueError("native library geometry or group map mismatch")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(spec, geom):
    """Load checked sidecars normally, or opt into a content-bound attached-library cache."""
    if spec.native is not None:
        _check_native(spec, geom)
    directory = os.environ.get("CRVFS_PREPARED_LIBRARY_CACHE")
    if directory:
        from .analyses.library_cache import load as cached_load
        lib = cached_load(spec, geom, directory, _load_uncached)
    else:
        lib = _load_uncached(spec, geom)
    if spec.native is not None:
        _check_native(spec, geom)
    return lib


def _load_uncached(spec, geom):
    """Attach checked topology and template sidecars in their reference order, then hard muons or anchors."""
    from . import tier1b
    from .analyses.tier1b import read_template_sidecar, read_topology_sidecar
    lib = tier1b.read_library(spec.library)
    if not (spec.topology or spec.templates):
        return lib
    digest = sha256(spec.library)
    if spec.library_sha256 and digest != spec.library_sha256:
        raise ValueError(f"tier-1b set {spec.name!r}: {spec.library} is not the pinned library")
    if spec.topology:
        lib = tier1b.attach_topology(lib, *read_topology_sidecar(spec.topology, library_sha256=digest), geom)
    if spec.templates:
        lib = tier1b.attach_templates(lib, read_template_sidecar(spec.templates, library_sha256=digest), geom)
    if spec.options.get("hard"):
        lib = tier1b.attach_hard(lib, geom)
    if spec.options.get("anchor"):
        lib = tier1b.attach_anchors(lib, geom)
    return lib
