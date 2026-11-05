# Explicit native secondary-library construction

This builder constructs root-coherent secondary assets for a named CRV geometry. It does not install a default,
certify physics, or make an extracted library's absolute shower positions reusable in another geometry.

**Paused checkpoint:** the native Run 1A library is not admitted or merge-ready. The whole-input multi-primary
construction guard remains; `native_identity` is isolated, not production-integrated. Saved raw one-/ten-event
capture gates passed, but not exact raw-pointer or native recording completeness. The new exact raw and recording
readers passed syntax-only checks; their user linked build and actual readback remain pending. No saved-raw
CrvSteps conversion has run. See the [pause handoff](run1a-pause.md), [raw-first gates](native-inclusive-capture.md)
and [implemented integrity contract](native-raw-integrity.md).

## Inputs and invocation

Use existing schema-v2 CrvStep donor tables and their corresponding full-generator streams. Legacy tables must
be re-exported by a compatible built dumper; do not fabricate event censuses or ProductIDs. Keep training and
validation populations separate, including the near-miss census.

```bash
python3 scripts/build_native_library.py \
  --donors '/inputs/run1a/donors/crvsteps.{s}.root' \
  --generators '/inputs/run1a/generators/gen.{s}.root' \
  --run-config run1a_v01 --shards 101 103 105 \
  --outdir /outputs/new-native-library
```

`{s}` is the reference sequencer `001400_<eight-digit subrun>`; `{k}` is the integer subrun. Paths and patterns
are canonicalized once. The destination must not exist, including as a dangling symlink. Each declared subrun
must agree with the actual input census; exposure is not inferred from filenames or generator-row counts.

The driver runs the existing tier-1b builder with roots, constructs topology and native near-miss/chord templates,
then actually attaches topology, templates and module-scope anchors. `build_check.json` retains counts, hashes,
anchor summaries and the explicit statement that physics validation is pending.

## Stage-local association and training-census checks

Every donor passes `genassoc.validate_donor`: step and parent pointers reference the selected SimParticle
collection, generator pointers reference that donor's selected generator collection, and every used ancestry
reaches a true primary. Full-generator inputs independently pass the schema-v2 full-stream validator.

`native_training.validate(step_paths, gen_paths, declared_shards=None)` checks explicit paired file lists. The
complete donor census must be a subset of its paired full-generator census. Full-generator censuses must be
mutually disjoint, including events with no generator rows. Repeated/resolved/hardlink aliases fail. Returned
receipts contain input hashes, local validation reports, actual run/subruns and complete-event digests/exposure.
The driver requires declarations and binds this receipt into construction and publication; before/after input
hash checks detect changes during that interval.

These are event-census checks, not cross-stage pointer matching. Compression can copy/reindex generators into a
new collection, so ProductIDs, indices and multiplicities must not be equated between donor and full streams.
The receipt explicitly leaves shared source provenance, a cross-stream generator-pointer join and filter
completeness unestablished. Record source lineage and compatible generation, geometry, field, cuts and selection
separately before using the assets as physics evidence. An omitted donor event is not automatically a measured
zero-response event.

## Construction-domain restriction

The current crossing builder aggregates by event/bar, and its descendant attachment and near-miss masks use event
scope. Those operations are not identity-safe for multiple primary muons in one event. Named construction therefore
requires at most one actual primary-muon SimParticle per donor event and at most one muon in each full-generator
event. `native_construction.require_supported` checks this separately from the stage-local census, before creating
the driver output directory. Compressed donor generator rows are not counted as distinct primary SimParticles.
A valid multi-muon census remains valid census evidence but the entire construction is refused: no event filtering,
resampling, nearest-primary matching or denominator change is permitted.

`construction_support` is source-bound in both manifests and checked on bundle admission. Existing bundles lacking
this evidence are not promoted by editing receipts; audit the actual sources first. Multiple-muon native production
requires identity-safe crossing, descendant and near-miss reconstruction plus a new validated library. This guard is
not that physics implementation, and does not demonstrate that a historical library is unaffected.

A read-only donor-local audit can quantify the historical behavior without rebuilding or redrawing:

```bash
python3 scripts/audit_native_primary_identity.py --config explicit-donors.json --out new-identity-audit.json
```

The config supplies `donors` (explicit paths), `run_config` and optional local `counters`. It compares historical
attachment with a diagnostic restricted to exact primary-key ancestry, retains same-bar/proximity mistakes and
near-miss exclusions, and does not establish a cross-stream generator-pointer join or closure.

## Donor-local identity repair development

`native_identity.crossings` and `native_identity.attach_descendants` are isolated donor-local helpers, not wired
into library construction or defaults. They aggregate by exact `(run, subrun, event, primary key, bar)`, retain
separate entry/exit/chord data, and restrict descendant candidates to their own ancestor before same-bar/distance
scoring. Descendants whose own ancestor has no crossing remain unattached even if another muon crosses the event.
The audit compares these helpers with its independent exact-ancestor diagnostic. This is not a new admitted bundle.

A complete near-miss repair additionally needs an authoritative mapping from donor primary/compressed generator
identities to full-stream source generator identities. It must preserve compressed aliases without multiplying
full-stream denominator rows, retain noncrossing companions of crossing muons, and bind source/extraction provenance
and filter completeness. Source payload similarity, equal event censuses and compressed/full index equality are
not that mapping. `native_identity.require_full_nearmiss_bridge()` explicitly refuses the unimplemented step.
Do not disable near-miss sampling, select single-muon events, fabricate a mapping, or relabel census evidence to
bypass this requirement. Existing guards remain in force until complete reconstruction and validation are supplied.

## Existing-source recovery investigation

The lack of a persisted compression remapping product does not by itself prove that source identity is
irrecoverable. An implementation-bound reconstruction may be possible when the original primary key survives:
direct `GenParticles` input gives no previous-stage key offset, the G4 primary helper selects its source generator
entry by primary track ID, and single-input SimParticle compression can preserve that key. This is not a generic
`key - 1` convention. It requires the actual producer versions, resolved input mode, absence of rekeying or mixing,
true primary status, untruncated ancestry and event/source lineage to be established together. Never infer a source
index by sorting retained particles or equate the compressed generator index to the full-stream index.

Even an exact candidate's payload agreement is only corroboration of a source-defined identity; it cannot replace
that identity contract. A retained original SimParticle collection with its original generator pointers would be a
different, explicit recovery route. Original generator collections alone do not restore overwritten pointers.

Filter completeness remains independent. A detector-step filter with nonzero energy or momentum requirements does
not guarantee retention of every primary crossing. A preceding G4-status filter may also remove generated events
with nonzero transport. Generator output containing those events does not measure their crossing status. Existing
full-generator exposure must therefore not be anti-joined against an incomplete crossing inventory and called an
inclusive near-miss denominator. An exporter operating only on already-filtered records cannot recover discarded
transport products. Preserve unknown crossing status explicitly; do not silently count it as a near-miss or change
the exposure/selection to make the construction pass.

`scripts/inspect_native_source_identity.sbatch` inventories explicit existing reference art shards with the
already-built runtime, hashes inputs before/after, records embedded configurations and product inventories, and
copies relevant pinned source files when available. It performs no transport, reconstruction or library build.
Site paths, runtime and filename owner are supplied through environment variables. Reused failed-job inventories
are explicitly unverified historical artifacts, not freshly attested by the current backing-source inspection.
Inspection results are not construction admission, and the current guard remains mandatory.

`scripts/audit_native_source_candidates.py --config paired-inputs.json --out new-candidates.json` tests the
conditional source-key candidate against explicit strict donor/full-generator pairs. It compares exact source
payloads after the integer identity lookup, reports missing candidates and disagreements without substitution,
and can probe explicit event tuples through optional `probe_events`. It emits only diagnostic JSON with source
identity, filter completeness and physics admission all false. It does not output a mapping or alter the builder.

## Geometry and exact identities

The named configuration supplies native counters and sector-to-library groups. Sparse geometries use quantiles
only for populated groups; near-miss closest approach excludes absent group boxes. No extracted absolute positions
or unrelated sidecar geometry are substituted.

Ancestry classification retains complete integer event/particle identities and resolves arbitrary finite depth
with pointer doubling. Broken or cyclic ancestry remains unknown. Anchor grouping uses full muon/root/module
tuples, not arithmetic packing with a presumed particle-key width. Module scope retains the strict_v5-style
root-coherent draw semantics.

## Bundle and loading

A complete bundle contains:

- `library.root` and `library.json`;
- `topology.npz` and `topology.npz.json`;
- `templates_v3.npz` and `templates_v3.npz.json`;
- `build_check.json`.

```python
from crvfastsim import runconfig, tier1b_assets

rc = runconfig.get('run1a_v01')
spec = tier1b_assets.native_bundle('/outputs/new-native-library', rc)
lib = tier1b_assets.load(spec, rc.geometry())
```

Resolution binds all seven files, configuration, counters bytes, geometry/group fingerprint and the build-time
training-census receipt. Sidecar donor/census source hashes must match the construction inputs, not merely their
paths. Loading checks bundle/geometry integrity again and attaches the placement data in order. Missing roots or
chords fail. Bundles missing current build-time evidence must be reconstructed, not retroactively relabeled.

The resolved spec has no installed set name. It explicitly enables context, containment, point/topology placement,
unvisited/near-miss templates, chords, anchors and native-position non-muon showers. Extracted defaults and bare
custom-library semantics remain separate.

## Tests and scientific limits

Synthetic ROOT integration tests exercise the real driver, sidecars, bundle resolution and anchor attachment,
including generator-empty events, relative paths and rejected input declarations. Separate regressions cover large
integer identities, deep/broken ancestry and mixed source/geometry/hash evidence. These establish software and
integrity behavior, not adequate donor support or independent-sample physics closure.

Before end-to-end physics use, quantify finite-library uncertainty and unsupported/clamped sampling, validate on
held-out native inputs, and match stock response, timing, conditions, field and selected population. GPU execution,
prepared contexts, throughput, track/layer efficiency, objective agreement and C4 require their own evidence.
