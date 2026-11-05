# Native secondary libraries

The secondary library must be built in the same geometry and production-cut configuration as the transported
sample. An extracted library's absolute shower positions and placement sidecars are not reusable in Run 1A.
Building an asset is not physics validation and does not install a new default.
The standalone [construction/loading contract](native-library-build.md) is separate from the response and campaign
diagnostics collected below. The native Run 1A library is **not admitted or merge-ready**; see the
[current pause handoff](run1a-pause.md) for the build/readback and inclusive-denominator gates.

**Construction support is currently restricted to at most one primary muon per event**, checked independently
in donor SimParticles and full generators. Valid multi-muon censuses are refused as a whole, not filtered.
Event/bar crossing aggregation and event-only descendant/near-miss masks require identity-safe reconstruction
before native multi-muon production. Existing library and response diagnostics do not establish that this issue
is absent. See the construction contract for the guard, source-bound evidence and read-only identity audit.

## Build

Supply checked schema-v2 CrvStep donor tables and the corresponding full-generator streams. Re-dump legacy tables
with an existing built `CrvStepTableDumper`; do not manufacture association metadata or bypass `genassoc.validate_donor`.
Keep donor, development and confirmation event identities disjoint, including the near-miss census.

```bash
python3 scripts/build_native_library.py \
  --donors '/inputs/run1a/donors/crvsteps.{s}.root' \
  --generators '/inputs/run1a/generator/gen.{s}.root' \
  --run-config run1a_v01 --shards 101 \
  --outdir /outputs/new-run1a-library
```

`{s}` expands to the reference run/subrun sequencer; `{k}` to the integer shard. The output directory must not
exist. The driver calls the strict tier-1b builder with roots, then reconstructs topology, native near-miss templates,
chords and module-scope anchors from those same donors. `build_check.json` records generated exposure, asset hash,
counts and anchor attachment statistics. It deliberately says that physics validation is pending.

`native_training.validate` admits the training **event censuses**, not a cross-stage particle join. Every donor
passes its local schema-v2 ProductID/ancestry checks and its full event census must be a subset of the explicitly
paired full-generator census. Full censuses must be disjoint, including events with no generator rows. Actual
run/subrun identities must match the driver's declarations; exposure is counted from the full census. Resolved
paths, input hashes, per-stage reports and census digests are bound into both manifests. Sidecar metadata retains
source hashes and counts, which bundle admission checks against the training receipt. Relative patterns are
canonicalized before construction. Consumers reject bundles without that build-time
census evidence; reconstruct them from the source tables rather than modifying historical receipts.

Donor generators may be compressed/reindexed collections, while full generators belong to an earlier stage.
Their ProductIDs, indices and multiplicities must not be equated across streams. The census receipt explicitly
leaves cross-stream generator pointers, common source provenance and filter completeness unestablished. Establish
source lineage, geometry, field, production cuts and selection semantics separately before making a physics claim.
An aggregate shower-reach denominator does not require a fabricated generator-pointer correspondence.

The lower-level builder also accepts `tier1b-build --run-config NAME --roots`. Its manifest records the configuration,
counters hash and build-time geometry/group-map fingerprint; bundles without that build-time evidence must be rebuilt,
not retroactively relabelled. Strict report paths/roles and full-generator exposure must match the declared training inputs.
Topology and template sidecars use that configuration's geometry and group map. Sidecar construction for legacy
manifests without `run_config` retains its extracted interpretation, but cannot enter this native-bundle path. Geometries with absent library groups use quantiles only for populated groups;
near-miss closest-approach selection excludes absent group boxes rather than borrowing another geometry's edges.

## Explicit ordinary-chain use

```bash
python3 -m crvfastsim runchain /inputs/run1a/gen.heldout.root \
  --run-config run1a_v01 --native-library-dir /outputs/new-run1a-library \
  --conditions-manifest /conditions/run1a/manifest.json --gdml /geometry/run1a.gdml \
  --outdir /outputs/heldout-chain
```

This is an explicit asset choice, not an installed set or default. The bundle must contain `library.root`,
`library.json`, `build_check.json`, `topology.npz`, `templates_v3.npz` and both sidecar JSON files. Resolution binds
all seven files, the named configuration, counters bytes and geometry/group map; loading checks those identities
again and attaches topology, templates and module-scope anchors. Missing root/chord support fails at attachment.
The near-miss census must use the library's training donor/generator population, not held-out validation rows.

The ordinary chain passes the strict-style placement options and enables native non-muon showers: their absolute
positions belong to this same geometry, unlike extracted showers. Bare `--library` semantics and the registry's
named shower defaults are unchanged. Output records the effective sampling options and bundle identity; merge
rejects different identities. Do not combine `--native-library-dir` with `--library`, `--tier1b` or `--tier1b-dir`.
Prepared contexts, benchmark callbacks and device tier-1b are not supported for this path. These checks establish
bundle integrity only, not donor support adequacy, independent-sample closure or stock input geometry certification.

The explicit native path requires one strict schema-v2 full-generator shard per invocation, disjoint from the
recorded actual training run/subrun pairs. Matching subrun numbers in different runs do not establish overlap.
Exposure and `--max-events` come from the complete event census, including events without generator rows. Packed
channel/FEB labels require a single run and bounded nonnegative event/subrun identities; ambiguous inputs fail.
Optional `--native-output new.npz` retains the full census, steps, particles, digis, detailed pulses and cluster
memberships, with full triples and an integrity sidecar. It requires strict conditions and refuses existing output,
`--compare`, `--emit`, prepared contexts and benchmark callbacks. Input, implementation and external asset bytes are
snapshotted before setup and rechecked around publication. This is a diagnostic record, not a response receipt,
benchmark admission, execution attestation or physics pass. Fixed absolute timing and finite-library limitations
remain. Compare only to reference outputs whose exact generated population and response convention match.

Ancestry classification uses full integer identities and logarithmic-depth parent traversal; broken/cyclic chains
remain unknown. Native constructors supply their configuration's sector-to-group map, not extracted sector indices.
Anchor groups retain complete muon/root/module tuples without arithmetic particle-key packing. These checks protect
software identity and do not address finite donor support or change an installed default.

`scripts/bench_native_chain.py --config timing.json --outdir NEW` measures repeated ordinary **host** native calls
without bypassing the prepared-context/benchmark-callback restrictions. The explicit config supplies `generator`,
`run_config`, `native_library_dir`, `conditions_manifest`, `gdml`, `seed`, `window` and `repeat` (at least two).
Every call reloads setup, publishes a separate detailed native record, and must preserve the full census, counts and
array fingerprint across repeats. Input/assets/sources are hashed around the measurement; native snapshots retain
both lexical asset paths and resolved targets so changed symlink bytes cannot evade verification. Reported call
elapsed includes output and is distinct from internal, potentially overlapping timers; never sum stage timers for
throughput. Imports, preflight and final verification are reported separately. An optional explicitly labeled
`stock_rate` is retained as an unmatched descriptive basis, not a matched stock speedup or GPU/physics certificate.

## Conditional diagnostics

`scripts/check_native_secondaries.py` samples the explicit native bundle on an independent reference's **true
primary-muon steps**, with a checked full-generator census. It refuses training-shard overlap and reports paired
full-event energy/step moments by actual sector and origin, including zero-step events. This diagnostic separates
secondary sampling from primary transport; it does not validate an end-to-end chain or include finite-library
uncertainty. Use independent sampling seeds, without tuning to the observed held-out result.
`scripts/summarize_native_secondaries.py` combines these records by seed with input hashes, retaining repeated
seeds as separate arms rather than extra exposure. It reports conditional event SE and descriptive shard SE only;
it supplies no acceptance gate and does not replace paired histogram/shape validation.

`scripts/check_named_coinc.py` feeds saved stock pulses to the named batched finder and compares exact cluster
memberships by full event identity, sector and sorted pulse indices. It retains the full event census and reports
fallback counts and unmatched signatures. A diagnostic NumPy replay on mismatch events can localize a backend
difference; it does not replace the candidate or turn a failed identity check into admission. An explicit
no-opposite-attachment arm diagnoses a historical stock implementation boundary, never a new default or tuned
physics correction. Strict conditions
verify the saved text-overlay bundle, not the originating stock geometry or full V-R3/E2E certification.
`scripts/summarize_named_reco.py --config controls.json --out NEW.json` combines explicit `coincidences` and
`pulses` report lists, hashes the reports, and keeps raw pulse-finding differences separate from the agreed
clean-waveform criteria. Report aggregation is not duplicate-event admission or a native-chain certificate.

## Validation requirements

Before an end-to-end claim: independently held-out step closure, V4 and V-R checks on the matching stock response,
paired CRV histogram comparisons with the stated null/error rule, and the frozen C4 selection where event populations
match. Include finite donor support, sampling-seed spread, exact timing-window and conditions identities, and the
reference's field/selection limitations. A field-off reference does not validate field-on cosmic transport.

No named asset set or automatic Run 1A chain default is installed by the builder. The extracted `strict_v5` default
and its sidecars remain separate.

`scripts/check_native_support.py --config support.json --out NEW.json` reads an explicit current `bundle` and
`run_config`, with optional `native_records` entries (`path`, declared `seed`). It reports training-file coverage
per momentum/path bin, actual muons/anchors, and raw near-miss/shower reach support before any sampling probability
clipping. Empty bins may reflect absent geometry groups; counts alone do not establish sufficient support. Frozen
record clamp/drop counts stay separated by seed and retain their declared library hash. Historical hashes that
differ from the audited bundle are explicitly unbound, not retroactively certified. No bootstrap uncertainty or
support-closure flag is supplied.

`scripts/measure_stock_logs.py records.json --out NEW.json` measures explicitly listed successful stock grid logs
against their actual strict full-generator censuses. Full-payload logs retain art Real/CPU timing. Explicit
`worker_summary` records use the preserved GNU elapsed time, validate subrun/count/completion markers and report
total CPU as unavailable when only user CPU was retained. A worker output-directory label is not expanded FHiCL
verification. Generation/Geant4/filtered-step output excludes CRV response/reconstruction, so its rate cannot be
advertised as a matched full-chain stock speedup. Mixed timing formats, failed/ambiguous logs and changed inputs fail.

## Retained histogram and component checks

`scripts/check_native_rebuild.py --reference OLD --candidate NEW --run-config NAME --out NEW.json` requires current
strict admission for the candidate, but reads an older bundle only through a separately labeled diagnostic path.
It checks the old configuration, geometry and library/sidecar hashes without fabricating a census receipt. Exact
numeric fingerprints cover every library field, sidecar array and current-code anchor array, including dtype,
shape and signed-zero bytes. Integer storage changes get separate exact-int64 value diagnostics but remain strict
mismatches. Both anchor arms use current code, not historical execution; no baseline response or transport is rerun.

`scripts/check_native_response.py` verifies a full native record against a strict generator census and the exact
union of accepted stock response events. It reads the highest ROOT cycles and requires their collection counts
to agree. It reports rates only; missing rejected-event response is an explicit selection assumption, not a fact
inferred from accepted donors.

`scripts/compare_native_histograms.py --config pairs.json --out new.json` takes explicit generator, donor, stock
file-list and seed-to-native-record mappings per shard (`run_config` plus `pairs`, each with `generator`, `donor`,
`stock`, `candidates`). Independent generator shards must have equal full-census exposure; repeated sampling seeds
stay separate. Fixed, declared histogram edges include underflow/overflow and zero-cluster events. Pulse and step
histograms cover every actual sector; cluster histograms retain their declared all-sector scope. The statistic in
`validation/paired_shards.py` uses paired batch variance, the Student mean-bias factor, shared sign flips across
histograms and gamma-tail family calibration. Independent-arm variances are retained for overlays. Identical,
one-empty and unsupported constant-null cases are explicit. This diagnostic neither reproduces every DQM histogram
nor certifies small-sample tail coverage, finite-library errors or end-to-end physics.

`scripts/check_named_v4.py` supplies per-sector component controls using the named response, strict conditions and
physical/status mask. It retains generator-event zeros, paired event PE uncertainty and the event-clustered V4 CDF
time rule. `--no-stock` is an independent-NumPy seed null, not a stock test. When Offline's Python lacks analysis
packages, `--prepare-oracle new.npz` exports a frozen numeric request; `scripts/run_named_v4_oracle.py` runs the
existing stock classes in a pinned Offline environment, and `--oracle-result result.npz` validates request/output
hashes and bitwise numeric input identity before comparing. No analysis dependency is installed into Offline.
Per-sector stock-class objects do not emulate historical art shared-LUT yield ownership. Truncated-step controls,
fixed-window SiPM behavior and a host component comparison do not establish stock-art, GPU or full V4 admission.

`scripts/check_named_waveforms.py` replays saved stock charges with phases joined by the complete event triple and
ROC/FEB, not FEB number alone. Missing mapped phases remain disclosed. A noise-free reference can establish
conditional sample identity; it does not establish noise-on per-channel KS behavior or event-relative timing.
`scripts/check_named_waveform_noise.py --config pairs.json` accepts explicit `pairs` of saved noisy `reference`
files with optional `nonoise_reference` controls. It retains unique full event censuses and strict conditions,
requires identical saved charges and compatible FEB phases for a noise-free control, and checks that control
against deterministic synthesis before using its residuals. Unmapped hardware and unknown fast phases are excluded
symmetrically and counted. The per-channel ADC check keeps the declared minimum of 200 digis in both arms and
p > 0.01; no eligible channels is unsupported, not a pass. Residual width, shape and adjacent-sample correlation
are descriptive conditional diagnostics, not independent-trial noise agreement or a full V-R1 verdict. Source,
waveform template, input and conditions hashes are retained; off-grid clocks, aliased baselines and input drift fail.
`scripts/audit_native_c4.py` replays the supplied cached rule and selected-track counts, then joins full event triples
to the strict generator census. Equal geometry names or event numbers in different runs are not matching populations.
Keep stock C4 when selection, field, timing or exact parentage is not shared; never transfer a field-off reference
verdict to a cached field-on selected-track analysis.

`scripts/check_named_sector_response.py` accepts a named `run_config`, a `reference` file list and labeled
`candidates` file lists. It requires exact saved event populations and assigns pulses by their bars' actual geometry
sectors, not a shared stock sector-type number. It retains zero-response events for the paired event PE ratio SE,
flow counts, and diagnostic median/spectrum comparisons. Histogram errors are independent-pulse approximations;
these checks do not certify event-clustered shape uncertainty, layer efficiency, an objective scan or full V-R4.
Historical diagnostic candidates remain explicit controls, never corrected defaults or physics admission.

`scripts/check_named_response_campaign.py --config pairs.json --out NEW.json` compares saved same-step responses
across at least four disjoint strict full-generator shards of equal declared exposure. Config fields are
`run_config`, `seed`, `conditions_manifest`, `generated_per_pair`, and `pairs` containing `generator`, `reference`
and `candidate` file lists. Candidate metadata must declare the exact named NumPy/noise-on/offspill configuration,
seed and strict conditions; historical shared-yield overrides are refused. Event populations must match within each
pair and belong to its generator census, including saved zero-response events. Accepted counts may differ across
shards. It retains raw response histograms, paired-shard family diagnostics and cluster-count ratio SE, plus
actual-sector pulse diagnostics. A separate full-generator zero overlay is explicitly diagnostic. Producer lineage
and identical G4 inputs need external evidence; matching metadata is not execution attestation. No track/layer,
objective, inclusive-selection or full V-R4 certificate follows from this check.

`scripts/audit_c4_ntuple.py` inventories one explicitly supplied cached-source ntuple's event and truth branches,
then joins complete event triples to the cached selection. Optional `--mca` invokes the original analysis adapter
on that one file and requires exact cached track momentum/time and coincidence fields. It hashes the cache,
adapter and original analysis sources, not a remote ROOT payload; record catalogue checksums separately. This
bounded stock-selection replay does not validate field-on native C4. The adapter's opt-in `include_lineage`
retains original track indices and their aligned `trkmcsim` genealogy through the analysis mask; ordinary selection
output is unchanged. SimParticle IDs and generator codes are not a strict GenParticle ProductID/key association,
and a single source-file audit does not certify the whole cached population. Optional `--lineage-output` retains
those original indices and genealogies in an exclusively published parquet with an output hash; it never overwrites
a cached selection or supplies missing generator associations.

`scripts/check_named_response_axes.py --config axes.json --out new.json` replays saved stock, named-default and
explicit historical-response pulse populations through attachment off/on. The config declares `run_config`,
`reference`, `default`, `historical`, `conditions_manifest` and `resolved_fcl`. It requires exact event populations,
strict conditions and saved producer metadata matching each declared response/finder axis. It requires
`MU2EFS_DEVICE_FLOAT=float64`, records precision/device provenance and rejects malformed saved pulse times/channels
before finder dispatch. This explicit diagnostic precision does not change production defaults. No response is redrawn,
no pulse is modified and no ROOT baseline is rewritten. It reports actual-sector pulse attribution, exact own-arm
membership checks, cluster-presence migrations, conditional event SE and raw fixed-edge histogram differences with
flows and event zeros. These are descriptive controls, not calibrated family tests or stock-art certification.
Between response populations the frozen draws are not exact common random variates; within each pulse population
only the finder attachment setting varies. Keep both default and historical results; no default promotion follows.

`scripts/check_named_stock_release.py --reference old.root --candidate new.root --expected-events N
--expected-event-sha256 DIGEST --out new.json` requires an independently pinned complete census and exact
persisted event triples, spill/window/bunch-time fields and all 5,120 calibration/status/hardware rows, including
dtypes and signed-zero bytes after key sorting. It also validates highest-cycle collection counts and memberships.
This checks persisted timing/calibration and collection integrity only. Release-level response changes are not a
yield-only A/B, and identical persisted conditions do not certify geometry execution, DB IOV or physics closure.

`scripts/compare_named_reference_release.py --config replacements.json --outdir NEW` accepts a hash-bound
`legacy_pairs` (`path`, `sha256`) and complete `replacement_stock` mapping from every original stock filename to
one distinct new filename. It preserves native inputs, seed declarations, edges and statistics, checks every old/new
persisted event timing/calibration population, then runs the existing paired diagnostic in the new output directory.
The original paired report stays separate and unchanged. A fixed-release result is not a yield-only A/B and cannot
supply missing rejected-event response, finite-library uncertainty or full named physics certification.

Histogram admission rejects accepted stock keys outside the generator census, candidate file aliases, changed
conditions/geometry/optical assets and inconsistent recorded seeds. A legacy record lacking seed/effective-transport
metadata needs an explicitly hashed producer-script declaration (`legacy_producer_script`, with `path`, `sha256`
and optional `additional_scripts`). That only labels historical evidence; it does not upgrade the record or attest
execution. The resulting `sampling_seed_contrast` discloses whether seeds are recorded or externally declared.
Future records preserve effective transport settings explicitly. Sparse one-empty histograms are descriptive
mismatches, not automatically family-calibrated rejections; their calibrated flag is unsupported.
