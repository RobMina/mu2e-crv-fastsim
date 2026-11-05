# Raw-first Run 1A inclusive capture: preparation and campaign gates

## Readiness

Pinned-runtime expansion and debug-config checks passed for both split schedules without transporting or
converting events in that preflight. Those checks establish configuration resolution only.
The bounded one-event and ten-event raw-first controls subsequently passed exact original/retained generator
census and payload checks, independent status-zero readback, normal completion and no truncation-warning markers.
These controls did not exercise empty/multi-muon events or actual overflow; raw-pointer integrity and native
attempted/saved-hit completeness remain unproven. No conversion was executed. No bulk transport, conversion,
limit change or library admission is authorized by these files. The earlier
one/ten-event inclusive controls and the separate two-muon control used the combined G4-plus-conversion path;
they are not runtime validation of this new split, overflow-failure retention or global denominator completeness.

## Separate persistence from conversion

1. `grid/fcl/ref_inclusive_capture_run1a.fcl` inherits the existing inclusive reference, schedules only `g4run`,
   and writes `inclusive-raw-run1a.art` with original generators, uncompressed SimParticles, StatusG4, raw CRV
   StepPointMCs, original subrun provenance and trigger results. No detector/status filter, compression or CrvSteps
   conversion runs. Transport physics, geometry, field and resource limits are inherited unchanged.
2. The art process must finish and close its output before any downstream conversion starts. Read the saved art
   independently for product inventory, status and a generator-only census. The generator-only dumper does not
   dereference raw-hit or SimParticle pointers, so it can preserve diagnostic evidence for damaged transport.
3. Audit the original and retained generator census and all payloads, independent status, transport completion and
   truncation warnings. Retain failed events as unknown. A missing/corrupt output or missing status cannot be
   replaced by an empty-hit observation, and a partial closed output cannot establish the requested full census.
4. `grid/fcl/convert_inclusive_capture_run1a.fcl` is a separate, unscheduled configuration candidate for saved raw
   input. Its process is `R3InclusiveSteps`; only `CrvSteps` runs, never G4. It keeps the original raw/status products
   and adds `CrvSteps::R3InclusiveSteps`. `art/CrvFastSim/fcl/dump_inclusive_capture_steps.fcl` targets those steps
   and the original `g4run::R3Inclusive` SimParticles. This conversion still needs separate integrity checks and
   explicit approval; the raw-capture runner never invokes it.

Two paths in one job are not this persistence boundary: a producer exception can prevent output retention.
The split removes that downstream exception from the raw job but does not guarantee survival of an exception
inside G4, art serialization, disk exhaustion or process termination. A real new overflow remains untested.

## Existing-tool bounded preparation

`scripts/run_inclusive_run1a_capture.sh` performs no submission, build, retry, fallback or deletion. Execution is
switched off by default; its only modes are `preflight`, `1`, and `10` within an explicit allocated runtime. Preflight
requests expansion/debug-validation of both schedules without event transport. Nonzero skip requires one event
and an expected exact event ID, independently verified in the original art input log before G4. Original and
transported generator dumps use consistent source ranges; the saved-raw dump must not apply the skip a second time.
The EXIT handler retains diagnostic products/logs and hashes whatever exists on both success and failure. Hashing
a partial art file records evidence, not usability or a complete event census. A successful wrapper exit means
commands/readback finished, not successful transport, passed audit or authorization to convert.

After an approved bounded capture, from the repository root run:

```bash
PYTHONPATH=. python3 scripts/audit_inclusive_run1a_capture.py \
  --directory "$RESULT" --events 1 --out "$RESULT/capture-audit.json"
```

Use `--events 10` only for that separately approved bound, or `--expected-event RUN SUBRUN EVENT` for a one-event
probe. The audit checks strict schema-v2 full-generator streams, exact original ProductID/index and all generator
payload columns, complete event census including empty events, unique per-event status and pre/post hashes. Valid
nonzero status/overflow/step-limit records produce diagnostic JSON with unknown transport rather than disappearing
from the census. Missing/malformed/ambiguous evidence refuses. `capture_checks_passed` describes only these bounded
retention checks; `raw_capture_review_gate_passed` also requires clean printed status and the log gate below.
Both are weaker than native raw completeness, pointer integrity, denominator closure or physics admission.
`conversion_eligible`, `raw_pointer_integrity_established`, `downstream_conversion_authorized` and both physics/
global-denominator flags remain false even for a clean capture. The audit is not a conversion-launch token.

## Status zero is necessary, not sufficient

Pinned source inspection shows that `Mu2eG4EventAction` sets status 1 for track step-limit or field-propagator
kills and status 10 for SimParticle overflow. The existing printer omits the separate field-propagator counter,
but a positive counter makes this source's status nonzero. Full serialized StatusG4 remains in raw art.

Separately, `CRVSD::ProcessHits` returns without saving hits after its effective `g4.stepsSizeLimit`. The sensitive
detector emits truncation warnings, but this condition is not included in the StatusG4 construction. Do not confuse
that effective geometry limit with the producer's `ResourceLimits.maxStepPointCollectionSize`; inspect both their
configuration and actual source wiring before claiming complete recording. No limit is changed here.

The raw FCL removes the G4 category message-count cap. The audit hashes the complete transport log, requires normal
art completion and conservatively quarantines the whole bounded capture on sensitive-detector truncation or
SimParticle-limit markers. It does not infer a per-event zero from an unattributed warning. Absence of these markers
is only a necessary diagnostic gate: effective SD limits, suppression settings, attempted/saved counts and raw
pointer/census integrity still require evidence. The existing strict CrvStep dumper validates converted pointer
associations, not all pre-conversion raw hit membership or attempted-hit counts. No native raw completeness
certificate is implemented. The exact saved-pointer and configuration readers described below are implemented
and syntax-checked, but their linked user build and actual saved-output readback remain pending. Neither source
implementation nor a syntax pass is a recording certificate or permission to build.

## Campaign definition to freeze before requesting a bulk launch

| Item | Required definition/gate |
|---|---|
| Source universe | Exact original generator art hashes, original process/tag, file and finished-event identities; no CRY regeneration, time-offset repetition, mixing or approximate association. |
| Development split | Preserve existing odd-subrun source membership as development only. All sources already inspected or used for decisions, including historical even validation studies, remain disclosed as such. |
| Confirmation split | Reserve fresh, disjoint generator identities before examining outcomes, or explicitly label reused even inputs as a reused validation set rather than fresh confirmation. New seeds alone do not make the source population unseen. |
| Internally consistent transport | Derive crossings, descendants, near-miss templates and status from the same new raw transports. Do not splice new crossings onto historical filtered descendants. |
| Randomness | Freeze a distinct seed per transport job and pinned RNG/runtime configuration, with event/file-range binding and immutable snapshots. No seed retries until a favorable status or response appears. |
| Exposure | Require complete requested finished-event census and full-generator identities, retaining empty events/noncrossing companions. Original full-subrun livetime cannot normalize a truncated probe. |
| Recording | Verify internal filtering is disabled, raw hit caps/warnings and pointer/product integrity, with one independent status per requested event. Preserve all failures as unknown. |
| Admission | One incomplete event blocks an exact inclusive denominator for that source universe. No event removal, exposure redefinition, automatic retry or silent resource-limit increase. Resolve any changed transport protocol separately and apply it consistently. |
| Construction | Integrate exact event + original primary + bar crossing/attachment/near-miss logic throughout builder and sidecars; keep the current multi-muon construction guard until that is complete. |
| Statistical validation | Freeze metrics, paired per-source-shard errors and decision thresholds before confirmation; retain timing, selection-migration, finite-library/clamp and V-R4/V-R5 gates. |
| Scope and resources | User-approved source ranges/events, site, CPUs/memory, wall and storage caps, concurrency and stop conditions, followed by measured sizing rather than extrapolation from tiny startup-dominated probes. |

A scale target preserving the historical source structure would be ten odd development and ten even reused-validation
files, each with 50,000 events (one million original events total). This is a planning option, not an authorization,
a statistically sized campaign, or a fresh held-out design. Input availability and unbiased recording must be verified;
any genuinely fresh confirmation source and its required event count are still to be agreed. Do not release this
option until raw-first runtime/integrity and resource gates pass. No bulk-capable submitter is supplied here.

## Completed bounded capture and saved-raw follow-up

The zero-transport configuration preflight passed: resolved capture has only G4, conversion only CrvSteps,
RootInput has no inherited firstRun, no output SelectEvents/filter is scheduled, and the transport configuration
matches the earlier combined control apart from the explicit source skip and seed. Reference geometry, field-off
configuration, cuts and resource limits are unchanged; G4 message-category cap removal is the intended diagnostic
delta. Runtime binary hashes also match that control. Subsequently authorized one-event and ten-event raw-only
controls completed with clean independent capture gates. That authorization is exhausted. Reuse their saved raw
outputs for any separately authorized integrity diagnostics or conversion; do not rerun G4. Actual overflow
retention and mixed crossing/noncrossing multi-muon coverage remain untested; no artificial empty generator event
or forced low-limit transport is implied. These controls establish the bounded capture route, not campaign sizing
or the missing native recording certificate.

### Existing-reader diagnostic candidate

`art/CrvFastSim/fcl/inspect_inclusive_capture_pointers.fcl` is a readback-only candidate, not an executed check or
launch wrapper. It extends the independent status reader with explicit raw-CRV and SimParticle printers and the
existing `PointerCheck` analyzer. Printer momentum cuts are disabled and all particles are listed; SimParticle
verbosity 2 includes daughter keys. `PointerCheck` uses verbosity 1, dereferencing enabled and no exclusions.
Its pinned implementation can return a failure without throwing at verbosity zero, so quiet mode is not safe.
Required explicit printer tags make missing selected products throw rather than letting `getMany` pass vacuously.

The installed checker tests internal SimParticle IDs and whether nonnull raw-hit, parent, generator and daughter
pointers dereference. It does not require their ProductIDs/keys to belong to the selected original collections,
enforce the all-PDG original-primary bijection, or export each pointer identity. Available pointers into the wrong
product can pass. Raw-hit and SimParticle printers omit required pointer ProductIDs; a generator-type name is not
an original generator identity. Complete logs and collection/event counts are necessary even for these partial
diagnostics. Success must not set `raw_pointer_integrity_established` or authorize conversion.

### Exact raw exporter implemented; build and readback pending

None of the inspected previously built interfaces supplies the full pre-conversion membership certificate.
The new standalone exporter and strict auditor implement the following contract; a linked user build and actual
saved-output checks are still required before claiming bounded pointer integrity:

- Explicit original generator, SimParticle and raw CRV tags, plus one event census including all zero-row events.
- Every raw hit index and its SimParticle pointer nullness, full-width ProductID/key, availability and membership.
- Every SimParticle map key/internal ID and every original-generator, parent and daughter pointer, including nulls,
  unavailable links and links outside the selected collections; preserve failures as diagnostics, never skip them.
- Exact all-PDG original-primary bijection and signed PDG; acyclic own-primary ancestry and reciprocal parent/daughter
  membership, without deriving original identity from key arithmetic, rank, time or payload similarity.
- Serialized status fields and exact saved-hit/particle counts bound to the original capture and exporter hashes.

The standalone exporter and strict auditor are now implemented under the [exact raw integrity contract](native-raw-integrity.md).
Both new C++ readers pass pinned-header syntax validation; an actual user build and saved-output readback are still
required. A reader of saved art cannot recover unpersisted attempted-hit counts. A separate configuration reader
reconstructs the effective geometry `g4.stepsSizeLimit` and compares original Geometry path/lines/hash evidence;
this is a source-derived cap gate, not measured attempted-versus-saved equality. Source defaults alone are not an
attestation of the capture's effective configuration. No limit or transport diagnostic is changed here.

### Retrospective converted-output audit, not an execution route

`scripts/audit_inclusive_run1a_conversion.py` is prepared for separately produced conversion outputs; it executes
neither art nor conversion. It has only synthetic validation so far. It freshly recomputes the capture audit into
a temporary receipt rather than trusting an existing JSON file, then requires its actual clean gate. Inventories
must have a single unique raw CRV collection with the expected process and no competing CrvSteps collection.
Converted output must retain the raw products and original trigger results and add the exact new-process steps
and trigger results. Independent converted status must match raw status exactly; normal conversion completion,
strict schema-v2 all-PDG original-primary bijection, ancestry and converted-step checks are mandatory.

For already produced evidence only, from the repository root:

```bash
PYTHONPATH=. python3 scripts/audit_inclusive_run1a_conversion.py \
  --raw-directory "$RAW_RESULT" --conversion-directory "$CONVERSION_RESULT" \
  --events 1 --out "$AUDIT_JSON"
```

The only other supported bound is 10; optional `--expected-event RUN SUBRUN EVENT` requires one event. All eleven
raw/converted evidence files, including both art files, and four audit-source files are hashed before/after.
Missing, empty, aliased, malformed or drifting evidence refuses without a report; outputs are exclusive. Opaque art
hashes bind bytes, not dump provenance or native validity. `conversion_checks_passed` is retrospective bounded
evidence, never a launch token. Raw-pointer integrity, native recording completeness, conversion eligibility,
authorization, physics admission and global denominator flags remain false. No conversion job or submitter has been
executed or supplied by this audit; approval and the unresolved pre-conversion integrity gates remain separate.
