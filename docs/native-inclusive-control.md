# Run 1A inclusive existing-generator control

## Scope

Bounded one/ten-event runtime controls passed, not an admitted sample or a bulk campaign. Replay original `cosmicTimeOffset`
primaries from the retained full-generator art using the pinned reference runtime. Do not regenerate CRY or apply
a second time offset. This is new stochastic G4 transport, not recovery of historical discarded trajectories.

`grid/fcl/ref_inclusive_run1a.fcl` inherits reference geometry, field, physics and transport limits, replaces the
source with RootInput, and schedules only G4 and CRV-step construction. Neither the G4-status filter nor the detector
filter runs. Output keeps original generator products, uncompressed SimParticles, StatusG4, raw CRV StepPointMCs,
CRV steps and trigger results. Original generator pointers remain available directly, without compression mapping.
`art/CrvFastSim/fcl/dump_inclusive_run1a.fcl` configures the existing dumper for these direct associations.

## Launch gates

1. Expand the FCL with the pinned runtime; check exact geometry, field, cuts, original process tags and scheduled modules.
2. Validate one event, then ten events, with one CPU and an exclusive output directory for each step.
3. Verify event census and original generator pointer identity, including empty generator events and muon companions.
4. Read StatusG4 independently; a failed transport is unknown, never a measured zero or a near-miss.
5. Exercise the known overflow-event primaries as a separate new transport control, not a historical replay claim.
6. Before scaling, measure pilot output size and completion time; do not extrapolate runtime from small event counts.
7. Establish a status extraction/audit path before library admission; the step dumper alone does not export StatusG4.

The initial control preserves the reference particle collection limit. If overflow occurs, retain the event and
refuse completeness. Any limit change must be explicit and separately compared, not silently mixed into references.

`scripts/run_inclusive_run1a_pilot.sh` requires an explicit compute allocation, built runtime, input, seed, bind
paths and exclusive output directory. It performs no build or submission. `INCLUSIVE_EVENTS` accepts only
`preflight` (default), `1`, or `10`; execution requires `INCLUSIVE_EXECUTE=1`. Preflight expands and validates the
configuration without transport. A successful exit is execution evidence, not physics admission. The caller must
inspect the expanded configuration before transport, and audit the one-event result before the ten-event launch.
For a separately approved one-event boundary probe, `INCLUSIVE_SKIP` selects an existing input entry and requires
`INCLUSIVE_EXPECT_EVENT=run:subrun:event`. Both the independent input dump and transport use the same skip count;
an exact art input-log event check runs before G4. The audit's `--expected-event RUN SUBRUN EVENT` additionally binds
the finished input and transported census to that identity. A source-order probe is not event selection for exposure.
The runner stops after a nonzero G4 status rather than passing damaged pointers to the step dumper or retrying.

Retain output art until status and association audits pass. Do not mix new crossing inventories with old filtered
descendant templates: the inclusive library must use internally consistent transport and ancestry.

`audit_inclusive_run1a.fcl` reads StatusG4 independently with the existing PrintModule and an explicit process tag;
a missing product throws. It also prints retained product provenance and trigger results. Per-event status, overflow
and step-limit counters must reconcile with the event census. This printer omits the separate field-propagator counter
and rounds times; it is not a dump of every serialized status field. Nonzero status remains unknown transport.
Status readback precedes the strict step dumper, which can itself refuse unresolved overflow pointers.

From the repository root, `PYTHONPATH=. python3 scripts/audit_inclusive_run1a_pilot.py --directory RESULT --events 1 --out audit.json` compares the complete
retained event census and generator payload to an independent first-N input dump, checks actual primary pointer
bijection across all PDGs, and validates strict step/parent associations and per-event status. Use `--events 10`
for the second control. The report remains bounded-pilot evidence, never global denominator or physics admission.
Any-CRV-step ancestor counts include secondary-only depositions and are explicitly not primary-crossing counts.
`scripts/inspect_inclusive_edge_inputs.py` scans explicitly supplied existing schema-v2 full-generator tables for
empty-generator events and exact boundary probes without transport. Its zero-based census row indices are only
candidate skip values: the independent art input event check is mandatory before running a selected probe. If no
empty-generator event exists, report that fact rather than synthesizing one or claiming transport coverage.
Original subrun livetime/count products are retained for provenance; their full-input exposure must not be assigned
to a truncated one/ten-event pilot or substituted for its finished-entry count.

The CRV-step producer selects module/instance without a process qualifier. Generator-only input provenance and its
first-event selected-product list must therefore show only the new transport's CRV StepPointMC collection. Overflow
can make CRV-step construction fail before output retention; such a failure is not evidence of zero crossings and
requires a separately scoped G4-only status-retaining control rather than silently increasing the particle limit.

## Bounded runtime evidence and remaining gates

The pinned-runtime one-event and ten-event controls completed with exact original-generator pointer/payload/census
checks, strict step/parent associations and independent status readback. The ten-event control retained all ten events,
including eight zero-step events, with ten status-zero records and no overflow or step-limit kills. These controls
establish the configuration/readback route only, not statistical transport validation or complete denominator coverage.
The initial controls had no empty-generator or multi-muon event. A separately approved one-event control subsequently
transported the known historical overflow-event primaries: both original muon identities and all parent associations
passed, with status zero, no overflow or step-limit kills, and zero CRV steps. This is new stochastic transport; it
does not establish the discarded historical trajectory's crossing status or prove overflow-failure retention. Nor does
it test a multi-muon event with one crossing and one noncrossing companion. The reference particle limit was unchanged.
A read-only scan found no empty-generator event in the supplied existing training census, so actual empty-event transport
was not exercised or synthesized. Synthetic tests cover these cases in the audit, not in actual transport. Any further
transport requires its own explicit scope. Keep the original particle limit unless a separate change is approved.

Campaign seeds, resources, successful complete transport and internally consistent training/validation inputs must
be frozen before bulk execution. Do not extrapolate campaign runtime or storage from these tiny controls. The combined
controls used existing built tools; the later exact saved-raw integrity route needs the new standalone readers and a
user build, still pending. Existing construction guards stay unchanged. No bulk transport or library admission is
implied by these pilot results.

## Raw-first follow-up preparation

[Raw-first capture and campaign gates](native-inclusive-capture.md) separates G4-only persistence from later CRV
conversion. Both split configurations pass pinned-runtime expansion/debug-config checks. Separately authorized
one-event and ten-event raw-only captures also passed exact original/retained generator and independent status/log
gates; conversion remains unexecuted and needs separate approval. Exact raw-pointer membership and native hit
recording completeness are still unproven. The source audit identifies warning-only CRV sensitive-detector hit
truncation not covered by StatusG4; status zero alone is not a raw-recording completeness certificate. Existing
successful controls and the construction guard are unchanged.
