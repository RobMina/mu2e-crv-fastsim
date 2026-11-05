# Run 1A CRV checkpoint — 2026-11-05

This is a resumable development checkpoint on `crv-run1a-r3`, not a merge-ready release or physics admission.
The branch retains its original base; integration with current main and current-main CI remain outstanding.
Inherited named conditions, receipts and response diagnostics are dependencies, not Run 1A closure.

## Established and unestablished

| Area | Established | Still open |
|---|---|---|
| Native library | Construction/loading interfaces, strict stage-local input checks and source-bound whole-input multiplicity refusal. | No admitted inclusive Run 1A library. Historical bundles have measured identity defects; the isolated `native_identity` helper is not production-integrated. |
| Inclusive transport | Approved combined controls and subsequent raw-only one-/ten-event controls completed their respective bounded gates. | Historical discarded transport is not recovered. Empty-event, mixed crossing/noncrossing multi-muon and real overflow-failure recording coverage remain untested. No further G4 is authorized. |
| Saved-raw integrity | Exact raw-pointer exporter/auditor, separate SimpleConfig reconstruction reader/comparator, readback-only FCLs and disabled runner implemented. Both C++ readers passed pinned Offline v13_39_00/p107 syntax-only checks. | User linked build, plugin loading and actual exact saved-art readback have not occurred. New audits have synthetic validation, not actual raw-export results. |
| Recording | Pinned cap and collection-handoff source inspected; configuration reconstruction route implemented. | Effective cap has not yet been read by the new module. Legacy hash limitations remain. Attempted/saved-hit equality, native recording completeness and global inclusive denominator completeness are unestablished. |
| Conversion | Separate saved-raw conversion candidate and retrospective audit implemented. | No saved-raw CrvSteps conversion executed; explicit separate authorization and integrity review required. |
| Validation/C4 | Conditional component and same-step response diagnostics; exact stock cached C4 fraction reproduced. | Full native V4/V-R1–V-R5, selected paired closure, timing, migration, finite-library effects and native C4 remain open. Keep stock C4. |

The latest recorded scoped regression passed **510 tests across 11 files**, followed by **six final wrapper tests**.
These are overlapping selections, not additive full-suite CI. Earlier frozen and construction-focused regressions
have different selections and source snapshots; they must not be relabeled as validation of this final checkpoint.
No linked build, transport, readback, conversion or allocated event job was performed for this documentation checkpoint.
The final local portability/runner/diagnostic check passed eight tests; it is not a physics run.

Native ordinary-host and stock transport timings cover different stages. No matched full-chain speedup,
production GPU cost or campaign sizing is established. Tiny startup-dominated controls cannot size a campaign.

## Resume in order

1. The user builds the isolated `art/` area with the pinned p107 setup; see [art helpers](../art/README.md).
   Wait for build confirmation, then verify source/binary hashes, plugin dependencies and loadability.
2. Deploy only the newly built helpers into a separate runtime. Preserve the old transport runtime, saved raw
   captures and all diagnostics. Under an explicit bounded saved-readback scope, read the existing raw files;
   do not rerun G4, truncate the saved event stream or retry until favorable.
3. Run the exact [saved-raw integrity](native-raw-integrity.md) and recording-reconstruction audits, and review
   their bounded claims independently. A clean wrapper exit is not a gate. A cap-disabled reconstruction is not
   measured attempted/saved equality. Failed transport/recording remains unknown, never a zero crossing.
4. Obtain separate approval before actual [saved-raw conversion](native-inclusive-capture.md). Retrospective
   conversion audit code is not permission to launch it.
5. Integrate exact event + original primary + bar logic throughout construction and sidecars, with authoritative
   original-generator identities and complete transport/recording coverage. Keep the whole-input multi-primary
   guard until that is complete. Do not filter events, redefine exposure or disable near-misses to pass.
6. Deliver the newly validated anchored `run1a_v01` native library first; then the V series and paired histograms;
   then C4 on the exact cached selection with matched field/source lineage. A bulk campaign remains unsized and
   unauthorized; previously inspected even inputs are reused validation, not fresh confirmation.

Private runtime paths, source inventories, results, hashes and operational handoff stay in the private work record,
not this source repository. Generated conditions/receipts can contain private paths and logs; do not publish them
unreviewed. This checkpoint changes no main/default/device/core behavior and opens no pull request.
