# Saved-raw exact integrity export contract

## Scope

The standalone `CrvRawIntegrityDumper` reads saved raw products only. It does not schedule transport or CrvSteps,
modify inputs, discard damaged events, certify attempted-hit counts, or change existing dumpers. A separate
read-only Python audit validates the export against the independent original full-generator stream. The exporter
records diagnostics rather than throwing on a bad embedded pointer. A missing/corrupt collection remains failed
census evidence; it is never an empty collection. Both standalone readers and their Python audits are implemented
and synthetic-tested; the C++ readers passed pinned-header syntax-only checks. Neither reader has a confirmed
linked build or an executed saved-art readback. This is an implemented contract, not a measured integrity result.

## ROOT schema v1

All trees live under `rawdump/`. Every row has `run`, `subrun`, `event` uint32. The event census includes zero-row
events. Counts/indices/map keys/embedded pointer keys are uint64, ProductIDs uint32, switches/status/PDGs int32,
and payload/timing values double. Null pointer keys are zero with `nonnull=0`; zero is not a null-key sentinel.
Integer values must never pass through floating point. No particle or pointer is removed on error.

| Tree | Additional columns |
|---|---|
| `events` | `schema_version` uint32=1; `gen_product_id`, `sim_product_id`, `raw_product_id`, `status_product_id` uint32; `gen_present`, `sim_present`, `raw_present`, `status_present`, `collection_error` int32; `n_gen`, `n_particles`, `n_raw`, `n_daughters` uint64; `status`, `nG4Tracks`, `overflowSimP`, `nStepLimit`, `nFieldPropagator` int32; `cpuTime`, `realTime` double; string `gen_tag`, `sim_tag`, `raw_tag`, `status_tag`. |
| `gen` | `idx` uint64; `gen_product_id` uint32; `pdg`, `genid` int32; `e`, `t`, `x`, `y`, `z`, `px`, `py`, `pz` double. |
| `particles` | `key`, `id`, `n_daughters` uint64; `sim_product_id` uint32; `pdg`, `isprim` int32; pointer groups `parent_*` and `gen_*`. |
| `raw` | `idx` uint64; `raw_product_id` uint32; `bar` uint32; pointer group `sim_*`. |
| `daughters` | `owner_key`, `idx` uint64; `sim_product_id` uint32; pointer group `daughter_*`. |

Each pointer group contains `nonnull` int32, `product_id` uint32, `key` uint64, and int32 `available`, `dereferenced`,
`member`, `matches_selected`, `error`. `member` compares ProductID and key against the explicitly selected collection,
without dereferencing; `matches_selected` additionally compares the resolved address to that collection element.
All pointer access exceptions set `error=1`; metadata is recorded before potentially throwing dereferences.
`available`, `dereferenced`, `member`, `matches_selected` are zero for null pointers. Null pointers retain their
serialized ProductID; a null pointer with nonzero ProductID is not canonical and is refused by the strict audit.
Collection read failures set `collection_error=1`, `*_present=0` and zero counts for missing products, which cannot
be admitted as empty events. Missing status has `status=-1` and zero counters/times but `status_present=0`.

## Exact audit obligations

- Bind original schema-v2 full-generator census/payload, raw export, independent raw status/inventory/log and saved
  raw art before and after audit; freshly recompute the bounded capture gate, do not trust an old receipt.
- Require exact event census and exact process-qualified tags, complete distinct generator indices and all payload
  columns equal to the independent original stream, including every PDG and empty events.
- Require all selected collections/status present and no collection errors. Compare every serialized status field
  available in the independent printer, require all error counters (including field propagation) zero and finite
  nonnegative serialized CPU/wall times. Compare counters exactly and times after rendering the serialized values
  to the independent printer's fixed three-decimal precision, never treating printed times as exact serialized values. Status zero is necessary, not a native recording certificate.
- Validate every raw-hit pointer's full ProductID/key and membership, availability and resolved target. Raw indices
  must be a complete distinct range per event, including zero-hit events. Do not infer primary crossings here.
- Validate all SimParticle map keys against internal IDs, every nonnull original-gen/parent/daughter pointer and
  count, signed primary PDG, exact all-PDG original-primary bijection, acyclic own-primary ancestry for ALL particles,
  and exact reciprocal parent/daughter edges. Daughter indices must be complete, distinct and owner-bound.
- Refuse missing/malformed evidence. Well-formed failed status/pointer diagnostics yield a failed integrity receipt
  with all event outcomes still unknown, never a filtered successful subset. Success is bounded saved-pointer
  integrity only, not historical trajectory recovery, native attempted/saved recording, conversion permission,
  library/physics admission or global denominator completeness.

## Recording is independent

A post-transport reader cannot recover unpersisted attempted-hit counts. Determine the effective geometry
`g4.stepsSizeLimit` from the exact pinned source/configuration resolution and bind it to the original captures;
do not confuse it with producer ResourceLimits. No new transport or limit change is implied by this contract.

## Implemented readers and execution boundary

`CrvRawIntegrityDumper_module.cc` implements the full export above; `dump_inclusive_raw_integrity.fcl` supplies the
exact tags and analyzer label `rawdump`. `audit_inclusive_run1a_raw_integrity.py` implements the strict whole-export
audit. Its `raw_pointer_integrity_established` can become true only for that bounded saved export after all checks;
all native-recording, conversion-authorization, physics and global-denominator flags remain false. Full uint64 keys,
including values above signed int64, remain exact throughout joins. Failed diagnostics are not a successful subset.

`CrvRecordingConfigDumper_module.cc` separately constructs the same `SimpleConfig` as the capture, without creating
a G4 world. Its single `recording/config` row records requested/resolved input path, replacement policy, recursive
line count, uint64 legacy hash, active resolved configuration text, explicit-limit presence and effective value with
zero fallback. `dump_inclusive_recording_config.fcl` schedules both readers, never transport or conversion.
`audit_inclusive_run1a_recording.py` compares that row to the original transport log's Geometry path/lines/hash;
it reports only a source-derived cap gate, not measured attempted/saved equality or native completeness.

The legacy Boost hash is not a cryptographic include-closure attestation. The pinned `SimpleConfig` also hashes the
old accumulated line rather than the newly read line in one continuation path. Matching the recorded hash is useful
reconstruction evidence but cannot alone attest identical original contents. Resolve and inspect the active limit,
keep SHA256/source bindings, and retain this limitation. A nonpositive reconstructed value and bound deployed cap
branch can establish the branch was disabled under that configuration, not a measured attempted-hit counter.

`scripts/run_inclusive_run1a_raw_integrity.sh` is disabled by default and accepts only declared one-/ten-event saved
captures in an allocated runtime. It requires the new built plugins in the supplied p107 area and refuses another
backing/stub. It reads the complete saved file without a new skip or event truncation, writes only an exclusive new
output directory, hashes raw art/transport log, sources and binaries before/after, and never transports, converts,
builds, submits, retries or deletes. The Python audit enforces the declared bound after readback. A wrapper exit zero
means only export execution, not an integrity gate or conversion authorization.

The user builds the isolated `art/` area. `scripts/syntax_check_inclusive_readers.sh` performs only `g++ -fsyntax-only`
against the pinned backing and makes no binary; it does not replace the user build, plugin loading or real readback.
After the user confirms the build, verify binaries/source dates and hashes, deploy only the new helper libraries to
a separate runtime, then read the already saved one-/ten-event raw files. Preserve their original snapshots/art/logs.
