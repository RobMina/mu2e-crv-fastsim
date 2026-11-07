# Cached C4 comparisons

> Historical diagnostic checkpoint: this branch preserves the development-input recovery and residual studies.
> Its retained-input implementation predates the reviewed fixes merged in
> [PR29](https://github.com/RobMina/mu2e-crv-fastsim/pull/29). Use current main for that path; do not merge this
> snapshot over the corrected implementation. Held-out results, response banks and production defaults stay frozen.

The cached track selection stays fixed. `python3 -m crvfastsim.veto` reproduces the stock veto rule;
`python3 -m crvfastsim.veto.paired` additionally compares a replacement coincidence collection:

```bash
python3 -m crvfastsim.veto.paired selection.parquet --card veto_stats.json \
  --fast fast.parquet --label 'Primary-only placeholder' --sectors 1 2 --outdir comparison
```

Both parquet files carry `run`, `subrun`, `event`; the reference also contains jagged `p_mid`, `t_mid` and `coincs`.
The candidate contains `coincs` with `time`, `timeStart`, `timeEnd`, `PEs`, `nLayers`, `sectorType`.
Every cached event must be present, including events with an empty fast response. Event rows may be reordered.
A `shard` column in the reference defines the paired error clusters; otherwise `(run, subrun)` is used.

Outputs are `numbers.json` and a momentum-binned inefficiency overlay with a paired pull panel, `crv_veto.png`.
Vetoed fraction and inefficiency are separately named: inefficiency is **one minus the vetoed fraction**.
Ratios and differences use paired cluster-robust delta-method errors. One shard cannot estimate a shard error;
zero reference counts have an undefined ratio. Neither case is replaced with a zero uncertainty.
The shape chi-square is the sum of squared paired bin pulls, without off-diagonal covariance or a fitted normalisation.

The denominator is the stock-selected, stock-gate-passing population. Fast gate migrations are reported separately,
not used to silently remove selected tracks. Sector counts are exclusive attribution to the highest-PE vetoing
coincidence, matching the existing veto CLI. They are **not** sector-conditioned inefficiencies. Supply every sector
type through `--sectors` to display sectors with zero response in both arms. In Run 1A, physical T1 and T2 both
have `sectorType=1`: type 2 is not physical T2, and a type table cannot separate their response.

## Primary-only placeholder

`crvfastsim.veto.primaries` reads saved original-track-index-aligned `trkmcsim` genealogy and the same source
EventNtuple's true and reconstructed bunch times. Generator-associated muons (`valid`, `gen == 38`, `abs(pdg) == 13`)
are deduplicated by event and SimParticle ID. Their stored start states, not a proximity match, provide the inputs.
The detector-frame origin in Mu2e coordinates is an explicit argument; it must match the source geometry.
`--files` with `--cached` and `--mca` recovers all source members using the original analysis selection.

```bash
python3 -m crvfastsim.veto.primaries --lineage selected_lineage.parquet --ntuple source.root \
  --detector-origin -3904 0 10171 --outdir primary_inputs
python3 -m crvfastsim.veto.primary_chain --selection primary_inputs/selection.parquet \
  --primaries primary_inputs/primaries.parquet --gdml run1a.gdml \
  --calib calib.csv --deviation photon.csv --outdir primary_response
```

The placeholder composes the existing Run 1A material transport, photon, SiPM, waveform and coincidence stages,
without tier-1b sampling. It converts step times to the true bunch frame, folds photon arrival times by the nominal
1695 ns microbunch period, digitises in 200–1500 ns, and subtracts the reconstructed bunch time from reconstructed
coincidence times. The period can be supplied with `--period`.

Add `--library-dir DIRECTORY` to use the `run1a_v01` root-coherent library and its topology/template sidecars.
This opt-in route needs the per-primary sampler. The default remains primary-only. `--split dev` uses source-file
shards 0–21 and `--split heldout` uses 22 onward; `--split-at` changes that boundary before comparing results.
The output also includes the exact selection subset. Choose settings on development files, then report held-out once.

This is a diagnostic: represented genealogy muons are not an all-primary census; primary-only has no secondary showers,
no magnetic-field deflection, and no production-era response validation. The existing SiPM pre-window approximation
also differs from the stock start margin. Zero represented-muon events remain in the response file. Do not use its
throughput per selected event as a rate per generated cosmic or as a stock-grid speedup.

## Saved-decision diagnostics

Start on development files and keep the held-out result frozen:

```bash
python3 -m crvfastsim.veto.diagnose --selection dev/selection.parquet --fast dev/fast.parquet \
  --primary primary_dev/fast.parquet --card veto_stats.json --outdir diagnosis
```

The per-track table preserves the stock denominator, distinguishes absent from ineligible coincidences and timing
failures, and counts represented generator muons. Each arm's representative coincidence maximises the signed margin
inside the fixed time window. The representatives are independent decision diagnostics, **not** truth matches.
Genealogy tables retain exact event/SimParticle identities, including generated non-muons excluded by the muon driver.
`--counters FILE --detector-origin X Y Z` adds a labelled truth-start straight-ray diagnostic, without material,
scattering or field deflection. Stock longitudinal cluster positions are only a physical-sector diagnostic, not an
exclusive attribution when sector footprints overlap. Event-level raw tables should stay with private outputs.

To locate losses before digitisation, `crvfastsim.veto.transport_diagnose` repeats only transport and optional library
sampling with the same event order and seed; `--seeds` permits a predeclared seed check. It requires `--selection`,
`--decisions diagnosis/tracks.parquet`, `--gdml`, `--detector-origin`, `--outdir`, and optionally `--library-dir`.
Step support in three distinct layers across the common Run 1A sector type is necessary but not sufficient for a
reconstructed coincidence. Counts at this boundary are not veto predictions, and extra seeds are not extra independent
input events. `--saved-steps FILE [...]` replaces transport replay with saved-step summaries and per-layer energy/time
tables; it does not require GDML or an origin. The layer timing diagnostic uses the nominal 1695 ns period.

`crvfastsim.veto.truth --files source_paths.txt --selection selection.parquet --outdir truth` reads the development
source files (shard <22 by default). It retains CRV reco/MC cluster index alignment and genealogy from every track.
Use `--truth truth/truth.parquet --decisions diagnosis/tracks.parquet --card veto_stats.json --selection selection.parquet
--outdir truth_report` for local ancestry summaries; `--support steps_seed_tracks.parquet` adds transport categories.
The highest-PE vetoing stock cluster selects its **native** MC summary, not a time- or position-matched truth particle.
These CRV MC summaries lack SimParticle IDs. Comparing their primary PDG/start coordinates against represented roots
is only a coverage diagnostic: incompatible signatures exclude a candidate; compatible signatures do not establish
identity. The union of all track genealogy is still not a complete generator collection. Do not fill missing inputs
from the vetoing cluster's truth, since that would condition the response on the reference outcome.

For an independent upstream check, `scripts/dump_c4_generators.fcl` uses an existing `CrvStepTableDumper` plugin
with steps disabled and the retained `compressDetStepMCs::CosmicS2` generator collection. Inspect actual source
products first; a compressed collection is not the full generated stream. Run `truth --generator-dump dump.root
--selection selection.parquet --shards 0 --outdir generator_check` to census it, retain empty events, and count exact
event-key overlap. Optional `--detector-origin X Y Z` compares start states at the ntuple's float32 precision and
lists retained generators without compatible genealogy states. It does not construct fast inputs or establish
particle identity. Check catalog ancestry and any resampling/time transformations before using upstream states;
never substitute a kinematic nearest-neighbour match for an event relation.

For a population-wide input census, `scripts/catalog_c4_generators.py --sources primary_summary.json --outdir catalog`
uses read-only MetaCat calls to follow the actual development ntuple/reco/digi/step ancestry. Run this in the data-handling
environment, separately from the analysis Python environment. `python3 -m crvfastsim.veto.inputs --selection selection.parquet
--ancestors catalog/ancestors.json --outdir plan` chooses ancestor files by shard/subrun without reading veto outcomes.
The catalogue script's `--locations plan/files.tsv` mode resolves known filenames and reads locality only; it never recalls tape.
`scripts/dump_c4_generator_files.py` runs the existing dumper sequentially, supports disjoint `--shards` assignments and
`--reuse NAME=ROOT`, and writes a completed-dump list. It requires an already configured runtime matching `--area`.

Compare with `python3 -m crvfastsim.veto.inputs --selection selection.parquet --dump-list completed_dumps.txt
--truth truth.parquet --primaries primaries.parquet --detector-origin X Y Z --outdir input_census`. The saved fast-input
check reproduces the original extraction exactly. Coverage distinguishes uncovered events from empty collections,
extra muons from excluded non-muons, missing genealogy states, duplicate stored generator states and ambiguous
compatibility. `--decisions tracks.parquet` adds decision attribution only after the independent census. A compressed
collection can retain CRV-contributing roots absent from track genealogy, but is still not the complete original CRY
stream. Its GenParticle indices are not downstream SimParticle IDs. These tools report input coverage, not a corrected
C4 prediction or a licence to feed veto-cluster truth back into fast inputs. Optional `--ancestry tracks_truth.parquet`
compares native CRV primary signatures only after the census; it never selects or constructs generator inputs.

## Retained-generator input path

Prepare inputs independently of veto decisions, retaining every species and the complete selected event census:

```bash
python3 -m crvfastsim.veto.generators --selection selection.parquet \
  --generators selected_upstream_generators.parquet --events selected_upstream_events.parquet \
  --outdir retained_dev
python3 -m crvfastsim.veto.primary_chain --selection retained_dev/selection.parquet \
  --primaries retained_dev/primaries.parquet --input-kind retained \
  --generator-events retained_dev/generator_events.parquet --split dev \
  --gdml run1a.gdml --calib calib.csv --deviation photon.csv --library-dir library --outdir response
```

Preparation defaults to development shards <22 and requires a census row for every selected event, including empty
collections. It validates generator counts and per-event ProductID/index uniqueness without deduplicating by kinematics.
Positions, momenta, global times and total energies remain in the upstream Mu2e frame at their stored precision.
The chain uses dense internal keys but saves `generators.parquet` and `generator_map.parquet` with original event keys,
`gen_product_id` and `source_idx`. All muons enter transport; all non-muons enter the existing shower sampler.
Non-muon inputs require the library rather than silently producing no response. The muon-only `represented` route
remains the default and retains its muon-mass energy calculation; `retained` preserves the supplied total energy.
The Slurm wrapper enables this path with `C4_INPUT_KIND=retained` and the prepared directory as `C4_INPUTS`.

This fixes input omissions, not the response model: retained compressed generators are not the original full CRY
stream, and non-muon showers remain species/energy-conditioned donor draws, not input position/direction-conditioned.
The report records sampler counts and this limitation. Validate on development events before any held-out evaluation;
changing the particle population changes RNG consumption, so event differences are not pure added-particle effects.

## Non-muon shower coverage

`python3 -m crvfastsim.veto.nonmuon --generators response/generators.parquet --library library/library.root
--sampler-source runtime/tier1b.py --outdir nonmuon` inspects every development non-muon input before any outcome join.
It uses the runtime sampler's mass map and KE bins and reports missing cells, zero-shower cells, donor support,
per-input reach probabilities, their expected shower count and the independent-draw probability of zero showers.
These are fixed-library sampling quantities, not uncertainties on the library or predicted veto counts.
Optional `--decisions diagnosis/tracks.parquet` joins saved decisions only after the coverage calculation.

For a fixed-seed check in the actual response runtime, omit `--sampler-source` and add
`--replay-selection response/selection.parquet --gdml run1a.gdml --seed 20261105`. This repeats transport/library only,
records the existing RNG draws without changing their consumption, and saves the complete sampler report for comparison.
It does not digitise, change library contents, scan seeds or replace the saved response. The portable wrapper is
`scripts/diagnose_c4_nonmuon.sbatch`. Held-out inputs are refused by this diagnostic.

The reach denominator describes the generated cosmic population, not the tracker-selected retained population.
Species/KE support alone therefore does not validate the conditional response. Sparse high-energy bins with no donor
showers receive zero probability, which is a finite-library limitation rather than a physical impossibility. Donor
positions, directions and relative times are copied without conditioning on the supplied primary position/direction.
Do not force showers, fit probabilities to stock veto outcomes, or infer that a different seed repairs this model.
