#!/bin/bash
# Execute only in an explicit compute allocation against an already-built runtime.
set -eo pipefail
: "${SLURM_JOB_ID:?}" "${INCLUSIVE_INPUT:?}" "${INCLUSIVE_OUT:?}" "${INCLUSIVE_CODE:?}"
: "${INCLUSIVE_RUNTIME:?}" "${INCLUSIVE_RUNNER:?}" "${INCLUSIVE_SEED:?}"
: "${CVMFSEXEC_DIST:?}" "${OFFLINE_IMAGE:?}" "${OFFLINE_BINDS:?}"
[[ ${INCLUSIVE_EXECUTE:-0} == 1 ]] || { echo 'Execution disabled: set INCLUSIVE_EXECUTE=1 after preflight approval'; exit 2; }
[[ $INCLUSIVE_SEED =~ ^[0-9]+$ ]] || exit 2
case ${INCLUSIVE_EVENTS:-preflight} in preflight|1|10) ;; *) exit 2 ;; esac
SKIP=${INCLUSIVE_SKIP:-0}
[[ $SKIP =~ ^(0|[1-9][0-9]*)$ && ${#SKIP} -le 8 ]] || exit 2
if [[ $SKIP != 0 || -n ${INCLUSIVE_EXPECT_EVENT:-} ]]; then
  [[ ${INCLUSIVE_EVENTS:-preflight} == 1 && ${INCLUSIVE_EXPECT_EVENT:-} =~ ^[0-9]+:[0-9]+:[0-9]+$ ]] || exit 2
fi
[[ -f $INCLUSIVE_INPUT && ! -e $INCLUSIVE_OUT && ! -L $INCLUSIVE_OUT ]] || exit 2
mkdir "$INCLUSIVE_OUT"
cd "$INCLUSIVE_OUT"
sha256sum "$INCLUSIVE_INPUT" > INPUT.sha256
cp "$INCLUSIVE_CODE/grid/fcl/"ref{_sim_extracted,_sim_run1a,_inclusive_run1a}.fcl .
cp "$INCLUSIVE_CODE/art/CrvFastSim/fcl/"{dump_inclusive_run1a,audit_inclusive_run1a}.fcl .
sha256sum *.fcl "$INCLUSIVE_RUNNER" "$INCLUSIVE_CODE/scripts/run_inclusive_run1a_pilot.sh" > SOURCE.sha256
printf '#include "ref_inclusive_run1a.fcl"\nservices.SeedService.baseSeed : %s\nsource.skipEvents : %s\n' "$INCLUSIVE_SEED" "$SKIP" > pilot.fcl
printf '#include "CrvFastSim/fcl/dump_gen.fcl"\nphysics.analyzers.dump.genParticles : "cosmicTimeOffset::Primary"\nservices.TFileService.fileName : "input-gen.root"\nsource.skipEvents : %s\n' "$SKIP" > dump_input.fcl
{
  printf 'set -eo pipefail\n'
  printf 'export TMPDIR="$(dirname "$OFFLINE_CACHE_PATH")/tmp" SQLITE_TMPDIR="$(dirname "$OFFLINE_CACHE_PATH")/sqlite"\nmkdir -p "$TMPDIR" "$SQLITE_TMPDIR"\n'
  printf 'fhicl-dump -c %q > expanded.fcl\n' "$INCLUSIVE_OUT/pilot.fcl"
  printf 'config_dumper -P %q > input-embedded.fcl\n' "$INCLUSIVE_INPUT"
  printf 'mu2e --print-description PrintModule > PrintModule.description.txt\n'
  printf 'mu2e --print-description CrvStepTableDumper > Dumper.description.txt\n'
  printf 'mu2e --debug-config validated.fcl -c %q > validation.log 2>&1\n' "$INCLUSIVE_OUT/pilot.fcl"
  printf 'executable=$(readlink -f "$(command -v mu2e)"); backing=${executable%%/build/*}\nprintf "%%s\\n" "$backing" > backing.txt\n'
  printf 'sha256sum "$executable" "$backing/build/$MUSE_STUB/Offline/lib/"*Mu2eG4* "$backing/build/$MUSE_STUB/Offline/lib/"*PrintModule* > BINARIES.sha256\n'
  printf 'sha256sum %q/build/$MUSE_STUB/CrvFastSim/lib/*CrvStepTableDumper* >> BINARIES.sha256\n' "$INCLUSIVE_RUNTIME"
  printf 'cp "$backing/Offline/Print/src/StatusG4Printer.cc" "$backing/Offline/MCDataProducts/src/StatusG4.cc" .\n'
  if [[ ${INCLUSIVE_EVENTS:-preflight} != preflight ]]; then
    printf 'mu2e -c %q -s %q -n %q > input-dump.log 2>&1\n' "$INCLUSIVE_OUT/dump_input.fcl" "$INCLUSIVE_INPUT" "$INCLUSIVE_EVENTS"
    if [[ -n ${INCLUSIVE_EXPECT_EVENT:-} ]]; then
      IFS=: read -r ER ES EE <<< "$INCLUSIVE_EXPECT_EVENT"
      printf 'awk -v expect=%q %q input-dump.log\n' "record. run: $ER subRun: $ES event: $EE at " '/Begin processing/ {n++; if(index($0,expect)==0) bad=1} END {exit !(n==1 && !bad)}'
      printf 'printf %q > EXPECTED_EVENT.txt\n' "$INCLUSIVE_EXPECT_EVENT"
    fi
    printf '"$OFFLINE_TIME" -v -o transport.time mu2e -c %q -s %q -n %q > transport.log 2>&1\n' "$INCLUSIVE_OUT/pilot.fcl" "$INCLUSIVE_INPUT" "$INCLUSIVE_EVENTS"
    printf 'product_sizes_dumper inclusive-run1a.art > products.txt\n'
    printf 'mu2e -c %q -s inclusive-run1a.art > status.log 2>&1\n' "$INCLUSIVE_OUT/audit_inclusive_run1a.fcl"
    printf 'awk -v expected=%q %q status.log\n' "$INCLUSIVE_EVENTS" '$1=="status:" {n++; if($2!=0 || $6!=0 || $8!=0) bad=1} END {exit !(n==expected && !bad)}'
    printf 'mu2e -c %q -s inclusive-run1a.art > dump.log 2>&1\n' "$INCLUSIVE_OUT/dump_inclusive_run1a.fcl"
  fi
  printf 'sha256sum -c BINARIES.sha256\n'
} > chain.sh
export OFFLINE_CHAIN="$INCLUSIVE_OUT/chain.sh" OFFLINE_LOG_PREFIX=inclusive_pilot
bash "$INCLUSIVE_RUNNER" "$INCLUSIVE_RUNTIME" "$INCLUSIVE_OUT/pilot.fcl"
sha256sum -c INPUT.sha256
sha256sum -c SOURCE.sha256
sha256sum *.fcl chain.sh *.txt *.cc > OUTPUTS.sha256
if [[ ${INCLUSIVE_EVENTS:-preflight} != preflight ]]; then
  sha256sum *.art *.root >> OUTPUTS.sha256
fi
printf 'INCLUSIVE_PILOT_EXECUTION_EXIT=0; PHYSICS_ADMITTED=false\n'
