#!/bin/bash
# Read existing bounded raw art only; never transport, convert, build, submit, retry or delete.
set -eo pipefail
: "${SLURM_JOB_ID:?}" "${INCLUSIVE_RAW_DIRECTORY:?}" "${INCLUSIVE_OUT:?}" "${INCLUSIVE_CODE:?}"
: "${INCLUSIVE_RUNTIME:?}" "${INCLUSIVE_RUNNER:?}"
: "${CVMFSEXEC_DIST:?}" "${OFFLINE_IMAGE:?}" "${OFFLINE_BINDS:?}"
[[ ${INCLUSIVE_READBACK_EXECUTE:-0} == 1 ]] || { echo 'Readback disabled; built exporter and scoped execution required'; exit 2; }
case ${INCLUSIVE_EVENTS:-} in 1|10) ;; *) exit 2 ;; esac
[[ -d $INCLUSIVE_RAW_DIRECTORY && ! -L $INCLUSIVE_RAW_DIRECTORY ]] || exit 2
RAW=$INCLUSIVE_RAW_DIRECTORY/inclusive-raw-run1a.art
[[ -f $RAW && ! -L $RAW && ! -e $INCLUSIVE_OUT && ! -L $INCLUSIVE_OUT ]] || exit 2
mkdir "$INCLUSIVE_OUT"
cd "$INCLUSIVE_OUT"
finalize() {
  rc=$?
  trap - EXIT
  set +e
  [[ ! -f INPUTS.sha256 ]] || { sha256sum -c INPUTS.sha256 > input-check.log 2>&1 || rc=1; }
  [[ ! -f SOURCES.sha256 ]] || { sha256sum -c SOURCES.sha256 > source-check.log 2>&1 || rc=1; }
  printf '%s\n' "$rc" > READBACK_WRAPPER_EXIT.txt
  shopt -s nullglob
  files=(*.fcl *.sh *.txt *.log *.root *.sha256)
  sha256sum "${files[@]}" > OUTPUTS.sha256 || rc=1
  [[ $rc != 0 ]] || printf 'RAW_EXPORT_READBACK_EXIT=0; EXACT_AUDITS_REQUIRED=true; CONVERSION_AUTHORIZED=false; PHYSICS_ADMITTED=false\n'
  exit "$rc"
}
trap finalize EXIT
sha256sum "$RAW" "$INCLUSIVE_RAW_DIRECTORY/transport.log" > INPUTS.sha256
for file in dump_inclusive_raw_integrity.fcl dump_inclusive_recording_config.fcl; do
  cp "$INCLUSIVE_CODE/art/CrvFastSim/fcl/$file" .
done
sha256sum *.fcl "$INCLUSIVE_RUNNER" "$INCLUSIVE_CODE/scripts/run_inclusive_run1a_raw_integrity.sh" > SOURCES.sha256
printf '#include "dump_inclusive_raw_integrity.fcl"\nphysics.analyzers.recording : { module_type : CrvRecordingConfigDumper inputFile : "Production/JobConfig/cosmic/geom_cosmic_run1_a.txt" allowReplacement : true }\nphysics.integrity : [ rawdump, recording ]\n' > readback.fcl
sha256sum readback.fcl >> SOURCES.sha256
{
  printf 'set -eo pipefail\n'
  printf 'export TMPDIR="$(dirname "$OFFLINE_CACHE_PATH")/tmp" SQLITE_TMPDIR="$(dirname "$OFFLINE_CACHE_PATH")/sqlite"\nmkdir -p "$TMPDIR" "$SQLITE_TMPDIR"\n'
  printf '[[ $MUSE_STUB == al9-prof-e29-p107 ]] || exit 2\n'
  printf 'executable=$(readlink -f "$(command -v mu2e)"); backing=${executable%%/build/*}\n'
  printf '[[ $backing == */Offline/v13_39_00 ]] || exit 2\n'
  printf 'for module in CrvRawIntegrityDumper CrvRecordingConfigDumper; do\n'
  printf 'lib=%q/build/$MUSE_STUB/CrvFastSim/lib/libmu2e_CrvFastSim_${module}_module.so\n' "$INCLUSIVE_RUNTIME"
  printf '[[ -f $lib ]] || exit 2\nsha256sum "$lib" >> BINARIES.sha256\ndone\n'
  printf 'sha256sum "$executable" "$backing/build/$MUSE_STUB/Offline/lib/libmu2e_ConfigTools.so" >> BINARIES.sha256\n'
  printf 'sha256sum "$backing/Offline/Mu2eG4/src/CRVSD.cc" "$backing/Offline/Mu2eG4/src/Mu2eG4SensitiveDetector.cc" "$backing/Offline/Mu2eG4/src/SensitiveDetectorHelper.cc" "$backing/Offline/Mu2eG4/src/Mu2eG4PerThreadStorage.cc" "$backing/Offline/ConfigTools/src/SimpleConfig.cc" > RECORDING_SOURCES.sha256\n'
  printf 'printf "%%s\\n" "$MU2E_SEARCH_PATH" > geometry-search-path.txt\n'
  printf 'mu2e --debug-config readback-validated.fcl -c %q > validation.log 2>&1\n' "$INCLUSIVE_OUT/readback.fcl"
  printf 'mu2e -c %q -s %q > exporter.log 2>&1\n' "$INCLUSIVE_OUT/readback.fcl" "$RAW"
  printf 'sha256sum -c BINARIES.sha256\nsha256sum -c RECORDING_SOURCES.sha256\n'
} > chain.sh
export OFFLINE_CHAIN="$INCLUSIVE_OUT/chain.sh" OFFLINE_LOG_PREFIX=inclusive_raw_integrity
bash "$INCLUSIVE_RUNNER" "$INCLUSIVE_RUNTIME" "$INCLUSIVE_OUT/readback.fcl"
