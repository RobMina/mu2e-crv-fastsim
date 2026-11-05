#!/bin/bash
# Syntax-only validation of new readback analyzers, never a compilation/link build or event run.
set -eo pipefail
AREA=${1:?usage: syntax_check_inclusive_readers.sh AREA EXCLUSIVE_OUTPUT_DIRECTORY}
OUT=${2:?usage: syntax_check_inclusive_readers.sh AREA EXCLUSIVE_OUTPUT_DIRECTORY}
[[ -d $AREA && ! -e $OUT && ! -L $OUT ]] || exit 2
mkdir "$OUT"
OUT=$(cd "$OUT" && pwd -P)
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
setup muse
cd "$AREA"
muse setup -q p107 || true
[[ $(readlink -f "$MUSE_WORK_DIR") == "$(pwd -P)" && $MUSE_STUB == al9-prof-e29-p107 ]] || exit 2
printf 'MUSE_WORK_DIR=%s\nMUSE_STUB=%s\n' "$MUSE_WORK_DIR" "$MUSE_STUB" > "$OUT/runtime.txt"
executable=$(readlink -f "$(command -v mu2e)")
backing=${executable%%/build/*}
[[ $backing == */Offline/v13_39_00 ]] || exit 2
sources=(CrvFastSim/src/CrvRawIntegrityDumper_module.cc CrvFastSim/src/CrvRecordingConfigDumper_module.cc)
sha256sum "${sources[@]}" > "$OUT/SOURCE.sha256"
include=(-I"$MUSE_WORK_DIR" -I"$backing")
IFS=: read -ra paths <<< "$MUSE_VIEW_INC"
for path in "${paths[@]}"; do [[ -z $path ]] || include+=(-I"$path"); done
printf '%s\n' "${include[@]}" > "$OUT/includes.txt"
rc=0
for source in "${sources[@]}"; do
  name=$(basename "$source")
  result=0
  g++ -std=c++20 -fsyntax-only -Wall -Wextra -Wno-unused-parameter "${include[@]}" "$source" > "$OUT/$name.txt" 2>&1 || result=$?
  printf '%s syntax_exit=%s\n' "$source" "$result" | tee -a "$OUT/results.txt"
  [[ $result == 0 ]] || rc=1
done
sha256sum -c "$OUT/SOURCE.sha256" > "$OUT/source-check.txt" || rc=1
printf '%s\n' "$rc" > "$OUT/SYNTAX_EXIT.txt"
exit "$rc"
