#!/bin/bash
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -J r3_source_identity
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:20:00
set -eo pipefail
if [[ ${1:-} == --inside ]]; then
  export TMPDIR="$(dirname "$OFFLINE_CACHE_PATH")/tmp"
  mkdir -p "$TMPDIR"
  if [[ ! -f inventory_reused_from.txt ]]; then
    while IFS= read -r input; do
      name=$(basename "$input" .art)
      config_dumper -P "$input" > "$name.embedded.fcl"
      product_sizes_dumper "$input" > "$name.products.txt" 2>&1
    done < inputs.txt
    for kind in dts gen; do
      input=$(grep "/$kind\..*_00000109.art$" inputs.txt)
      mu2e -c "$IDENTITY_CODE/inspect_products.fcl" -s "$input" -n 1 > "$kind.inventory.log" 2>&1
    done
  fi
  executable=$(readlink -f "$(command -v mu2e)")
  backing=${executable%/build/*}
  [[ $backing != "$executable" ]] || exit 2
  printf '%s\n' "$backing" > backing.txt
  sha256sum "$executable" > EXECUTABLE.sha256
  mkdir backing_sources
  while IFS= read -r source; do
    if [[ -f "$backing/$source" ]]; then
      cp "$backing/$source" "backing_sources/$(basename "$source")"
    else
      printf '%s\n' "$source" >> missing_backing_sources.txt
    fi
  done < "$IDENTITY_CODE/backing_sources.txt"
  sha256sum backing_sources/* > BACKING_SOURCES.sha256
  for binary in "$backing/build/$MUSE_STUB/Offline/lib/"*CompressDetStepMCs* "$backing/build/$MUSE_STUB/Offline/lib/"*DetectorStepFilter* "$backing/build/$MUSE_STUB/Offline/lib/"*FilterStatusG4* "$backing/build/$MUSE_STUB/Offline/lib/"*Mu2eG4*; do
    [[ ! -f "$binary" ]] || sha256sum "$binary"
  done > BACKING_BINARIES.sha256
  exit 0
fi
: "${IDENTITY_ROOT:?}" "${IDENTITY_CODE:?}" "${IDENTITY_RUNTIME:?}" "${IDENTITY_OWNER:?}"
: "${IDENTITY_RUNNER:?}" "${CVMFSEXEC_DIST:?}" "${OFFLINE_IMAGE:?}"
[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
module load apptainer
OUT="$IDENTITY_ROOT/source_identity_${SLURM_JOB_ID}"
mkdir "$OUT"
cp "$IDENTITY_CODE/SOURCE.sha256" "$OUT/CODE.sha256"
cd "$IDENTITY_CODE"
sha256sum -c "$OUT/CODE.sha256"
cd "$OUT"
for k in 101 103 105 107 109 111 113 115 117 119; do
  sequence=$(printf '001400_%08d' "$k")
  for kind in dts gen; do
    if [[ $k == 101 ]]; then
      input="$IDENTITY_ROOT/inputs/$kind.$sequence.art"
    else
      input="$IDENTITY_ROOT/inputs_training_odd/$kind.$IDENTITY_OWNER.CosmicCRYrun1aRef.v1.$sequence.art"
    fi
    printf '%s\n' "$input" >> inputs.txt
    sha256sum "$input" >> INPUTS.sha256
  done
done
if [[ -n ${IDENTITY_REUSE:-} ]]; then
  cmp INPUTS.sha256 "$IDENTITY_REUSE/INPUTS.sha256"
  printf '%s\n' "$IDENTITY_REUSE" > inventory_reused_from.txt
  sha256sum "$IDENTITY_REUSE/CODE.sha256" > REUSED_CODE.sha256
  printf '%s\n' 'historical_inventory_not_originally_hash_attested; current backing evidence is separate' > REUSE_SCOPE.txt
  while IFS= read -r input; do
    name=$(basename "$input" .art)
    for suffix in embedded.fcl products.txt; do
      [[ -s "$IDENTITY_REUSE/$name.$suffix" ]] || exit 2
      cp "$IDENTITY_REUSE/$name.$suffix" .
    done
  done < inputs.txt
  for kind in dts gen; do
    [[ -s "$IDENTITY_REUSE/$kind.inventory.log" ]] || exit 2
    cp "$IDENTITY_REUSE/$kind.inventory.log" .
  done
  sha256sum *.fcl *.products.txt *.inventory.log > REUSED_OUTPUTS.sha256
fi
sha256sum "$IDENTITY_RUNNER" > RUNNER.sha256
{
  printf 'export IDENTITY_CODE=%q\n' "$IDENTITY_CODE"
  printf 'bash %q --inside\n' "$IDENTITY_CODE/inspect_native_source_identity.sbatch"
} > run.sh
export OFFLINE_CHAIN="$OUT/run.sh" OFFLINE_LOG_PREFIX=identity_inventory
export OFFLINE_BINDS="$IDENTITY_ROOT:$IDENTITY_ROOT"
bash "$IDENTITY_RUNNER" "$IDENTITY_RUNTIME" "$IDENTITY_CODE/inspect_products.fcl"
sha256sum -c INPUTS.sha256
cd "$IDENTITY_CODE"
sha256sum -c "$OUT/CODE.sha256"
cmp SOURCE.sha256 "$OUT/CODE.sha256"
cd "$OUT"
[[ ! -f REUSED_OUTPUTS.sha256 ]] || sha256sum -c REUSED_OUTPUTS.sha256
sha256sum *.fcl *.products.txt *.inventory.log BACKING_SOURCES.sha256 BACKING_BINARIES.sha256 EXECUTABLE.sha256 > OUTPUTS.sha256
printf 'R3_SOURCE_IDENTITY_EXIT=0\n'
