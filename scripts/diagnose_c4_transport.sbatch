#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J c4-diagnose
set -eo pipefail
module load apptainer
: "${C4_WORK:?existing C4 output area}"
: "${C4_CRV:?diagnostic code snapshot}"
: "${C4_GDML:?source-matching GDML}"
: "${C4_SIF:?project image}"
: "${C4_PARQUET:?parquet site-packages}"
: "${C4_OFFLINE:?Offline search path}"
: "${C4_LIBRARY:?Run 1A library directory}"
: "${C4_DEV:?saved library development output}"
: "${C4_CARD:?stock veto card}"
O="$C4_WORK/diagnosis_transport_${SLURM_JOB_ID:?}"
mkdir -p "$O"
exec_container=(apptainer exec --cleanenv -B "$(dirname "$C4_WORK")" --env
  "PYTHONPATH=$C4_CRV:$C4_WORK/code/core:$C4_PARQUET,PYTHONDONTWRITEBYTECODE=1,JAX_PLATFORMS=cpu,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,CRVFS_NO_FETCH=1,MU2E_SEARCH_PATH=$C4_OFFLINE" "$C4_SIF")
"${exec_container[@]}" python3 -m crvfastsim.veto.diagnose --selection "$C4_DEV/selection.parquet" \
  --fast "$C4_DEV/fast.parquet" --card "$C4_CARD" --outdir "$O/decisions"
"${exec_container[@]}" python3 -m crvfastsim.veto.transport_diagnose --selection "$C4_DEV/selection.parquet" \
  --decisions "$O/decisions/tracks.parquet" --gdml "$C4_GDML" --detector-origin -3904 0 10171 \
  --library-dir "$C4_LIBRARY" --seeds 20261105 20261106 20261107 --outdir "$O/steps"
