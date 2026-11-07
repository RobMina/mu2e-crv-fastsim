#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -J c4-nonmuon
set -eo pipefail
module load apptainer
: "${C4_WORK:?existing C4 output area}"
: "${C4_CRV:?actual retained runtime snapshot}"
: "${C4_GDML:?source-matching GDML}"
: "${C4_SIF:?project image}"
: "${C4_PARQUET:?parquet site-packages}"
: "${C4_OFFLINE:?Offline search path}"
: "${C4_LIBRARY:?Run 1A library directory}"
: "${C4_DEV:?saved retained development output}"
O="$C4_WORK/diagnosis_nonmuon_${SLURM_JOB_ID:?}"
mkdir -p "$O"
apptainer exec --cleanenv -B "$(dirname "$C4_WORK")" --env \
  "PYTHONPATH=$C4_CRV:$C4_WORK/code/core:$C4_PARQUET,PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,CRVFS_NO_FETCH=1,MU2E_SEARCH_PATH=$C4_OFFLINE" "$C4_SIF" \
  python3 -m crvfastsim.veto.nonmuon --generators "$C4_DEV/generators.parquet" \
  --library "$C4_LIBRARY/library.root" --replay-selection "$C4_DEV/selection.parquet" \
  --gdml "$C4_GDML" --seed 20261105 --outdir "$O"
