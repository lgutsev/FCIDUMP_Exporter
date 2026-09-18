#!/bin/bash
#PBS -N GJOB
#PBS -A ALLOCATION
#PBS -q single
#PBS -l nodes=1:ppn=8
#PBS -l walltime=4:00:00
#PBS -j oe

# Gaussian 16 + formchk job template.
# Usage: set NAME below (or pass it in), then qsub this script.
# Every path is derived from $PBS_O_WORKDIR -- nothing is hardcoded.

cd "$PBS_O_WORKDIR" || exit 1

NAME="${NAME:-CHANGEME}"

module load gaussian

export GAUSS_SCRDIR="${GAUSS_SCRDIR:-/var/scratch/$USER}"
mkdir -p "$GAUSS_SCRDIR"

g16 "${NAME}.gjf" || exit 1
formchk "${NAME}.chk" "${NAME}.fch" || exit 1

# Optional: probe the matrix-element file right here, where gauopen lives.
if [ -n "$GAUOPEN_PATH" ]; then
    PYTHONPATH="$GAUOPEN_PATH:$PYTHONPATH" \
        python3 scripts/inspect_mat.py "${NAME}.mat" --json "${NAME}.probe.json"
fi
