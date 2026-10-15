#!/bin/bash
# One Gaussian job, plus formchk, plus the matrix-element probe. Slurm, LONI.
#
#   cd gaussian
#   sbatch --export=ALL,NAME=probe_ch2_rohf job_template.slurm
#
# Use this when you want a single job rather than the whole set; run_probes.slurm
# is the one to submit for the M0 deliverable. Submit from the directory holding
# the .gjf: Slurm copies the batch script, so $0 is not this file.
#
# Resources below suit the cheap probes (%nprocshared=1, %mem<=2GB). For the
# porphine inputs pass --cpus-per-task=8 --time=12:00:00, matching the
# %nprocshared in the .gjf -- Gaussian will not notice a mismatch, it will just
# oversubscribe the allocation. No --mem: QB4's sbatch filter rejects it.

#SBATCH --job-name=g16dump-job
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=g16dump-%x-%j.out

set -eo pipefail
module load gaussian/g16-c01
set -u

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch}"

NAME="${NAME:?pass the job basename, e.g. --export=ALL,NAME=probe_ch2_rohf}"
if [ ! -f "${NAME}.gjf" ]; then
  echo "${NAME}.gjf is not in $(pwd)" >&2
  exit 1
fi

export GAUSS_SCRDIR="${TMPDIR:-/tmp}/g16dump_${SLURM_JOB_ID}"
mkdir -p "$GAUSS_SCRDIR"
trap 'rm -rf "$GAUSS_SCRDIR"' EXIT

g16 "${NAME}.gjf"
formchk "${NAME}.chk" "${NAME}.fch"

# The probe: g16dump's own numpy reader, from the repository this directory is
# in; no gauopen. Exit 1 means it read the file and found something extract
# cannot use, and the report says what.
PYTHONPATH="..:${PYTHONPATH:-}" python3 -m g16dump inspect "${NAME}.mat" \
  --json "${NAME}.probe.json" || echo "probe exited $?; see the report above"
