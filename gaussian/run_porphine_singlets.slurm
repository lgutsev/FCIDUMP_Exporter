#!/bin/bash
# The porphine tier, RHF singlets only, under Slurm on LONI. No --export needed,
# so the dispatch desk's submit_smokes.sh can submit it as it finds it:
#
#   cd gaussian
#   sbatch run_porphine_singlets.slurm
#
# Runs probe_fe_porphine_singlet and probe_ni_porphine_singlet through
# run_probes.sh with PORPHINE=singlets. The cheap set already ran (job 1072685),
# and the ROHF triplets are left out because extract refuses them (README.md,
# "M0 outcome"). --cpus-per-task matches the %nprocshared=8 in both inputs.
#
# Done means both .log files in this directory end in "Normal termination".
# The deliverable is probe_results_<host>_<date>.tar.gz, left in this directory.

#SBATCH --job-name=g16dump-porphine
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=g16dump-porphine-%j.out

# module is a shell function that trips -u; load first, tighten after.
set -eo pipefail
module load gaussian/g16-c01
set -u

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch, from the gaussian/ directory}"

if [ ! -f ./run_probes.sh ]; then
  echo "run_probes.sh is not here. Submit from the repository's gaussian/" >&2
  echo "directory: cd gaussian && sbatch run_porphine_singlets.slurm" >&2
  exit 1
fi

export GAUSS_SCRDIR="${TMPDIR:-/tmp}/g16dump_${SLURM_JOB_ID}"
mkdir -p "$GAUSS_SCRDIR"
trap 'rm -rf "$GAUSS_SCRDIR"' EXIT

echo "g16root:     ${g16root:-unset}"
echo "GAUSS_SCRDIR $GAUSS_SCRDIR"
echo

set +e
PORPHINE=singlets bash ./run_probes.sh
status=$?
set -e
echo "run_probes.sh exited ${status}"
exit 0
