#!/bin/bash
# The ROHF-route experiment under Slurm on LONI. No --export needed, so the
# dispatch desk's submit_smokes.sh can submit it as it finds it:
#
#   cd gaussian
#   sbatch run_rohf_route.slurm
#
# Five ROHF probes rerun with IOp(8/5=1) (do not pseudo-canonicalize ROHF
# orbitals) and/or IOp(8/40=2) (treat the ROHF window as RHF, transforming only
# the alpha orbitals), through run_probes.sh with PROBESET=rohf_route. Each is
# seconds long on one core. See README.md, "M0 outcome" and "The ROHF route".
#
# Done means all five .log files in this directory end in "Normal termination".
# A job Gaussian rejects (an IOp it does not accept) ends in "Error termination";
# that is a result too -- bring the archive back either way.
# The deliverable is probe_results_<host>_<date>.tar.gz, left in this directory.

#SBATCH --job-name=g16dump-rohf-route
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=g16dump-rohf-route-%j.out

# module is a shell function that trips -u; load first, tighten after.
set -eo pipefail
module load gaussian/g16-c01
set -u

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch, from the gaussian/ directory}"

if [ ! -f ./run_probes.sh ]; then
  echo "run_probes.sh is not here. Submit from the repository's gaussian/" >&2
  echo "directory: cd gaussian && sbatch run_rohf_route.slurm" >&2
  exit 1
fi

export GAUSS_SCRDIR="${TMPDIR:-/tmp}/g16dump_${SLURM_JOB_ID}"
mkdir -p "$GAUSS_SCRDIR"
trap 'rm -rf "$GAUSS_SCRDIR"' EXIT

echo "g16root:     ${g16root:-unset}"
echo "GAUSS_SCRDIR $GAUSS_SCRDIR"
echo

set +e
PROBESET=rohf_route bash ./run_probes.sh
status=$?
set -e
echo "run_probes.sh exited ${status}"
exit 0
