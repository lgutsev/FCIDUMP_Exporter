#!/bin/bash
# Run the M0 probe set under Slurm, on LONI.
#
#   cd gaussian
#   sbatch run_probes.slurm
#
# Submit from THIS directory: Slurm copies the batch script to a spool area, so
# $0 is not this file and the script locates everything through
# $SLURM_SUBMIT_DIR instead. The work is done by run_probes.sh, which is the
# same script you would run interactively; this file only supplies the module
# and the scratch directory. The probe step reads each .mat with g16dump's own
# numpy reader (python3 -m g16dump inspect), so gauopen is not needed -- it is
# not installed on QB4 anyway (dispatch desk search, 2026-10-07).
#
# The cheap set is 10 jobs, every one of them seconds long -- the probe .gjf
# files ask for %nprocshared=1 and at most %mem=2GB, so one core is the right
# request; the 2 hours below is deliberate headroom (the set runs in minutes).
#
# For the porphine tier, override on the command line (sbatch flags beat the
# #SBATCH lines below) and match --cpus-per-task to the %nprocshared=8 in those
# inputs:
#
#   sbatch --export=ALL,PORPHINE=1 --cpus-per-task=8 \
#          --time=12:00:00 run_probes.slurm
#
# QB4's sbatch filter rejects --mem: memory comes with the cores, so there is no
# #SBATCH --mem line and the command above passes none.
#
# The deliverable is probe_results_<host>_<date>.tar.gz, left in this directory.

#SBATCH --job-name=g16dump-probes
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=g16dump-probes-%j.out

# module is a shell function that trips -u; load first, tighten after. This is
# the ordering the working Gaussian jobs on this cluster already use.
set -eo pipefail
module load gaussian/g16-c01
set -u

cd "${SLURM_SUBMIT_DIR:?submit this with sbatch, from the gaussian/ directory}"

# -f, not -x, and run it with bash: files committed from Windows lose the
# executable bit, and that stopped every submission before job 1072685.
if [ ! -f ./run_probes.sh ]; then
  echo "run_probes.sh is not here. Submit from the repository's gaussian/" >&2
  echo "directory: cd gaussian && sbatch run_probes.slurm" >&2
  exit 1
fi

# Node-local scratch. Gaussian's .rwf is the largest thing these jobs write and
# it has no reason to touch the shared filesystem.
export GAUSS_SCRDIR="${TMPDIR:-/tmp}/g16dump_${SLURM_JOB_ID}"
mkdir -p "$GAUSS_SCRDIR"
trap 'rm -rf "$GAUSS_SCRDIR"' EXIT

echo "g16root:     ${g16root:-unset}"
echo "GAUSS_SCRDIR $GAUSS_SCRDIR"
echo

# run_probes.sh keeps going after a failed job on purpose, so a non-zero exit
# from one route must not kill the array of remaining work.
set +e
bash ./run_probes.sh
status=$?
set -e
echo "run_probes.sh exited ${status}"
exit 0
