#!/usr/bin/env bash
# Run the M0 probe jobs and collect everything needed to answer the open
# questions, in one command, on the cluster where Gaussian and gauopen live.
#
#   cd gaussian
#   GAUOPEN_PATH=/path/to/gauopen ./run_probes.sh
#
# Produces probe_results_<host>_<date>.tar.gz next to this script. That archive
# is the whole deliverable -- send it back and nothing else is needed.
#
# Nothing here is hardcoded to a machine. Every path is derived from the script
# location or from the environment, and the script keeps going after a failed
# job so that one broken route does not cost you the whole set.

set -uo pipefail

cd "$(dirname "$0")"
OUT="probe_results_$(hostname -s 2>/dev/null || echo host)_$(date +%Y%m%d)"
rm -rf "${OUT}" && mkdir -p "${OUT}"

# Ordered cheapest-first and most-informative-first, so an interrupted run still
# leaves the jobs that matter most. See README.md for what each one answers.
JOBS=(
  probe_h2_smoke             # FIRST: 2 basis functions. Is ITran 5? Is there a
                             #   Fock record at all? If this one fails nothing
                             #   below is worth reading.
  probe_ch2_uhf              # SECOND: the only job that writes BETA blocks, so
                             #   it is the reference for what a beta record
                             #   looks like when one exists.
  probe_h2o_rhf              # does the route work, closed shell
  probe_h2o_nowindow         # control: proves Window= actually restricts
  probe_ch2_rohf             # the Fock question, open shell
  probe_nh_rohf              # a second open shell, degenerate pi pair
  probe_h2o_frozen_virtual   # the only job with frozen virtuals
  probe_h2o_rks              # is a Kohn-Sham matrix written as a Fock matrix
)

# Tier 2: the real thing. 37 atoms, ~260 basis functions, minutes-to-hours
# rather than seconds, so it is opt-in. Run the set above first and only come
# here once ITran=5 has been confirmed on something cheap -- there is no sense
# spending a porphine SCF to discover the route was wrong.
#
#   PORPHINE=1 ./run_probes.sh
#
if [ -n "${PORPHINE:-}" ]; then
  JOBS+=(probe_ni_porphine_singlet probe_ni_porphine_triplet)
fi

PY="${PYTHON:-python3}"
echo "gaussian probes -> ${OUT}/"
echo

for name in "${JOBS[@]}"; do
  if [ ! -f "${name}.gjf" ]; then
    echo "[skip] ${name}.gjf not found"
    continue
  fi

  echo "=== ${name} ==="
  if g16 "${name}.gjf"; then
    echo "  g16 ok"
  else
    echo "  g16 FAILED (exit $?) -- keeping the log and moving on"
  fi
  cp -f "${name}.log" "${OUT}/" 2>/dev/null

  # The .fch is needed by the rebuilt-Fock path, which is the documented way out
  # if Gaussian turns out to store a Roothaan operator rather than F^a/F^b.
  if [ -f "${name}.chk" ]; then
    formchk "${name}.chk" "${name}.fch" > /dev/null 2>&1 \
      && echo "  formchk ok" || echo "  formchk FAILED"
  fi

  if [ -f "${name}.mat" ]; then
    ls -l "${name}.mat" | awk '{print "  .mat is " $5 " bytes"}'
    if [ -n "${GAUOPEN_PATH:-}" ]; then
      PYTHONPATH="${GAUOPEN_PATH}:${PYTHONPATH:-}" \
        "${PY}" ../scripts/inspect_mat.py "${name}.mat" \
          --json "${OUT}/${name}.probe.json" > "${OUT}/${name}.probe.txt" 2>&1 \
        && echo "  probe ok" \
        || echo "  probe FAILED -- see ${OUT}/${name}.probe.txt"
    else
      echo "  probe SKIPPED (set GAUOPEN_PATH to run inspect_mat.py here)"
    fi
  else
    echo "  no .mat was produced"
  fi
  echo
done

# Environment, so a surprise in the output can be traced to a version.
{
  echo "date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:     $(hostname 2>/dev/null)"
  echo "g16:      $(command -v g16 || echo 'not found')"
  echo "g16root:  ${g16root:-unset}"
  echo "python:   $(${PY} --version 2>&1)"
  echo "gauopen:  ${GAUOPEN_PATH:-unset}"
} > "${OUT}/environment.txt"

# The .mat files themselves are small for these systems and are the ground
# truth, so include them when they are not large.
for name in "${JOBS[@]}"; do
  if [ -f "${name}.mat" ]; then
    size=$(stat -c%s "${name}.mat" 2>/dev/null || echo 0)
    if [ "${size}" -lt 20000000 ]; then
      cp -f "${name}.mat" "${OUT}/"
    else
      # A porphine .mat is expected to land here. The probe JSON is the
      # deliverable; the raw file is only a convenience for the small jobs.
      echo "${name}.mat omitted from the archive, ${size} bytes"         >> "${OUT}/environment.txt"
    fi
  fi
done

tar czf "${OUT}.tar.gz" "${OUT}"
echo "=============================================================="
echo "Done. Send back: ${OUT}.tar.gz"
echo "  $(find "${OUT}" -type f | wc -l) files, $(du -sh "${OUT}.tar.gz" | cut -f1)"
echo
echo "If nothing else survives, the two files that matter most are"
echo "  ${OUT}/probe_h2o_rhf.probe.json  and  ${OUT}/probe_ch2_rohf.probe.json"
echo "=============================================================="
