#!/usr/bin/env bash
# Run the M0 probe jobs and collect everything needed to answer the open
# questions, in one command, on the cluster where Gaussian lives.
#
#   cd gaussian
#   bash run_probes.sh
#
# Each .mat is probed with g16dump's own numpy reader (python3 -m g16dump
# inspect, from the repository one level up). Nothing beyond numpy is needed.
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
  probe_ch2_rohf_ccpvdz      # MOKIT gate: pure d functions, ROHF
  probe_h2o_rks_631gs        # MOKIT gate: Cartesian d and SP shells, KS
)

# Tier 2: the real thing. 37 atoms, ~260 basis functions, minutes-to-hours
# rather than seconds, so it is opt-in. Run the set above first and only come
# here once ITran=5 has been confirmed on something cheap -- there is no sense
# spending a porphine SCF to discover the route was wrong.
#
#   PORPHINE=1 bash run_probes.sh
#
# PORPHINE=singlets runs the two RHF singlets and nothing else: the cheap set has
# already run (job 1072685), and the ROHF triplets are refused by extract until
# an ROHF route exists (README.md, "M0 outcome").
# PROBESET=rohf_route runs only the ROHF-route experiment: the ROHF probes again
# with IOp(8/5) and IOp(8/40), to see whether Gaussian can be made to transform
# over the ROHF orbitals instead of per-spin semicanonical ones (README.md,
# "M0 outcome"). Seconds, one core.
if [ "${PROBESET:-}" = rohf_route ]; then
  JOBS=(probe_ch2_rohf_iop_both probe_ch2_rohf_iop_nopc probe_ch2_rohf_iop_rhfwin
        probe_ch2_rohf_ccpvdz_iop_both probe_nh_rohf_iop_both)
fi

case "${PORPHINE:-}" in
  "") ;;
  singlets) JOBS=(probe_fe_porphine_singlet probe_ni_porphine_singlet) ;;
  # The ROHF triplets, for extract --integrals pyscf (their Gaussian integrals
  # are semicanonical and unused; the orbitals, overlap and .fch are what count).
  triplets) JOBS=(probe_fe_porphine_triplet probe_ni_porphine_triplet) ;;
  *) JOBS+=(probe_fe_porphine_singlet probe_fe_porphine_triplet
            probe_ni_porphine_singlet probe_ni_porphine_triplet) ;;
esac

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
    # The .fch is small and is half of every rebuilt-Fock test; ship it.
    cp -f "${name}.fch" "${OUT}/" 2>/dev/null
  fi

  if [ -f "${name}.mat" ]; then
    ls -l "${name}.mat" | awk '{print "  .mat is " $5 " bytes"}'
    PYTHONPATH="..:${PYTHONPATH:-}" "${PY}" -m g16dump inspect "${name}.mat" \
      --json "${OUT}/${name}.probe.json" > "${OUT}/${name}.probe.txt" 2>&1
    case $? in
      0) echo "  probe ok" ;;
      1) if [ -f "${OUT}/${name}.probe.json" ]; then  # read; extract cannot use it as is
           echo "  probe: read, but extract cannot use it as is -- see ${OUT}/${name}.probe.txt"
         else echo "  probe FAILED -- see ${OUT}/${name}.probe.txt"; fi ;;
      *) echo "  probe FAILED -- see ${OUT}/${name}.probe.txt" ;;
    esac
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
  echo "numpy:    $(${PY} -c 'import numpy; print(numpy.__version__)' 2>&1)"
} > "${OUT}/environment.txt"

# The .mat files are the ground truth, so ship them. Most of a .mat is record
# padding and compresses about 15x in the archive, so only a huge one stays out.
for name in "${JOBS[@]}"; do
  if [ -f "${name}.mat" ]; then
    size=$(stat -c%s "${name}.mat" 2>/dev/null || echo 0)
    if [ "${size}" -lt 500000000 ]; then
      cp -f "${name}.mat" "${OUT}/"
    else
      # The probe JSON is the deliverable; the raw file is a convenience.
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
