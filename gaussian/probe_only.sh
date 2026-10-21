#!/bin/bash
# Probe the .mat files an earlier run_probes.slurm left in this folder, without
# running Gaussian again. Seconds, so the login node is fine:
#
#   cd gaussian && bash probe_only.sh
#
# Uses g16dump's own numpy reader (python3 -m g16dump inspect, from the
# repository one level up); gauopen is not needed. Writes
# probe_only_<date>/<name>.probe.json and .probe.txt, then probe_only_<date>.tar.gz.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY="${PYTHON:-python3}"
OUT="probe_only_$(date +%Y%m%d)"
mkdir -p "$OUT"
echo "python: $(${PY} --version 2>&1), numpy $(${PY} -c 'import numpy; print(numpy.__version__)' 2>&1)"
n=0
for mat in *.mat; do
  [ -f "$mat" ] || { echo "no .mat files here: run run_probes.slurm first" >&2; exit 2; }
  name=${mat%.mat}
  PYTHONPATH="..:${PYTHONPATH:-}" "${PY}" -m g16dump inspect "$mat" \
    --json "$OUT/$name.probe.json" > "$OUT/$name.probe.txt" 2>&1
  case $? in
    0) echo "  $name: probe ok"; n=$((n + 1)) ;;
    1) if [ -f "$OUT/$name.probe.json" ]; then  # read it; extract cannot use it as is
         echo "  $name: read, but extract cannot use it as is, see $OUT/$name.probe.txt"; n=$((n + 1))
       else echo "  $name: probe FAILED, see $OUT/$name.probe.txt"; fi ;;
    *) echo "  $name: probe FAILED, see $OUT/$name.probe.txt" ;;
  esac
done
{ echo "python: $(${PY} --version 2>&1)"; echo "host: $(hostname)"; } > "$OUT/environment.txt"
tar czf "$OUT.tar.gz" "$OUT"
echo "$n files read; bring back $(pwd)/$OUT.tar.gz"
