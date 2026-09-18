#!/usr/bin/env bash
# Run pytest with a marker expression, treating "nothing was collected" as a
# pass rather than a failure.
#
# The package is being built milestone by milestone, so a job can legitimately
# find no test matching its marker expression. pytest reports that as exit code
# 5, which would otherwise turn an empty-but-healthy suite into a red build.
# Every other non-zero exit code is a real failure and is passed through.
#
# Usage: run_tests.sh "<marker expression>" [extra pytest args...]

set -uo pipefail

MARKERS="${1:?marker expression required}"
shift

echo "pytest -m '${MARKERS}'"
python -m pytest -m "${MARKERS}" --color=yes -ra "$@"
rc=$?

if [ "${rc}" -eq 5 ]; then
  echo "::notice::No test matched -m '${MARKERS}'. Treating an empty selection as a pass."
  exit 0
fi

exit "${rc}"
