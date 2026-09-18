#!/usr/bin/env bash
# legacy/ is regression and reference material: the original scripts exactly as
# they were received, kept so results can be compared against them. Nothing in
# the rebuild may edit it.
#
# The check is a checksum manifest rather than a git diff so that it works on a
# shallow checkout and on any branch, with no base revision to compare against.
#
# If you are seeing this fail and the change to legacy/ really is intended,
# regenerate the manifest deliberately and say why in the commit message:
#
#   .github/scripts/check_legacy_untouched.sh --regenerate

set -euo pipefail

cd "$(dirname "$0")/../.."
MANIFEST=".github/legacy.sha256"

generate() {
  find legacy -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
}

if [ "${1:-}" = "--regenerate" ]; then
  generate > "${MANIFEST}"
  echo "Regenerated ${MANIFEST} ($(wc -l < "${MANIFEST}") files)."
  exit 0
fi

if [ ! -f "${MANIFEST}" ]; then
  echo "::error::${MANIFEST} is missing; cannot verify legacy/ is untouched." >&2
  exit 1
fi

if generate | diff -u "${MANIFEST}" - > /tmp/legacy-diff.txt; then
  echo "legacy/ matches the recorded manifest ($(wc -l < "${MANIFEST}") files)."
else
  echo "::error::legacy/ has changed. It is reference material and must stay byte-identical." >&2
  cat /tmp/legacy-diff.txt >&2
  exit 1
fi
