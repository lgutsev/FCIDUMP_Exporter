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

# Deterministic on every platform, which is fiddlier than it looks.
#
# Binary mode is requested explicitly (-b). On Windows, sha256sum's *text* mode
# strips CR while reading, which changes the digest of every file under
# legacy/. Binary mode then prints "hash *path" whereas GNU coreutils on Linux
# defaults to "hash  path", so the marker is rewritten to two spaces to leave
# one format everywhere.
#
# The manifest is compared with its CR characters stripped for the same family
# of reasons: it is marked `-text` in .gitattributes, so it keeps whatever line
# endings it was committed with, on whichever platform that was.
generate() {
  find legacy -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum -b \
    | sed 's/ \*/  /'
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

if generate | diff -u <(tr -d '\r' < "${MANIFEST}") - > /tmp/legacy-diff.txt; then
  echo "legacy/ matches the recorded manifest ($(wc -l < "${MANIFEST}") files)."
else
  echo "::error::legacy/ has changed. It is reference material and must stay byte-identical." >&2
  cat /tmp/legacy-diff.txt >&2
  exit 1
fi
