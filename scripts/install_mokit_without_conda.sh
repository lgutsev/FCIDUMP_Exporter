#!/bin/bash
# Install MOKIT's prebuilt conda package into an existing pip Python, without
# conda. Used for the validation tests (tests/test_extract.py) in environments
# that have pip but no conda -- e.g. a cloud container.
#
# On a cluster with conda, prefer the normal route instead:
#     conda install mokit -c mokit -c conda-forge
#
# What this does, and why each step is needed:
#   1. downloads the mokit .conda archive matching this Python (cpXY),
#   2. unpacks it into $PREFIX (a .conda file is a zip holding a zstd tarball),
#   3. installs libgfortran5 + OpenBLAS from apt -- the Fortran extensions link
#      against them; they need only GFORTRAN_8/GFORTRAN_10 symbols, which
#      Ubuntu 24.04's libgfortran provides even though the conda metadata asks
#      for a newer one,
#   4. puts the package on sys.path with a .pth file, and
#   5. links MOKIT's executables onto PATH -- load_mol_from_fch shells out to
#      `bas_fch2py`, and fails with "not found" without it.
#
# Usage: scripts/install_mokit_without_conda.sh [version] [prefix]
set -euo pipefail

VERSION="${1:-1.2.9rc1}"
PREFIX="${2:-/opt/mokit_env}"
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')
CHANNEL="https://conda.anaconda.org/mokit/linux-64"

echo "looking for mokit ${VERSION} built for py${PYVER}"
FILE=$(curl -sS -m 120 "${CHANNEL}/repodata.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
pk = {**d.get('packages', {}), **d.get('packages.conda', {})}
hits = [k for k, v in pk.items() if v['name'] == 'mokit' and v['version'] == '${VERSION}'
        and 'py${PYVER}' in v['build']]
print(sorted(hits)[-1] if hits else '')
")
if [ -z "$FILE" ]; then
    echo "no mokit ${VERSION} build for py${PYVER} on the mokit channel" >&2
    exit 1
fi

mkdir -p "$PREFIX/pkgs"
curl -sS -m 600 -L -o "$PREFIX/pkgs/$FILE" "${CHANNEL}/${FILE}"
python3 -m pip install -q zstandard
python3 - "$PREFIX" "$PREFIX/pkgs/$FILE" <<'PY'
import io, sys, tarfile, zipfile, zstandard
prefix, archive = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(archive)
for name in z.namelist():
    if name.startswith("pkg-") and name.endswith(".tar.zst"):
        stream = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(z.read(name)))
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            tar.extractall(f"{prefix}/prefix")
PY

apt-get install -y -q libgfortran5 libopenblas0-openmp >/dev/null \
    || { apt-get update -q >/dev/null && apt-get install -y -q libgfortran5 libopenblas0-openmp >/dev/null; }

SITE="$PREFIX/prefix/lib/python${PYVER:0:1}.${PYVER:1}/site-packages"
TARGET=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "$SITE" > "$TARGET/mokit_opt.pth"

for exe in "$PREFIX"/prefix/bin/*; do
    [ -x "$exe" ] && ln -sf "$exe" /usr/local/bin/
done

python3 -c "
from mokit.lib.gaussian import load_mol_from_fch, gen_fcidump
import mokit; print('mokit', mokit.__file__)"
command -v bas_fch2py >/dev/null && echo "bas_fch2py on PATH"
