#!/usr/bin/env bash
set -euo pipefail

IM_VERSION="${IM_VERSION:-7.1.1-47}"
IM_PREFIX="${IM_PREFIX:-/opt/im7}"
ODB_SHA="${ODB_SHA:-7ddb8df411ad9bd2fff139acafe0ed16aa1694fe}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDM_UPSTREAM="${CDM_UPSTREAM:-$HERE/_upstream}"
PY="${PYTHON:-python3}"

log() { printf '\n== %s\n' "$*"; }
die() { printf 'SETUP FAIL: %s\n' "$*" >&2; exit 1; }

log "apt: TeX Live subset, Ghostscript, ImageMagick build/runtime libraries"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -q
sudo apt-get install -y -q --no-install-recommends \
  texlive-latex-base texlive-latex-recommended texlive-latex-extra ghostscript \
  build-essential pkg-config zlib1g-dev libpng-dev libjpeg-dev libtiff-dev libwebp-dev \
  libfreetype-dev libfontconfig-dev liblcms2-dev libxml2-dev

if [ -x "$IM_PREFIX/bin/magick" ] && "$IM_PREFIX/bin/magick" -version | head -1 | grep -q "ImageMagick $IM_VERSION "; then
  log "ImageMagick $IM_VERSION found in $IM_PREFIX (cache hit), skipping build"
else
  [ "${REQUIRE_IM_CACHE:-0}" = 1 ] && die "REQUIRE_IM_CACHE=1 but no cached ImageMagick $IM_VERSION in $IM_PREFIX"
  log "building ImageMagick $IM_VERSION into $IM_PREFIX"
  sudo mkdir -p "$IM_PREFIX"
  sudo chown "$(id -u):$(id -g)" "$IM_PREFIX"
  src="$(mktemp -d)"
  curl -sSfL "https://github.com/ImageMagick/ImageMagick/archive/refs/tags/${IM_VERSION}.tar.gz" -o "$src/im.tar.gz"
  sha256sum "$src/im.tar.gz"
  tar -xzf "$src/im.tar.gz" -C "$src"
  (
    cd "$src/ImageMagick-${IM_VERSION}"
    ./configure --prefix="$IM_PREFIX" --enable-shared --disable-static --without-modules \
      --disable-docs --without-perl --without-magick-plus-plus --without-x \
      LDFLAGS="-Wl,-rpath,$IM_PREFIX/lib" >/dev/null
    make -j"$(nproc)" >/dev/null
    make install >/dev/null
  )
  rm -rf "$src"
fi

log "linking magick into /usr/local/bin"
sudo ln -sf "$IM_PREFIX/bin/magick" /usr/local/bin/magick
echo "$IM_PREFIX/lib" | sudo tee /etc/ld.so.conf.d/im7.conf >/dev/null
sudo ldconfig
if ldd "$IM_PREFIX/bin/magick" | grep -q "not found"; then
  ldd "$IM_PREFIX/bin/magick"
  die "magick has unresolved shared libraries"
fi
magick -version | head -1 | grep -q "ImageMagick $IM_VERSION " || die "magick is not $IM_VERSION: $(magick -version | head -1)"
magick -version | head -4

log "policy.xml: allow PDF read|write"
POLICY="$(magick -list policy | sed -n 's/^ *Path: \(.*policy\.xml\)$/\1/p' | grep -v '\[built-in\]' | head -1 || true)"
if [ -z "$POLICY" ]; then
  CFG="$(magick -list configure | sed -n 's/^CONFIGURE_PATH *//p' | head -1)"
  POLICY="${CFG%/}/policy.xml"
fi
[ -f "$POLICY" ] || die "policy.xml not found (looked at '$POLICY')"
echo "policy file: $POLICY"
sed -i -E 's#<policy domain="coder" rights="none" pattern="PDF" */>#<policy domain="coder" rights="read|write" pattern="PDF" />#' "$POLICY"
if ! grep -q 'rights="read|write" pattern="PDF"' "$POLICY"; then
  sed -i 's#</policymap>#  <policy domain="coder" rights="read|write" pattern="PDF" />\n</policymap>#' "$POLICY"
fi
grep -n 'pattern="PDF"' "$POLICY"
magick -list policy
magick -list format | grep -E '^ *PDF ' || die "magick has no PDF format"

log "TeX Live / xcolor (\\mathcolor needs xcolor >= 2.14)"
pdflatex --version | head -1
XC="$(kpsewhich xcolor.sty)" || die "xcolor.sty not found"
grep -m1 -A1 'ProvidesPackage{xcolor}' "$XC" || true
gs --version

log "OmniDocBench CDM sources @ $ODB_SHA"
rm -rf "$CDM_UPSTREAM"
mkdir -p "$CDM_UPSTREAM"
git -C "$CDM_UPSTREAM" init -q
git -C "$CDM_UPSTREAM" remote add origin https://github.com/opendatalab/OmniDocBench.git
git -C "$CDM_UPSTREAM" config core.sparseCheckout true
mkdir -p "$CDM_UPSTREAM/.git/info"
printf '%s\n' /src/metrics/cdm/ /LICENSE > "$CDM_UPSTREAM/.git/info/sparse-checkout"
git -C "$CDM_UPSTREAM" fetch -q --depth 1 --filter=blob:none origin "$ODB_SHA"
git -C "$CDM_UPSTREAM" checkout -q FETCH_HEAD
[ "$(git -C "$CDM_UPSTREAM" rev-parse HEAD)" = "$ODB_SHA" ] || die "upstream sha mismatch"
[ -f "$CDM_UPSTREAM/src/metrics/cdm/modules/latex2bbox_color.py" ] || die "upstream CDM files missing"

log "python requirements"
"$PY" -m pip install -q -r "$HERE/requirements.txt"

log "self-check: pdflatex with \\mathcolor -> PDF -> magick PNG"
chk="$(mktemp -d)"
cat > "$chk/t.tex" <<'TEX'
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{xcolor}
\pagestyle{empty}
\begin{document}
\begin{displaymath}
\mathcolor[RGB]{0,0,255}{x}^{\mathcolor[RGB]{255,0,0}{2}}
\end{displaymath}
\end{document}
TEX
(cd "$chk" && pdflatex -interaction=nonstopmode t.tex >/dev/null) || { tail -30 "$chk/t.log"; die "pdflatex self-check failed"; }
magick -density 200 -quality 100 "$chk/t.pdf[0]" "$chk/t.png" || die "magick PDF->PNG failed"
magick identify "$chk/t.png"
"$PY" - "$chk/t.png" <<'PYCHK'
import sys
import numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB")).reshape(-1, 3)
blue = int(((a[:, 0] == 0) & (a[:, 1] == 0) & (a[:, 2] == 255)).sum())
red = int(((a[:, 0] == 255) & (a[:, 1] == 0) & (a[:, 2] == 0)).sum())
white = int((a == 255).all(axis=1).sum())
print(f"pixels={len(a)} white={white} exact_blue={blue} exact_red={red}")
if blue == 0 or red == 0 or white < len(a) // 2:
    sys.exit("colour self-check failed: exact token colours or white background missing")
PYCHK
rm -rf "$chk"
log "CDM setup OK"
