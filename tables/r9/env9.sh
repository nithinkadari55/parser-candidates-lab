#!/usr/bin/env bash
# Round 9 model environments. usage: env9.sh GROUP  -> prints the python to use on the last line of $GITHUB_ENV (R9_PY)
# Groups: ort (the scoring env itself), mineru, pptable, paddle2 (TableMaster), mmocr (MuTabNet), lore, trace.
set -euo pipefail
G="$1"; CODE="${R9_CODE:-$HOME/r9code}"; mkdir -p "$CODE"
mkvenv() { "$1" -m venv /tmp/menv; /tmp/menv/bin/pip install -q --upgrade pip wheel "setuptools<70"; echo "R9_PY=/tmp/menv/bin/python" >> "$GITHUB_ENV"; }
P=/tmp/menv/bin/pip
case "$G" in
  ort)
    echo "R9_PY=$(command -v python)" >> "$GITHUB_ENV" ;;
  mineru)
    mkvenv python3.11
    $P install -q mineru==4.2.3 ;;
  pptable)
    mkvenv python3.11
    $P install -q paddlepaddle==3.2.2 paddleocr==3.7.0 lxml ;;
  paddle2)
    mkvenv python3.10
    $P install -q "numpy<2" paddlepaddle==2.6.2 paddleocr==2.7.3 ;;
  mmocr)
    mkvenv python3.10
    $P install -q "numpy<2" torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
    $P install -q mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
    $P install -q "mmdet<3" "mmocr<1" apted distance lxml more-itertools "positional-encodings[pytorch]" rotary-embedding-torch tqdm "urllib3<2" yapf==0.40.1 \
      "git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer"
    git clone -q https://github.com/JG1VPP/MuTabNet.git "$CODE/mutabnet" && git -C "$CODE/mutabnet" checkout -q c45a54c && git -C "$CODE/mutabnet" log -1 --format=%H ;;
  lore)
    mkvenv python3.10
    $P install -q "numpy<2" torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
    $P install -q opencv-python-headless==4.10.0.84 pycocotools shapely scipy progress matplotlib "Pillow<11" tqdm numba
    git clone -q --filter=blob:none --sparse https://github.com/AlibabaResearch/AdvancedLiterateMachinery.git "$CODE/alm"
    git -C "$CODE/alm" sparse-checkout set DocumentUnderstanding/LORE-TSR && git -C "$CODE/alm" log -1 --format=%H
    ln -sfn "$CODE/alm/DocumentUnderstanding/LORE-TSR" "$CODE/lore"
    cp tables/r9/models9/dcn_shim.py "$CODE/lore/src/lib/models/networks/DCNv2/dcn_v2.py" ;;
  trace)
    mkvenv python3.10
    $P install -q "numpy<2" torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
    $P install -q opencv-python-headless==4.10.0.84 scipy scikit-image shapely Polygon3 pretrainedmodels PyPDF2 pdf2image XlsxWriter tensorboardX "Pillow<11"
    git clone -q https://github.com/naver-ai/trace.git "$CODE/trace" && git -C "$CODE/trace" log -1 --format=%H ;;
  *) echo "unknown group $G"; exit 1 ;;
esac
# every model environment also needs the small helpers cand9.py infer imports
if [ "$G" != "ort" ]; then /tmp/menv/bin/pip install -q pillow pyyaml; fi
