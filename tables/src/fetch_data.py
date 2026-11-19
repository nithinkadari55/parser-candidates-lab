"""Download benchmark data into $TB_WORK/data/raw (never committed; cached by actions/cache)."""
import os, sys, zipfile, hashlib, shutil, urllib.request
from paths import RAW
from huggingface_hub import hf_hub_download

HF = {
    "ptn": ("docling-project/PubTabNet_OTSL", ["data/val-00000-of-00001-2d685be412ba1e61.parquet"]),
    "ftn": ("docling-project/FinTabNet_OTSL", ["data/test-00000-of-00002-7d6f82c6161b0377.parquet", "data/test-00001-of-00002-7104c7ad37a00357.parquet"]),
    "pt1m": ("docling-project/PubTables-1M_OTSL", ["data/test-00000-of-00032-0cf42e31c8819e12.parquet", "data/test-00001-of-00032-cfac8e72b328378a.parquet"]),
    "pt2": ("kensho/PubTables-v2", [f"PubTables-v2_Cropped-Tables_test_{p}.tar.gz" for p in ("tables", "words", "xml-annotations", "images")]),
    "scitsr": ("bevaya/SciTSR-cc-by-nc-sa", ["data/test-00000-of-00001.parquet"]),
    "scitsr_pd": ("bevaya/SciTSR-pd", ["data/test-00000-of-00001.parquet"]),
}
TABREC = "https://ndownloader.figshare.com/files/40332520"
ICDAR = "http://www.tamirhassan.com/html/files/icdar2013-competition-dataset-with-gt.zip"
OURS_COMMIT = "cfb8df247f555b61251f36925a6a6312c19a6513"
OURS = {"docling-rs-demotion-repro.pdf": "8d4bc24f7aca473c76023a3dd3bf396e", "2305.03393v1.pdf": "7426a095f957bb4b6def0a9af897c4c6", "2206.01062.pdf": "d1a7bef8653663d15ab83a8f32e02151"}


def get(url, dst):
    if os.path.exists(dst): return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "table-bench/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst + ".tmp", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    os.replace(dst + ".tmp", dst)


def main(which):
    if which in HF:
        rid, files = HF[which]
        for fn in files: hf_hub_download(rid, fn, repo_type="dataset", local_dir=os.path.join(RAW, which))
    elif which == "tabrec":
        get(TABREC, os.path.join(RAW, "tabrec", "tabrecset.zip"))
    elif which == "icdar":
        z = os.path.join(RAW, "icdar2013", "icdar2013.zip"); get(ICDAR, z)
        x = os.path.join(RAW, "icdar2013", "x")
        if not os.path.isdir(x): zipfile.ZipFile(z).extractall(x)
    elif which == "ours":
        for fn, md5 in OURS.items():
            dst = os.path.join(RAW, "ours", fn)
            get(f"https://raw.githubusercontent.com/docling-project/docling.rs/{OURS_COMMIT}/tests/data/pdf/sources/{fn}", dst)
            h = hashlib.md5(open(dst, "rb").read()).hexdigest()
            assert h == md5, (fn, h)
    print("fetched", which)


if __name__ == "__main__":
    for w in sys.argv[1:]: main(w)
