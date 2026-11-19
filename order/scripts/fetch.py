import hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = os.path.dirname(HERE)
ODB = os.environ.get("ODB_DIR", os.path.join(os.path.expanduser("~"), "odb"))
PPDIR = os.environ.get("PP_DIR", os.path.join(os.path.expanduser("~"), "ppdlv3"))
PDFDIR = os.environ.get("PDF_DIR", os.path.join(os.path.expanduser("~"), "pdfs"))
PP_REPO, PP_REV = "PaddlePaddle/PP-DocLayoutV3_onnx", "46bbdf188bb0a772c08aed74882ce7e51a8f1ea6"
PP_SHA = "45bf71750b00739a41fc209f132eb104a4d6b5bb29483c9078164d8b87cf28ba"
DOCLING_REV = "5e600e8d37a11f70b62f795ea1e3ce3a4052e551"
DOCLINGRS_REV = "cfb8df247f555b61251f36925a6a6312c19a6513"
PDFS = ["table_mislabeled_as_picture.pdf", "elsevier-00.pdf", "2206.01062.pdf", "amt_handbook_sample.pdf",
        "2203.01017v2.pdf", "2305.03393v1.pdf", "redp5110_sampled.pdf"]


def odb():
    from huggingface_hub import hf_hub_download
    os.makedirs(ODB, exist_ok=True)
    hf_hub_download("opendatalab/OmniDocBench", "OmniDocBench.json", repo_type="dataset", local_dir=ODB)
    d = json.load(open(os.path.join(ODB, "OmniDocBench.json"), encoding="utf-8"))
    en = [p["page_info"]["image_path"] for p in d if p["page_info"]["page_attribute"].get("language") == "english"]

    def get(n):
        if os.path.exists(os.path.join(ODB, "images", n)):
            return 0
        for _ in range(4):
            try:
                hf_hub_download("opendatalab/OmniDocBench", "images/" + n, repo_type="dataset", local_dir=ODB)
                return 0
            except Exception as e:
                err = e
        print("FAIL", n, err, flush=True)
        return 1
    with ThreadPoolExecutor(8) as ex:
        bad = sum(ex.map(get, en))
    print("odb english images", len(en), "failed", bad)
    if bad:
        sys.exit(1)


def pp():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(PP_REPO, "inference.onnx", revision=PP_REV, local_dir=PPDIR)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print("ppdlv3", p, h)
    if h != PP_SHA:
        sys.exit("sha mismatch")


def vendor():
    os.makedirs(os.path.join(ORDER, "vendor"), exist_ok=True)
    url = f"https://raw.githubusercontent.com/docling-project/docling/{DOCLING_REV}/docling/models/postprocessing/reading_order_rb.py"
    urllib.request.urlretrieve(url, os.path.join(ORDER, "vendor", "docling_ro.py"))
    print("vendored", url)


def pdfs():
    os.makedirs(PDFDIR, exist_ok=True)
    for f in PDFS:
        dst = os.path.join(PDFDIR, f)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/docling-project/docling.rs/{DOCLINGRS_REV}/tests/data/pdf/sources/{f}", dst)
    print("pdfs", os.listdir(PDFDIR))


DRD = os.environ.get("DRD_DIR", os.path.join(os.path.expanduser("~"), "drdoc"))
LMODELS = os.environ.get("LMODELS_DIR", os.path.join(os.path.expanduser("~"), "lmodels"))
DRD_REV = "7a2bc3882dff68e883fb55d10d4df22865ce2b07"
EGRET = ("ningpp/docling-layout-egret-medium-ONNX", "06b27d85c5a004eb024ca82f78e094fb7bf47e47", "model.onnx")
PPL = ("Ralutet/PP-DocLayout-L-onnx-int8", "361fe8f94b5bbd2fe83d1cf3eb4b78203ba934b5", "pp-doclayout-l.onnx")


def drdoc():
    import time
    from huggingface_hub import snapshot_download
    for k in range(8):
        try:
            snapshot_download("2077AIDataFoundation/DrDocBench", repo_type="dataset", revision=DRD_REV, local_dir=DRD,
                              allow_patterns=["dev/*/*/json/*.json", "dev/*/*/images/*.jpg", "README.md"], max_workers=4)
            break
        except Exception as e:
            print("retry", k, str(e)[:200], flush=True)
            time.sleep(60)
    else:
        sys.exit("drdoc download failed")
    import glob
    print("drdoc json", len(glob.glob(os.path.join(DRD, "dev", "*", "*", "json", "*.json"))),
          "images", len(glob.glob(os.path.join(DRD, "dev", "*", "*", "images", "*.jpg"))))


def lmodels():
    from huggingface_hub import hf_hub_download
    for sub, (repo, rev, fn) in (("egret", EGRET), ("ppl", PPL)):
        p = hf_hub_download(repo, fn, revision=rev, local_dir=os.path.join(LMODELS, sub))
        print(sub, p, os.path.getsize(p))


if __name__ == "__main__":
    for what in sys.argv[1:]:
        globals()[what]()
