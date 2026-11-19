import hashlib, json, os, shutil, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MANIFEST = json.load(open(os.path.join(C.ROOT, "dlrs_models.json"), encoding="utf-8"))


def odb():
    from huggingface_hub import hf_hub_download
    os.makedirs(C.ODB, exist_ok=True)
    hf_hub_download(C.ODB_REPO, "OmniDocBench.json", repo_type="dataset", revision=C.ODB_REV, local_dir=C.ODB)
    sel = C.selection_index()

    def get(n):
        if os.path.exists(os.path.join(C.ODB, "images", n)):
            return 0
        err = None
        for _ in range(4):
            try:
                hf_hub_download(C.ODB_REPO, "images/" + n, repo_type="dataset", revision=C.ODB_REV, local_dir=C.ODB)
                return 0
            except Exception as e:
                err = e
                time.sleep(3)
        print("FAIL", n, err, flush=True)
        return 1
    with ThreadPoolExecutor(8) as ex:
        bad = sum(ex.map(get, [s["img"] for s in sel]))
    print("odb selected", len(sel), "failed", bad)
    if bad:
        sys.exit(1)


def corpus():
    tmp = os.path.join(C.HOME, "dlrs-src")
    if not os.path.isdir(tmp):
        subprocess.check_call(["git", "clone", "--filter=blob:none", "--no-checkout", "--sparse",
                               "https://github.com/docling-project/docling.rs", tmp])
        subprocess.check_call(["git", "-C", tmp, "sparse-checkout", "set", "--no-cone",
                               "/tests/data/pdf/sources/", "/tests/data/scanned/sources/", "/tests/data/docx/sources/",
                               "/tests/data/pptx/sources/", "/tests/data/xlsx/sources/"])
        subprocess.check_call(["git", "-C", tmp, "checkout", C.DOCLINGRS_REV])
    for sub in ("pdf", "scanned", "docx", "pptx", "xlsx"):
        src = os.path.join(tmp, "tests", "data", sub, "sources")
        dst = os.path.join(C.CORPUS, sub)
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(src)):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        print(sub, len(os.listdir(dst)), sorted(os.listdir(dst)))


def _get(url, dst, sha=None):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and (sha is None or _sha(dst) == sha):
        return
    for i in range(5):
        try:
            urllib.request.urlretrieve(url, dst + ".part")
            os.replace(dst + ".part", dst)
            break
        except Exception as e:
            print("retry", url, e, flush=True)
            time.sleep(5 * (i + 1))
    else:
        sys.exit("download failed " + url)
    if sha and _sha(dst) != sha:
        sys.exit(f"sha mismatch {dst}")


def _sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def dlrs():
    for m in MANIFEST["base"]:
        _get(m["url"], os.path.join(C.DLRS_MODELS, "models", m["path"]), m.get("sha256"))
    for name in MANIFEST["records"]:
        _get(MANIFEST["release"] + "/" + name, os.path.join(C.DLRS_MODELS, "records", name))
    print("dlrs base models", sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(os.path.join(C.DLRS_MODELS, "models")) for f in fs))


def dlrs_formula():
    for m in MANIFEST["formula"]:
        _get(m["url"], os.path.join(C.DLRS_MODELS, "models", m["path"]), m.get("sha256"))
    print("dlrs formula models ok")


if __name__ == "__main__":
    for what in sys.argv[1:]:
        globals()[what]()
