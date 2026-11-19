import gzip, io, json, os, sys, tarfile, time, urllib.error, urllib.request, zipfile
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
UNIMER_REV = "2343ddd963290469da36ca83e3a56c66e068add9"
UA = {"User-Agent": "formula-bench/0.1 (research benchmark; local evaluation only)"}


def get(url, limit=None):
    for k in range(8):
        time.sleep(3.5)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                n = r.headers.get("Content-Length")
                if limit and n and int(n) > limit:
                    return None
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            print("retry", url, e.code, flush=True)
            time.sleep(30 * (k + 1))
        except (urllib.error.URLError, TimeoutError) as e:
            print("retry", url, e, flush=True)
            time.sleep(30 * (k + 1))
    raise RuntimeError("failed " + url)


def unimer():
    z = hf_hub_download("wanderkid/UniMER_Dataset", "UniMER-Test.zip", repo_type="dataset", revision=UNIMER_REV,
                        local_dir=os.path.join(D, "unimer"))
    out = os.path.join(D, "unimer", "x")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(out)
    os.remove(z)


def arxiv():
    papers = json.load(open(os.path.join(ROOT, "papers.json")))
    ok = []
    for p in papers:
        aid = p["id"]
        pd = os.path.join(D, "arxiv", aid)
        os.makedirs(os.path.join(pd, "src"), exist_ok=True)
        try:
            src = get(f"https://arxiv.org/e-print/{aid}", limit=30_000_000)
        except Exception as e:
            print("skip", aid, e, flush=True)
            continue
        if src is None:
            print("skip too big", aid, flush=True)
            continue
        if src[:2] == b"\x1f\x8b":
            src = gzip.decompress(src)
        if src[:4] == b"%PDF":
            print("skip pdf-only", aid, flush=True)
            continue
        try:
            with tarfile.open(fileobj=io.BytesIO(src)) as tf:
                mem = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith((".tex", ".sty", ".bbl", ".def", ".cls", ".ltx"))]
                tf.extractall(os.path.join(pd, "src"), members=mem, filter="data")
        except tarfile.ReadError:
            open(os.path.join(pd, "src", "main.tex"), "wb").write(src)
        except Exception as e:
            print("skip bad archive", aid, e, flush=True)
            continue
        try:
            pdf = get(f"https://arxiv.org/pdf/{aid}", limit=60_000_000)
        except Exception as e:
            print("skip", aid, e, flush=True)
            continue
        if not pdf or pdf[:4] != b"%PDF":
            print("no pdf", aid)
            continue
        open(os.path.join(pd, "paper.pdf"), "wb").write(pdf)
        ok.append(p)
        print("ok", aid, flush=True)
    json.dump(ok, open(os.path.join(D, "arxiv", "papers.json"), "w"), indent=0)
    print("papers", len(ok))


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    if "unimer" in sys.argv:
        unimer()
    if "arxiv" in sys.argv:
        arxiv()
