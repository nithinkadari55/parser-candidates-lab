import os, sys, io, json, tarfile, zipfile, urllib.request, collections
from huggingface_hub import hf_hub_download

W = os.environ.get("TB_WORK", "work")
os.makedirs(W, exist_ok=True)


def short(v, n=300):
    s = json.dumps(v); return s[:n]


def main():
    p = hf_hub_download("kensho/PubTables-v2", "PubTables-v2_Cropped-Tables_test_tables.tar.gz", repo_type="dataset", local_dir=os.path.join(W, "probe"))
    with tarfile.open(p) as t:
        k = 0
        for m in t:
            if m.isfile():
                d = json.loads(t.extractfile(m).read())
                print("== pt2 table", m.name, type(d), len(d))
                tb = d[0] if isinstance(d, list) else d
                for key, v in tb.items(): print("  ", key, short(v, 400))
                k += 1
                if k >= 2: break
    z = os.path.join(W, "probe", "tabrecset.zip")
    if not os.path.exists(z):
        urllib.request.urlretrieve("https://ndownloader.figshare.com/files/40332520", z)
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
        c = collections.Counter("/".join(n.split("/")[:3]) for n in names if not n.endswith("/"))
        print("== tabrecset counts", dict(c))
        js = [n for n in names if "TSR_TCR_annotation/english" in n and n.endswith(".json")][:2]
        for n in js:
            d = json.loads(f.read(n))
            print("--", n, list(d.keys()))
            for s in d.get("shapes", [])[:4]: print("   shape", short(s, 500))
            print("   other", {k: short(v, 200) for k, v in d.items() if k not in ("shapes", "imageData")})
        ims = [n for n in names if "image/english" in n][:3]; print(ims)
        oc = [n for n in names if "OCR" in n or "ocr" in n][:5]; print(oc)


if __name__ == "__main__":
    main()
