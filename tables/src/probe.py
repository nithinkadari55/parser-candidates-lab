import os, sys, io, json, tarfile, zipfile, urllib.request
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

W = os.environ.get("TB_WORK", "work")
os.makedirs(W, exist_ok=True)


def head_tar(rid, fn, n=3, show=1500):
    p = hf_hub_download(rid, fn, repo_type="dataset", local_dir=os.path.join(W, "probe"))
    print("==", fn, os.path.getsize(p))
    with tarfile.open(p) as t:
        k = 0; names = []
        for m in t:
            names.append(m.name)
            if m.isfile() and k < n:
                k += 1
                data = t.extractfile(m).read()
                print("--", m.name, len(data))
                if not m.name.endswith((".jpg", ".png")): print(data[:show].decode("utf-8", "replace"))
            if len(names) > 20000: break
        print("members seen", len(names), names[:5])


def main():
    for part in ["tables", "words", "xml-annotations"]:
        head_tar("kensho/PubTables-v2", f"PubTables-v2_Cropped-Tables_test_{part}.tar.gz", n=2, show=2500)
    for rid in ["bevaya/SciTSR-cc-by-nc-sa", "bevaya/SciTSR-pd"]:
        p = hf_hub_download(rid, "data/test-00000-of-00001.parquet", repo_type="dataset", local_dir=os.path.join(W, "probe"))
        t = pq.read_table(p); print("==", rid, t.num_rows, t.schema.names)
        r = t.slice(0, 1).to_pylist()[0]
        for k, v in r.items():
            if k in ("image", "pdf"): print(" ", k, type(v), (list(v.keys()) if isinstance(v, dict) else len(v)))
            else: print(" ", k, str(v)[:600])
    z = os.path.join(W, "probe", "tabrecset.zip")
    if not os.path.exists(z):
        urllib.request.urlretrieve("https://ndownloader.figshare.com/files/40332520", z)
    print("== tabrecset", os.path.getsize(z))
    with zipfile.ZipFile(z) as f:
        names = f.namelist(); print(len(names))
        dirs = sorted(set("/".join(n.split("/")[:3]) for n in names))
        print(dirs[:60])
        js = [n for n in names if n.endswith(".json")][:2]
        for n in js:
            print("--", n); print(f.read(n)[:3000].decode("utf-8", "replace"))
        other = [n for n in names if not n.endswith((".json", ".jpg", ".png", "/"))][:10]
        print("other", other)


if __name__ == "__main__":
    main()
