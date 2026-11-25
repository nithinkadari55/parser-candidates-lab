"""Round 8 probe: download small parts of each new dataset and print their structure (keys, counts, shapes; no table
content), so the prep code is written against the real files.  usage: inspect8.py NAME [NAME ...]"""
import os, sys, io, json, re, glob, tarfile, zipfile, subprocess, collections, hashlib, urllib.request, time

WORK = os.environ.get("R8_SCRATCH", "/tmp/r8probe")
os.makedirs(WORK, exist_ok=True)


def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd]).returncode


def hf(repo, fn):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fn, repo_type="dataset", local_dir=os.path.join(WORK, repo.replace("/", "_")))


def shape(v, depth=0):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {shape(x, depth + 1)}" for k, x in list(v.items())[:25]) + "}"
    if isinstance(v, list):
        return f"list[{len(v)}]" + (f" of {shape(v[0], depth + 1)}" if v and depth < 4 else "")
    if isinstance(v, str):
        return f"str({len(v)})"
    return type(v).__name__


def tar_summary(path, n=8):
    c = collections.Counter(); first = collections.defaultdict(list)
    with tarfile.open(path, "r|*") as t:
        for m in t:
            d = os.path.dirname(m.name); ext = os.path.splitext(m.name)[1]
            c[(d, ext, m.isfile())] += 1
            if len(first[d]) < n: first[d].append(os.path.basename(m.name))
    for (d, ext, f), v in sorted(c.items()):
        print(f"  {v:7d} {d}/*{ext} file={f}  e.g. {first[d][:3]}", flush=True)


def fintabnet_c():
    p = hf("bsmock/FinTabNet.c", "FinTabNet.c-PDF_Annotations.tar.gz")
    print("PDF_Annotations", os.path.getsize(p)); tar_summary(p)
    with tarfile.open(p, "r|*") as t:
        for m in t:
            if m.isfile() and m.name.endswith(".json"):
                d = json.load(t.extractfile(m)); print("pdf annotation json", m.name.split("/")[-1][:12], shape(d)); break
    t0 = time.time()
    p = hf("bsmock/FinTabNet.c", "FinTabNet.c-Structure.tar.gz")
    print("Structure", os.path.getsize(p), "download s", round(time.time() - t0)); t0 = time.time()
    tar_summary(p); print("list s", round(time.time() - t0))
    seen = set()
    with tarfile.open(p, "r|*") as t:
        for m in t:
            if not m.isfile(): continue
            ext = os.path.splitext(m.name)[1]; d = os.path.dirname(m.name)
            key = (d, ext)
            if key in seen or ext not in (".xml", ".json", ".txt"): continue
            seen.add(key)
            raw = t.extractfile(m).read()
            if ext == ".xml":
                import xml.etree.ElementTree as ET
                r = ET.fromstring(raw)
                names = collections.Counter(o.find("name").text for o in r.iter("object"))
                print("xml", d, "size", r.find("size") is not None and [r.find("size").find(k).text for k in ("width", "height")], dict(names), "filename tag", r.find("filename") is not None)
            elif ext == ".json":
                print("json", d, shape(json.loads(raw)))
            else:
                print("txt", d, m.name, "lines", raw.count(b"\n"))
    os.remove(p)


def synfin():
    import pyarrow.parquet as pq
    p = hf("ethanbradley/synfintabs", "data/test-00000-of-00002.parquet")
    f = pq.ParquetFile(p); print("synfin rows", f.metadata.num_rows, f.schema_arrow)
    t = f.read_row_group(0).to_pylist()[:3]
    from PIL import Image
    for r in t:
        im = Image.open(io.BytesIO(r["image"]["bytes"]))
        rows = r["rows"]
        print("synfin item: image", im.size, "bbox", r["bbox"], "theme", r["theme"], "rows", len(rows), "cells/row", [len(x["cells"]) for x in rows][:12],
              "labels", collections.Counter(c.get("label") for x in rows for c in x["cells"]),
              "html tags", collections.Counter(re.findall(r"<(/?\w+)", r["representations"]["html"])),
              "spans", re.findall(r"(?:row|col)span=\"?\d+", r["representations"]["html"])[:6],
              "words/cell", [len(c["words"]) for c in rows[0]["cells"]], "ocr words", len(r["ocr_results"]["words"]))
        print("  first row cell bboxes", [c["bbox"] for c in rows[0]["cells"]][:6], "row bbox", rows[0]["bbox"])
    readme = hf("ethanbradley/synfintabs", "README.md")
    print("README license lines", [l for l in open(readme, encoding="utf-8") if "licen" in l.lower()])


def pt2():
    for part in ("cross-page-table-pairs", "single-page-table-continuations", "tables", "xml-annotations"):
        p = hf("kensho/PubTables-v2", f"PubTables-v2_Full-Documents_test_{part}.tar.gz")
        print("== full docs", part, os.path.getsize(p)); tar_summary(p, 4)
        with tarfile.open(p, "r|*") as t:
            k = 0
            for m in t:
                if not m.isfile(): continue
                raw = t.extractfile(m).read()
                if m.name.endswith(".json"):
                    d = json.loads(raw); print("  json", m.name.split("/")[-1], shape(d))
                    if isinstance(d, list) and d and isinstance(d[0], dict): print("   keys", list(d[0].keys()))
                    if isinstance(d, dict): print("   keys", list(d.keys())[:20])
                elif m.name.endswith(".xml"):
                    import xml.etree.ElementTree as ET
                    r = ET.fromstring(raw); print("  xml", m.name.split("/")[-1], collections.Counter(o.find("name").text for o in r.iter("object")))
                else:
                    print("  other", m.name, raw[:200])
                k += 1
                if k >= 2: break
    p = hf("kensho/PubTables-v2", "PubTables-v2_Cropped-Tables_test_tables.tar.gz")
    names = []
    with tarfile.open(p, "r|*") as t:
        for m in t:
            if m.isfile(): names.append(os.path.basename(m.name))
    print("cropped test tables", len(names), names[:5])
    with tarfile.open(p, "r|*") as t:
        for m in t:
            if m.isfile():
                d = json.load(t.extractfile(m)); print("cropped table json", shape(d)); print("  keys", list(d[0].keys()) if isinstance(d, list) else list(d.keys())); break


def ctdar():
    d = os.path.join(WORK, "ctdar")
    if not os.path.isdir(d):
        sh(f"git clone -q --depth 1 https://github.com/cndplab-founder/ICDAR2019_cTDaR.git {d}")
    import xml.etree.ElementTree as ET
    for sub in ("training/TRACKB1/ground_truth", "test_ground_truth/TRACKB1", "test_ground_truth/TRACKB2", "test/TRACKB1", "test/TRACKB2"):
        xs = sorted(glob.glob(os.path.join(d, sub, "*.xml")))
        t0 = [x for x in xs if os.path.basename(x).startswith("cTDaR_t0")]; t1 = [x for x in xs if os.path.basename(x).startswith("cTDaR_t1")]
        nt = nc = nocell = 0; attrs = collections.Counter()
        for x in xs:
            r = ET.parse(x).getroot()
            for tb in r.iter("table"):
                nt += 1; cells = list(tb.iter("cell")); nc += len(cells)
                if not cells: nocell += 1
                for c in cells[:3]: attrs.update(c.attrib.keys()); attrs.update("child:" + ch.tag for ch in c)
        print(sub, "xml", len(xs), "t0", len(t0), "t1", len(t1), "tables", nt, "cells", nc, "tables without cells", nocell, dict(attrs))
    for x in sorted(glob.glob(os.path.join(d, "test_ground_truth/TRACKB2/*.xml")))[:1] + sorted(glob.glob(os.path.join(d, "test_ground_truth/TRACKB2/cTDaR_t1*.xml")))[:1]:
        r = ET.parse(x).getroot(); print("example", os.path.basename(x), r.tag, r.attrib, [(c.tag, c.attrib) for c in list(r)[:2]])
    imgs = collections.Counter(os.path.splitext(p)[1].lower() for p in glob.glob(os.path.join(d, "**", "*.*"), recursive=True))
    print("files by ext", dict(imgs))


def wtw():
    base = "https://www.modelscope.cn/api/v1/datasets/yichao666/WTW_test/repo"
    def tree(root, page):
        u = f"{base}/tree?Revision=master&Root={root}&Recursive=false&PageNumber={page}&PageSize=500"
        return json.load(urllib.request.urlopen(u, timeout=120))["Data"]
    for root in ("html", "images"):
        n = 0; page = 1; tot = None
        while True:
            d = tree(root, page); fs = d.get("Files") or []
            tot = d.get("TotalCount", tot); n += len(fs)
            if len(fs) < 500: break
            page += 1
        print("wtw", root, "listed", n, "total", tot)
    t0 = time.time(); ok = 0
    for f in ("html/000cce9ca593055d4618466e823e6d7c_table0.html", "images/000cce9ca593055d4618466e823e6d7c_table0.jpg"):
        r = urllib.request.urlopen(f"{base}?Revision=master&FilePath={f}", timeout=120).read(); ok += 1
        print("wtw file", f.split("/")[0], len(r), "bytes")
        if f.endswith(".jpg"):
            from PIL import Image
            print("  image size", Image.open(io.BytesIO(r)).size)
        else:
            print("  html tags", collections.Counter(re.findall(rb"<(/?\w+)", r)))
    print("wtw 2 files in", round(time.time() - t0, 1), "s")
    sh("pip -q install modelscope==1.31.0 && python -c \"import modelscope; print('modelscope', modelscope.__version__)\" || true")


def tabrec():
    z = os.path.join(WORK, "tabrec.zip")
    t0 = time.time()
    req = urllib.request.Request("https://ndownloader.figshare.com/files/40332520", headers={"User-Agent": "table-bench/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(z, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b: break
            f.write(b)
    print("tabrec zip", os.path.getsize(z), "s", round(time.time() - t0))
    zz = zipfile.ZipFile(z); c = collections.Counter()
    for n in zz.namelist():
        parts = n.split("/")
        c["/".join(parts[:min(4, len(parts) - 1)])] += 1
    for k, v in sorted(c.items()): print(f"  {v:6d} {k}")
    os.remove(z)


def lcnet():
    from huggingface_hub import hf_hub_download, model_info
    p = hf_hub_download("PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx", "inference.onnx")
    import onnxruntime as ort
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    print("pplcnet", os.path.getsize(p), hashlib.sha256(open(p, "rb").read()).hexdigest(), [(i.name, i.shape, i.type) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()])
    info = model_info("PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx")
    print("card license", (info.card_data or {}).get("license") if info.card_data else None, "sha", info.sha, "files", [x.rfilename for x in info.siblings])


def ctdar_exif():
    """EXIF orientation of the cTDaR images and whether the answer-key table boxes fit the raw or the turned image,
    plus ink density inside the table boxes for both readings (counts only)"""
    from PIL import Image, ImageOps
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(WORK, "ctdar")
    if not os.path.isdir(d): sh(f"git clone -q --depth 1 https://github.com/cndplab-founder/ICDAR2019_cTDaR.git {d}")
    import prep8
    imgs = {}
    for p in glob.glob(os.path.join(d, "**", "*.*"), recursive=True):
        if os.path.splitext(p)[1].lower() in (".jpg", ".png", ".tiff", ".tif"): imgs.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    xmls = {}
    for sub in ("test_ground_truth/TRACKB2", "test_ground_truth/TRACKB1", "training/TRACKB1/ground_truth"):
        for x in sorted(glob.glob(os.path.join(d, sub, "*.xml"))): xmls.setdefault(os.path.basename(x)[:-4], x)
    c = collections.Counter(); ink = collections.defaultdict(list)
    for k in sorted(xmls):
        kind = "hist" if k.startswith("cTDaR_t0") else "mod"
        _, tabs = prep8.ctdar_tables(xmls[k])
        im = Image.open(imgs[k]); o = im.getexif().get(274, 1); c[(kind, "exif", o)] += 1
        raw = im.convert("L"); tr = ImageOps.exif_transpose(im).convert("L")
        for name, img in (("raw", raw), ("transposed", tr)):
            W, H = img.size; a = np.asarray(img) < 128
            fits = all(tb[2] <= W + 2 and tb[3] <= H + 2 for tb, _ in tabs)
            c[(kind, name, "fits", fits)] += 1
            if fits and tabs:
                ins = np.mean([a[int(tb[1]):int(tb[3]), int(tb[0]):int(tb[2])].mean() for tb, _ in tabs])
                ink[(kind, name, o)].append(ins / max(1e-6, a.mean()))
        if c[(kind, "shown")] < 3:
            c[(kind, "shown")] += 1
            print(kind, "image", raw.size, "exif", o, "first table box", [round(v) for v in tabs[0][0]] if tabs else None, "cells", len(tabs[0][1]) if tabs else 0,
                  "grid", (max(x["r"] + x["rs"] for x in tabs[0][1]), max(x["c"] + x["cs"] for x in tabs[0][1])) if tabs else None)
    for k, v in sorted(c.items(), key=str): print(k, v)
    for k, v in sorted(ink.items(), key=str): print("ink ratio inside tables / page", k, "n", len(v), "median", round(float(np.median(v)), 2))


if __name__ == "__main__":
    for w in sys.argv[1:]:
        print("\n######", w, flush=True)
        try:
            globals()[w]()
        except Exception as e:
            import traceback; traceback.print_exc()
        sys.stdout.flush()
