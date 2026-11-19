"""Dedicated timing job: same pages, one stack at a time, each in its own process (own venv).

usage: timing.py OUT   (env: PY_DL, PY_PP = python executables of the two venvs)
"""
import hashlib, json, os, statistics, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))


def tree_size(d):
    sizes = {}
    if not os.path.isdir(d):
        return sizes
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isdir(p):
            fs = [os.path.join(r, f) for r, _, x in os.walk(p) for f in x]
            sizes[name] = dict(bytes=sum(os.path.getsize(f) for f in fs), files=sorted(os.path.relpath(f, p) for f in fs))
        else:
            sizes[name] = dict(bytes=os.path.getsize(p))
    return sizes


def pick():
    sel = C.selection_index()
    imgs = []
    for b in ("simple", "medium", "complex", "edge"):
        imgs += [s for s in sel if s["bucket"] == b][1:4]
    warm = os.path.join(C.ODB, "images", sel[0]["img"])
    pdfs = [os.path.join(C.CORPUS, "pdf", "2206.01062.pdf"), os.path.join(C.CORPUS, "pdf", "amt_handbook_sample.pdf"),
            os.path.join(C.CORPUS, "scanned", "ocr_test.pdf")]
    return warm, [os.path.join(C.ODB, "images", s["img"]) for s in imgs], pdfs


def run(tag, py, script, extra, paths, out):
    d = os.path.join(out, tag)
    t = time.perf_counter()
    rc = subprocess.call([py, os.path.join(HERE, script)] + extra + ["files", "0", "1", d] + paths)
    wall = time.perf_counter() - t
    recs = []
    for f in os.listdir(d):
        if f.startswith("results-") and f.endswith(".jsonl"):
            recs = [json.loads(l) for l in open(os.path.join(d, f), encoding="utf-8")]
    return dict(tag=tag, rc=rc, wall=round(wall, 1), recs=recs, dir=d)


def md_hashes(d):
    out = {}
    md = os.path.join(d, "md")
    for f in sorted(os.listdir(md)) if os.path.isdir(md) else []:
        out[f] = hashlib.sha1(open(os.path.join(md, f), "rb").read()).hexdigest()
    return out


def summarize(r, warm_name, img_names, pdf_names):
    files = {x["file"]: x for x in r["recs"] if "file" in x}
    meta = next((x["meta"] for x in r["recs"] if "meta" in x), {})
    done = next((x for x in r["recs"] if x.get("done")), {})
    img = [files[n]["sec"] for n in img_names if n in files and "sec" in files[n]]
    per_pdf = {n: dict(sec=files[n]["sec"], pages=files[n]["pages"], sec_per_page=round(files[n]["sec"] / max(1, files[n]["pages"]), 3))
               for n in pdf_names if n in files and "sec" in files[n]}
    return dict(tag=r["tag"], rc=r["rc"], wall=r["wall"], load_sec=meta.get("load_sec"), peak_rss_mb=done.get("peak_rss_mb"),
                warmup_sec=files.get(warm_name, {}).get("sec"), odb_images=len(img),
                odb_sec_per_page_median=round(statistics.median(img), 3) if img else None,
                odb_sec_per_page_mean=round(statistics.mean(img), 3) if img else None, pdfs=per_pdf,
                errors=[x for x in r["recs"] if x.get("error")])


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    warm, imgs, pdfs = pick()
    paths = [warm] + imgs + pdfs
    names = [os.path.basename(p) for p in paths]
    py_dl, py_pp = os.environ["PY_DL"], os.environ["PY_PP"]
    runs = [("dlrs_a", py_dl, "run_dlrs.py", []), ("dlrs_b", py_dl, "run_dlrs.py", []),
            ("pp_a", py_pp, "run_pp.py", []), ("pp_b", py_pp, "run_pp.py", []),
            ("pp_threads4", py_pp, "run_pp.py", ["--threads", "4", "--tag", "pp_t4"])]
    res = {}
    for tag, py, script, extra in runs:
        r = run(tag, py, script, extra, paths, out)
        res[tag] = summarize(r, names[0], names[1:1 + len(imgs)], names[1 + len(imgs):])
        res[tag]["md"] = md_hashes(r["dir"])
        print(tag, json.dumps({k: v for k, v in res[tag].items() if k != "md"})[:1500], flush=True)
    det = {}
    for s in ("dlrs", "pp"):
        a, b = res[f"{s}_a"]["md"], res[f"{s}_b"]["md"]
        det[s] = dict(files=len(a), identical=sum(1 for k in a if b.get(k) == a[k]), differing=sorted(k for k in a if b.get(k) != a[k]))
    sizes = dict(dlrs=tree_size(os.path.join(C.DLRS_MODELS, "models")),
                 pp=tree_size(os.path.join(C.HOME, ".paddlex", "official_models")))
    tot = {k: sum(v["bytes"] for v in d.values()) for k, d in sizes.items()}
    json.dump(dict(set=names, runs=res, determinism=det, model_sizes=sizes, model_total_bytes=tot,
                   cpu=os.cpu_count()), open(os.path.join(out, "timing.json"), "w"), indent=1)
    print("determinism", det, "model bytes", tot)


if __name__ == "__main__":
    main()
