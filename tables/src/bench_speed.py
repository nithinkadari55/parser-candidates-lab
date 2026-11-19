import os, sys, json, time, statistics, resource
from PIL import Image
import paths
from paths import SETS, OUT, MODELS

FILES = {
    "tatr32": ["tatr/model.onnx"], "tatr16": ["tatr/model_fp16.onnx"], "tatr": ["tatr/model_quantized.onnx"], "tatr4": ["tatr/model_int4.onnx"],
    "slanet1m": ["slanet1m/slanet_1m.onnx"], "tcn": ["tcn/model.onnx"], "potatr": ["potatr/potatr.onnx"], "potatr8": ["potatr/potatr_int8.onnx"],
    "tfv2": ["tfv2/model.safetensors"], "tfv2q": ["tfv2_onnx/enc_int8.onnx", "tfv2_onnx/dec_int8.onnx", "tfv2_onnx/box_int8.onnx"],
}


def rss_mb():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS:"): return int(l.split()[1]) / 1024
    return 0.0


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sample_ids(n_per_set=50):
    out = []
    for s in ["ptn", "ftn", "pt1m", "pt2", "tabrec"]:
        p = os.path.join(SETS, s, "index.json")
        if not os.path.exists(p): continue
        ids = json.load(open(p))
        step = max(1, len(ids) // n_per_set)
        out += [(s, i) for i in ids[::step][:n_per_set]]
    return out


def pct(v, q): return v[int(q * (len(v) - 1))]


def disk_mb(model):
    tot = 0
    for f in FILES.get(model, []):
        p = os.path.join(MODELS, f)
        if os.path.exists(p): tot += os.path.getsize(p)
        if os.path.exists(p + ".data"): tot += os.path.getsize(p + ".data")
    return round(tot / 1e6, 1)


def bench_model(model, threads):
    import bench_models as BM
    base = rss_mb()
    t = time.perf_counter(); M = BM.MODELS_[model](threads); load = time.perf_counter() - t
    after_load = rss_mb()
    pad = getattr(M, "PAD", 20)
    items = []
    for s, i in sample_ids():
        gt = BM.load_gt(s, i)
        im, (ox, oy) = BM.prep_image(Image.open(os.path.join(SETS, s, i + ".png")).convert("RGB"), "pad", pad)
        items.append((im, BM.chunks(gt, pad, ox, oy)))
    M(*items[0]); M(*items[1])
    ts = []; tm = []
    for im, tk in items:
        t = time.perf_counter(); _, m = M(im, tk); ts.append((time.perf_counter() - t) * 1000); tm.append(m * 1000)
    ts.sort(); tm.sort()
    return dict(model=model, threads=threads, n=len(ts), disk_mb=disk_mb(model), load_s=round(load, 2), median_ms=round(statistics.median(ts), 1), p95_ms=round(pct(ts, 0.95), 1),
                mean_ms=round(sum(ts) / len(ts), 1), model_only_median_ms=round(statistics.median(tm), 1),
                rss_after_load_minus_base_mb=round(after_load - base), peak_rss_mb=round(peak_mb()), base_rss_mb=round(base))


def bench_tf(threads):
    from make_pdfs import make_multi
    base = rss_mb()
    import docling_rs as d
    from docling_rs.options import PdfFormatOption, PdfPipelineOptions, InputFormat
    po = PdfPipelineOptions(do_ocr=False); po.accelerator_options.num_threads = threads
    t = time.perf_counter()
    conv = d.DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}, do_ocr=False)
    tmp = os.path.join(OUT, "speedpdf"); os.makedirs(tmp, exist_ok=True)
    ids = sample_ids()
    singles = []
    for k, (s, i) in enumerate(ids):
        p = os.path.join(tmp, f"one_{k}.pdf"); make_multi(s, [i], p, "letter"); singles.append(p)
    w = os.path.join(tmp, "warm.pdf"); make_multi(ids[0][0], [ids[0][1]], w, "letter")
    conv.convert(w); load = time.perf_counter() - t
    after_load = rss_mb()
    ts = []
    for p in singles:
        t = time.perf_counter(); conv.convert(p); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    multi = []; by = {}
    for s, i in ids: by.setdefault(s, []).append(i)
    for s, lst in by.items():
        for k in range(0, len(lst), 25):
            p = os.path.join(tmp, f"multi_{s}_{k}.pdf"); ch = lst[k:k + 25]; make_multi(s, ch, p, "letter")
            t = time.perf_counter(); conv.convert(p); multi.append((time.perf_counter() - t) * 1000 / len(ch))
    cache = os.path.expanduser("~/.cache/docling.rs")
    tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(cache) for f in fs if "table" in os.path.join(r, f).lower() or "layout" in f.lower()) if os.path.isdir(cache) else 0
    return dict(model="tableformer", note="docling-rs 2.84.0 full PDF pipeline (layout + TableFormer, no OCR), one table per page", threads=threads, n=len(ts),
                disk_mb_layout_plus_tableformer=round(tot / 1e6, 1), load_plus_first_convert_s=round(load, 2),
                median_ms=round(statistics.median(ts), 1), p95_ms=round(pct(ts, 0.95), 1), per_page_in_25_page_docs_median_ms=round(statistics.median(multi), 1),
                rss_after_load_minus_base_mb=round(after_load - base), peak_rss_mb=round(peak_mb()), base_rss_mb=round(base))


def main():
    model = sys.argv[1]; threads = int(sys.argv[2])
    r = bench_tf(threads) if model == "tableformer" else bench_model(model, threads)
    print(json.dumps(r), flush=True)
    os.makedirs(os.path.join(OUT, "speed"), exist_ok=True)
    if os.environ.get("DOCLING_RS_TIMING"): return
    open(os.path.join(OUT, "speed", "speed.jsonl"), "a").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
