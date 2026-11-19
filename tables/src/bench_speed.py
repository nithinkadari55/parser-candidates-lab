import os, sys, json, time, statistics, resource
from PIL import Image
import paths
from paths import SETS, OUT


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sample_ids(n_per_set=40):
    out = []
    for s in ["ptn", "ftn", "pt1m"]:
        ids = json.load(open(os.path.join(SETS, s, "index.json")))
        step = max(1, len(ids) // n_per_set)
        out += [(s, i) for i in ids[::step][:n_per_set]]
    return out


def pct(v, q): return v[int(q * (len(v) - 1))]


def bench_onnx(model, threads):
    import bench_models as BM
    base = peak_mb()
    t = time.perf_counter(); M = BM.MODELS_[model](threads); load = time.perf_counter() - t
    items = []
    for s, i in sample_ids():
        gt = BM.load_gt(s, i)
        items.append((BM.padimg(Image.open(os.path.join(SETS, s, i + ".png")).convert("RGB"), 20), BM.chunks(gt, 20)))
    M(*items[0]); M(*items[1])
    ts = []; tm = []
    for im, tk in items:
        t = time.perf_counter(); _, m = M(im, tk); ts.append((time.perf_counter() - t) * 1000); tm.append(m * 1000)
    ts.sort(); tm.sort()
    return dict(model=model, threads=threads, n=len(ts), load_s=round(load, 2), median_ms=round(statistics.median(ts), 1), p95_ms=round(pct(ts, 0.95), 1),
                model_only_median_ms=round(statistics.median(tm), 1), model_only_p95_ms=round(pct(tm, 0.95), 1),
                peak_rss_mb=round(peak_mb()), python_base_rss_mb=round(base))


def bench_tf(threads):
    from make_pdfs import make_multi
    base = peak_mb()
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
    ts = []
    for p in singles:
        t = time.perf_counter(); conv.convert(p); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    multi = []
    by = {}
    for s, i in ids: by.setdefault(s, []).append(i)
    for s, lst in by.items():
        for k in range(0, len(lst), 20):
            p = os.path.join(tmp, f"multi_{s}_{k}.pdf"); ch = lst[k:k + 20]; make_multi(s, ch, p, "letter")
            t = time.perf_counter(); conv.convert(p); multi.append((time.perf_counter() - t) * 1000 / len(ch))
    return dict(model="tableformer (docling-rs 2.84.0 full PDF pipeline: layout + TableFormer, no OCR)", threads=threads, n=len(ts),
                load_plus_first_convert_s=round(load, 2), single_table_pdf_median_ms=round(statistics.median(ts), 1), single_table_pdf_p95_ms=round(pct(ts, 0.95), 1),
                per_page_in_20_page_docs_median_ms=round(statistics.median(multi), 1), peak_rss_mb=round(peak_mb()), python_base_rss_mb=round(base))


def main():
    model = sys.argv[1]; threads = int(sys.argv[2])
    r = bench_tf(threads) if model == "tableformer" else bench_onnx(model, threads)
    print(json.dumps(r))
    os.makedirs(os.path.join(OUT, "speed"), exist_ok=True)
    if os.environ.get("DOCLING_RS_TIMING"): return
    open(os.path.join(OUT, "speed", "speed.jsonl"), "a").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
