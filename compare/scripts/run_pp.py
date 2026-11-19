"""PaddleOCR PP-StructureV3 (paddleocr 3.7.0 / paddlex 3.7.2), default settings on CPU.

usage: run_pp.py odb|own|files K N OUTDIR [--engine onnxruntime] [--no-formula] [--threads N] [paths...]
"""
import argparse, json, os, re, sys, time, traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def make(engine=None, no_formula=False, threads=None, no_mkldnn=False):
    from paddleocr import PPStructureV3
    kw = dict(device="cpu")
    if engine:
        kw["engine"] = engine
    if no_formula:
        kw["use_formula_recognition"] = False
    if threads:
        kw["cpu_threads"] = threads
    if no_mkldnn:
        kw["enable_mkldnn"] = False
    return PPStructureV3(**kw), kw


def trim(j):
    keep = {k: j.get(k) for k in ("input_path", "page_index", "page_count", "width", "height", "model_settings", "parsing_res_list")}
    keep["tables"] = [dict(pred_html=t.get("pred_html"), cell_box_list=t.get("cell_box_list")) for t in j.get("table_res_list") or []]
    keep["formulas"] = [dict(rec_formula=t.get("rec_formula")) for t in j.get("formula_res_list") or []]
    keep["layout_boxes"] = [dict(label=b.get("label"), score=round(float(b.get("score", 0)), 3), coordinate=[round(float(x), 1) for x in b.get("coordinate", [])])
                            for b in (j.get("layout_det_res") or {}).get("boxes", [])]
    return keep


def page_feats(j, md):
    blocks = j.get("parsing_res_list") or []
    lab = Counter(b.get("block_label") for b in blocks)
    htmls = [t.get("pred_html") or "" for t in j.get("tables") or []]
    hl = Counter(len(m.group(1)) for m in re.finditer(r"(?m)^(#{1,6}) ", md or ""))
    return dict(
        labels=dict(lab), titles=lab["doc_title"], headings=lab["paragraph_title"] + lab["abstract_title"] + lab["reference_title"] + lab["content_title"], md_heading_levels={str(k): v for k, v in sorted(hl.items())},
        tables=lab["table"], tables_with_spans=sum(1 for x in htmls if re.search(r"(rowspan|colspan)=\"?[2-9]", x)),
        tables_with_th=sum(1 for x in htmls if "<th" in x or "<thead" in x), table_cells=sum(x.count("<td") + x.count("<th") for x in htmls),
        pictures=lab["image"] + lab["chart"], captions=lab["figure_title"] + lab["figure_table_chart_title"] + lab["table_title"] + lab["chart_title"],
        vision_footnotes=lab["vision_footnote"], footnotes=lab["footnote"], page_headers=lab["header"] + lab["header_image"],
        page_footers=lab["footer"] + lab["footer_image"], numbers=lab["number"], formulas=lab["formula"],
        formulas_with_latex=sum(1 for b in blocks if b.get("block_label") == "formula" and (b.get("block_content") or "").strip()),
        lists=sum(1 for b in blocks if "list" in (b.get("block_label") or "")),
        blocks=len(blocks), blocks_with_bbox=sum(1 for b in blocks if b.get("block_bbox") and len(b["block_bbox"]) == 4),
        has_page_index=j.get("page_index") is not None)


def merge(fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            if isinstance(v, bool):
                out[k] = out.get(k, True) and v
            elif isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + v
            elif isinstance(v, dict):
                d = out.setdefault(k, {})
                for a, b in v.items():
                    d[a] = d.get(a, 0) + b
    return out


def words_of(j):
    w = []
    for b in j.get("parsing_res_list") or []:
        w += C.norm_words(C.strip_html(b.get("block_content")))
    return w


def text_of(j):
    return " ".join(C.strip_html(b.get("block_content")) for b in j.get("parsing_res_list") or [])


def run(pipe, path):
    t = time.perf_counter()
    pages = []
    for r in pipe.predict_iter(path):
        j = trim(r.json["res"])
        md = r.markdown.get("markdown_texts", "")
        pages.append((j, md, time.perf_counter() - t))
    return pages, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("k", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--engine")
    ap.add_argument("--no-formula", action="store_true")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--no-mkldnn", action="store_true")
    ap.add_argument("--tag", default="pp")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "md"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "json"), exist_ok=True)
    t0 = time.perf_counter()
    pipe, kw = make(a.engine, a.no_formula, a.threads, a.no_mkldnn)
    load = time.perf_counter() - t0
    from importlib.metadata import version
    vers = {p: version(p) for p in ("paddleocr", "paddlex")}
    try:
        vers["paddlepaddle"] = version("paddlepaddle")
    except Exception:
        vers["paddlepaddle"] = None
    with open(os.path.join(a.out, f"results-{a.tag}-{a.mode}-{a.k}.jsonl"), "w", encoding="utf-8") as f:
        C.write_jsonl(f, dict(meta=dict(stack=a.tag, mode=a.mode, versions=vers, init_kwargs=kw, load_sec=round(load, 2))))
        if a.mode == "odb":
            for s in C.shard(C.selection_index(), a.k, a.n):
                p = os.path.join(C.ODB, "images", s["img"])
                try:
                    pages, dt = run(pipe, p)
                except Exception as e:
                    traceback.print_exc()
                    C.write_jsonl(f, dict(img=s["img"], bucket=s["bucket"], error=repr(e)[:400]))
                    continue
                j, md, _ = pages[0]
                open(os.path.join(a.out, "md", os.path.splitext(s["img"])[0] + ".md"), "w", encoding="utf-8").write(md)
                C.write_jsonl(f, dict(img=s["img"], bucket=s["bucket"], sec=round(dt, 3), pages=1, feats=page_feats(j, md),
                                      model_settings=j.get("model_settings")))
                print(s["img"], s["bucket"], round(dt, 2), flush=True)
        else:
            paths = C.balanced(C.corpus_pdfs(), a.k, a.n) if a.mode == "own" else a.paths
            for p in paths:
                name = os.path.basename(p)
                try:
                    pages, dt = run(pipe, p)
                except Exception as e:
                    traceback.print_exc()
                    C.write_jsonl(f, dict(file=name, error=repr(e)[:400]))
                    print(name, "ERROR", e, flush=True)
                    continue
                js = [x[0] for x in pages]
                feats = merge(page_feats(j, md) for j, md, _ in pages)
                rec = dict(file=name, sec=round(dt, 3), pages=len(pages), feats=feats, model_settings=js[0].get("model_settings") if js else None)
                if p.lower().endswith(".pdf"):
                    ref = C.pdf_words(p)
                    words = [w for j in js for w in words_of(j)]
                    rec["text_layer_words"] = len(ref)
                    rec["text_f1_vs_layer"] = round(C.bag_f1(words, ref), 4) if ref else None
                    rec["page_texts"] = {str((j.get("page_index") or 0) + 1): text_of(j) for j in js}
                json.dump(js, open(os.path.join(a.out, "json", name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
                open(os.path.join(a.out, "md", name + ".md"), "w", encoding="utf-8").write("\n\n".join(md for _, md, _ in pages))
                C.write_jsonl(f, rec)
                print(name, len(pages), round(dt, 2), rec.get("text_f1_vs_layer"), flush=True)
        C.write_jsonl(f, dict(done=True, peak_rss_mb=C.peak_rss_mb(), wall_sec=round(time.perf_counter() - t0, 1)))


if __name__ == "__main__":
    main()
