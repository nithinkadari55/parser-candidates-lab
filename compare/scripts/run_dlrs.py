"""docling.rs 2.84.0 (PyPI wheel) with our harness options.

usage: run_dlrs.py odb|odb_formula|own|office|files K N OUTDIR [paths...]
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import dl_doc

M = os.path.join(C.DLRS_MODELS, "models")
ENV = {
    "DOCLING_RS_MODELS_DIR": M,
    "DOCLING_RS_FP32": "1",
    "DOCLING_OCR_REC_ONNX": os.path.join(M, "ocr_rec_v6_medium.onnx"),
    "DOCLING_OCR_DICT": os.path.join(M, "ocr_rec_v6_dict.txt"),
    "DOCLING_RS_PDF_WORKERS": "1",
    "DOCLING_RS_PDF_INTRA": "4",
    "DOCLING_RS_TF_INTRA": "4",
    "DOCLING_RS_OCR_SESSIONS": "2",
    "DOCLING_RS_OCR_ENGINE": "ppocr",
    "DOCLING_RS_OCR_LANG": "en",
}
OPTS = dict(do_ocr=True, do_table_structure=True, heading_hierarchy=True, replace_broken_text_layer=True,
            use_embedded_picture_images=True, ocr_textline_orientation=True, do_picture_classification=True,
            do_picture_ocr=True, ocr_engine="ppocr", ocr_lang="en", ocr_scale=3.0)


def converter(formula=False):
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    import docling_rs
    opts = dict(OPTS, do_formula_enrichment=formula)
    return docling_rs.DocumentConverter(**opts), opts


def convert(conv, path):
    t = time.perf_counter()
    doc = conv.convert(path).document.export_to_dict()
    return doc, time.perf_counter() - t


def main():
    mode, k, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(os.path.join(out, "md"), exist_ok=True)
    os.makedirs(os.path.join(out, "json"), exist_ok=True)
    t0 = time.perf_counter()
    conv, opts = converter(formula=mode == "odb_formula")
    load = time.perf_counter() - t0
    from importlib.metadata import version
    meta = dict(stack="dlrs", mode=mode, version=version("docling-rs"), options=opts, env=ENV, load_sec=round(load, 2))
    with open(os.path.join(out, f"results-{mode}-{k}.jsonl"), "w", encoding="utf-8") as f:
        C.write_jsonl(f, dict(meta=meta))
        if mode in ("odb", "odb_formula"):
            sel = C.selection_index()
            if mode == "odb_formula":
                sel = [s for s in sel if s["has_formula"]]
            for s in C.shard(sel, k, n):
                p = os.path.join(C.ODB, "images", s["img"])
                try:
                    doc, dt = convert(conv, p)
                except Exception as e:
                    C.write_jsonl(f, dict(img=s["img"], bucket=s["bucket"], error=repr(e)[:400]))
                    continue
                md = dl_doc.to_markdown(doc)
                open(os.path.join(out, "md", os.path.splitext(s["img"])[0] + ".md"), "w", encoding="utf-8").write(md)
                feats, _ = dl_doc.features(doc)
                C.write_jsonl(f, dict(img=s["img"], bucket=s["bucket"], sec=round(dt, 3), pages=1, feats=feats))
                print(s["img"], s["bucket"], round(dt, 2), flush=True)
        else:
            paths = C.balanced(C.corpus_pdfs(), k, n) if mode == "own" else sys.argv[5:]
            for p in paths:
                name = os.path.basename(p)
                try:
                    doc, dt = convert(conv, p)
                except Exception as e:
                    C.write_jsonl(f, dict(file=name, error=repr(e)[:400]))
                    print(name, "ERROR", e, flush=True)
                    continue
                feats, words = dl_doc.features(doc)
                rec = dict(file=name, sec=round(dt, 3), pages=feats["pages"], feats=feats)
                if p.lower().endswith(".pdf"):
                    ref = C.pdf_words(p)
                    rec["text_layer_words"] = len(ref)
                    rec["text_f1_vs_layer"] = round(C.bag_f1(words, ref), 4) if ref else None
                    rec["page_texts"] = {str(i): dl_doc.page_text(doc, i) for i in range(1, feats["pages"] + 1)}
                json.dump(dl_doc.strip_images(doc), open(os.path.join(out, "json", name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
                open(os.path.join(out, "md", name + ".md"), "w", encoding="utf-8").write(dl_doc.to_markdown(doc))
                C.write_jsonl(f, rec)
                print(name, rec["pages"], round(dt, 2), rec.get("text_f1_vs_layer"), flush=True)
        C.write_jsonl(f, dict(done=True, peak_rss_mb=C.peak_rss_mb(), wall_sec=round(time.perf_counter() - t0, 1)))


if __name__ == "__main__":
    main()
