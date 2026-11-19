"""Office files: docling.rs (DocumentConverter) vs PaddleOCR `doc2md` (the only Office path in PaddleOCR 3.7.0).

usage: office.py dlrs|pp IN_DIRS... --out OUT
"""
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MARKERS = {
    "synthetic.docx": ["BODYMARK", "HEADINGMARK", "COMMENTMARK", "HEADERMARK", "FOOTERMARK"],
    "synthetic.pptx": ["SLIDE1MARK", "VISIBLEBODYMARK", "NOTESMARK", "HIDDENSLIDEMARK", "SLIDE3MARK", "CHARTTITLEMARK", "CatAlpha", "22.75", "SeriesMark"],
    "synthetic.xlsx": ["VisibleSheetMark", "MERGEDMARK", "HiddenSheetMark", "HIDDENCELLMARK"],
}
EXT = (".docx", ".pptx", ".xlsx", ".xlsm")


def files(dirs):
    out = []
    for d in dirs:
        out += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(EXT)]
    return out


def dl_hits(doc, marker):
    def res(ref):
        _, kind, idx = ref.split("/")
        return doc[kind][int(idx)]
    hits = []
    for kind in ("texts", "groups", "tables", "pictures"):
        for it in doc.get(kind, []):
            parent = it.get("parent", {}).get("$ref")
            plab = res(parent).get("label") if parent and parent.count("/") == 2 else parent
            if kind == "tables":
                for c in (it.get("data") or {}).get("table_cells") or []:
                    if marker in (c.get("text") or ""):
                        hits.append(dict(kind=kind, label=it.get("label"), layer=it.get("content_layer"), parent=plab,
                                         row_span=c.get("row_span"), col_span=c.get("col_span"), column_header=c.get("column_header")))
            else:
                s = (it.get("text") or "") + " " + (it.get("name") or "")
                if marker in s:
                    hits.append(dict(kind=kind, label=it.get("label"), layer=it.get("content_layer"), parent=plab))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stack")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stack == "dlrs":
        import run_dlrs, dl_doc
        conv, _ = run_dlrs.converter()
    else:
        from paddleocr._doc2md import convert as d2m
    rows = []
    for p in files(a.dirs):
        name = os.path.basename(p)
        t = time.perf_counter()
        rec = dict(file=name, stack=a.stack)
        try:
            if a.stack == "dlrs":
                doc = conv.convert(p).document.export_to_dict()
                feats, _ = dl_doc.features(doc)
                layers = {}
                for kind in ("texts", "tables", "pictures", "groups"):
                    for it in doc.get(kind, []):
                        layers[it.get("content_layer", "body")] = layers.get(it.get("content_layer", "body"), 0) + 1
                rec.update(ok=True, feats=feats, layers=layers, groups=sorted({g.get("label") for g in doc.get("groups", [])}),
                           chart_tables=sum(1 for pic in doc.get("pictures", []) for an in pic.get("annotations") or [] if an.get("kind") == "tabular_chart_data") +
                           sum(1 for pic in doc.get("pictures", []) if (pic.get("meta") or {}).get("tabular_chart")))
                for m in MARKERS.get(name, []):
                    rec.setdefault("markers", {})[m] = dl_hits(doc, m)
                json.dump(dl_doc.strip_images(doc), open(os.path.join(a.out, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
            else:
                r = d2m(p)
                md = r.markdown
                open(os.path.join(a.out, name + ".md"), "w", encoding="utf-8").write(md)
                rec.update(ok=True, md_chars=len(md), images=len(r.images or {}), tables=md.count("<table"),
                           spans=len(re.findall(r"(rowspan|colspan)=\"?[2-9]", md)), th=md.count("<th"),
                           headings=len(re.findall(r"(?m)^#{1,6} ", md)))
                for m in MARKERS.get(name, []):
                    lines = [ln.strip()[:160] for ln in md.splitlines() if m in ln]
                    rec.setdefault("markers", {})[m] = lines
        except Exception as e:
            rec.update(ok=False, error=repr(e)[:400])
        rec["sec"] = round(time.perf_counter() - t, 3)
        rows.append(rec)
        print(name, rec.get("ok"), rec.get("error", ""), flush=True)
    with open(os.path.join(a.out, f"office-{a.stack}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            C.write_jsonl(f, r)


if __name__ == "__main__":
    main()
