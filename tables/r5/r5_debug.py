"""Round 5 check: the harness's table pass on the corpus PDFs with gold table shapes.

usage: r5_debug.py PDF_DIR SELECTOR_JSON WORK_DIR WORDS_TOOL
For every one-box table docling-rs 2.84.0 (fp32 layout, as the harness) finds in 2206.01062, redp5110_sampled and
elsevier-00: the run-time crop (layout box + 4 pt), docling.rs words and crop (r5words), TATR fp16, the round-3
features and the exported round-5 trees. Prints TableFormer's and TATR's shapes, the selector probability and pick
next to the gold shapes of the harness corpus test (eval/parse/gold, rows x cols per page)."""
import os, sys, json, math, subprocess, copy
os.environ.setdefault("DOCLING_RS_FP32", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "r4"))
from r4_common import *
import numpy as np
from PIL import Image
import r3_select as R3
from r5_sel import leaf_sum
from r5_tatr import tatr_call

GOLD = {
    "2206.01062": [(4, 14, 12), (6, 14, 6), (7, 13, 5), (7, 14, 5), (8, 16, 5)],
    "elsevier-00": [(7, 18, 5), (11, 18, 5), (12, 18, 5), (14, 5, 13)],
    "redp5110_sampled": [(8, 5, 3), (9, 13, 6), (11, 4, 2), (12, 10, 3)],
}
MARGIN_PT = 4.0


def tl(b, ph):
    if b.get("coord_origin") == "BOTTOMLEFT": return [b["l"], ph - b["t"], b["r"], ph - b["b"]]
    return [b["l"], b["t"], b["r"], b["b"]]


def shape(cells):
    if not cells: return (0, 0)
    return (max(c["r1"] for c in cells), max(c["c1"] for c in cells))


def main():
    pdf_dir, sel_path, work, tool = sys.argv[1:5]
    model = json.load(open(sel_path))
    import docling_rs as d
    from docling_rs.options import PdfFormatOption, PdfPipelineOptions, InputFormat
    po = PdfPipelineOptions(do_ocr=False)
    conv = d.DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}, do_ocr=False)
    import bench_models as BM
    M = BM.TATR16(0)
    os.makedirs(work, exist_ok=True)
    jobs, meta = [], {}
    for stem in GOLD:
        path = os.path.join(pdf_dir, stem + ".pdf")
        dd = conv.convert(path).document.export_to_dict()
        heights = {int(k): v["size"]["height"] for k, v in dd["pages"].items()}
        widths = {int(k): v["size"]["width"] for k, v in dd["pages"].items()}
        for i, t in enumerate(dd["tables"]):
            if t.get("label") != "table" or len(t.get("prov") or []) != 1: continue
            p = t["prov"][0]; pg = p["page_no"]; ph = heights[pg]
            b = tl(p["bbox"], ph)
            crop = [max(0.0, b[0] - MARGIN_PT), max(0.0, b[1] - MARGIN_PT), min(widths[pg], b[2] + MARGIN_PT), min(ph, b[3] + MARGIN_PT)]
            tid = f"{stem}_t{i}"
            jobs.append(dict(set="dbg", id=tid, pdf=path, page=pg - 1, crop_pt=crop))
            meta[tid] = dict(stem=stem, page=pg, table=t, ph=ph, crop=crop, box=b)
    jp = os.path.join(work, "jobs.jsonl")
    with open(jp, "w") as f:
        for j in jobs: f.write(json.dumps(j) + "\n")
    subprocess.run([tool, jp, work], check=True)
    rows = []
    for tid, m in meta.items():
        w = json.load(open(os.path.join(work, "dbg", tid + ".json")))
        if "error" in w: print(tid, "words error", w["error"]); continue
        words = [dict(text=x["text"], bbox=x["bbox"], gi=None) for x in w["words"]]
        im = Image.open(os.path.join(work, "dbg", tid + ".png")).convert("RGB")
        im2, (ox, oy) = BM.prep_image(im, "pad", 20)
        toks = BM.chunks(dict(words=words), 20, ox, oy)
        try:
            cb, _, cap = tatr_call(M, im2, toks)
            for c in cb: c["bbox"] = [c["bbox"][0] + ox, c["bbox"][1] + oy, c["bbox"][2] + ox, c["bbox"][3] + oy]
        except Exception as e:
            cb = []; print(tid, "tatr error", repr(e)[:200])
        L, T = m["crop"][0], m["crop"][1]
        from run_tableformer import cells_of
        ca = cells_of(m["table"], m["ph"], L, T, S)
        fa, fb = R3.feats(ca, words), R3.feats(cb, words)
        x = {"a_" + k: v for k, v in fa.items()}; x.update({"b_" + k: v for k, v in fb.items()})
        x.update({"d_" + k: fb[k] - fa[k] for k in fa})
        x["agree"] = R3.agreement(ca, cb); x["n_words"] = len(words)
        vec = np.array([x[k] for k in model["features"]], np.float32)
        margin = float(np.float32(model["base_margin"]) + leaf_sum(model["trees"], vec))
        prob = 1 / (1 + math.exp(-margin))
        data = m["table"]["data"]
        tf_shape = (data.get("num_rows"), data.get("num_cols"))
        tatr_shape = shape(cb)
        hdr = sorted(set(c["r0"] for c in cb if c.get("header")))
        rows.append(dict(id=tid, stem=m["stem"], page=m["page"], tf=tf_shape, tatr=tatr_shape, prob=prob, pick="tatr" if (cb and prob > 0.5) else "tableformer",
                         words=len(words), box=[round(v, 1) for v in m["box"]], agree=x["agree"], tatr_header_rows=hdr,
                         tatr_objects=len(cap["objs"]) if cb else 0))
    for stem, gold in GOLD.items():
        print(f"\n## {stem}")
        used = set()
        for (page, gr, gc) in gold:
            cand = [r for r in rows if r["stem"] == stem and r["page"] == page and r["id"] not in used]
            if not cand: print(f"gold p{page} {gr}x{gc}: no table found"); continue
            r = cand[0]; used.add(r["id"])
            final = r["tatr"] if r["pick"] == "tatr" else r["tf"]
            flag = "OK" if tuple(final) == (gr, gc) else "WRONG"
            tf_ok = "ok" if tuple(r["tf"]) == (gr, gc) else "x"
            print(f"gold p{page} {gr}x{gc} | TableFormer {r['tf'][0]}x{r['tf'][1]} ({tf_ok}) | TATR {r['tatr'][0]}x{r['tatr'][1]} | p(TATR)={r['prob']:.3f} -> {r['pick']} => {flag} | "
                  f"agree {r['agree']:.3f} words {r['words']} box {r['box']} TATR header rows {r['tatr_header_rows']}")
        for r in rows:
            if r["stem"] == stem and r["id"] not in used:
                print(f"(no gold) {r['id']} p{r['page']} TableFormer {r['tf']} TATR {r['tatr']} p={r['prob']:.3f} {r['pick']}")
    json.dump(rows, open(os.path.join(work, "debug_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
