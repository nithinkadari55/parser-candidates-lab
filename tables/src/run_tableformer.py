import os, sys, json, time, argparse
import paths
from paths import SETS, RAW, OUT
from bench_models import shard_ids, set_ids, load_gt
from make_pdfs import make_multi


def tl(b, ph):
    if b.get("coord_origin") == "BOTTOMLEFT": return [b["l"], ph - b["t"], b["r"], ph - b["b"]]
    return [b["l"], b["t"], b["r"], b["b"]]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def cells_of(t, ph, ox, oy, s):
    out = []
    for c in t["data"]["table_cells"]:
        b = c.get("bbox")
        if not b: continue
        x0, y0, x1, y1 = tl(b, ph)
        out.append(dict(r0=c["start_row_offset_idx"], r1=c["end_row_offset_idx"], c0=c["start_col_offset_idx"], c1=c["end_col_offset_idx"],
                        bbox=[(x0 - ox) * s, (y0 - oy) * s, (x1 - ox) * s, (y1 - oy) * s], header=bool(c.get("column_header"))))
    return out


def page_heights(dd):
    return {int(k): v["size"]["height"] for k, v in dd["pages"].items()}


def run_images(conv, s, ids, of, page, chunk=20):
    tmpd = os.path.join(OUT, "tfpdf"); os.makedirs(tmpd, exist_ok=True)
    with open(of, "w", encoding="utf-8") as f:
        for k in range(0, len(ids), chunk):
            ch = ids[k:k + chunk]
            pdf = os.path.join(tmpd, f"{s}_{k}.pdf"); offs = make_multi(s, ch, pdf, page)
            t = time.perf_counter(); r = conv.convert(pdf); dt = (time.perf_counter() - t) * 1000
            dd = r.document.export_to_dict(); ph = page_heights(dd)
            bypage = {}
            for t_ in dd["tables"]:
                if not t_["prov"]: continue
                p = t_["prov"][0]; b = p["bbox"]
                bypage.setdefault(p["page_no"], []).append((abs((b["r"] - b["l"]) * (b["t"] - b["b"])), t_))
            for pi, tid in enumerate(ch, start=1):
                lst = sorted(bypage.get(pi, []), key=lambda x: -x[0])
                rec = dict(id=tid, ms_total=dt / len(ch), n_tables_on_page=len(lst), page=page)
                ox, oy = offs[pi - 1]
                rec["cells"] = cells_of(lst[0][1], ph[pi], ox, oy, 1.0) if lst else []
                if not lst: rec["err"] = "no table detected"
                f.write(json.dumps(rec) + "\n")
            f.flush(); os.remove(pdf)
            print(s, k, len(ch), round(dt / 1000, 1), "s", flush=True)


def run_icdar(conv, shard, nshards, of):
    ids = set_ids("icdar"); gts = {i: load_gt("icdar", i) for i in ids}
    bypage = {}
    for i, g in gts.items(): bypage.setdefault((g["filename"], g["page"]), []).append(i)
    keys = sorted(bypage)
    keys = [k for j, k in enumerate(keys) if j % nshards == shard]
    with open(of, "w", encoding="utf-8") as f:
        for fn, pg in keys:
            t = time.perf_counter(); r = conv.convert(os.path.join(SETS, "icdar_pdf", fn), page_range=(pg, pg)); dt = (time.perf_counter() - t) * 1000
            dd = r.document.export_to_dict(); ph = page_heights(dd)
            tabs = [t_ for t_ in dd["tables"] if t_["prov"]]
            for tid in bypage[(fn, pg)]:
                g = gts[tid]; L, T, R, B = g["crop_pt"]; S = g["scale"]
                best = None; bv = 0
                for t_ in tabs:
                    p = t_["prov"][0]
                    v = iou(tl(p["bbox"], ph.get(p["page_no"], list(ph.values())[0])), [L, T, R, B])
                    if v > bv: bv = v; best = t_
                rec = dict(id=tid, ms_total=dt / len(bypage[(fn, pg)]), match_iou=bv, n_tables_on_page=len(tabs))
                if best is not None and bv > 0.1:
                    p = best["prov"][0]
                    rec["cells"] = cells_of(best, ph.get(p["page_no"], list(ph.values())[0]), L, T, S)
                else:
                    rec["cells"] = []; rec["err"] = "no matching table"
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print("icdar", fn, pg, round(dt / 1000, 1), "s", len(tabs), flush=True)


def run_scitsr(conv, ids, of):
    with open(of, "w", encoding="utf-8") as f:
        for tid in ids:
            g = load_gt("scitsr", tid); sx, sy = g["pdf_scale"]
            t = time.perf_counter(); r = conv.convert(os.path.join(SETS, "scitsr", tid + ".pdf")); dt = (time.perf_counter() - t) * 1000
            dd = r.document.export_to_dict(); ph = page_heights(dd)
            tabs = [t_ for t_ in dd["tables"] if t_["prov"]]
            rec = dict(id=tid, ms_total=dt, n_tables_on_page=len(tabs))
            if tabs:
                best = max(tabs, key=lambda t_: len(t_["data"]["table_cells"]))
                p = best["prov"][0]
                cells = cells_of(best, ph.get(p["page_no"], list(ph.values())[0]), 0, 0, 1.0)
                for c in cells: c["bbox"] = [c["bbox"][0] * sx, c["bbox"][1] * sy, c["bbox"][2] * sx, c["bbox"][3] * sy]
                rec["cells"] = cells
            else:
                rec["cells"] = []; rec["err"] = "no table detected"
            f.write(json.dumps(rec) + "\n"); f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="ptn,ftn,pt1m,icdar,pt2,scitsr,tabrec,rot")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--page", default="letter", choices=["letter", "tight"])
    a = ap.parse_args()
    import docling_rs as d
    from docling_rs.options import PdfFormatOption, PdfPipelineOptions, InputFormat
    po = PdfPipelineOptions(do_ocr=False)
    if a.threads: po.accelerator_options.num_threads = a.threads
    conv = d.DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}, do_ocr=False)
    os.makedirs(os.path.join(OUT, "pred"), exist_ok=True)
    for s in a.sets.split(","):
        of = os.path.join(OUT, "pred", f"tableformer_{s}_s{a.shard}.jsonl")
        if s == "icdar": run_icdar(conv, a.shard, a.nshards, of)
        elif s == "scitsr": run_scitsr(conv, shard_ids(s, a.shard, a.nshards), of)
        else: run_images(conv, s, shard_ids(s, a.shard, a.nshards), of, a.page)
        print("done tableformer", s, a.shard, flush=True)


if __name__ == "__main__":
    main()
