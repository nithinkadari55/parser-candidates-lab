import argparse
import ctypes
import json
import os
import sys

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

MATH_MARKS = (
    "CMMI", "CMSY", "CMEX", "CMBSY", "MSAM", "MSBM", "EUFM", "EUSM", "EURM", "RSFS",
    "LMMATH", "LATINMODERNMATH", "STIXMATH", "STIXTWOMATH", "XITSMATH", "MATHJAX",
    "TXMI", "TXSY", "TXEX", "PXMI", "PXSY", "PXEX", "MTMI", "MTSY", "RMMI", "ESINT",
    "CAMBRIAMATH", "MATHITALIC", "MATHSYMBOL", "MATHEXTENSION",
)


def font_name(tp, i):
    flags = ctypes.c_int(0)
    n = pdfium_c.FPDFText_GetFontInfo(tp.raw, i, None, 0, ctypes.byref(flags))
    if n <= 0:
        return ""
    buf = ctypes.create_string_buffer(n)
    pdfium_c.FPDFText_GetFontInfo(tp.raw, i, buf, n, ctypes.byref(flags))
    return buf.value.decode("utf-8", "replace")


def is_math_font(name):
    up = name.upper().split("+")[-1].replace("-", "").replace(" ", "")
    return any(m in up for m in MATH_MARKS)


def page_chars(tp):
    chars = []
    for i in range(tp.count_chars()):
        ch = tp.get_text_range(i, 1)
        if not ch or ch.isspace() or ord(ch[0]) < 32:
            continue
        l, b, r, t = tp.get_charbox(i)
        if r <= l or t <= b:
            continue
        chars.append({"ch": ch, "box": (l, b, r, t), "font": font_name(tp, i)})
    return chars


def rows_of(chars):
    chars = sorted(chars, key=lambda c: -(c["box"][3] - c["box"][1]))
    rows = []
    for c in chars:
        cy = (c["box"][1] + c["box"][3]) / 2
        for row in rows:
            if row["lo"] <= cy <= row["hi"]:
                row["chars"].append(c)
                break
        else:
            rows.append({"lo": c["box"][1], "hi": c["box"][3], "chars": [c]})
    rows.sort(key=lambda r: -r["hi"])
    return rows


def segments_of(row, gap=20.0):
    cs = sorted(row["chars"], key=lambda c: c["box"][0])
    segs = [[cs[0]]]
    for c in cs[1:]:
        if c["box"][0] - segs[-1][-1]["box"][2] > gap:
            segs.append([c])
        else:
            segs[-1].append(c)
    return segs


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def prose_words(seg):
    words, run = 0, 0
    for c in seg + [None]:
        if c is not None and c["ch"].isalpha() and not is_math_font(c["font"]):
            run += 1
            continue
        if run >= 4:
            words += 1
        run = 0
    return words


def display_blocks(chars):
    cands = []
    for row in rows_of(chars):
        for seg in segments_of(row):
            n = len(seg)
            m = sum(is_math_font(c["font"]) for c in seg)
            if n >= 2 and m / n >= 0.4 and prose_words(seg) <= 1:
                cands.append(seg)
    blocks = []
    for seg in cands:
        box = union([c["box"] for c in seg])
        merged = False
        for blk in blocks:
            bb = blk["box"]
            v_gap = max(bb[1] - box[3], box[1] - bb[3])
            h_overlap = min(bb[2], box[2]) - max(bb[0], box[0])
            if v_gap < 6.0 and h_overlap > 0:
                blk["chars"].extend(seg)
                blk["box"] = union([bb, box])
                merged = True
                break
        if not merged:
            blocks.append({"chars": list(seg), "box": box})
    return [b for b in blocks if len(b["chars"]) >= 6]


def text_of(chars):
    cs = sorted(chars, key=lambda c: (-round((c["box"][1] + c["box"][3]) / 4), c["box"][0]))
    return "".join(c["ch"] for c in cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--per-pdf", type=int, default=6)
    ap.add_argument("--dump-pages", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for pdf_path in args.pdf:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        doc = pdfium.PdfDocument(pdf_path)
        found = []
        fonts = {}
        for pi in range(min(len(doc), args.max_pages)):
            page = doc[pi]
            mediabox = [round(v, 3) for v in page.get_mediabox()]
            tp = page.get_textpage()
            chars = page_chars(tp)
            for c in chars:
                fonts[c["font"]] = fonts.get(c["font"], 0) + 1
            if pi < args.dump_pages:
                print(f"--- {stem} page {pi} mediabox {mediabox}", file=sys.stderr)
                for row in rows_of(chars)[:60]:
                    for seg in segments_of(row):
                        b = union([c["box"] for c in seg])
                        m = sum(is_math_font(c["font"]) for c in seg)
                        print(f"  [{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}] math={m}/{len(seg)} "
                              f"{''.join(c['ch'] for c in seg)[:90]}", file=sys.stderr)
            if pi == 0:
                mx0, my0, mx1, my1 = mediabox
                rows.append({"id": f"{stem}_p0_blank", "pdf": os.path.abspath(pdf_path), "page_index": 0,
                             "bbox": [mx0 + 2, my0 + 2, mx0 + 20, my0 + 12], "mediabox": mediabox})
            for k, blk in enumerate(display_blocks(chars)):
                found.append((len(blk["chars"]), pi, k, mediabox, blk))
            tp.close()
            page.close()
        doc.close()
        chosen = sorted(found, key=lambda f: -f[0])[:args.per_pdf]
        got = len(chosen)
        for _, pi, k, mediabox, blk in sorted(chosen, key=lambda f: (f[1], f[2])):
            rows.append({"id": f"{stem}_p{pi}_eq{k}", "pdf": os.path.abspath(pdf_path), "page_index": pi,
                         "bbox": [round(v, 3) for v in blk["box"]], "mediabox": mediabox,
                         "pdfium_text": text_of(blk["chars"])})
        top = sorted(fonts.items(), key=lambda kv: -kv[1])[:12]
        print(f"{stem}: regions={got} fonts={top}", file=sys.stderr)

    first = next((r for r in rows if "pdfium_text" in r), rows[0])
    rows.append({"id": "err_missing_pdf", "pdf": "/nonexistent/x.pdf", "page_index": 0,
                 "bbox": [0, 0, 10, 10], "mediabox": [0, 0, 612, 792]})
    rows.append({"id": "err_bad_page", "pdf": first["pdf"], "page_index": 9999,
                 "bbox": [0, 0, 10, 10], "mediabox": first["mediabox"]})
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.write("{not json\n")
    print(f"wrote {len(rows) + 1} lines to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
