import argparse
import json
import os
import sys

import pypdfium2 as pdfium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_pdfmath import page_ctm, page_geometry  # noqa: E402

from pdfmath.detection import find_equations  # noqa: E402
from pdfmath.extraction.pdfminer_backend import extract_pages  # noqa: E402
from pdfmath.units import pt_to_bp  # noqa: E402


def region_to_user(box, ctm):
    a, b, c, d, e, f = ctm
    det = a * d - b * c
    xs, ys = [], []
    for X in (pt_to_bp(box.x0), pt_to_bp(box.x1)):
        for Y in (pt_to_bp(box.y0), pt_to_bp(box.y1)):
            u, v = X - e, Y - f
            xs.append((d * u - c * v) / det)
            ys.append((-b * u + a * v) / det)
    return [min(xs), min(ys), max(xs), max(ys)]


def snap(textpage, user_box, tol=0.5):
    x0, y0, x1, y1 = user_box
    found = []
    text = []
    for i in range(textpage.count_chars()):
        l, b, r, t = textpage.get_charbox(i)
        if r <= l or t <= b:
            continue
        cx, cy = 0.5 * (l + r), 0.5 * (b + t)
        if x0 - tol <= cx <= x1 + tol and y0 - tol <= cy <= y1 + tol:
            found.append((l, b, r, t))
            text.append(textpage.get_text_range(i, 1))
    if not found:
        return None, ""
    return [min(f[0] for f in found), min(f[1] for f in found),
            max(f[2] for f in found), max(f[3] for f in found)], "".join(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--display", type=int, default=6)
    ap.add_argument("--inline", type=int, default=4)
    args = ap.parse_args()

    rows = []
    for pdf_path in args.pdf:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        doc = pdfium.PdfDocument(pdf_path)
        n_pages = min(len(doc), args.max_pages)
        geoms = page_geometry(pdf_path, set(range(n_pages)))
        want = {"display": args.display, "text": args.inline}
        got = {"display": 0, "text": 0}
        fonts = {}
        for page in extract_pages(pdf_path, list(range(1, n_pages + 1))):
            for g in page.glyphs:
                fonts[g.font.base_name] = fonts.get(g.font.base_name, 0) + 1
            pi = page.page - 1
            ctm = page_ctm(*geoms[pi])
            pg = doc[pi]
            mediabox = list(pg.get_mediabox())
            tp = pg.get_textpage()
            for k, r in enumerate(find_equations(page, inline=True)):
                if got[r.style] >= want[r.style]:
                    continue
                user = region_to_user(r.bbox, ctm)
                snapped, text = snap(tp, user)
                if snapped is None:
                    continue
                got[r.style] += 1
                rows.append({"id": f"{stem}_p{pi}_{r.style}{k}", "pdf": os.path.abspath(pdf_path),
                             "page_index": pi, "bbox": [round(v, 3) for v in snapped],
                             "mediabox": mediabox, "ref_bbox_pt": r.bbox.as_list(),
                             "ref_style": r.style, "pdfium_text": text})
            if pi == 0:
                mx0, my0, mx1, my1 = mediabox
                rows.append({"id": f"{stem}_p0_blank", "pdf": os.path.abspath(pdf_path),
                             "page_index": 0, "bbox": [mx0 + 300, my1 - 20, mx0 + 330, my1 - 8],
                             "mediabox": mediabox})
            tp.close()
            pg.close()
        doc.close()
        top = sorted(fonts.items(), key=lambda kv: -kv[1])[:15]
        print(f"{stem}: pages={n_pages} regions={got} fonts={top}", file=sys.stderr)

    first = rows[0]
    rows.append({"id": "err_missing_pdf", "pdf": "/nonexistent/x.pdf", "page_index": 0,
                 "bbox": [0, 0, 10, 10], "mediabox": [0, 0, 612, 792]})
    rows.append({"id": "err_bad_page", "pdf": first["pdf"], "page_index": 9999,
                 "bbox": [0, 0, 10, 10], "mediabox": first["mediabox"]})
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} regions to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
