import argparse
import json
import statistics
import sys

from pdfmath.extraction.pdfminer_backend import extract_pages
from pdfmath.fonts.mathparams import Style
from pdfmath.geometry.bbox import BBox
from pdfmath.latex.serializer import to_latex
from pdfmath.parse.driver import parse_region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    regions = [json.loads(l) for l in open(args.regions, encoding="utf-8") if l.strip()]
    outs = [json.loads(l) for l in open(args.out, encoding="utf-8") if l.strip()]
    assert len(regions) == len(outs), (len(regions), len(outs))
    by_id = {o["id"]: o for o in outs}
    assert all(r["id"] in by_id for r in regions)

    need = {}
    for r in regions:
        if "ref_bbox_pt" in r:
            need.setdefault(r["pdf"], set()).add(r["page_index"] + 1)
    pages = {}
    for pdf, nums in need.items():
        for p in extract_pages(pdf, sorted(nums)):
            pages[(pdf, p.page)] = p

    statuses = {}
    match = total = 0
    for r in regions:
        o = by_id[r["id"]]
        statuses[o["status"]] = statuses.get(o["status"], 0) + 1
        ref = None
        if "ref_bbox_pt" in r:
            page = pages[(r["pdf"], r["page_index"] + 1)]
            style = Style.TEXT if r["ref_style"] == "text" else Style.DISPLAY
            _, tree, _ = parse_region(page, BBox(*r["ref_bbox_pt"]), style)
            ref = to_latex(tree).latex
            total += 1
            match += int(ref == o["latex"])
        print(json.dumps({"id": r["id"], "status": o["status"], "conf": o["confidence"], "ms": o["ms"],
                          "latex": o["latex"], "same_as_pdfmath_own_box": None if ref is None else ref == o["latex"],
                          "pdfium_text": r.get("pdfium_text"), "error": o["error"]}, ensure_ascii=False))
        if ref is not None and ref != o["latex"]:
            print(f"    ref: {ref}")
    ms = [o["ms"] for o in outs if o["status"] == "ok"]
    print(f"statuses={statuses}")
    print(f"latex identical to pdfmath on its own detected box: {match}/{total}")
    if ms:
        ms.sort()
        print(f"ok ms: n={len(ms)} median={statistics.median(ms):.1f} "
              f"p90={ms[int(0.9 * (len(ms) - 1))]:.1f} max={ms[-1]:.1f}")
    expect = {"err_missing_pdf": "error", "err_bad_page": "error"}
    bad = [k for k, v in expect.items() if by_id.get(k, {}).get("status") != v]
    blanks = [r["id"] for r in regions if r["id"].endswith("_blank") and by_id[r["id"]]["status"] != "no_math"]
    if not statuses.get("ok") or bad or blanks:
        print(f"FAIL ok={statuses.get('ok')} bad={bad} blanks_not_no_math={blanks}")
        sys.exit(1)


if __name__ == "__main__":
    main()
