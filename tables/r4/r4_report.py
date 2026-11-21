"""Assemble tables/r4/REPORT.md from the r4-sel, r4-miss and r4-xpage artifacts. usage: r4_report.py ROOT RUN_ID"""
import os, sys, json, glob


def find(root, name):
    m = glob.glob(os.path.join(root, "**", name), recursive=True)
    return m[0] if m else None


def load(root, name, js=False):
    p = find(root, name)
    if not p: return None
    return json.load(open(p)) if js else open(p, encoding="utf-8").read()


def main():
    root, run = sys.argv[1], sys.argv[2]
    sel, miss, xp, lay = load(root, "sel.md"), load(root, "miss.md"), load(root, "xpage.md"), load(root, "layout.md")
    ls = load(root, "layout_summary.json", True)
    pp, ps = load(root, "pp.md"), load(root, "pp_summary.json", True)
    ss, ms, xs = load(root, "sel_summary.json", True), load(root, "miss_summary.json", True), load(root, "xpage_summary.json", True)
    V = ["# Round 4 table tests (gaps 15, 14, 6)", "", f"GitHub Actions run {run} in nithinkadari55/parser-candidates-lab (workflow tables-r4.yml). "
         "Evaluated artifacts may come from several runs (see the report job's extra_runs input). All model work ran on 4-vCPU ubuntu-latest runners; datasets were downloaded inside the jobs and never committed or uploaded.", "", "## Verdicts", ""]
    if ss:
        lines = []
        for k, v in ss.items():
            g = v.get("gain")
            if not g: continue
            w = "wins" if g[1] > 0 else "loses" if g[2] < 0 else "no significant difference"
            lines.append(f"{k}: {w} ({g[0]:+.4f}, CI [{g[1]:+.4f}, {g[2]:+.4f}] vs best single = {'TableFormer' if v['best'] == 'va' else 'TATR'})")
        allg = ss.get("all", {}).get("gain")
        verdict = "yes" if allg and allg[1] > 0 else "no" if allg and allg[2] < 0 else "not significantly"
        V.append(f"- **Gap 15 (selector on real PDF words): does the selector still win? {verdict}.** Retrained selector vs the better single model per dataset: " + "; ".join(lines) + ".")
    if ms:
        best = None
        for ds in ("all",):
            for k, v in ms.get(ds, {}).get("variants", {}).items():
                score = v["rec5"] / max(1, ms[ds]["misses"])
                if v["prec5"] >= 0.5 and v["false100"] <= 5 and (best is None or score > best[0]): best = (score, k, v)
        if best:
            V.append(f"- **Gap 14 (missed tables):** best proposer meeting precision >= 0.5 and <= 5 false tables / 100 pages: {best[1]} recovers {best[2]['rec5']} of "
                     f"{ms['all']['misses']} heron misses (incremental recall {best[0]:.3f}), precision {best[2]['prec5']:.3f}, {best[2]['false100']:.1f} false tables / 100 pages.")
        else:
            V.append("- **Gap 14 (missed tables): no proposer reaches precision >= 0.5 with <= 5 false tables / 100 pages; do not add it to the pipeline as is** (see the per-variant table).")
    if xs:
        r = xs.get("r4", {}); t = xs.get("textract (accuracy 98, default)", {}); s = xs.get("stitcher", {})
        V.append(f"- **Gap 6 (cross-page tables):** our rule set link P {r.get('P', 0):.3f} / R {r.get('R', 0):.3f}, false-merge rate {r.get('false_merge', 0):.4f}, exact multi-part assembly "
                 f"{r.get('exact_multi', 0):.3f}; Table Stitcher port P {s.get('P', 0):.3f} / R {s.get('R', 0):.3f}; Textract rules P {t.get('P', 0):.3f} / R {t.get('R', 0):.3f}. "
                 f"Region splitting: {json.dumps(xs.get('split', {}))}.")
    if ls:
        parts = []
        for ds, v in ls.items():
            d = v["f1_diff"]
            arm = v.get("arm", {}); a = arm.get("RF-DETR-Large|0.5|uncovered regions"); y = arm.get("YOLOX-L|0.5|uncovered regions")
            parts.append(f"{ds}: mean-class mAP50-95 RF-DETR {v['rf_map']:.3f} vs heron INT8 {v['heron_map']:.3f}, paired page F1 diff {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}], "
                         f"{v['rf_ms']:.0f} vs {v['heron_ms']:.0f} ms/page" + (f"; missed tables (uncovered regions, >= 0.5): RF-DETR recovers {a['rec']} of {v['misses']} (precision {a['prec']:.3f}, {a['false100']:.1f} false/100 pages)" if a else "")
                         + (f", YOLOX-L {y['rec']} (precision {y['prec']:.3f}, {y['false100']:.1f} false/100 pages)" if y else ""))
        V.append("- **Layout: RF-DETR-Large (test only, not a replacement):** " + "; ".join(parts) + ".")
    if ps:
        parts = []
        for ds, v in ps.items():
            d = v["f1_diff"]; a = v.get("arm", {}).get("PP-DocLayout_plus-L|0.5|uncovered regions")
            parts.append(f"{ds}: shared-class mAP50-95 PP {v['map'].get('pp', 0):.3f} vs heron {v['map'].get('heron', 0):.3f}, paired page F1 diff {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}], "
                         f"{v['ms'].get('pp', 0):.0f} vs {v['ms'].get('heron', 0):.0f} ms/page" + (f"; missed tables (uncovered, >= 0.5): {a['rec']} of {v['misses']} recovered, precision {a['prec']:.3f}, {a['false100']:.1f} false/100 pages" if a else ""))
        V.append("- **Layout: PP-DocLayout_plus-L (test only, OmniDocBench decides):** " + "; ".join(parts) + ".")
    V += ["", "## Sources checked from the actual files", "",
          "- SPARTAN: the real repository is github.com/shlok-n/spartan (MIT, commit 66ada02, v2.2.0). github.com/IvoLeist/spartan-table-extraction is a GitHub fork of it with the same commits. Called unmodified (detect_tables + extract_pdf_words, 300 DPI, OCR off).",
          "- pdf-inspector: github.com/firecrawl/pdf-inspector (MIT, v1.27.0). Its Python wheel exposes only pages_with_tables, not table boxes, so the line/rect proposal idea (detect_lines.rs, detect_rects.rs constants) was ported to Python on pdfplumber paths.",
          "- Table Stitcher: github.com/PebbleRoad/table-stitcher (MIT). AWS rules: aws-samples/amazon-textract-response-parser t_tables.py (Apache-2.0).",
          "- KDD 2026 bundle: HF marijanic/table-extraction-scientific-datasets (card MIT; source PDFs keep their own terms). PubTables subset: 23,567 PDFs (35 GB tar, streamed); Table-BRGM 26 PDFs; ICDAR-2013 (corrected) 67 PDFs.",
          "- SciTSR: Academic-Hammer/SciTSR (MIT code); data from the README Google Drive link (SciTSR.tar.gz, 802 MB), test/pdf + test/structure; arXiv PDFs keep their own terms.",
          "- FinTabNet.c: HF bsmock/FinTabNet.c (CDLA-Permissive-2.0) ships only JSON annotations (no page PDFs); the original IBM FinTabNet download (dax-cdn.cdn.appdomain.cloud) did not respond, so FinTabNet was not tested.",
          "- DocLayNet v1.2 test (CDLA-Permissive-1.0) carries the page PDFs; heron INT8 detections and GT from order run 44477997398 (joined by page hash).",
          "- OmniDocBench: original PDF pages only on the v1_0 branch (ori_pdfs, 981 pages); README: research purposes only, not commercial; used for internal testing only.",
          "- SCI-3000: Zenodo 8165783, open access; record licence CC BY 4.0 while the LICENSE file inside the zip says CC BY-SA 4.0. Both allow research use. Read by HTTP range requests (no full download).",
          "- PubTables-v2 Full Documents test: HF kensho/PubTables-v2, CDLA-Permissive-2.0.",
          "- RF-DETR-Large DocLayNet: HF nie3e/rf-detr-large-doclay @fecf537, model card licence apache-2.0 (no separate LICENSE file in the model repo), base Roboflow/rf-detr-large card apache-2.0, roboflow/rf-detr code Apache-2.0; trained on DocLayNet v1.2 (CDLA-Permissive-1.0). Run through transformers 5.22.2 RfDetrForObjectDetection (code read: sigmoid + top-k post-processing, no NMS).", ""]
    for part in (sel, miss, xp, lay, pp):
        if part: V += [part, ""]
    txt = "\n".join(V)
    open(os.path.join(root, "REPORT.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
