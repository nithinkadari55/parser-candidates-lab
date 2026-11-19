"""Merge every job's artifacts into summary.md + summary.json.

usage: aggregate.py ART_DIR OUT_DIR
"""
import glob, json, os, re, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ORDER_CASES = json.load(open(os.path.join(C.ROOT, "data", "order_expected.json"), encoding="utf-8"))
BUCKETS = ["simple", "medium", "complex", "edge"]
METHODS = [("dlrs", "docling.rs (our options)"), ("dlrs_formula", "docling.rs + CodeFormulaV2"), ("pp", "PP-StructureV3 default")]


def jl(pattern):
    rows, seen = [], {}
    for f in sorted(glob.glob(pattern, recursive=True)):
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            k = r.get("img") or r.get("file")
            if k:
                k = (re.sub(r"-\d+\.jsonl$", "", os.path.basename(f)), k)
                if k in seen:
                    rows[seen[k]] = r
                    continue
                seen[k] = len(rows)
            rows.append(r)
    return rows


def last(pattern):
    p = sorted(glob.glob(pattern, recursive=True))
    return p[-1] if p else None


def norm(s):
    return " " + re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", (s or "").lower())).strip() + " "


def score_text(text, chains):
    t = norm(text)
    pos, found, total = {}, 0, 0
    for ci, ch in enumerate(chains):
        for si, sn in enumerate(ch):
            total += 1
            q = norm(sn).strip()
            k = t.find(" " + q + " ")
            if k < 0:
                k = t.find(q)
            if k >= 0:
                pos[(ci, si)] = k
                found += 1
    good = pairs = 0
    for ci, ch in enumerate(chains):
        for a in range(len(ch)):
            for b in range(a + 1, len(ch)):
                if (ci, a) in pos and (ci, b) in pos:
                    pairs += 1
                    good += pos[(ci, a)] < pos[(ci, b)]
    inter = 0
    for c1 in range(len(chains)):
        for c2 in range(c1 + 1, len(chains)):
            p1 = [v for (c, s), v in pos.items() if c == c1]
            p2 = [v for (c, s), v in pos.items() if c == c2]
            if p1 and p2 and not (max(p1) < min(p2) or max(p2) < min(p1)):
                inter = 1
    return dict(found=found, total=total, pair_acc=round(good / pairs, 3) if pairs else 0.0,
                exact=int(found == total and good == pairs and not inter))


def f(x, d=3):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(f(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    art, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    S, md = {}, ["# compare: docling.rs 2.84.0 vs PP-StructureV3 (PaddleOCR 3.7.0)", ""]

    # ODB accuracy (official evaluator)
    ev = {}
    for m, _ in METHODS:
        p = last(os.path.join(art, "**", f"{m}-summary.json"))
        if p:
            ev[m] = json.load(open(p, encoding="utf-8"))
    S["odb_eval"] = ev
    md += ["## OmniDocBench (official end2end evaluator, quick_match)", ""]
    specs = [("Text edit distance (lower better)", "text_block", "Edit_dist"),
             ("Table TEDS (higher better)", "table", "TEDS"), ("Table TEDS structure only", "table", "TEDS_structure_only"),
             ("Table edit distance", "table", "Edit_dist"), ("Display-formula edit distance", "display_formula", "Edit_dist"),
             ("Reading-order edit distance", "reading_order", "Edit_dist")]
    for title, el, metric in specs:
        rows = []
        for m, label in METHODS:
            pg = (((ev.get(m) or {}).get("official") or {}).get(el) or {}).get("page", {}).get(metric, {})
            rows.append([label] + [pg.get("ALL")] + [pg.get(f"bucket: {b}") for b in BUCKETS])
        md += [f"### {title}", "", table(["method", "all"] + BUCKETS, rows), ""]
    rows = []
    for m, label in METHODS:
        b = (ev.get(m) or {}).get("formula_bleu") or {}
        rows.append([label, b.get("ALL")] + [b.get(x) for x in BUCKETS])
    md += ["### Display-formula BLEU (higher better; evaluate.bleu on the official matched pairs)", "",
           table(["method", "all"] + BUCKETS, rows), ""]

    # ODB run stats
    md += ["## OmniDocBench run stats (accuracy shards, 4-vCPU runners)", ""]
    rows = []
    for tag, pat in (("dlrs", "results-odb-*.jsonl"), ("dlrs_formula", "results-odb_formula-*.jsonl"), ("pp", "results-pp-odb-*.jsonl")):
        recs = jl(os.path.join(art, "**", pat))
        pages = [r for r in recs if "img" in r]
        ok = [r for r in pages if "sec" in r]
        rss = [r["peak_rss_mb"] for r in recs if r.get("done") and r.get("peak_rss_mb")]
        loads = [r["meta"]["load_sec"] for r in recs if "meta" in r]
        by = defaultdict(list)
        for r in ok:
            by[r["bucket"]].append(r["sec"])
        S.setdefault("odb_runs", {})[tag] = dict(pages=len(pages), ok=len(ok), errors=[r for r in pages if "error" in r][:10],
                                                  sec_mean={b: statistics.mean(v) for b, v in by.items()}, peak_rss_mb=max(rss) if rss else None,
                                                  meta=next((r["meta"] for r in recs if "meta" in r), None))
        rows.append([tag, len(pages), len(ok), statistics.mean([r["sec"] for r in ok]) if ok else None] +
                    [statistics.mean(by[b]) if by[b] else None for b in BUCKETS] + [max(rss) if rss else None, max(loads) if loads else None])
    md += [table(["stack", "pages", "ok", "s/page", "s simple", "s medium", "s complex", "s edge", "peak RSS MB", "load s"], rows), ""]
    pp_settings = [r.get("model_settings") for r in jl(os.path.join(art, "**", "results-pp-odb-*.jsonl")) if r.get("model_settings")]
    if pp_settings:
        S["pp_model_settings"] = pp_settings[0]
        md += ["PP-StructureV3 model_settings (from its own JSON): `" + json.dumps(pp_settings[0]) + "`", ""]

    # own corpus
    md += ["## docling.rs test-corpus PDFs", ""]
    own = {}
    for tag, pat in (("dlrs", "results-own-*.jsonl"), ("pp", "results-pp-own-*.jsonl")):
        for r in jl(os.path.join(art, "**", pat)):
            if "file" in r:
                own.setdefault(r["file"], {})[tag] = r
    S["own"] = {k: {t: {x: y for x, y in v.items() if x != "page_texts"} for t, v in d.items()} for k, d in own.items()}
    rows = []
    for name in sorted(own):
        d, p = own[name].get("dlrs", {}), own[name].get("pp", {})
        rows.append([name, d.get("pages") or p.get("pages"), d.get("text_layer_words"), d.get("text_f1_vs_layer"), p.get("text_f1_vs_layer"),
                     d.get("sec"), p.get("sec"), "ERR" if "error" in d else "", "ERR" if "error" in p else ""])
    md += [table(["file", "pages", "layer words", "F1 dlrs", "F1 pp", "s dlrs", "s pp", "dlrs err", "pp err"], rows), ""]
    keys_d = ["titles", "headings", "list_items", "nested_lists", "tables", "tables_with_spans", "tables_with_col_header", "tables_with_row_header",
              "pictures", "pictures_captioned", "pictures_classified", "footnotes", "page_headers", "page_footers", "formulas", "formulas_with_text", "items_with_prov", "items_total"]
    keys_p = ["titles", "headings", "lists", "tables", "tables_with_spans", "tables_with_th", "pictures", "captions", "vision_footnotes", "footnotes",
              "page_headers", "page_footers", "formulas", "formulas_with_latex", "blocks_with_bbox", "blocks"]
    tot_d, tot_p = defaultdict(int), defaultdict(int)
    for name in own:
        for k in keys_d:
            tot_d[k] += (own[name].get("dlrs", {}).get("feats") or {}).get(k, 0) or 0
        for k in keys_p:
            v = (own[name].get("pp", {}).get("feats") or {}).get(k, 0)
            tot_p[k] += v if isinstance(v, (int, float)) else 0
    lv_d, lv_p = defaultdict(int), defaultdict(int)
    for name in own:
        for k, v in ((own[name].get("dlrs", {}).get("feats") or {}).get("heading_levels") or {}).items():
            lv_d[str(k)] += v
        for k, v in ((own[name].get("pp", {}).get("feats") or {}).get("md_heading_levels") or {}).items():
            lv_p[str(k)] += v
    S["own_totals"] = dict(dlrs=tot_d, pp=tot_p, dlrs_heading_levels=lv_d, pp_md_heading_levels=lv_p)
    md += ["Totals over all corpus PDFs (structure emitted):", "", "docling.rs: `" + json.dumps(tot_d) + "`", "",
           "docling.rs heading levels (section_header level -> count): `" + json.dumps(lv_d) + "`", "",
           "PP-StructureV3: `" + json.dumps(tot_p) + "`", "", "PP-StructureV3 Markdown heading depth (# count -> count): `" + json.dumps(lv_p) + "`", ""]
    rows = []
    for c in ORDER_CASES:
        r = [c["pdf"], c["page"]]
        for tag in ("dlrs", "pp"):
            pt = ((own.get(c["pdf"]) or {}).get(tag) or {}).get("page_texts") or {}
            s = score_text(pt.get(str(c["page"]), ""), c["chains"]) if pt else None
            S.setdefault("order_cases", []).append(dict(pdf=c["pdf"], page=c["page"], stack=tag, score=s))
            r += [f"{s['found']}/{s['total']}" if s else None, s["pair_acc"] if s else None, s["exact"] if s else None]
        rows.append(r)
    md += ["Reading order on hand-checked pages (snippets found, pair order accuracy, fully correct):", "",
           table(["pdf", "page", "dlrs found", "dlrs pair acc", "dlrs exact", "pp found", "pp pair acc", "pp exact"], rows), ""]

    # office
    md += ["## Office files", ""]
    off = {}
    for tag in ("dlrs", "pp"):
        for r in jl(os.path.join(art, "**", f"office-{tag}.jsonl")):
            off.setdefault(r["file"], {})[tag] = r
    S["office"] = off
    rows = [[n, "ok" if off[n].get("dlrs", {}).get("ok") else ("ERR " + off[n].get("dlrs", {}).get("error", "")[:60] if off[n].get("dlrs") else None),
             "ok" if off[n].get("pp", {}).get("ok") else ("ERR " + off[n].get("pp", {}).get("error", "")[:60] if off[n].get("pp") else None)] for n in sorted(off)]
    md += [table(["file", "docling.rs", "PaddleOCR doc2md"], rows), ""]
    for n in sorted(off):
        if n.startswith("synthetic"):
            md += [f"### {n} markers", ""]
            mk = sorted(set((off[n].get("dlrs", {}).get("markers") or {}).keys()) | set((off[n].get("pp", {}).get("markers") or {}).keys()))
            rows = []
            for m in mk:
                dh = (off[n].get("dlrs", {}).get("markers") or {}).get(m)
                ph = (off[n].get("pp", {}).get("markers") or {}).get(m)
                rows.append([m, json.dumps(dh)[:200] if dh is not None else None, json.dumps(ph)[:200] if ph is not None else None])
            md += [table(["marker", "docling.rs hits (kind/label/layer)", "doc2md lines"], rows), ""]

    # timing
    tm = last(os.path.join(art, "**", "timing.json"))
    if tm:
        T = json.load(open(tm))
        S["timing"] = T
        md += ["## Timing job (one clean runner, sequential)", "", f"cpu={T.get('cpu')} set={T.get('set')}", ""]
        rows = []
        for tag, r in T["runs"].items():
            pdfs = r.get("pdfs") or {}
            rows.append([tag, r.get("rc"), r.get("load_sec"), r.get("warmup_sec"), r.get("odb_sec_per_page_median"), r.get("odb_sec_per_page_mean")] +
                        [pdfs.get(k, {}).get("sec_per_page") for k in ("2206.01062.pdf", "amt_handbook_sample.pdf", "ocr_test.pdf")] + [r.get("peak_rss_mb")])
        md += [table(["run", "rc", "load s", "warm-up s", "ODB s/page median", "ODB s/page mean", "2206.01062 s/page", "amt_handbook s/page", "ocr_test (scan) s/page", "peak RSS MB"], rows), ""]
        md += ["Determinism (same process config run twice, Markdown byte-identical): `" + json.dumps(T.get("determinism")) + "`", "",
               "Model bytes on disk: `" + json.dumps(T.get("model_total_bytes")) + "`", ""]
        rows = [[stack, name, v["bytes"] / 1e6, ", ".join(x for x in v.get("files", []) if not x.startswith("."))[:120]]
                for stack, d in T.get("model_sizes", {}).items() for name, v in d.items()]
        md += [table(["stack", "model", "MB", "files"], rows), ""]

    # onnx / windows / licences
    for label, pat in (("PP-StructureV3 without paddlepaddle (engine=onnxruntime)", "results-pp_onnx*-files-*.jsonl"), ("Windows", "results-*win*.jsonl")):
        recs = jl(os.path.join(art, "**", pat))
        if recs:
            S.setdefault("extra", {})[label] = recs
            md += [f"## {label}", ""] + ["- `" + json.dumps({k: v for k, v in r.items() if k not in ("page_texts", "feats")})[:400] + "`" for r in recs] + [""]
    for f_ in glob.glob(os.path.join(art, "**", "*-errors.txt"), recursive=True):
        md += [f"## {os.path.basename(f_)}", "", "```", open(f_, encoding="utf-8", errors="replace").read()[-3000:], "```", ""]
    for f_ in sorted(glob.glob(os.path.join(art, "**", "licenses-*.json"), recursive=True)):
        L = json.load(open(f_))
        S.setdefault("licenses", {})[os.path.basename(f_)] = L
        md += [f"## {os.path.basename(f_)}", "", table(["model", "licence"], [[k, v.get("license") or v.get("readme_license")] for k, v in L.items()]), ""]
    json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
