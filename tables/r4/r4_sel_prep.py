"""Gap 15 data: real source PDFs, our own pdfplumber words, answer-key HTML only as the scoring target.

usage: r4_sel_prep.py {pt,brgm,icdar,scitsr}
Writes $TB_WORK/data/sets/r4<src>/: <id>.json (html tokens, cell texts, pdfplumber words in crop pixels, crop box),
<doc>_p<page>.pdf (the single source page) and index.json. Nothing here is committed or uploaded."""
import os, sys, io, re, json, glob, random, subprocess, collections, unicodedata
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from r4_common import *
import numpy as np
import pypdfium2 as pdfium, pdfplumber
from prep_sets import tag, stratify

KDD = "marijanic/table-extraction-scientific-datasets"
HFURL = f"https://huggingface.co/datasets/{KDD}/resolve/main/"
SCITSR_URL = "https://drive.usercontent.google.com/download?id=1qXaJblBg9sbPN0xknWsYls1aGGtlp4ZN&export=download&confirm=t"
TARGET_PT = int(os.environ.get("R4_PT_TABLES", "3000"))
LIMIT = int(os.environ.get("TB_LIMIT") or 0)
NFC = lambda s: unicodedata.normalize("NFC", s)


def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(["bash", "-o", "pipefail", "-c", cmd]).returncode


def img_size(xml):
    try:
        r = ET.parse(xml).getroot(); s = r.find("size")
        return float(s.find("width").text), float(s.find("height").text)
    except Exception:
        return None


def html_tags(toks, texts, im=None, extra=None):
    return tag(dict(html=toks, cells=[[dict(tokens=list(t), bbox=[0, 0, 0, 0]) for t in texts]]), im, extra)


def crop_words(pw, off, crop):
    L, T, R, B = crop; out = []
    for w in pw:
        x0, y0, x1, y1 = w["x0"] - off[0], w["top"] - off[1], w["x1"] - off[0], w["bottom"] - off[1]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if L <= cx <= R and T <= cy <= B:
            out.append(dict(text=w["text"], bbox=[(x0 - L) * S, (y0 - T) * S, (x1 - L) * S, (y1 - T) * S], gi=None, upright=w["upright"]))
    return out


def write_page_pdf(doc, pi, path):
    if os.path.exists(path): return
    new = pdfium.PdfDocument.new(); new.import_pages(doc, [pi]); new.save(path); new.close()


def process_kdd_doc(job):
    """job = (setname, doc, pdf_path, [(tid, page_from_name, html, (imgw, imgh) or None)])"""
    setname, doc, pdf_path, items = job
    out, stats = [], collections.Counter()
    try:
        pdf = pdfium.PdfDocument(pdf_path); plb = pdfplumber.open(pdf_path)
    except Exception as e:
        return out, collections.Counter({"pdf_open_error": len(items)})
    cache = {}

    def page_data(pi):
        if pi not in cache:
            pg = pdf[pi]; pp = plb.pages[pi]
            gray = np.asarray(pg.render(scale=S).to_pil().convert("L"))
            pw = plumber_words(pp)
            off, ink = render_offset(pp, pg, pw, gray, S) if pw else ((0.0, 0.0), 0.0)
            cache[pi] = (pg, pp, gray, pw, off, ink)
        return cache[pi]
    for tid, pno, html, isz in items:
        try:
            toks, texts = html_to_tokens(html); coords = table_coords(html)
            if not coords or not toks: stats["no_html_coords"] += 1; continue
            gtok = set().union(*[tokens(t) for t in texts]) if texts else set()
            best = None
            for pi in dict.fromkeys([pno, pno - 1]):
                if pi < 0 or pi >= len(pdf): continue
                pg, pp, gray, pw, off, ink = page_data(pi)
                wpt, hpt = pg.get_size()
                iw, ih = isz if isz else ((1000.0 * wpt / max(wpt, hpt)), (1000.0 * hpt / max(wpt, hpt)))
                if abs(iw / ih - wpt / hpt) > 0.03 * (wpt / hpt): continue
                sc = iw / wpt
                bb = [coords[0] / sc, coords[1] / sc, coords[2] / sc, coords[3] / sc]
                inside = [w["text"] for w in pw if bb[0] <= (w["x0"] + w["x1"]) / 2 - off[0] <= bb[2] and bb[1] <= (w["top"] + w["bottom"]) / 2 - off[1] <= bb[3]]
                ov = jaccard(set().union(*[tokens(t) for t in inside]) if inside else set(), gtok)
                if best is None or ov > best[0]: best = (ov, pi, bb)
            if best is None: stats["page_size_mismatch"] += 1; continue
            ov, pi, bb = best
            if ov < 0.2: stats["text_overlap_below_0.2"] += 1; continue
            stats["page_idx_" + ("as_named" if pi == pno else "minus1")] += 1
            pg, pp, gray, pw, off, ink = page_data(pi)
            if pg.get_rotation(): stats["rotated_page_skipped"] += 1; continue
            wpt, hpt = pg.get_size()
            crop = [max(0.0, bb[0] - MARGIN), max(0.0, bb[1] - MARGIN), min(wpt, bb[2] + MARGIN), min(hpt, bb[3] + MARGIN)]
            words = crop_words(pw, off, crop)
            if not words: stats["no_words_in_crop"] += 1; continue
            X0, Y0, X1, Y1 = int(round(crop[0] * S)), int(round(crop[1] * S)), int(round(crop[2] * S)), int(round(crop[3] * S))
            from PIL import Image
            im = Image.fromarray(gray[Y0:Y1, X0:X1]).convert("RGB")
            T = html_tags(toks, texts, im)
            fn = re.sub(r"[^A-Za-z0-9._-]", "_", f"{doc}_p{pi}") + ".pdf"
            write_page_pdf(pdf, pi, os.path.join(SETS, setname, fn))
            gt = dict(id=tid, filename=doc, doc=doc, pdf=fn, page_index=pi, crop_pt=crop, scale=S, offset=list(off), ink_fit=ink, text_overlap=ov,
                      html=toks, cells=[dict(text=t, bbox=[0, 0, 0, 0]) for t in texts], words=words, tags=T,
                      words_src="pdfplumber 0.11.10 extract_words(use_text_flow=False, keep_blank_chars=False)")
            json.dump(gt, open(os.path.join(SETS, setname, tid + ".json"), "w", encoding="utf-8"))
            out.append((tid, T["bucket"])); stats["kept"] += 1
        except Exception as e:
            stats["error"] += 1; print("skip", tid, repr(e)[:200], flush=True)
    plb.close(); pdf.close()
    return out, stats


def kdd_items(root, setname, pdf_lookup):
    """html/<doc>_<page>_<k>.html + test/<doc>_<page>.xml -> per-document jobs"""
    by = collections.defaultdict(list)
    for h in sorted(glob.glob(os.path.join(root, "html", "*.html"))):
        stem = NFC(os.path.basename(h)[:-5])
        doc, page, k = stem.rsplit("_", 2)
        xml = os.path.join(root, "test", f"{doc}_{page}.xml")
        if not os.path.exists(xml):
            cand = [x for x in glob.glob(os.path.join(root, "test", "*.xml")) if NFC(os.path.basename(x)) == NFC(f"{doc}_{page}.xml")]
            xml = cand[0] if cand else xml
        tid = setname + "_" + re.sub(r"[^A-Za-z0-9]", "_", unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode())
        by[doc].append((tid, int(page), open(h, encoding="utf-8").read(), img_size(xml) if os.path.exists(xml) else None))
    jobs = []; missing = 0
    for doc, items in sorted(by.items()):
        p = pdf_lookup(doc)
        if p is None: missing += len(items); continue
        jobs.append((setname, doc, p, items))
    print(setname, "documents", len(by), "tables", sum(len(v) for v in by.values()), "tables without a source PDF", missing, flush=True)
    return jobs


def run_jobs(setname, jobs, extra_stats=None):
    os.makedirs(os.path.join(SETS, setname), exist_ok=True)
    if LIMIT: jobs = jobs[: max(1, LIMIT // 2)]
    kept, stats = [], collections.Counter(extra_stats or {})
    with Pool(4) as p:
        for out, st in p.imap_unordered(process_kdd_doc if setname != "r4scitsr" else process_scitsr, jobs, chunksize=1):
            kept += out; stats.update(st)
    ids = sorted(t for t, _ in kept)
    json.dump(ids, open(os.path.join(SETS, setname, "index.json"), "w"))
    json.dump(dict(stats), open(os.path.join(SETS, setname, "prep_stats.json"), "w"))
    print(setname, "kept", len(ids), "buckets", dict(collections.Counter(b for _, b in kept)), "stats", dict(stats), flush=True)


def hf_get(fn, dst):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(KDD, fn, repo_type="dataset", local_dir=dst)


def prep_small(which):
    raw = os.path.join(RAW, "kdd"); os.makedirs(raw, exist_ok=True)
    tgz = hf_get({"brgm": "table-brgm.tar.gz", "icdar": "icdar-2013.tar.gz"}[which], raw)
    top = {"brgm": "brgm", "icdar": "icdar"}[which]
    sh(f"tar -xzf '{tgz}' -C '{raw}' --wildcards '{top}/html/*' '{top}/test/*' '{top}/pdfs/*'")
    root = os.path.join(raw, top)
    pdfs = {NFC(os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(root, "pdfs", "*.pdf"))}
    run_jobs("r4" + which, kdd_items(root, "r4" + which, lambda d: pdfs.get(NFC(d))))


def prep_pt():
    raw = os.path.join(RAW, "kddpt"); os.makedirs(raw, exist_ok=True)
    for fn in ("pubtables/html.tar.gz", "pubtables/test.tar.gz"):
        p = hf_get(fn, raw); sh(f"tar -xzf '{p}' -C '{raw}'")
    by = collections.defaultdict(list)
    for h in glob.glob(os.path.join(raw, "html", "*.html")):
        stem = os.path.basename(h)[:-5]; doc = stem.split("_")[0]
        by[doc].append(h)
    rng = random.Random(42); pool = []
    for doc in sorted(by):
        hs = sorted(by[doc]); rng.shuffle(hs)
        for h in hs[:2]:
            try:
                toks, texts = html_to_tokens(open(h, encoding="utf-8").read())
                pool.append((h, html_tags(toks, texts)))
            except Exception:
                pass
    chosen = stratify(pool, int(TARGET_PT * 1.15), random.Random(42))
    print("pubtables html pool", len(pool), "chosen", len(chosen), dict(collections.Counter(T["bucket"] for _, T in chosen)), flush=True)
    keep = set(h for h, _ in chosen)
    docs = sorted(set(os.path.basename(h).split("_")[0] for h in keep))
    want = set(docs); need = min(len(want), LIMIT) if LIMIT else len(want)
    pdfdir = os.path.join(raw, "pdf"); os.makedirs(pdfdir, exist_ok=True)
    import tarfile, urllib.request, time as _t
    got = {}; seen = 0; t0 = _t.time()
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(urllib.request.Request(HFURL + "pubtables/pdfs.tar.gz", headers={"User-Agent": "table-bench/1.0"}), timeout=600)
            with tarfile.open(fileobj=resp, mode="r|gz", bufsize=1 << 20) as tf:
                for m in tf:
                    if not m.isfile(): continue
                    seen += 1; d = os.path.basename(m.name)[:-4]
                    if d in want and d not in got:
                        p = os.path.join(pdfdir, d + ".pdf")
                        with open(p, "wb") as fo: fo.write(tf.extractfile(m).read())
                        got[d] = p
                        if len(got) >= need: break
                    if seen % 2000 == 0: print("pubtables stream", seen, "members", len(got), "kept", round(_t.time() - t0), "s", flush=True)
            break
        except Exception as e:
            print("pubtables stream error", repr(e)[:200], "attempt", attempt, "kept so far", len(got), flush=True)
            if len(got) >= 0.8 * need: break
    print("pubtables pdf stream: members read", seen, "wanted", len(want), "got", len(got), "seconds", round(_t.time() - t0), flush=True)
    for h in list(glob.glob(os.path.join(raw, "html", "*.html"))):
        if h not in keep: os.remove(h)
    jobs = kdd_items(raw, "r4pt", lambda d: got.get(d))
    run_jobs("r4pt", jobs)
    ids = json.load(open(os.path.join(SETS, "r4pt", "index.json")))
    if len(ids) > TARGET_PT:
        rng = random.Random(7); keep_ids = sorted(rng.sample(ids, TARGET_PT))
        for t in set(ids) - set(keep_ids): os.remove(os.path.join(SETS, "r4pt", t + ".json"))
        json.dump(keep_ids, open(os.path.join(SETS, "r4pt", "index.json"), "w"))
        print("r4pt trimmed to", len(keep_ids), flush=True)


def process_scitsr(job):
    import prep_scitsr
    setname, items = job
    out, stats = [], collections.Counter()
    for sid, pdf_path, st_path, comp in items:
        try:
            cells_in = json.load(open(st_path, encoding="utf-8"))["cells"]
            toks, cells = prep_scitsr.build(cells_in)
            texts = [c["text"] for c in cells]
            pdf = pdfium.PdfDocument(pdf_path); pg = pdf[0]
            with pdfplumber.open(pdf_path) as plb:
                pp = plb.pages[0]; pw = plumber_words(pp)
                if not pw: stats["no_words"] += 1; pdf.close(); continue
                gray = np.asarray(pg.render(scale=S).to_pil().convert("L"))
                off, ink = render_offset(pp, pg, pw, gray, S)
            wpt, hpt = pg.get_size()
            xs0 = [w["x0"] - off[0] for w in pw]; ys0 = [w["top"] - off[1] for w in pw]; xs1 = [w["x1"] - off[0] for w in pw]; ys1 = [w["bottom"] - off[1] for w in pw]
            crop = [max(0.0, min(xs0) - 8), max(0.0, min(ys0) - 8), min(wpt, max(xs1) + 8), min(hpt, max(ys1) + 8)]
            words = crop_words(pw, off, crop)
            X0, Y0, X1, Y1 = [int(round(v * S)) for v in crop]
            from PIL import Image
            im = Image.fromarray(gray[Y0:Y1, X0:X1]).convert("RGB")
            T = html_tags(toks, texts, im, dict(scitsr_comp=comp))
            tid = "r4scitsr_" + re.sub(r"[^A-Za-z0-9]", "_", sid)
            fn = tid + ".pdf"
            import shutil
            shutil.copy(pdf_path, os.path.join(SETS, setname, fn))
            doc = re.split(r"v\d", sid)[0]
            gt = dict(id=tid, filename=sid, doc=doc, pdf=fn, page_index=0, crop_pt=crop, scale=S, offset=list(off), ink_fit=ink,
                      html=toks, cells=[dict(text=t, bbox=[0, 0, 0, 0]) for t in texts], words=words, tags=T,
                      words_src="pdfplumber 0.11.10 extract_words(use_text_flow=False, keep_blank_chars=False)")
            json.dump(gt, open(os.path.join(SETS, setname, tid + ".json"), "w", encoding="utf-8"))
            out.append((tid, T["bucket"])); stats["kept"] += 1; pdf.close()
        except Exception as e:
            stats["error"] += 1; print("skip", sid, repr(e)[:200], flush=True)
    return out, stats


def prep_scitsr_full():
    raw = os.path.join(RAW, "scitsr_full"); os.makedirs(raw, exist_ok=True)
    tgz = os.path.join(raw, "SciTSR.tar.gz")
    if not os.path.exists(tgz):
        assert sh(f"curl -sL --retry 5 --retry-delay 15 -o '{tgz}' '{SCITSR_URL}'") == 0
    print("SciTSR.tar.gz bytes", os.path.getsize(tgz), flush=True)
    sh(f"tar -xzf '{tgz}' -C '{raw}' --wildcards '*test/pdf/*' '*test/structure/*' '*COMP.list' && rm -f '{tgz}'")
    pdfs = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(raw, "**", "test", "pdf", "*.pdf"), recursive=True)}
    sts = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(raw, "**", "test", "structure", "*.json"), recursive=True)}
    comp = set()
    for f in glob.glob(os.path.join(raw, "**", "*COMP.list"), recursive=True):
        comp |= set(l.strip() for l in open(f) if l.strip())
    ids = sorted(set(pdfs) & set(sts))
    print("scitsr test pdf", len(pdfs), "structure", len(sts), "paired", len(ids), "COMP listed", len(comp), "COMP in test", len(comp & set(ids)), flush=True)
    if LIMIT: ids = ids[:LIMIT]
    chunks = [ids[i::40] for i in range(40)]
    jobs = [("r4scitsr", [(s, pdfs[s], sts[s], s in comp) for s in ch]) for ch in chunks if ch]
    os.makedirs(os.path.join(SETS, "r4scitsr"), exist_ok=True)
    run_jobs("r4scitsr", jobs)


if __name__ == "__main__":
    w = sys.argv[1]
    if w == "pt": prep_pt()
    elif w in ("brgm", "icdar"): prep_small(w)
    elif w == "scitsr": prep_scitsr_full()
    import shutil
    shutil.rmtree(RAW, ignore_errors=True)
