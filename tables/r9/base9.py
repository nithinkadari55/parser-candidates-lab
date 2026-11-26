"""Round 9 baselines and the shared inputs. For each table docling found (tf9.py), on its run-time crop:

  words   text-layer pages: docling-pdf 2.84.0 word cells (tables/r5/words, r5words); scans: PP-OCRv6 small on the crop
          (on the turned crop when PP-LCNet turns it), words kept on the upright crop
  turn    text pages: the glyph-angle vote (harness); scans: PP-LCNet_x1_0_doc_ori when label != 0 and confidence
          >= 0.75 (the milestone-14 gate), else none
  TATR    v1.1 fp16 on the turned crop, cells turned back (tables/r8/tatr8.tatr)
  features the milestone-14 selector's 56 features (r3_select.feats on TableFormer and TATR + agreement + n_words)
Scores TableFormer and TATR (score8.score_cells, the run-time words fill the cells).

usage: base9.py SHARD NSHARDS TF_DIR WORDS_BIN
writes $TB_WORK/out/r9/pub/base/s<k>.jsonl (ids, scores, features, timings; no table text: uploaded in clear)
       $TB_WORK/out/r9/bundle/s<k>/ (crop PNGs, words, cells, answer keys: uploaded only encrypted)"""
import os, sys, json, glob, time, gzip, shutil, subprocess, collections, platform, resource
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common9 import *
from PIL import Image
import bench_models as BM
import score8 as S8
import r3_select as R3
import tatr8 as T8
from multiprocessing import Pool

GATE = 0.75


def cpu_name():
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return platform.processor()


def features(tf_cells, tatr_cells, words):
    fa, fb = R3.feats(tf_cells, words), R3.feats(tatr_cells, words)
    x = {"a_" + k: v for k, v in fa.items()}; x.update({"b_" + k: v for k, v in fb.items()})
    x.update({"d_" + k: fb[k] - fa[k] for k in fa})
    x["agree"] = R3.agreement(tf_cells, tatr_cells); x["n_words"] = len(words)
    return x


def _score(job):
    key, gt, words, cells = job
    return key, S8.score_cells(gt, words, cells)


def strip(cs, text=False):
    out = []
    for c in cs:
        d = dict(r0=c["r0"], r1=c["r1"], c0=c["c0"], c1=c["c1"], bbox=[float(v) for v in c["bbox"]], header=bool(c.get("header")))
        if text: d["text"] = c.get("text", "")
        out.append(d)
    return out


def main():
    shard, n, tf_dir, words_bin = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    pubd = os.path.join(R9OUT, "pub", "base"); bdir = os.path.join(R9OUT, "bundle", f"s{shard}")
    os.makedirs(pubd, exist_ok=True); os.makedirs(os.path.join(bdir, "img"), exist_ok=True); os.makedirs(os.path.join(bdir, "gt"), exist_ok=True)
    tf = {}
    for p in glob.glob(os.path.join(tf_dir, "**", "tf_*_s*.jsonl"), recursive=True):
        for r in jl(p): tf[(r["set"], r["id"])] = r
    keys = sorted(k for k in tf if mine(k[1], shard, n))
    print("tf records", len(tf), "mine", len(keys), flush=True)
    jobs = os.path.join(bdir, "..", f"jobs_s{shard}.jsonl"); wdir = os.path.join(WORK, "r9words", f"s{shard}")
    with open(jobs, "w", encoding="utf-8") as f:
        for s, tid in keys:
            r = tf[(s, tid)]
            if not r.get("found"): continue
            g = load_gt(s, tid)
            f.write(json.dumps(dict(set=s, id=tid, pdf=os.path.join(SETS, s, g["pdf"]), page=0, crop_pt=r["crop_pt_rt"])) + "\n")
    t0 = time.time(); subprocess.run([words_bin, jobs, wdir], check=True); print("r5words s", round(time.time() - t0), flush=True)
    M = BM.TATR16(4); lc = T8.Lcnet(); ocr = T8.Ocr()
    cpu = cpu_name(); meta_rows, bundle_rows, score_jobs = [], [], []; nerr = 0
    for k, (s, tid) in enumerate(keys):
        r = tf[(s, tid)]; g = load_gt(s, tid); doc = doc_of(s, g)
        meta = dict(id=tid, set=s, type=table_type(s, g), lang=table_lang(s, g), doc=doc, kind=set_kind9(s), bucket=g["tags"]["bucket"],
                    tags={t: bool(g["tags"].get(t)) for t in S8.SC.TAGS + ["scitsr_comp", "words_aligned", "no_gt_text"] if t in g["tags"]},
                    line_type=g["tags"].get("line_type"), src_meta=g.get("src_meta"), rot=g.get("rot"),
                    gt_rows=g["tags"].get("rows"), gt_cols=g["tags"].get("cols"), n_span=g["tags"].get("n_span"),
                    found=bool(r.get("found")), match_iou=r.get("match_iou"), ms_tf=r.get("ms_tf"), ms_convert=r.get("ms_convert"),
                    tf_err=r.get("err"), has_gt_text=not g["tags"].get("no_gt_text"), cpu=cpu)
        if not r.get("found"):
            meta_rows.append(meta); continue
        try:
            wj = json.load(open(os.path.join(wdir, s, tid + ".json"), encoding="utf-8"))
            if "error" in wj: raise RuntimeError("r5words: " + str(wj["error"]))
            png = os.path.join(wdir, s, tid + ".png")
            im = Image.open(png).convert("RGB"); W, H = im.size; scale = wj["scale"]; cx, cy = r["crop_pt_rt"][0], r["crop_pt_rt"][1]
            tf_cells = [dict(c, bbox=[(c["bbox"][0] - cx) * scale, (c["bbox"][1] - cy) * scale, (c["bbox"][2] - cx) * scale, (c["bbox"][3] - cy) * scale])
                        for c in r["cells_pt"]]
            t1 = time.perf_counter(); lab, conf = lc(im); meta["ms_lcnet"] = (time.perf_counter() - t1) * 1000
            meta.update(lcnet=lab, lcnet_conf=conf, crop_w=W, crop_h=H, scale=scale)
            if set_kind9(s) == "scan":
                turn = lab if (lab and conf >= GATE) else 0
                t1 = time.perf_counter()
                if turn:
                    imT = T8.turn_image(im, turn); words = T8.unturn_words(ocr(imT), turn, W, H)
                else:
                    words = ocr(im)
                meta["ms_ocr"] = (time.perf_counter() - t1) * 1000; meta["turn_src"] = "lcnet" if turn else None
            else:
                words = [dict(text=x["text"], bbox=x["bbox"]) for x in wj["words"]]
                t1 = time.perf_counter(); turn, ng = T8.glyph_turn(os.path.join(SETS, s, g["pdf"]), 0, r["crop_pt_rt"])
                meta.update(n_glyphs=ng, ms_vote=(time.perf_counter() - t1) * 1000, turn_src="glyph" if turn else None)
            meta["turn"] = turn
            t1 = time.perf_counter(); cells, ms_m, ms_t = T8.tatr(M, im, words, turn)
            meta.update(ms_tatr_model=ms_m, ms_tatr=ms_t, n_words=len(words), n_tf_cells=len(tf_cells), n_tatr_cells=len(cells))
            t1 = time.perf_counter(); meta["x"] = features(tf_cells, cells, words); meta["ms_feat"] = (time.perf_counter() - t1) * 1000
            score_jobs.append(((tid, "tf"), g, words, tf_cells)); score_jobs.append(((tid, "tatr"), g, words, cells))
            shutil.copy(png, os.path.join(bdir, "img", tid + ".png"))
            json.dump(g, open(os.path.join(bdir, "gt", tid + ".json"), "w", encoding="utf-8"))
            bundle_rows.append(dict(id=tid, set=s, kind=set_kind9(s), W=W, H=H, scale=scale, turn=turn, words=words,
                                    cells=dict(tf=strip(tf_cells), tatr=strip(cells))))
        except Exception as e:
            nerr += 1; meta["err"] = repr(e)[:300]; print("error", s, tid, meta["err"], flush=True)
        meta_rows.append(meta)
        if k % 250 == 0: print("progress", k, len(keys), flush=True)
    print("inference done; tables", len(meta_rows), "errors", nerr, "score jobs", len(score_jobs), flush=True)
    score_jobs.sort(key=lambda j: -len(j[3]))
    scores = collections.defaultdict(dict)
    with Pool(4) as p:
        for i, (key, res) in enumerate(p.imap_unordered(_score, score_jobs, chunksize=1)):
            scores[key[0]][key[1]] = res
    for m in meta_rows: m["scores"] = scores.get(m["id"], {})
    dump_jsonl(os.path.join(pubd, f"s{shard}.jsonl"), meta_rows)
    with gzip.open(os.path.join(bdir, "rows.jsonl.gz"), "wt", encoding="utf-8") as f:
        for r in bundle_rows: f.write(json.dumps(r) + "\n")
    json.dump(dict(shard=shard, tables=len(meta_rows), found=sum(m["found"] for m in meta_rows), errors=nerr, cpu=cpu,
                   peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), open(os.path.join(pubd, f"summary_s{shard}.json"), "w"))
    print("shard", shard, "tables", len(meta_rows), "errors", nerr, flush=True)
    if meta_rows and nerr > 0.02 * max(1, sum(m["found"] for m in meta_rows)):
        sys.exit(f"too many errors: {nerr}")


if __name__ == "__main__":
    main()
