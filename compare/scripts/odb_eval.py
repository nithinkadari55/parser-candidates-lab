"""Score one method's Markdown with the official OmniDocBench end2end evaluator.

usage: odb_eval.py METHOD PRED_MD_DIR ODB_EVAL_REPO OUT_DIR
Adds a `bucket` page attribute to the selected pages, so the official page split reports per bucket.
Formula BLEU is computed here (per bucket) from the official matched + normalised formula pairs.
Only scores leave the job: per-sample files hold ground-truth text (research licence, not redistributed).
"""
import json, os, shutil, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CFG = """end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist]
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: 3
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {gt}
    prediction:
      data_path: {pred}
    match_method: quick_match
    match_workers: 3
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""


def img_of(img_id):
    return img_id if img_id.endswith((".jpg", ".png")) else "_".join(img_id.split("_")[:-1])


def bleu_by_bucket(samples, buckets):
    import evaluate
    bleu = evaluate.load("bleu")
    groups = {}
    for s in samples:
        b = buckets.get(img_of(s.get("img_id", "")))
        gt = s.get("norm_gt") or s.get("gt") or ""
        pr = s.get("norm_pred") or s.get("pred") or ""
        for key in ("ALL", b):
            if key:
                groups.setdefault(key, []).append((gt, pr))
    out = {}
    for k, pairs in groups.items():
        if not any(p.split() for _, p in pairs) or not any(g.split() for g, _ in pairs):
            out[k] = 0.0
            continue
        r = bleu.compute(predictions=[g for g, _ in pairs], references=[p for _, p in pairs])
        out[k] = r["bleu"]
    return out


def main():
    method, pred, repo, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    pred = os.path.abspath(pred)
    work = os.path.abspath(os.path.join(out, method))
    os.makedirs(work, exist_ok=True)
    gt = os.path.join(work, "gt_selected.json")
    n = C.filtered_gt(gt)
    print("gt pages", n, "pred md", len(os.listdir(pred)))
    cfg = os.path.join(work, "cfg.yaml")
    open(cfg, "w").write(CFG.format(gt=gt, pred=pred))
    lp = os.path.join(work, "eval.log")
    with open(lp, "w") as log:
        rc = subprocess.call([sys.executable, "pdf_validation.py", "--config", cfg], cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    lines = open(lp, encoding="utf-8", errors="replace").read().splitlines()
    for line in lines[-120:]:
        print(line[:300])
    print("eval rc", rc)
    save = os.path.basename(pred) + "_quick_match"
    res = {}
    mr = os.path.join(repo, "result", save + "_metric_result.json")
    if os.path.exists(mr):
        res["official"] = json.load(open(mr, encoding="utf-8"))
    buckets = {s["img"]: s["bucket"] for s in C.selection_index()}
    fr = os.path.join(repo, "result", save + "_display_formula_result.json")
    if os.path.exists(fr):
        try:
            res["formula_bleu"] = bleu_by_bucket(json.load(open(fr, encoding="utf-8")), buckets)
        except Exception as e:
            res["formula_bleu_error"] = repr(e)[:300]
    res["rc"] = rc
    res["gt_pages"] = n
    res["pred_files"] = len(os.listdir(pred))
    json.dump(res, open(os.path.join(out, f"{method}-summary.json"), "w", encoding="utf-8"), indent=1, default=str)
    shutil.rmtree(work, ignore_errors=True)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
