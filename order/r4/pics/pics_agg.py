"""Gap 10 report section. usage: pics_agg.py ARTDIR OUT.md"""
import glob
import json
import os
import sys
from collections import Counter

from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pics_data import CLASSES  # noqa: E402


def main():
    art, out = sys.argv[1], sys.argv[2]
    R = {}
    for f in glob.glob(os.path.join(art, "**", "pics-*.json"), recursive=True):
        d = json.load(open(f))
        R[d["kind"]] = d
    man = glob.glob(os.path.join(art, "**", "pics-manifest-summary.json"), recursive=True)
    L = ["# Round 4, gap 10: picture classification", ""]
    if man:
        s = json.load(open(man[0]))
        L.append("Data sources (crops kept per split and class): " + json.dumps(s.get("counts", {})))
        L.append("")
        L.append("Source status: " + "; ".join(f"{k}: {v}" for k, v in s.get("status", {}).items()))
        L.append("")
    rows = []
    for k in ("mnv3", "dinov2"):
        if k in R:
            for v in ("sklearn", "onnx_fp32", "onnx_int8"):
                if v in R[k]:
                    rows.append((f"{k} head ({v})", R[k][v], R[k][v].get("mb"), R[k][v].get("ms_p50")))
    if "docling_v2.5" in R:
        d = R["docling_v2.5"]
        rows.append(("docling DocumentFigureClassifier-v2.5", d, d["mb"], d["ms_p50"]))
    L.append("| model | macro-F1 | " + " | ".join(CLASSES) + " | MB | ms/crop (1 thread) |")
    L.append("|---|---|" + "---|" * len(CLASSES) + "---|---|")
    for name, d, mb, ms in rows:
        rep = d["report"]
        L.append(f"| {name} | {d['macro_f1']:.3f} | " + " | ".join(f"{rep[c]['f1-score']:.2f}" for c in CLASSES)
                 + f" | {'-' if mb is None else f'{mb:.1f}'} | {'-' if ms is None else f'{ms:.1f}'} |")
    L.append("")
    if "docling_v2.5" in R:
        d = R["docling_v2.5"]
        lab = dict(zip(d["test_ids"], d["labels"]))
        dp = dict(zip(d["test_ids"], d["pred"]))
        for k in ("mnv3", "dinov2"):
            if k not in R or "onnx_int8" not in R[k]:
                continue
            for v in ("onnx_fp32", "onnx_int8"):
                op = dict(zip(R[k]["test_ids"], R[k][v]["pred"]))
                ids = [i for i in op if i in dp]
                w = sum(1 for i in ids if op[i] == lab[i] and dp[i] != lab[i])
                l_ = sum(1 for i in ids if dp[i] == lab[i] and op[i] != lab[i])
                p = binomtest(w, w + l_).pvalue if w + l_ else 1.0
                L.append(f"- {k} {v} vs docling on {len(ids)} crops: ours right / docling wrong {w}, reverse {l_}, "
                         f"sign test p = {p:.2g}")
        L.append("")
        srcs = Counter()
        if "mnv3" in R:
            for s_, i in zip(R["mnv3"]["test_src"], R["mnv3"]["test_ids"]):
                srcs[s_] += 1
            L.append("Test crops by source: " + ", ".join(f"{k} {v}" for k, v in srcs.most_common()))
            L.append("")
        L.append("DocLayNet picture crops (no labels; domain check only) - predicted class shares:")
        L.append("")
        L.append("| model | " + " | ".join(CLASSES) + " |")
        L.append("|---|" + "---|" * len(CLASSES))
        for name, preds in [("docling", d["domain_pred"])] + [(f"{k} int8", R[k]["onnx_int8"]["domain_pred"])
                                                               for k in ("mnv3", "dinov2") if k in R and "onnx_int8" in R[k]]:
            c = Counter(preds)
            n = max(1, len(preds))
            L.append(f"| {name} | " + " | ".join(f"{c[i] / n:.2f}" for i in range(len(CLASSES))) + " |")
        L.append("")
        L.append("docling raw 26-class predictions on the test crops: " + json.dumps(d.get("raw26", {})))
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
