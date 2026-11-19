import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import difficulty
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for mf in ["own12", "unimer", "unimer_hwe", "arxiv", "timing"]:
    p = os.path.join(ROOT, "manifests", mf + ".json")
    if not os.path.exists(p):
        continue
    it = json.load(open(p, encoding="utf-8"))
    c = collections.Counter(x["set"] + ("/" + x["kind"] if "kind" in x else "") + (f"/lowres{x['lowres']}" if x.get("lowres") else "") for x in it)
    d = collections.Counter(difficulty(x["gt"]) for x in it)
    print(mf, len(it), dict(c), dict(d))
ax = json.load(open(os.path.join(ROOT, "manifests", "arxiv.json"), encoding="utf-8"))
print("arxiv papers with crops:", len({x["paper"] for x in ax}), "unique formulas:", len({(x["paper"], x["gt"]) for x in ax}))
print("cdm subset:", len(json.load(open(os.path.join(ROOT, "manifests", "cdm_subset.json")))))
