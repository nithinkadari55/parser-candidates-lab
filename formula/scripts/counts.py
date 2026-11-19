import collections, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for mf in ["own12", "unimer", "arxiv", "timing"]:
    it = json.load(open(os.path.join(ROOT, "manifests", mf + ".json"), encoding="utf-8"))
    c = collections.Counter(x["set"] + ("/" + x["kind"] if "kind" in x else "") + ("/lowres" if x.get("lowres") else "") for x in it)
    print(mf, len(it), dict(c))
ax = json.load(open(os.path.join(ROOT, "manifests", "arxiv.json"), encoding="utf-8"))
print("arxiv papers with crops:", len({x["paper"] for x in ax}))
