import json, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rnd = random.Random(11)
A = [x["id"] for x in json.load(open(os.path.join(ROOT, "manifests", "arxiv.json"), encoding="utf-8")) if not x.get("lowres")]
U = json.load(open(os.path.join(ROOT, "manifests", "unimer.json"), encoding="utf-8"))
ids = rnd.sample(A, min(1500, len(A)))
for s in ["unimer_spe", "unimer_cpe", "unimer_sce"]:
    ids += rnd.sample([x["id"] for x in U if x["set"] == s], 500)
json.dump(sorted(ids), open(os.path.join(ROOT, "manifests", "cdm_subset.json"), "w"), indent=0)
print("cdm subset", len(ids))
