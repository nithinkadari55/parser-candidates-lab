import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import difficulty
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = json.load(open(os.path.join(ROOT, "manifests", "unimer.json"), encoding="utf-8"))
A = [x for x in json.load(open(os.path.join(ROOT, "manifests", "arxiv.json"), encoding="utf-8")) if not x.get("lowres")]
rnd = random.Random(42)
out = []
for d in ["simple", "medium", "complex"]:
    for src in (U, A):
        c = [x for x in src if difficulty(x["gt"]) == d]
        out += rnd.sample(c, min(55, len(c)))
json.dump(out, open(os.path.join(ROOT, "manifests", "timing.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("timing items", len(out))
