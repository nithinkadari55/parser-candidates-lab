import json, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "unimer", "x", "UniMER-Test")
out = []
for s, n in [("spe", 300), ("cpe", 300), ("sce", 100)]:
    lines = open(os.path.join(D, s + ".txt"), encoding="utf-8", newline="").read().replace("\r\n", "\n").split("\n")
    ims = sorted(os.listdir(os.path.join(D, s)))
    rnd = random.Random(20261118)
    pick = sorted(rnd.sample(ims, n))
    for im in pick:
        out.append({"id": f"{s}/{im[:-4]}", "set": "unimer_" + s, "image": os.path.relpath(os.path.join(D, s, im), ROOT).replace("\\", "/"), "gt": lines[int(im[:-4])].strip()})
json.dump(out, open(os.path.join(ROOT, "manifests", "unimer.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
gt = json.load(open(os.path.join(ROOT, "gt.json"), encoding="utf-8"))
own = [{"id": k, "set": "own12", "image": f"crops/{k}.png", "gt": v} for k, v in gt.items() if not k.startswith("n")]
json.dump(own, open(os.path.join(ROOT, "manifests", "own12.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(len(out), len(own))
