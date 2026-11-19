import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "unimer", "x", "UniMER-Test")
os.makedirs(os.path.join(ROOT, "manifests"), exist_ok=True)


def subset(s):
    lines = open(os.path.join(D, s + ".txt"), encoding="utf-8", newline="").read().replace("\r\n", "\n").split("\n")
    out = []
    for im in sorted(os.listdir(os.path.join(D, s))):
        out.append({"id": f"{s}/{im[:-4]}", "set": "unimer_" + s,
                    "image": os.path.relpath(os.path.join(D, s, im), ROOT).replace("\\", "/"), "gt": lines[int(im[:-4])].strip()})
    return out


main = subset("spe") + subset("cpe") + subset("sce")
json.dump(main, open(os.path.join(ROOT, "manifests", "unimer.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
hwe = subset("hwe")
json.dump(hwe, open(os.path.join(ROOT, "manifests", "unimer_hwe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
gt = json.load(open(os.path.join(ROOT, "gt.json"), encoding="utf-8"))
own = [{"id": k, "set": "own12", "image": f"crops/{k}.png", "gt": v} for k, v in gt.items() if not k.startswith("n")]
json.dump(own, open(os.path.join(ROOT, "manifests", "own12.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(len(main), len(hwe), len(own))
