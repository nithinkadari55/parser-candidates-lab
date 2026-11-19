import os, json, random
from PIL import Image
from paths import SETS as SETDIR
from prep_sets import bucket

PER = 150


def rot_box(b, W):
    return [b[1], W - b[2], b[3], W - b[0]]


def main():
    rng = random.Random(7)
    out = os.path.join(SETDIR, "rot"); os.makedirs(out, exist_ok=True)
    index = []
    for s in ("ptn", "ftn"):
        ids = json.load(open(os.path.join(SETDIR, s, "index.json")))
        rng.shuffle(ids)
        for tid in ids[:PER]:
            gt = json.load(open(os.path.join(SETDIR, s, tid + ".json"), encoding="utf-8"))
            im = Image.open(os.path.join(SETDIR, s, tid + ".png")).convert("RGB")
            W = im.size[0]
            for c in gt["cells"]:
                if c.get("bbox") and any(c["bbox"]): c["bbox"] = rot_box(c["bbox"], W)
            T = dict(gt["tags"]); T["rotated"] = True; T["bucket"] = bucket(T)
            nid = "rot_" + tid.replace("_", "")
            gt.update(id=nid, tags=T, rot=90, source=tid)
            im.rotate(90, expand=True).save(os.path.join(out, nid + ".png"))
            json.dump(gt, open(os.path.join(out, nid + ".json"), "w", encoding="utf-8"))
            index.append(nid)
    json.dump(sorted(index), open(os.path.join(out, "index.json"), "w"))
    print("rot", len(index))


if __name__ == "__main__":
    main()
