"""Model licences from the downloaded files themselves.

PP: each ~/.paddlex/official_models/<name>/ is a Hugging Face snapshot of PaddlePaddle/<name>; read the
`license:` field of its README.md front matter and any LICENSE file. docling.rs: the release's models.json.
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def pp():
    root = os.path.join(C.HOME, ".paddlex", "official_models")
    out = {}
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, name)
        rec = dict(files=sorted(f for f in os.listdir(d)))
        rm = os.path.join(d, "README.md")
        if os.path.exists(rm):
            txt = open(rm, encoding="utf-8", errors="replace").read()
            m = re.match(r"---\s*\n(.*?)\n---", txt, re.S)
            fm = m.group(1) if m else ""
            lic = re.search(r"(?m)^license:\s*(.+)$", fm)
            rec["readme_license"] = lic.group(1).strip() if lic else None
        for f in os.listdir(d):
            if f.upper().startswith("LICENSE"):
                rec["license_file_head"] = open(os.path.join(d, f), encoding="utf-8", errors="replace").read(200)
        out[name] = rec
    return out


def dlrs():
    p = os.path.join(C.DLRS_MODELS, "records", "models.json")
    if not os.path.exists(p):
        return {}
    m = json.load(open(p, encoding="utf-8"))
    return {a["id"]: dict(name=a.get("name"), license=a.get("license"), upstream=(a.get("upstream") or {}).get("repository"),
                          revision=(a.get("upstream") or {}).get("revision")) for a in m.get("artifacts", [])}


if __name__ == "__main__":
    what, out = sys.argv[1], sys.argv[2]
    json.dump(globals()[what](), open(out, "w"), indent=1)
    print(open(out).read()[:4000])
