import json
import sys
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def merge(fields):
    out = {}
    for f in fields:
        name = f.get("name") or ""
        m = out.setdefault(name, {"name": name, "value": None, "widgets": [], "export": None, "type": None})
        if m["value"] in (None, "") and f.get("value") not in (None, ""):
            m["value"] = f.get("value")
        if m["type"] is None:
            m["type"] = f.get("type")
        if m["export"] is None:
            m["export"] = f.get("export")
        for w in f.get("widgets") or []:
            if w not in m["widgets"]:
                m["widgets"].append(w)
    return out


def value_ok(gt, ft, got):
    if gt is None or gt == "":
        if got in (None, "", False):
            return True
        return ft == "Btn" and got == "Off"
    if isinstance(got, bool):
        return got == (gt != "Off")
    if isinstance(gt, list):
        return isinstance(got, list) and got == gt
    return got == gt


def rect_eq(a, b):
    return a is not None and b is not None and all(abs(x - y) <= 0.6 for x, y in zip(a, b))


def nonascii(v):
    s = json.dumps(v, ensure_ascii=False) if v is not None else ""
    return any(ord(c) > 127 for c in s)


def score(gt_fields, res):
    got = merge(res.get("fields", []))
    provides_as = any(w.get("as") is not None for f in res.get("fields", []) for w in f.get("widgets") or [])
    gt_names = {f["name"] for f in gt_fields}
    s = {"total": len(gt_fields), "found": 0, "value": 0, "page_n": 0, "page": 0, "rect_n": 0, "rect": 0, "as_n": 0, "as": 0,
         "na_n": 0, "na": 0, "exp_n": 0, "exp": 0, "extra": len([n for n in got if n not in gt_names]), "fails": []}
    for f in gt_fields:
        g = got.get(f["name"])
        if f["value"] is not None and nonascii(f["value"]) or nonascii(f["name"]):
            s["na_n"] += 1
        if g is None:
            s["fails"].append(f"missing {f['name']!r}")
            continue
        s["found"] += 1
        ok = value_ok(f["value"], f["type"], g["value"])
        if ok:
            s["value"] += 1
            if f["value"] is not None and nonascii(f["value"]) or nonascii(f["name"]):
                s["na"] += 1
        else:
            s["fails"].append(f"value {f['name']!r}: want {f['value']!r} got {g['value']!r}")
        if f.get("export") is not None:
            s["exp_n"] += 1
            if g.get("export") == f["export"]:
                s["exp"] += 1
        gw = g["widgets"]
        for w in f["widgets"]:
            match = [x for x in gw if rect_eq(w["rect"], x.get("rect"))]
            if any(x.get("rect") is not None for x in gw):
                s["rect_n"] += 1
                if match:
                    s["rect"] += 1
                else:
                    s["fails"].append(f"rect {f['name']!r}: want {w['rect']} got {[x.get('rect') for x in gw]}")
            if any(x.get("page") is not None for x in gw):
                s["page_n"] += 1
                cand = match or gw
                if any(x.get("page") == w["page"] for x in cand):
                    s["page"] += 1
                else:
                    s["fails"].append(f"page {f['name']!r}: want {w['page']} got {[x.get('page') for x in cand]}")
            if w["as"] is not None and provides_as:
                s["as_n"] += 1
                cand = match or gw
                if any(x.get("as") == w["as"] for x in cand):
                    s["as"] += 1
                else:
                    s["fails"].append(f"AS {f['name']!r}: want {w['as']!r} got {[x.get('as') for x in cand]}")
    return s


def frac(a, n):
    return f"{a}/{n}" if n else "-"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    gt_dir = Path(sys.argv[1])
    res_dir = Path(sys.argv[2])
    print("| tool | os | file | status | found | value | non-ASCII | page | rect | AS | export | extra | ms(median) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    fails = []
    for rp in sorted(res_dir.rglob("*_forms*.json")):
        try:
            rep = load(rp)
        except Exception as e:
            print(f"| {rp.name} | | | unreadable: {e} |")
            continue
        for r in rep["results"]:
            gt = load(gt_dir / (Path(r["file"]).stem + ".json"))["fields"]
            if r["status"] != "ok":
                print(f"| {rep['tool']} | {rep['os']} | {r['file']} | {r['status']}: {str(r.get('error'))[:80]} | 0/{len(gt)} | | | | | | | | |")
                continue
            s = score(gt, r)
            print(f"| {rep['tool']} | {rep['os']} | {r['file']} | ok | {s['found']}/{s['total']} | {s['value']}/{s['total']} | {frac(s['na'], s['na_n'])} | {frac(s['page'], s['page_n'])} | {frac(s['rect'], s['rect_n'])} | {frac(s['as'], s['as_n'])} | {frac(s['exp'], s['exp_n'])} | {s['extra']} | {r.get('ms_median', 0):.2f} |")
            for x in s["fails"]:
                fails.append(f"{rep['tool']} {rep['os']} {r['file']}: {x}")
    print()
    for x in fails:
        print(x)
