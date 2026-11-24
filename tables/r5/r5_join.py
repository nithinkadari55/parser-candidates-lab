"""Round 5b: join the extra rows TATR makes out of multi-line cells.

TATR often splits a row whose cells hold several lines into one grid row per line, with the row label as a spanning
cell over them (elsevier-00 p7: "Geographic coordinates" over its S and E lines). The join merges grid row i into
row i-1 when, conservatively:
  every cell that starts at row i is one row high and sits under a cell of the same columns that ends at row i
  (so each can be appended to that cell), no cell in rows i-1 or i is a column header, and either
  (a) "span": some cell covers both rows i-1 and i (the label spanning the continuation), or
  (b) "blank": no cell spans rows i-1 and i, row i's first-column cell is empty while the one above is not,
      every non-empty cell of row i sits under a non-empty cell, and row i has at least one non-empty cell.
Variant "a" applies rule (a) only, "ab" both. Rows are tried top to bottom; a merged row is tried again.
Cells: dicts with r0, r1, c0, c1, bbox, header, text (the r5_tatr / bench_models output with text).

usage: r5_join.py PRED_ROOT OUT_DIR VARIANT SHARD [FIXTURE_JSON]
  reads tatr16r5_<set>_s<SHARD>.jsonl, writes tatr16j<VARIANT>_<set>_s<SHARD>.jsonl"""
import os, sys, json, glob, copy


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _try_merge(cells, i, variant):
    starting = [c for c in cells if c["r0"] == i]
    if not starting:
        return None
    spanning = [c for c in cells if c["r0"] < i < c["r1"]]
    if any(c.get("header") for c in cells if c["r0"] <= i < c["r1"] or c["r0"] <= i - 1 < c["r1"]):
        return None
    partners = []
    for c in starting:
        if c["r1"] != i + 1:
            return None
        above = [p for p in cells if p["r1"] == i and p["c0"] == c["c0"] and p["c1"] == c["c1"]]
        if len(above) != 1:
            return None
        partners.append((c, above[0]))
    if spanning:
        rule = "span"
    else:
        if variant != "ab":
            return None
        first = [(c, p) for c, p in partners if c["c0"] == 0]
        if len(first) != 1 or first[0][0]["text"].strip() or not first[0][1]["text"].strip():
            return None
        filled = [(c, p) for c, p in partners if c["text"].strip()]
        if not filled or any(not p["text"].strip() for c, p in filled):
            return None
        rule = "blank"
    out = []
    drop = set(id(c) for c, _ in partners)
    grow = {id(p): c for c, p in partners}
    for c in cells:
        if id(c) in drop:
            continue
        n = dict(c)
        if id(c) in grow:
            below = grow[id(c)]
            n["r1"] = i + 1
            if below["text"].strip():
                n["text"] = (c["text"] + " " + below["text"]).strip() if c["text"].strip() else below["text"]
            n["bbox"] = _union(c["bbox"], below["bbox"])
        out.append(n)
    for n in out:
        if n["r0"] > i:
            n["r0"] -= 1
        if n["r1"] > i:
            n["r1"] -= 1
    return out, rule


def join_rows(cells, variant):
    """Returns (joined cells, list of applied rules)."""
    cells = [dict(c) for c in cells]
    if variant == "none" or not cells:
        return cells, []
    rules = []
    i = 1
    while i < max(c["r1"] for c in cells):
        merged = _try_merge(cells, i, variant)
        if merged is None:
            i += 1
            continue
        cells, rule = merged
        rules.append(rule)
    return cells, rules


def main():
    root, out_dir, variant, shard = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    fixture = sys.argv[5] if len(sys.argv) > 5 else None
    os.makedirs(out_dir, exist_ok=True)
    fix, counts = [], {}
    for p in sorted(glob.glob(os.path.join(root, "**", f"tatr16r5_*_s{shard}.jsonl"), recursive=True)):
        base = os.path.basename(p)
        setname = base.split("_")[1]
        of = os.path.join(out_dir, base.replace("tatr16r5_", f"tatr16j{variant}_"))
        n = changed = 0
        with open(p, encoding="utf-8") as f, open(of, "w", encoding="utf-8") as g:
            for line in f:
                rec = json.loads(line)
                before = rec.get("cells") or []
                after, rules = join_rows(before, variant)
                n += 1
                if not rules and fixture is not None and sum(1 for t in fix if not t["rules"]) < 10 and 0 < len(before) <= 60:
                    fix.append(dict(id=rec["id"], set=setname, rules=[], before=before, after=after))
                if rules:
                    changed += 1
                    for r in rules: counts[r] = counts.get(r, 0) + 1
                    if fixture is not None and sum(1 for t in fix if t["rules"]) < 40 and len(before) <= 150:
                        fix.append(dict(id=rec["id"], set=setname, rules=rules, before=before, after=after))
                rec["cells"] = after; rec["join_rules"] = rules
                g.write(json.dumps(rec) + "\n")
        print(f"{base}: {changed} of {n} tables joined", flush=True)
    print("rules applied", counts, flush=True)
    if fixture is not None:
        json.dump(dict(variant=variant, source="tables/r5/r5_join.py", tables=fix), open(fixture, "w"))


if __name__ == "__main__":
    main()
