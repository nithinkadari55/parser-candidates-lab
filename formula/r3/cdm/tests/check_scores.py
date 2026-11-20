import json
import sys

pairs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
scores = {}
for l in open(sys.argv[2], encoding="utf-8"):
    r = json.loads(l)
    scores[r["id"]] = r

fails = []
print(f"{'id':42} {'status':17} {'cdm_f1':>7} {'recall':>7} {'prec':>7}  check")
for p in pairs:
    s = scores.get(p["id"])
    if s is None:
        fails.append(f"{p['id']}: no output row")
        print(f"{p['id']:42} MISSING")
        continue
    check = ""
    if "expect_f1" in p:
        ok = s["status"] == "ok" and s["cdm_f1"] == p["expect_f1"]
        check = f"f1=={p['expect_f1']} {'PASS' if ok else 'FAIL'}"
    elif "expect_f1_lt" in p:
        ok = s["status"] == "ok" and s["cdm_f1"] < p["expect_f1_lt"]
        check = f"f1<{p['expect_f1_lt']} {'PASS' if ok else 'FAIL'}"
    elif "expect_status" in p:
        ok = s["status"] == p["expect_status"]
        check = f"status=={p['expect_status']} {'PASS' if ok else 'FAIL'}"
    else:
        ok = True
    if not ok:
        fails.append(f"{p['id']}: {check} got {s}")
    fmt = lambda v: "None" if v is None else f"{v:.3f}"
    print(f"{p['id']:42} {s['status']:17} {fmt(s['cdm_f1']):>7} {fmt(s['recall']):>7} {fmt(s['precision']):>7}  {check}")
    if s.get("error"):
        print(f"{'':42} error: {s['error'][:200]}")
extra = set(scores) - {p["id"] for p in pairs}
if extra:
    fails.append(f"unexpected ids in output: {sorted(extra)}")
if fails:
    print("\nCHECK FAILURES:")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print(f"\nall {len(pairs)} rows present; all expectations met")
