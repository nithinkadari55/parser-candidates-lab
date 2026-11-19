import json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import ALL, load
from texnorm import normalize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(os.environ.get("PROBE_N", "24"))
items = json.load(open(os.path.join(ROOT, "manifests", "timing.json"), encoding="utf-8"))
items = items[::max(1, len(items) // N)][:N]
names = sys.argv[1].split(",") if len(sys.argv) > 1 else ALL
outs = {}
for name in names:
    print("=" * 20, name, flush=True)
    try:
        m = load(name, 4)
    except Exception:
        traceback.print_exc()
        continue
    for attr in ("sess", "enc", "dec", "st", "dy"):
        s = getattr(m, attr, None)
        if s is not None and hasattr(s, "get_inputs"):
            print(attr, "in", [(i.name, i.type, i.shape) for i in s.get_inputs()][:8])
            print(attr, "out", [(o.name, o.type, o.shape) for o in s.get_outputs()][:6])
    tot, em, ok = 0.0, 0, []
    for it in items:
        t = time.perf_counter()
        try:
            p, n, r = m(os.path.join(ROOT, it["image"]))
        except Exception:
            traceback.print_exc()
            break
        dt = time.perf_counter() - t
        tot += dt
        e = normalize(p) == normalize(it["gt"])
        em += e
        ok.append(p)
        print(f"{it['id'][:40]:40s} {dt:6.2f}s tok={n:4d} run={int(r)} exact={int(e)} | {p[:90]}", flush=True)
    outs[name] = ok
    print(f"SUMMARY {name}: n={len(ok)} mean_s={tot / max(len(ok), 1):.2f} exact={em}", flush=True)
if "p2t15_int8" in outs and "p2t15_int8_split" in outs:
    a, b = outs["p2t15_int8"], outs["p2t15_int8_split"]
    print("split identical:", sum(x == y for x, y in zip(a, b)), "/", min(len(a), len(b)))
