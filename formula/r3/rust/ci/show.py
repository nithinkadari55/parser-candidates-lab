import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out", nargs="+", required=True, help="tool=path")
    args = ap.parse_args()

    expected = []
    with open(args.regions, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                expected.append((json.loads(line)["id"], json.loads(line).get("pdfium_text")))
            except (ValueError, KeyError):
                expected.append((f"line:{n}", None))

    keys = ("id", "tool", "latex", "confidence", "status", "error", "ms")
    failed = False
    results = {}
    for spec in args.out:
        tool, path = spec.split("=", 1)
        rows = load(path)
        ids = [r.get("id") for r in rows]
        bad_keys = [r for r in rows if tuple(r.keys()) != keys or r["tool"] != tool]
        if ids != [e[0] for e in expected]:
            print(f"FAIL {tool}: output ids do not match input ids one-to-one in order", file=sys.stderr)
            failed = True
        if bad_keys:
            print(f"FAIL {tool}: {len(bad_keys)} rows with wrong keys/tool, e.g. {bad_keys[0]}", file=sys.stderr)
            failed = True
        statuses = {}
        for r in rows:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        ok_latex = sum(1 for r in rows if r["status"] == "ok" and r["latex"])
        ms = sorted(r["ms"] for r in rows if r["status"] in ("ok", "picture", "no_math"))
        med = ms[len(ms) // 2] if ms else None
        print(f"{tool}: {len(rows)} rows, statuses={statuses}, ok-with-latex={ok_latex}, median ms={med}")
        if ok_latex == 0:
            print(f"FAIL {tool}: no region produced LaTeX", file=sys.stderr)
            failed = True
        results[tool] = {r.get("id"): r for r in rows}

    for rid, text in expected:
        print(f"\n== {rid}" + (f"   pdfium: {text}" if text else ""))
        for tool, by_id in results.items():
            r = by_id.get(rid)
            if r is None:
                print(f"  {tool:10s} MISSING")
                continue
            conf = "" if r["confidence"] is None else f" conf={r['confidence']:.3f}"
            body = r["latex"] if r["latex"] is not None else (r["error"] or "")
            print(f"  {tool:10s} [{r['status']}{conf} {r['ms']:.2f}ms] {body}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
