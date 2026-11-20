import json
import sys

src, dst = sys.argv[1], sys.argv[2]
rows = json.load(open(src, encoding="utf-8"))
with open(dst, "w", encoding="utf-8") as f:
    for r in rows:
        i = r["img_id"]
        for row in (
            {"id": f"ex/{i}/gt-vs-pred", "gt": r["gt"], "pred": r["pred"]},
            {"id": f"ex/{i}/gt-vs-gt", "gt": r["gt"], "pred": r["gt"], "expect_f1": 1.0},
            {"id": f"ex/{i}/pred-vs-pred", "gt": r["pred"], "pred": r["pred"], "expect_f1": 1.0},
        ):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
