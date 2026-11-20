import argparse
import json
import logging
import math
import multiprocessing as mp
import multiprocessing.connection as mpc
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
from collections import deque

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
STATUSES = ("ok", "gt_render_fail", "pred_render_fail", "error")
MAX_ITER, MIN_SAMPLES, RESIDUAL_THRESHOLD, MAX_TRIALS = 5, 2, 30, 500


class _ErrorLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if "ERROR" in msg:
            self.messages.append(msg)


def _row(sid_id, status, f1=None, recall=None, precision=None, error=None):
    return {"id": sid_id, "cdm_f1": f1, "recall": recall, "precision": precision, "status": status, "error": error}


def _boxes(side_dir, sid):
    path = os.path.join(side_dir, "bbox", sid + ".jsonl")
    if not os.path.exists(path):
        return None
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if json.loads(line).get("bbox"):
                n += 1
    return n


def _render(latex2bbox_color, colors, latex, sid, root, side, handler):
    side_dir = os.path.join(root, side)
    tmp = os.path.join(root, "tmp_" + side)
    for d in (os.path.join(side_dir, "bbox"), os.path.join(side_dir, "vis"), tmp):
        os.makedirs(d, exist_ok=True)
    handler.messages.clear()
    latex2bbox_color((latex, sid, side_dir, tmp, colors))
    n = _boxes(side_dir, sid)
    if n is None:
        why = handler.messages[-1] if handler.messages else "no bbox output"
        return side_dir, f"{side} render failed: {why}"
    if n == 0:
        return side_dir, f"{side} rendered no visible token boxes"
    return side_dir, None


def score_one(item, work_root):
    from cdm.cdm import _get_total_color_list, process_single_image
    from cdm.modules.latex2bbox_color import latex2bbox_color

    sid, orig_id, gt, pred = item
    root = os.path.join(work_root, sid)
    handler = _ErrorLog()
    logger = logging.getLogger()
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    try:
        colors = _get_total_color_list()
        gt_dir, gt_err = _render(latex2bbox_color, colors, gt, sid, root, "gt", handler)
        if gt_err:
            return _row(orig_id, "gt_render_fail", error=gt_err)
        pred_dir, pred_err = _render(latex2bbox_color, colors, pred, sid, root, "pred", handler)
        if pred_err:
            return _row(orig_id, "pred_render_fail", 0.0, 0.0, 0.0, pred_err)
        res = process_single_image(
            (sid, gt_dir, pred_dir, os.path.join(root, "vis_match"), MAX_ITER, MIN_SAMPLES, RESIDUAL_THRESHOLD, MAX_TRIALS),
            save_vis=False,
        )
        if res is None:
            return _row(orig_id, "error", error="process_single_image returned None after gt rendered")
        m = res[1]
        vals = [float(m["F1_score"]), float(m["recall"]), float(m["precision"])]
        if not all(math.isfinite(v) for v in vals):
            return _row(orig_id, "error", error=f"non-finite metric {m}")
        return _row(orig_id, "ok", *vals)
    except Exception as e:
        return _row(orig_id, "error", error=f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _child(conn, item, work_root):
    try:
        out = score_one(item, work_root)
    except BaseException as e:
        out = _row(item[1], "error", error=f"{type(e).__name__}: {e}")
    conn.send(out)
    conn.close()


def _descendants(pid):
    children = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except Exception:
            continue
        children.setdefault(ppid, []).append(int(p))
    out, stack = [], [pid]
    while stack:
        for c in children.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


def _kill_tree(pid):
    for p in _descendants(pid) + [pid]:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_all(items, workers, work_root, sample_timeout):
    ctx = mp.get_context("fork")
    pending = deque(items)
    running = {}
    results = {}
    while pending or running:
        while pending and len(running) < workers:
            item = pending.popleft()
            r, w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child, args=(w, item, work_root))
            p.start()
            w.close()
            running[r] = (p, item, time.monotonic())
        ready = mpc.wait(list(running), timeout=1.0)
        for r in ready:
            p, item, _ = running.pop(r)
            try:
                results[item[0]] = r.recv()
            except EOFError:
                p.join(5)
                results[item[0]] = _row(item[1], "error", error=f"worker exited without result (exitcode {p.exitcode})")
            r.close()
            p.join(30)
        now = time.monotonic()
        for r, (p, item, t0) in list(running.items()):
            if now - t0 > sample_timeout:
                _kill_tree(p.pid)
                p.join(5)
                running.pop(r)
                r.close()
                results[item[0]] = _row(item[1], "error", error=f"timeout after {sample_timeout}s")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--sample-timeout", type=float, default=600.0)
    ap.add_argument("--cdm-src", default=os.path.join(HERE, "_upstream", "src", "metrics"))
    a = ap.parse_args()

    if not os.path.isfile(os.path.join(a.cdm_src, "cdm", "modules", "latex2bbox_color.py")):
        sys.exit(f"CDM sources not found under {a.cdm_src}; run setup_cdm.sh")
    sys.path.insert(0, a.cdm_src)
    import cdm.modules.latex2bbox_color  # noqa: F401
    from cdm.cdm import _get_total_color_list

    _get_total_color_list()

    items, seen = [], set()
    with open(a.input, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            for k in ("id", "gt", "pred"):
                if not isinstance(r.get(k), str):
                    sys.exit(f"{a.input}:{ln}: field '{k}' missing or not a string")
            if r["id"] in seen:
                sys.exit(f"{a.input}:{ln}: duplicate id {r['id']!r}")
            seen.add(r["id"])
            items.append((f"s{len(items):07d}", r["id"], r["gt"], r["pred"]))

    work_root = tempfile.mkdtemp(prefix="cdm_work_")
    t0 = time.perf_counter()
    try:
        results = run_all(items, max(1, a.workers), work_root, a.sample_timeout)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    wall = time.perf_counter() - t0

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        for sid, orig_id, _, _ in items:
            if sid in results:
                f.write(json.dumps(results[sid], ensure_ascii=False) + "\n")

    got = {}
    with open(a.output, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            got[r["id"]] = r
    missing = [i[1] for i in items if i[1] not in got or got[i[1]].get("status") not in STATUSES]
    counts = {s: sum(1 for r in got.values() if r["status"] == s) for s in STATUSES}
    scored = [r["cdm_f1"] for r in got.values() if r["status"] in ("ok", "pred_render_fail")]
    n = len(items)
    print(
        json.dumps({
            "n": n,
            "workers": a.workers,
            "wall_s": round(wall, 3),
            "s_per_sample": round(wall / n, 4) if n else None,
            "status_counts": counts,
            "mean_cdm_f1_over_ok_and_pred_fail": round(sum(scored) / len(scored), 4) if scored else None,
            "missing": len(missing),
        }),
        file=sys.stderr,
    )
    if missing:
        print("MISSING ids (no per-sample score and no explicit status):", file=sys.stderr)
        for i in missing:
            print("  " + i, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
