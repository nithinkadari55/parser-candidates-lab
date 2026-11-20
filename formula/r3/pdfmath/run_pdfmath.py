import argparse
import json
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

TOOL = "pdfmath"


class ItemTimeout(BaseException):
    pass


def _on_alarm(signum, frame):
    raise ItemTimeout()


class _Deadline:
    def __init__(self, seconds):
        self.seconds = seconds
        self.enabled = seconds and seconds > 0 and hasattr(signal, "setitimer")

    def __enter__(self):
        if self.enabled:
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        if self.enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return False


def _result(item_id, latex=None, confidence=None, status="error", error=None, ms=0.0):
    return {"id": item_id, "tool": TOOL, "latex": latex, "confidence": confidence,
            "status": status, "error": error, "ms": round(float(ms), 3)}


def page_ctm(mediabox, rotate):
    x0, y0, x1, y1 = [float(v) for v in mediabox]
    if rotate == 90:
        return (0.0, -1.0, 1.0, 0.0, -y0, x1)
    if rotate == 180:
        return (-1.0, 0.0, 0.0, -1.0, x1, y1)
    if rotate == 270:
        return (0.0, 1.0, -1.0, 0.0, y1, -x0)
    return (1.0, 0.0, 0.0, 1.0, -x0, -y0)


def user_bbox_to_region(bbox, ctm):
    from pdfmath.geometry.bbox import BBox
    from pdfmath.units import bp_to_pt
    a, b, c, d, e, f = ctm
    xs, ys = [], []
    for x in (float(bbox[0]), float(bbox[2])):
        for y in (float(bbox[1]), float(bbox[3])):
            xs.append(a * x + c * y + e)
            ys.append(b * x + d * y + f)
    return BBox(bp_to_pt(min(xs)), bp_to_pt(min(ys)), bp_to_pt(max(xs)), bp_to_pt(max(ys)))


def page_geometry(pdf_path, wanted):
    from pdfminer.pdfpage import PDFPage
    out = {}
    with open(pdf_path, "rb") as fh:
        for i, page in enumerate(PDFPage.get_pages(fh, check_extractable=False)):
            if i in wanted:
                out[i] = (page.mediabox, int(page.rotate or 0) % 360)
            if len(out) == len(wanted):
                break
    return out


def _validate(item):
    if not isinstance(item.get("pdf"), str) or not item["pdf"]:
        return "missing pdf"
    pi = item.get("page_index")
    if not isinstance(pi, int) or isinstance(pi, bool) or pi < 0:
        return "page_index must be a non-negative int"
    bb = item.get("bbox")
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return "bbox must be [x0, y0, x1, y1]"
    try:
        vals = [float(v) for v in bb]
    except (TypeError, ValueError):
        return "bbox values must be numbers"
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return "bbox must have x1 > x0 and y1 > y0"
    return None


def process_pdf(pdf_path, items, timeout):
    from pdfmath.detection.equations import find_displayed_equations
    from pdfmath.extraction.pdfminer_backend import extract_pages
    from pdfmath.fonts.mathparams import Style
    from pdfmath.latex.serializer import to_latex
    from pdfmath.parse.driver import parse_region

    results = {}
    by_page = {}
    for idx, item in items:
        by_page.setdefault(item["page_index"], []).append((idx, item))

    t0 = time.perf_counter()
    try:
        with _Deadline(timeout):
            geoms = page_geometry(pdf_path, set(by_page))
    except ItemTimeout:
        ms = (time.perf_counter() - t0) * 1000.0
        for idx, item in items:
            results[idx] = _result(item["id"], error=f"timeout after {timeout}s reading page tree", ms=ms)
        return results
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000.0
        for idx, item in items:
            results[idx] = _result(item["id"], error=f"{type(exc).__name__}: {exc}", ms=ms)
        return results
    tree_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(items))

    for pi in list(by_page):
        if pi not in geoms:
            for idx, item in by_page.pop(pi):
                results[idx] = _result(item["id"], error=f"page_index {pi} out of range", ms=tree_ms)

    remaining = sorted(by_page)
    while remaining:
        gen = extract_pages(pdf_path, [p + 1 for p in remaining])
        restart = False
        for pi in list(remaining):
            group = by_page[pi]
            t_page = time.perf_counter()
            page = None
            page_err = None
            try:
                with _Deadline(timeout):
                    page = next(gen)
            except ItemTimeout:
                page_err = f"timeout after {timeout}s extracting page"
                restart = True
            except StopIteration:
                page_err = "page not returned by extractor"
            except Exception as exc:
                page_err = f"extract: {type(exc).__name__}: {exc}"
                restart = True
            remaining.remove(pi)
            if page is not None and page.page != pi + 1:
                page_err = f"extractor returned page {page.page}, expected {pi + 1}"
                page = None
            if page is None:
                ms = (time.perf_counter() - t_page) * 1000.0 / len(group) + tree_ms
                for idx, item in group:
                    results[idx] = _result(item["id"], error=page_err, ms=ms)
                if restart:
                    break
                continue

            displays = None
            try:
                with _Deadline(timeout):
                    displays = [r.bbox for r in find_displayed_equations(page)]
            except (ItemTimeout, Exception):
                displays = None
            shared_ms = (time.perf_counter() - t_page) * 1000.0 / len(group) + tree_ms
            ctm = page_ctm(*geoms[pi])

            for idx, item in group:
                t_item = time.perf_counter()
                try:
                    with _Deadline(timeout):
                        region = user_bbox_to_region(item["bbox"], ctm)
                        if displays is None:
                            style = Style.DISPLAY
                        elif any(d.contains_point(region.cx, region.cy) for d in displays):
                            style = Style.DISPLAY
                        else:
                            style = Style.TEXT
                        sub = page.in_region(region)
                        if not sub.glyphs and not sub.rules:
                            res = _result(item["id"], status="no_math")
                        else:
                            src, tree, ctx = parse_region(page, region, style)
                            rendered = to_latex(tree)
                            latex = rendered.latex
                            if latex is None or not latex.strip():
                                res = _result(item["id"], status="no_math")
                            else:
                                res = _result(item["id"], latex=latex,
                                              confidence=round(float(tree.structural_confidence()), 6),
                                              status="ok")
                except ItemTimeout:
                    res = _result(item["id"], error=f"timeout after {timeout}s")
                except Exception as exc:
                    tb = traceback.extract_tb(exc.__traceback__)
                    where = f" at {os.path.basename(tb[-1].filename)}:{tb[-1].lineno}" if tb else ""
                    res = _result(item["id"], error=f"{type(exc).__name__}: {exc}{where}")
                res["ms"] = round((time.perf_counter() - t_item) * 1000.0 + shared_ms, 3)
                results[idx] = res
        if restart:
            gen.close()
            continue
        gen.close()
        for pi in remaining:
            for idx, item in by_page[pi]:
                results[idx] = _result(item["id"], error="page not returned by extractor", ms=tree_ms)
        remaining = []
    return results


def _crash_results(items, msg):
    return {idx: _result(item["id"], error=msg) for idx, item in items}


def run_groups(groups, workers, timeout):
    results = {}
    if workers <= 1:
        for pdf, items in groups.items():
            results.update(process_pdf(pdf, items, timeout))
        return results
    pending = dict(groups)
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_pdf, pdf, items, timeout): pdf for pdf, items in pending.items()}
            for fut in as_completed(futs):
                pdf = futs[fut]
                try:
                    results.update(fut.result())
                    pending.pop(pdf, None)
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    results.update(_crash_results(pending.pop(pdf), f"worker: {type(exc).__name__}: {exc}"))
    except BrokenProcessPool:
        for pdf, items in list(pending.items()):
            try:
                with ProcessPoolExecutor(max_workers=1) as ex:
                    results.update(ex.submit(process_pdf, pdf, items, timeout).result())
            except Exception as exc:
                results.update(_crash_results(items, f"worker crashed: {type(exc).__name__}: {exc}"))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    lines = []
    with open(args.input, "r", encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                lines.append(raw)

    results = {}
    groups = {}
    ids = {}
    for idx, raw in enumerate(lines):
        try:
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError("line is not a JSON object")
            ids[idx] = item.get("id")
        except Exception as exc:
            results[idx] = _result(None, error=f"bad input line: {exc}")
            continue
        err = _validate(item)
        if err is None and not os.path.isfile(item["pdf"]):
            err = f"pdf not found: {item['pdf']}"
        if err is not None:
            results[idx] = _result(item.get("id"), error=err)
            continue
        groups.setdefault(item["pdf"], []).append((idx, item))

    results.update(run_groups(groups, max(1, args.workers), args.timeout))

    with open(args.output, "w", encoding="utf-8") as out:
        for idx in range(len(lines)):
            res = results.get(idx) or _result(ids.get(idx), error="no result produced")
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
