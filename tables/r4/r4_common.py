import os, sys, re, json, random
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path: sys.path.insert(0, SRC)
import paths
from paths import WORK, SETS, OUT, RAW

R4OUT = os.path.join(OUT, "r4")
SEL_SETS = ["r4pt", "r4brgm", "r4icdar", "r4scitsr"]
SEL_NAMES = {"r4pt": "KDD PubTables (PMC PDFs)", "r4brgm": "KDD Table-BRGM (FR/EN reports)", "r4icdar": "KDD ICDAR-2013 (corrected)", "r4scitsr": "SciTSR test (original PDFs)"}
S = 2.0
MARGIN = 4.0


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def area(b): return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b): return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def cover(a, boxes):
    """fraction of a covered by the union of boxes (raster approximation on a 64x64 grid)"""
    if area(a) <= 0 or not boxes: return 0.0
    import numpy as np
    n = 64; xs = np.linspace(a[0], a[2], n); ys = np.linspace(a[1], a[3], n)
    X, Y = np.meshgrid(xs, ys); m = np.zeros_like(X, bool)
    for b in boxes: m |= (X >= b[0]) & (X <= b[2]) & (Y >= b[1]) & (Y <= b[3])
    return float(m.mean())


def html_to_tokens(html):
    """KDD table HTML -> (tokens in the PubTabNet structure format, cell texts)"""
    import lxml.html
    root = lxml.html.fromstring(html)
    tab = root if root.tag == "table" else root.find(".//table")
    toks, texts = [], []
    head_rows = tab.findall("./thead/tr")
    body_rows = [tr for tr in tab.iter("tr") if tr not in head_rows]
    for part, rows in (("thead", head_rows), ("tbody", body_rows)):
        if not rows: continue
        toks.append(f"<{part}>")
        for tr in rows:
            toks.append("<tr>")
            for td in tr:
                if td.tag not in ("td", "th"): continue
                cs = int(re.sub(r"\D", "", td.get("colspan") or "1") or 1); rs = int(re.sub(r"\D", "", td.get("rowspan") or "1") or 1)
                if cs > 1 or rs > 1:
                    toks.append("<td")
                    if cs > 1: toks.append(f' colspan="{cs}"')
                    if rs > 1: toks.append(f' rowspan="{rs}"')
                    toks.append(">")
                else:
                    toks.append("<td>")
                toks.append("</td>")
                texts.append(re.sub(r"\s+", " ", td.text_content() or "").strip())
            toks.append("</tr>")
        toks.append(f"</{part}>")
    return toks, texts


def table_coords(html):
    m = re.search(r'coords="\[([^\]]+)\]"', html)
    return [float(v) for v in m.group(1).split(",")] if m else None


def plumber_words(pp):
    return [dict(text=w["text"], x0=float(w["x0"]), top=float(w["top"]), x1=float(w["x1"]), bottom=float(w["bottom"]), upright=bool(w.get("upright", True)))
            for w in pp.extract_words(keep_blank_chars=False, use_text_flow=False)]


def render_offset(pp, page, words, gray=None, scale=1.0):
    """pdfplumber coordinates -> pdfium render space (pt from the top-left of the rendered area).
    Candidates as in prep_scitsr; the one whose word boxes cover the most ink wins."""
    import numpy as np
    cl, cb, cr, ct = page.get_cropbox(); ml, mb, mr, mt = page.get_mediabox()
    bx0, btop = float(pp.bbox[0]), float(pp.bbox[1])
    cands = [(0.0, 0.0), (cl - ml - bx0, (mt - ct) - btop), (-bx0, -btop), (cl - ml, mt - ct)]
    if gray is None:
        gray = np.asarray(page.render(scale=scale).to_pil().convert("L"))
    H, W = gray.shape
    sample = words[:: max(1, len(words) // 150)][:150]
    best = None
    for ox, oy in dict.fromkeys(cands):
        hit = 0
        for w in sample:
            x0, y0, x1, y1 = int((w["x0"] - ox) * scale), int((w["top"] - oy) * scale), int((w["x1"] - ox) * scale) + 1, int((w["bottom"] - oy) * scale) + 1
            if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H: continue
            patch = gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
            if patch.size and patch.min() < 140: hit += 1
        if best is None or hit > best[0]: best = (hit, (ox, oy))
    return best[1], (best[0] / max(1, len(sample)))


def tokens(text):
    return set(t.lower() for t in re.findall(r"[^\W\d_]+", text or ""))


def jaccard(a, b):
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))


def cluster_boot(rows, key_a, key_b, group="doc", n=2000, seed=0):
    """mean(a-b) with a 95% bootstrap CI that resamples documents (clusters)."""
    by = {}
    for r in rows: by.setdefault(r[group], []).append(r[key_a] - r[key_b])
    docs = list(by); rng = random.Random(seed)
    if not docs: return None
    tot = [sum(by[d]) for d in docs]; cnt = [len(by[d]) for d in docs]
    mean = sum(tot) / sum(cnt)
    bs = []
    for _ in range(n):
        s = c = 0
        for _ in docs:
            k = rng.randrange(len(docs)); s += tot[k]; c += cnt[k]
        bs.append(s / c)
    bs.sort()
    return mean, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def wilcoxon_p(x, y):
    from scipy.stats import wilcoxon
    try: return float(wilcoxon(x, y, zero_method="zsplit").pvalue)
    except Exception: return 1.0


def jl(path):
    out = []
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            l = l.strip()
            if l: out.append(json.loads(l))
    return out
