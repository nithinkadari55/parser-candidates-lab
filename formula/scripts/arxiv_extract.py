import json, os, re, sys, unicodedata, random
import numpy as np
import pypdfium2 as pdfium
from PIL import Image
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texnorm import tokenize, OPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "arxiv")
OUT = os.path.join(ROOT, "data", "arxiv_crops")
WL = set(json.load(open(os.path.join(ROOT, "manifests", "cmd_whitelist.json"))))
MAX_DISP, MAX_INL = 12, 4
ENV_OK = {"aligned", "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix", "Bmatrix", "cases", "split",
          "gathered", "smallmatrix", "subarray"}


def strip_comments(s):
    return re.sub(r"(?<!\\)%[^\n]*", "", s)


def read_tex(path, base, depth=0):
    try:
        s = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""
    s = strip_comments(s)
    if depth > 5:
        return s

    def inc(m):
        name = m.group(2).strip()
        p = os.path.join(base, name)
        if not os.path.exists(p):
            p += ".tex"
        return read_tex(p, base, depth + 1) if os.path.exists(p) else ""
    return re.sub(r"\\(input|include)\s*\{([^}]*)\}", inc, s)


def find_main(src):
    cands = []
    for dp, _, fs in os.walk(src):
        for f in fs:
            if f.endswith(".tex"):
                p = os.path.join(dp, f)
                t = open(p, encoding="utf-8", errors="replace").read()
                if r"\begin{document}" in t:
                    cands.append((len(t), p))
    return max(cands)[1] if cands else None


def brace(s, i):
    if i >= len(s) or s[i] != "{":
        return None, i
    d = 0
    for j in range(i, len(s)):
        if s[j] == "\\":
            continue
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            d += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            d -= 1
            if d == 0:
                return s[i + 1:j], j + 1
    return None, i


def skipws(s, i):
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def macros(s):
    M = {}
    for m in re.finditer(r"\\(?:re)?newcommand\*?\s*", s):
        i = skipws(s, m.end())
        if s[i:i + 1] == "{":
            name, i = brace(s, i)
        else:
            mm = re.match(r"\\[a-zA-Z]+", s[i:])
            if not mm:
                continue
            name, i = mm.group(0), i + mm.end()
        if not name:
            continue
        name = name.strip()
        i = skipws(s, i)
        n = 0
        mm = re.match(r"\[(\d)\]", s[i:])
        if mm:
            n = int(mm.group(1)); i = skipws(s, i + mm.end())
        if s[i:i + 1] == "[":
            continue
        body, _ = brace(s, i)
        if body is not None and re.fullmatch(r"\\[a-zA-Z]+", name):
            M[name] = (n, body)
    for m in re.finditer(r"\\def\s*(\\[a-zA-Z]+)\s*((?:#\d)*)\s*\{", s):
        body, _ = brace(s, m.end() - 1)
        if body is not None:
            M[m.group(1)] = (len(m.group(2)) // 2, body)
    for m in re.finditer(r"\\DeclareMathOperator(\*?)\s*\{?\s*(\\[a-zA-Z]+)\s*\}?\s*\{([^}]*)\}", s):
        M[m.group(2)] = (0, r"\operatorname%s{%s}" % (m.group(1), m.group(3)))
    return M


def expand(f, M):
    for _ in range(12):
        changed = False
        out, i = [], 0
        while i < len(f):
            mm = re.match(r"\\[a-zA-Z]+", f[i:])
            if mm and mm.group(0) in M:
                n, body = M[mm.group(0)]
                j = i + mm.end()
                args = []
                ok = True
                for _ in range(n):
                    j = skipws(f, j)
                    if f[j:j + 1] == "{":
                        a, j = brace(f, j)
                        if a is None:
                            ok = False; break
                    elif j < len(f):
                        m2 = re.match(r"\\[a-zA-Z]+|.", f[j:])
                        a = m2.group(0); j += m2.end()
                    else:
                        ok = False; break
                    args.append(a)
                if not ok:
                    return None
                b = body
                for k, a in enumerate(args):
                    b = b.replace("#%d" % (k + 1), a)
                out.append("{" + b + "}" if n else b + (" " if re.match(r"[a-zA-Z]", f[j:j + 1]) else ""))
                i = j
                changed = True
            elif mm:
                out.append(mm.group(0)); i += mm.end()
            else:
                out.append(f[i]); i += 1
        f = "".join(out)
        if not changed:
            return f
    return None


DISP_RE = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|displaymath|flalign\*?)\}(.*?)\\end\{\1\}|\\\[(.*?)\\\]|\$\$(.*?)\$\$", re.S)


def clean(f):
    f = re.sub(r"\\label\s*\{[^}]*\}", "", f)
    f = re.sub(r"\\tag\*?\s*\{[^}]*\}", "", f)
    f = re.sub(r"\\(nonumber|notag)\b", "", f)
    f = re.sub(r"\s+", " ", f).strip()
    return f


def ok_formula(f):
    if any(x in f for x in ["\\ref", "\\cite", "\\eqref", "\\color", "\\includegraphics", "\\intertext", "\\textcolor",
                            "\\hyperref", "\\url", "#", "\\label", "\\newline", "\\footnote", "\\qed", "\\par", "\\tikz",
                            "\\xymatrix", "\\begin{tabular", "\\hline", "\\multicolumn"]):
        return False
    for e in re.findall(r"\\begin\{([^}]*)\}", f):
        if e not in ENV_OK:
            return False
    for t in tokenize(f):
        if t.startswith("\\") and len(t) > 2 and t not in WL:
            return False
    return f.count("{") == f.count("}") or True


GREEK = {n: unicodedata.normalize("NFKC", c) for n, c in zip(
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega".split(),
    "αβγδϵεζηθϑικλμνξπϖρϱσςτυϕφχψωΓΔΘΛΞΠΣΥΦΨΩ")}
SYMS = {r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\in": "∈", r"\infty": "∞",
        r"\partial": "∂", r"\nabla": "∇"}
OPT = {"greek": False, "sym": False}


def keep_char(c):
    if len(c) != 1:
        return False
    if c.isascii():
        return c.isalnum() or c in "=+"
    if OPT["greek"] and c in GREEK_SET:
        return True
    if OPT["sym"] and c in SYM_SET:
        return True
    return False


GREEK_SET = set(GREEK.values())
SYM_SET = set(SYMS.values())


def signature(f):
    toks = tokenize(f)
    sig = []
    skip = 0
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in (r"\begin", r"\end"):
            i += 1
            d = 0
            while i < len(toks):
                if toks[i] == "{": d += 1
                if toks[i] == "}":
                    d -= 1
                    if d == 0:
                        break
                i += 1
            i += 1
            if t == r"\begin" and i < len(toks) and toks[i] == "{" and "array" in f:
                d = 0
                while i < len(toks):
                    if toks[i] == "{": d += 1
                    if toks[i] == "}":
                        d -= 1
                        if d == 0:
                            break
                    i += 1
                i += 1
            continue
        if t in OPS:
            sig.extend(t[1:])
        elif t in (r"\neq", r"\ne"):
            sig.append("=")
        elif OPT["greek"] and t[1:] in GREEK:
            sig.append(GREEK[t[1:]])
        elif OPT["sym"] and t in SYMS:
            sig.append(SYMS[t])
        elif len(t) == 1 and keep_char(t):
            sig.append(t)
        i += 1
    return "".join(sig)


EDGE_MAP = {")": ")", "]": "]", "(": "(", "[": "[", r"\}": "}", r"\{": "{", r"\rangle": "⟩", r"\langle": "⟨",
            "|": "|∣", r"\|": "‖∥", r"\Vert": "‖∥", r"\lvert": "|∣", r"\rvert": "|∣",
            r"\lVert": "‖∥", r"\rVert": "‖∥", "-": "-−", r"\rbrace": "}", r"\lbrace": "{"}
EDGE_IGNORE = {"{", "}", "^", "_", r"\left", r"\right", r"\big", r"\Big", r"\bigl", r"\bigr", r"\Bigl", r"\Bigr", r"\,", r"\;", r"\!", r"\:", r"\ ", r"\mathbf", r"\mathrm", r"\mathcal", r"\mathbb", r"\bm", r"\boldsymbol", r"\mathit", r"\mathsf", r"\mathfrak", r"\hat", r"\bar", r"\tilde", r"\widehat", r"\widetilde", r"\vec", r"\dot", r"\ddot", r"\overline", r"\text", r"\operatorname"}


def edge_glyphs(f):
    toks = tokenize(f)
    sigpos = [i for i, t in enumerate(toks) if signature(t)]
    if not sigpos:
        return None
    res = []
    for seg in (toks[:sigpos[0]], toks[sigpos[-1] + 1:]):
        out = []
        for t in seg:
            if t in EDGE_IGNORE:
                continue
            if t in EDGE_MAP:
                out.append(EDGE_MAP[t])
            else:
                return None
        res.append(out)
    return res


def page_chars(page):
    import pypdfium2.raw as R
    tp = page.get_textpage()
    n = tp.count_chars()
    chars = []
    for k in range(n):
        u = R.FPDFText_GetUnicode(tp.raw, k)
        c = unicodedata.normalize("NFKC", chr(u)) if 0 < u < 0x110000 else ""
        try:
            box = tp.get_charbox(k, loose=False)
        except Exception:
            box = None
        chars.append((c, box))
    return chars


def ink_grow(bw, x0, y0, x1, y1, gx, gy, limx=None):
    H, W = bw.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)
    changed = True
    while changed:
        changed = False
        for side in range(4):
            gap = 0
            while True:
                if side == 0:
                    if y0 - 1 - gap < 0: break
                    strip = bw[y0 - 1 - gap, x0:x1 + 1]
                elif side == 1:
                    if y1 + 1 + gap >= H: break
                    strip = bw[y1 + 1 + gap, x0:x1 + 1]
                elif side == 2:
                    if x0 - 1 - gap < 0: break
                    strip = bw[y0:y1 + 1, x0 - 1 - gap]
                else:
                    if x1 + 1 + gap >= W: break
                    strip = bw[y0:y1 + 1, x1 + 1 + gap]
                g = gy if side < 2 else gx
                if strip.any():
                    if side == 0: y0 = y0 - 1 - gap
                    elif side == 1: y1 = y1 + 1 + gap
                    elif side == 2: x0 = x0 - 1 - gap
                    else: x1 = x1 + 1 + gap
                    gap = 0
                    changed = True
                else:
                    gap += 1
                    if gap > g:
                        break
    return x0, y0, x1, y1


def main():
    os.makedirs(OUT, exist_ok=True)
    papers = json.load(open(os.path.join(D, "papers.json")))
    rnd = random.Random(7)
    items, stats = [], {}
    for p in papers:
        aid = p["id"]
        pd = os.path.join(D, aid)
        mainf = find_main(os.path.join(pd, "src"))
        if not mainf or not os.path.exists(os.path.join(pd, "paper.pdf")):
            continue
        s = read_tex(mainf, os.path.dirname(mainf))
        pre = s.split(r"\begin{document}")[0]
        body = s.split(r"\begin{document}", 1)[-1]
        sty = ""
        for dp, _, fs in os.walk(os.path.join(pd, "src")):
            for f in fs:
                if f.endswith(".sty") or (f.endswith(".tex") and os.path.join(dp, f) != mainf):
                    sty += strip_comments(open(os.path.join(dp, f), encoding="utf-8", errors="replace").read())
        try:
            pdf = pdfium.PdfDocument(os.path.join(pd, "paper.pdf"))
        except Exception:
            continue
        pages = []
        for pi in range(len(pdf)):
            ch = page_chars(pdf[pi])
            pages.append(ch)
        alltxt = "".join(c for ch in pages for c, _ in ch)
        OPT["greek"] = sum(c in GREEK_SET for c in alltxt) >= 30
        OPT["sym"] = sum(c in SYM_SET for c in alltxt) >= 15
        pages = [(ch, [k for k, (c, b) in enumerate(ch) if keep_char(c) and b], None) for ch in pages]
        pages = [(ch, idx, "".join(ch[k][0] for k in idx)) for ch, idx, _ in pages]
        M = macros(sty + "\n" + pre + "\n" + body)
        disp = []
        for m in DISP_RE.finditer(body):
            env = m.group(1)
            raw = m.group(2) if env else (m.group(3) if m.group(3) is not None else m.group(4))
            if raw is None:
                continue
            f = expand(clean(raw), M)
            if f is None:
                continue
            f = clean(f)
            if env and env.rstrip("*") in ("align", "eqnarray", "flalign"):
                if r"\\" in f or "&" in f:
                    f = r"\begin{aligned} " + f.replace("&=&", "&=") + r" \end{aligned}"
            elif env and env.rstrip("*") in ("gather",):
                if r"\\" in f:
                    f = r"\begin{gathered} " + f + r" \end{gathered}"
            elif env and env.rstrip("*") == "multline":
                if r"\\" in f:
                    f = r"\begin{gathered} " + f + r" \end{gathered}"
            elif r"\\" in f and not re.search(r"\\begin\{", f):
                continue
            if r"\begin{split}" in f:
                f = f.replace(r"\begin{split}", r"\begin{aligned}").replace(r"\end{split}", r"\end{aligned}")
            if not f or not ok_formula(f):
                continue
            sig = signature(f)
            if len(sig) < 6:
                continue
            disp.append((f, sig, m.start()))
        body_noenv = DISP_RE.sub(" ", body)
        inl = []
        for m in re.finditer(r"(?<![\\$])\$([^$]{3,120}?)\$(?!\$)|\\\((.{3,120}?)\\\)", body_noenv):
            raw = m.group(1) or m.group(2)
            if "^" not in raw and "_" not in raw:
                continue
            f = expand(clean(raw), M)
            if f is None:
                continue
            f = clean(f)
            if not ok_formula(f) or r"\\" in f:
                continue
            sig = signature(f)
            if len(sig) < 5:
                continue
            inl.append((f, sig, m.start()))
        cand = []
        seen = set()
        for kind, lst in (("disp", disp), ("inl", inl)):
            for f, sig, pos in lst:
                if (kind, sig) in seen:
                    continue
                seen.add((kind, sig))
                best = None
                for pi, (ch, idx, S) in enumerate(pages):
                    if not S:
                        continue
                    if kind == "inl":
                        occ = [m.start() for m in re.finditer(re.escape(sig), S)]
                        if len(occ) == 1:
                            best = (100, pi, occ[0], occ[0] + len(sig)) if best is None else (-1, None, None, None)
                        elif len(occ) > 1:
                            best = (-1, None, None, None)
                        continue
                    al = fuzz.partial_ratio_alignment(sig, S, score_cutoff=88)
                    if al and (best is None or al.score > best[0]):
                        best = (al.score, pi, al.dest_start, al.dest_end)
                if not best or best[0] < 0 or best[1] is None:
                    continue
                cand.append((kind, f, sig, best))
        stats[aid] = {"disp_src": len(disp), "inl_src": len(inl), "matched": len(cand), "greek": OPT["greek"], "sym": OPT["sym"]}
        rnd.shuffle(cand)
        nd = ni = 0
        rendered = {}
        for kind, f, sig, (score, pi, a, b) in cand:
            if kind == "disp" and nd >= MAX_DISP or kind == "inl" and ni >= MAX_INL:
                continue
            ch, idx, S = pages[pi]
            win = S[a:b]
            ops = Levenshtein.editops(sig, win)
            changed = {d for _, _, d in ops}
            matched = [idx[a + k] for k in range(len(win)) if k not in changed]
            if kind == "inl":
                k0, k1 = idx[a], idx[b - 1]
                before = ch[k0 - 1][0] if k0 > 0 else " "
                after = ch[k1 + 1][0] if k1 + 1 < len(ch) else " "
                if before.isalnum() or after.isalnum():
                    continue
                ext = edge_glyphs(f)
                if ext is None:
                    continue
                lead, trail = ext
                ok = True
                for g in reversed(lead):
                    j = k0 - 1
                    while j >= 0 and ch[j][0] in ("\r", "\n", ""):
                        j -= 1
                    if j < 0 or ch[j][0] not in g:
                        ok = False
                        break
                    k0 = j
                for g in trail:
                    j = k1 + 1
                    while j < len(ch) and ch[j][0] in ("\r", "\n", ""):
                        j += 1
                    if j >= len(ch) or ch[j][0] not in g:
                        ok = False
                        break
                    k1 = j
                if not ok:
                    continue
                matched = list(range(k0, k1 + 1))
            boxes = [ch[k][1] for k in matched if ch[k][1]]
            if not boxes:
                continue
            L = min(bx[0] for bx in boxes); R = max(bx[2] for bx in boxes)
            Bt = min(bx[1] for bx in boxes); T = max(bx[3] for bx in boxes)
            for scale in ([3.0, 1.5] if kind == "disp" else [3.0]):
                key = (pi, scale)
                if key not in rendered:
                    img = pdf[pi].render(scale=scale).to_pil().convert("L")
                    rendered[key] = (img, np.asarray(img) < 200, pdf[pi].get_height())
                img, bw, ph = rendered[key]
                x0, x1 = int(L * scale) - 1, int(R * scale) + 1
                y0, y1 = int((ph - T) * scale) - 1, int((ph - Bt) * scale) + 1
                if kind == "disp":
                    gx, gy = int(4.5 * scale), int(2.2 * scale)
                else:
                    gx, gy = 0, int(1.0 * scale)
                x0, y0, x1, y1 = ink_grow(bw, x0, y0, x1, y1, gx, gy)
                pad = int(2 * scale) if kind == "disp" else 2
                padx = pad if kind == "disp" else 0
                X0, Y0, X1, Y1 = x0 - padx, y0 - pad, x1 + padx + 1, y1 + pad + 1
                inside = "".join(c for c, bx in ch if bx and keep_char(c)
                                 and X0 <= (bx[0] + bx[2]) / 2 * scale <= X1 and Y0 <= (ph - (bx[1] + bx[3]) / 2) * scale <= Y1)
                sim = Levenshtein.normalized_similarity(sig, inside)
                if kind == "disp" and (sim < 0.9 or sorted(inside) != sorted(sig)):
                    break
                if kind == "inl" and inside != sig:
                    break
                if x1 - x0 < 8 or y1 - y0 < 6:
                    break
                crop = img.crop((X0, Y0, X1, Y1))
                if kind == "disp":
                    lowres = scale == 1.5
                    if lowres and nd % 3 != 0:
                        continue
                    if not lowres:
                        nd += 1
                else:
                    lowres = False
                    ni += 1
                cid = f"{aid}_{kind}{nd if kind == 'disp' else ni:02d}{'_lr' if lowres else ''}"
                fn = os.path.join(OUT, cid + ".png")
                crop.save(fn)
                items.append({"id": "arxiv/" + cid, "set": "arxiv", "image": os.path.relpath(fn, ROOT).replace("\\", "/"),
                              "gt": f, "kind": kind, "lowres": lowres, "paper": aid, "page": pi + 1, "match": score,
                              "sim": round(sim, 3)})
        stats[aid]["kept"] = sum(1 for it in items if it["paper"] == aid)
        print(aid, stats[aid], flush=True)
    json.dump(items, open(os.path.join(ROOT, "manifests", "arxiv.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("total", len(items), "papers", len({it["paper"] for it in items}))


if __name__ == "__main__":
    main()
