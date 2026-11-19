import os, re, sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texnorm import norm_tokens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIUM = {r"\frac", r"\dfrac", r"\tfrac", r"\sqrt", r"\sum", r"\int", r"\iint", r"\iiint", r"\oint", r"\prod", r"\binom",
          r"\bigcup", r"\bigcap", r"\coprod", r"\over", r"\overset", r"\underset", r"\stackrel", r"\bigoplus",
          r"\bigotimes", r"\choose", r"\lim", r"\sup", r"\inf", r"\max", r"\min", r"\limsup", r"\liminf"}
BIGOP = {r"\sum", r"\prod", r"\int", r"\iint", r"\iiint", r"\oint", r"\bigcup", r"\bigcap", r"\coprod", r"\lim",
         r"\max", r"\min", r"\sup", r"\inf", r"\limsup", r"\liminf", r"\bigoplus", r"\bigotimes"}
GREEK = set(r"\alpha \beta \gamma \delta \epsilon \varepsilon \zeta \eta \theta \vartheta \iota \kappa \lambda \mu \nu \xi \pi \varpi \rho \varrho \sigma \varsigma \tau \upsilon \phi \varphi \chi \psi \omega \Gamma \Delta \Theta \Lambda \Xi \Pi \Sigma \Upsilon \Phi \Psi \Omega".split())
MATRIX = re.compile(r"\\begin\s*\{\s*(matrix|pmatrix|bmatrix|vmatrix|Vmatrix|Bmatrix|smallmatrix|cases|dcases|array)\s*\}")
MULTI = re.compile(r"\\begin\s*\{\s*(aligned|align\*?|gathered|gather\*?|split|eqnarray\*?|multline\*?|alignedat)\s*\}")


def raw_tokens(s):
    return re.findall(r"\\[a-zA-Z]+\*?|\\.|[^\s]", s)


def nested_frac(s):
    t = raw_tokens(s)
    n = len(t)
    for i in range(n):
        if t[i] in (r"\frac", r"\dfrac", r"\tfrac", r"\cfrac"):
            j = i + 1
            for _ in range(2):
                if j < n and t[j] == "{":
                    d, k = 0, j
                    while k < n:
                        if t[k] == "{":
                            d += 1
                        elif t[k] == "}":
                            d -= 1
                            if d == 0:
                                break
                        if t[k] in (r"\frac", r"\dfrac", r"\tfrac", r"\cfrac") and k > j:
                            return True
                        k += 1
                    j = k + 1
                else:
                    j += 1
    return False


def difficulty(gt):
    t = raw_tokens(gt)
    if r"\begin" in t or r"\\" in t or nested_frac(gt) or len(norm_tokens(gt)) > 100:
        return "complex"
    if any(x in MEDIUM for x in t):
        return "medium"
    return "simple"


def tags(it):
    gt = it["gt"]
    t = raw_tokens(gt)
    ts = set(t)
    try:
        w, h = Image.open(os.path.join(ROOT, it["image"])).size
    except Exception:
        w, h = 1, 1
    f = []
    if w / h >= 10:
        f.append("very wide (w/h>=10)")
    if h / w >= 0.5:
        f.append("very tall (h/w>=0.5)")
    lr = it.get("lowres") or 0
    if lr == 1.5:
        f.append("low-res 1.5x render")
    if lr == 1.0:
        f.append("low-res 1x render")
    if it.get("kind") == "inl":
        f.append("inline tight crop")
    if ts & {r"\text", r"\textrm", r"\mbox", r"\textit", r"\textbf", r"\textsc"}:
        f.append("\\text inside")
    if ts & {r"\mathcal", r"\mathfrak", r"\mathscr", r"\mathbb", r"\cal", r"\Bbb"}:
        f.append("calligraphic/fraktur/blackboard")
    for i, x in enumerate(t[:-1]):
        if x in BIGOP and (t[i + 1] in ("_", "^") or (t[i + 1] == r"\limits" and i + 2 < len(t) and t[i + 2] in ("_", "^"))):
            f.append("big operator with limits")
            break
    if MATRIX.search(gt):
        f.append("matrix/cases/array")
    if MULTI.search(gt) or r"\\" in t:
        f.append("aligned/multi-line")
    if nested_frac(gt):
        f.append("nested fraction")
    if len(norm_tokens(gt)) > 100:
        f.append("long (>100 tokens)")
    if sum(1 for x in t if x in GREEK) >= 3:
        f.append("greek-heavy (>=3)")
    return f
