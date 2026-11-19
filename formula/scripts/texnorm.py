import re

TOK_RE = re.compile(r"\\[a-zA-Z]+\*?|\\.|[^\s]", re.S)

DROP = {
    r"\left", r"\right", r"\middle", r"\big", r"\Big", r"\bigg", r"\Bigg", r"\bigl", r"\bigr", r"\Bigl", r"\Bigr",
    r"\biggl", r"\biggr", r"\Biggl", r"\Biggr", r"\bigm", r"\Bigm", r"\biggm", r"\Biggm",
    r"\limits", r"\nolimits", r"\displaystyle", r"\textstyle", r"\scriptstyle", r"\scriptscriptstyle",
    r"\,", r"\;", r"\:", r"\!", r"\ ", r"\quad", r"\qquad", r"\enspace", r"\thinspace", r"\medspace",
    r"\thickspace", r"\negthinspace", r"\nonumber", r"\notag", r"\rm", r"\it", r"\nobreak", r"\allowbreak",
    r"\relax", "~", r"\mathop", r"\mathord", r"\mathbin", r"\mathrel", r"\mathpunct", r"\mathopen", r"\mathclose",
    r"\protect", r"\hfill", r"\hfil", r"\strut", r"\phantom", r"\vphantom", r"\hphantom", r"\cr", "&",
}
SWITCH = {r"\cal": r"\mathcal", r"\bf": r"\mathbf", r"\Bbb": r"\mathbb", r"\frak": r"\mathfrak", r"\sf": r"\mathsf",
          r"\tt": r"\mathtt", r"\mit": None, r"\em": None}
DROP_WITH_ARG = {r"\label", r"\tag", r"\tag*", r"\hspace", r"\hspace*", r"\vspace", r"\vspace*", r"\kern", r"\mkern",
                 r"\hskip", r"\mskip", r"\phantom", r"\vphantom", r"\hphantom"}
UNWRAP = {r"\mathrm", r"\mathit", r"\mathnormal", r"\operatorname", r"\operatorname*", r"\text", r"\textrm", r"\textit",
          r"\textnormal", r"\mbox", r"\hbox", r"\emph", r"\textup", r"\textsf", r"\mathup", r"\upright"}
SYN = {
    r"\le": r"\leq", r"\ge": r"\geq", r"\ne": r"\neq", r"\neq": r"\neq", r"\to": r"\rightarrow", r"\gets": r"\leftarrow",
    r"\lbrace": r"\{", r"\rbrace": r"\}", r"\lbrack": "[", r"\rbrack": "]", r"\vert": "|", r"\lvert": "|", r"\rvert": "|",
    r"\mid": "|", r"\Vert": r"\|", r"\lVert": r"\|", r"\rVert": r"\|", r"\lt": "<", r"\gt": ">", r"\dots": r"\ldots",
    r"\dotsc": r"\ldots", r"\dotso": r"\ldots", r"\dotsb": r"\cdots", r"\dotsm": r"\cdots", r"\dotsi": r"\cdots",
    r"\dfrac": r"\frac", r"\tfrac": r"\frac", r"\cfrac": r"\frac", r"\ast": "*", r"\colon": ":", r"\bm": r"\mathbf",
    r"\boldsymbol": r"\mathbf", r"\pmb": r"\mathbf", r"\textbf": r"\mathbf",r"\land": r"\wedge",
    r"\lor": r"\vee", r"\lnot": r"\neg", r"\iff": r"\Leftrightarrow", r"\implies": r"\Rightarrow",
    r"\leqslant": r"\leq", r"\geqslant": r"\geq", r"\owns": r"\ni", r"\dbinom": r"\binom", r"\tbinom": r"\binom",
    r"\varnothing": r"\emptyset", r"\textendash": "-", r"\textemdash": "-", r"\hat": r"\hat", r"\widehat": r"\hat",
    r"\widetilde": r"\tilde", r"\overline": r"\bar", r"\to": r"\rightarrow", r"\lbrace": r"\{",
}
OPS = {r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc", r"\log", r"\ln", r"\lg", r"\exp", r"\max", r"\min", r"\lim",
       r"\sup", r"\inf", r"\det", r"\arg", r"\dim", r"\ker", r"\deg", r"\gcd", r"\Pr", r"\sinh", r"\cosh", r"\tanh",
       r"\coth", r"\arcsin", r"\arccos", r"\arctan", r"\liminf", r"\limsup", r"\hom", r"\exp", r"\mod", r"\bmod", r"\Tr",
       r"\tr"}
ARITY = {"^": 1, "_": 1, r"\frac": 2, r"\binom": 2, r"\sqrt": 1, r"\stackrel": 2, r"\overset": 2, r"\underset": 2,
         r"\mathbf": 1, r"\mathbb": 1, r"\mathcal": 1, r"\mathfrak": 1, r"\mathscr": 1, r"\mathsf": 1, r"\mathtt": 1,
         r"\hat": 1, r"\bar": 1, r"\tilde": 1, r"\vec": 1, r"\dot": 1, r"\ddot": 1, r"\dddot": 1, r"\check": 1,
         r"\breve": 1, r"\acute": 1, r"\grave": 1, r"\mathring": 1, r"\underline": 1, r"\overbrace": 1,
         r"\underbrace": 1, r"\overrightarrow": 1, r"\overleftarrow": 1, r"\overleftrightarrow": 1, r"\xrightarrow": 1,
         r"\xleftarrow": 1, r"\boxed": 1, r"\cancel": 1, r"\not": 1, r"\substack": 1, r"\begin": 1, r"\end": 1}
MATRIX_ENVS = {"pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "Bmatrix": (r"\{", r"\}"), "vmatrix": ("|", "|"),
               "Vmatrix": (r"\|", r"\|"), "smallmatrix": (None, None), "matrix": (None, None), "array": (None, None),
               "subarray": (None, None)}
ALIGN_ENVS = {"aligned", "align", "align*", "split", "gathered", "gather", "gather*", "eqnarray", "eqnarray*",
              "multline", "multline*", "alignedat", "alignat", "alignat*", "flalign", "flalign*", "equation",
              "equation*", "array", "matrix", "smallmatrix", "subarray"}


def tokenize(s):
    return TOK_RE.findall(s)


def parse(toks, i=0, close=None):
    out = []
    while i < len(toks):
        t = toks[i]
        if t == "{":
            sub, i = parse(toks, i + 1, "}")
            out.append(("G", sub))
            continue
        if t == "}":
            if close == "}":
                return out, i + 1
            i += 1
            continue
        out.append(t)
        i += 1
    return out, i


def take_arg(nodes, i):
    while i < len(nodes) and isinstance(nodes[i], str) and nodes[i] in DROP:
        i += 1
    if i >= len(nodes):
        return None, i
    return nodes[i], i + 1


def as_list(n):
    return n[1] if isinstance(n, tuple) else [n]


def norm_nodes(nodes):
    for k, n in enumerate(nodes):
        if isinstance(n, str) and n in SWITCH:
            rest = list(nodes[k + 1:])
            nodes = list(nodes[:k]) + ([SWITCH[n], ("G", rest)] if SWITCH[n] else [("G", rest)])
            break
    out = []
    i = 0
    while i < len(nodes):
        n = nodes[i]
        if isinstance(n, tuple):
            out.extend(norm_nodes(n[1]))
            i += 1
            continue
        t = SYN.get(n, n)
        if t in (r"\left", r"\right", r"\middle") and i + 1 < len(nodes) and nodes[i + 1] == ".":
            i += 2
            continue
        if t in DROP_WITH_ARG:
            i += 1
            if t in (r"\hspace", r"\hspace*", r"\vspace", r"\vspace*", r"\tag", r"\label", r"\phantom", r"\vphantom",
                     r"\hphantom", r"\tag*"):
                _, i = take_arg(nodes, i)
            continue
        if t in DROP:
            i += 1
            continue
        if t in UNWRAP:
            a, i = take_arg(nodes, i + 1)
            if a is not None:
                out.extend(norm_nodes(as_list(a)))
            continue
        if t in OPS:
            out.extend(list(t[1:]))
            i += 1
            continue
        if t == "'":
            k = 0
            while i < len(nodes) and nodes[i] == "'":
                k += 1
                i += 1
            out.append("^")
            out.append(("G", [r"\prime"] * k))
            continue
        if t in (r"\begin", r"\end"):
            a, j = take_arg(nodes, i + 1)
            env = "".join(x if isinstance(x, str) else "" for x in as_list(a)) if a is not None else ""
            i = j
            if t == r"\begin" and env in ("array", "subarray", "alignedat", "alignat", "alignat*"):
                if i < len(nodes) and isinstance(nodes[i], tuple):
                    i += 1
            if t == r"\begin" and i < len(nodes) and nodes[i] == "[" and env in ("array", "aligned", "gathered"):
                while i < len(nodes) and nodes[i] != "]":
                    i += 1
                i += 1
            if env in MATRIX_ENVS and MATRIX_ENVS[env][0]:
                d = MATRIX_ENVS[env]
                if t == r"\begin":
                    out += [d[0], r"\begin", ("G", ["M"])]
                else:
                    while out and out[-1] == r"\\":
                        out.pop()
                    out += [r"\end", ("G", ["M"]), d[1]]
                continue
            if env in ALIGN_ENVS or env in MATRIX_ENVS:
                env = "M"
            if t == r"\end":
                while out and out[-1] == r"\\":
                    out.pop()
            out += [t, ("G", list(env))]
            continue
        if t == r"\sqrt" and i + 1 < len(nodes) and nodes[i + 1] == "[":
            j = i + 2
            idx = []
            while j < len(nodes) and nodes[j] != "]":
                idx.append(nodes[j])
                j += 1
            a, k = take_arg(nodes, j + 1)
            out += [r"\sqrt", "[", *norm_nodes(idx), "]", ("G", norm_nodes(as_list(a)) if a is not None else [])]
            i = k
            continue
        if t in (r"\underset", r"\overset", r"\stackrel"):
            x, i = take_arg(nodes, i + 1)
            y, i = take_arg(nodes, i)
            if x is None or y is None:
                continue
            out.extend(norm_nodes(as_list(y)))
            out += ["_" if t == r"\underset" else "^", ("G", norm_nodes(as_list(x)))]
            continue
        if t in ARITY:
            out.append(t)
            i += 1
            for _ in range(ARITY[t]):
                a, i = take_arg(nodes, i)
                if a is None:
                    break
                if isinstance(a, str) and SYN.get(a, a) in UNWRAP:
                    b, i = take_arg(nodes, i)
                    a = b if b is not None else ("G", [])
                elif isinstance(a, str) and SYN.get(a, a) in ARITY and SYN.get(a, a) not in ("^", "_"):
                    sub = [a]
                    for _ in range(ARITY[SYN.get(a, a)]):
                        b, i = take_arg(nodes, i)
                        if b is None:
                            break
                        sub.append(b)
                    a = ("G", sub)
                out.append(("G", norm_nodes(as_list(a))))
            continue
        out.append(t)
        i += 1
    return canon(out)


def canon(out):
    k = 0
    while k + 3 < len(out):
        if out[k] == "^" and out[k + 2] == "_" and isinstance(out[k + 1], tuple) and isinstance(out[k + 3], tuple):
            out[k], out[k + 1], out[k + 2], out[k + 3] = "_", out[k + 3], "^", out[k + 1]
            k += 4
        else:
            k += 1
    res = []
    k = 0
    while k < len(out):
        if out[k] == r"\begin" and k + 1 < len(out) and out[k + 1] == ("G", ["M"]):
            d, j = 0, k
            while j < len(out):
                if out[j] == r"\begin":
                    d += 1
                elif out[j] == r"\end":
                    d -= 1
                    if d == 0:
                        break
                j += 1
            body = out[k + 2:j]
            if j < len(out) and r"\\" not in body:
                res.extend(body)
                k = j + 2
                continue
        res.append(out[k])
        k += 1
    return res


def flatten(nodes):
    r = []
    for n in nodes:
        if isinstance(n, tuple):
            r.append("{")
            r.extend(flatten(n[1]))
            r.append("}")
        else:
            r.append(n)
    return r


def merge_scripts(toks):
    s = " ".join(toks)
    s = re.sub(r"(?:^| )[\^_] \{ \}", "", s)
    s = s.replace(". . .", r"\ldots").replace(r"\cdot \cdot \cdot", r"\cdots")
    s = re.sub(r"\^ \{ ((?:\\prime )+)\} \^ \{ ((?:\\prime )+)\}", r"^ { \1\2}", s)
    return s.split()


def norm_tokens(s):
    s = s.replace(r"\begin{align*}", r"\begin{aligned}").replace(r"\end{align*}", r"\end{aligned}")
    nodes, _ = parse(tokenize(s))
    toks = flatten(norm_nodes(nodes))
    toks = merge_scripts(toks)
    while toks and toks[-1] == r"\\":
        toks.pop()
    return toks


def normalize(s):
    return "".join(norm_tokens(s))


if __name__ == "__main__":
    import sys
    tests = [
        (r"x^2", r"x^{2}"), (r"{x}^{2}", r"x^{2}"), (r"\left( a \right)", r"(a)"), (r"\mathrm{d}x", r"dx"),
        (r"\sum\limits_{i=1}^n", r"\sum_{i=1}^{n}"), (r"\frac12", r"\frac{1}{2}"), (r"\le", r"\leq"),
        (r"\operatorname*{max}_x", r"\max_x"), (r"f'(x)", r"f^{\prime}(x)"), (r"\begin{pmatrix}a\\b\end{pmatrix}",
        r"\left(\begin{array}{c}a\\b\end{array}\right)"), (r"\frac{a}{bc}", r"\frac{ab}{c}"),
        (r"\begin{align*} a&=b\\ c&=d \end{align*}", r"\begin{aligned} a&=b\\ c&=d\\ \end{aligned}"),
        (r"\left. x \right|_0", r"x|_{0}"), (r"\boldsymbol{x}", r"\mathbf x"), (r"\text{if } x", r"\mathrm{if}x"),
    ]
    for a, b in tests:
        print(normalize(a) == normalize(b), normalize(a), "|", normalize(b))
