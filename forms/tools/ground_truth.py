import json
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import ArrayObject, IndirectObject, NameObject


def norm_val(v):
    if v is None:
        return None
    if isinstance(v, IndirectObject):
        v = v.get_object()
    if isinstance(v, NameObject):
        return str(v)[1:]
    if isinstance(v, ArrayObject):
        return [norm_val(x) for x in v]
    if isinstance(v, bytes):
        return v.decode("latin-1")
    return str(v)


def rect(r):
    if r is None:
        return None
    r = [float(x) for x in r.get_object()]
    return [round(min(r[0], r[2]), 2), round(min(r[1], r[3]), 2), round(max(r[0], r[2]), 2), round(max(r[1], r[3]), 2)]


def export_of(d, ft, value):
    opt = d.get("/Opt")
    if opt is None or value is None:
        return None
    opt = [o.get_object() for o in opt.get_object()]
    if ft == "Btn":
        if isinstance(value, str) and value.isdigit() and int(value) < len(opt):
            return norm_val(opt[int(value)])
        return None
    if ft == "Ch":
        m = {}
        for o in opt:
            if isinstance(o, ArrayObject) and len(o) >= 2:
                m[norm_val(o[0])] = norm_val(o[1])
        if not m:
            return None
        if isinstance(value, list):
            return [m.get(x, x) for x in value]
        return m.get(value, value)
    return None


def extract(path):
    r = PdfReader(path)
    annot_page = {}
    page_index = {}
    for i, p in enumerate(r.pages):
        page_index[p.indirect_reference.idnum] = i + 1
        for a in p.get("/Annots") or []:
            if isinstance(a, IndirectObject):
                annot_page[a.idnum] = i + 1
    out = []
    root = r.trailer["/Root"]
    af = root.get("/AcroForm")
    if af is None:
        return out
    af = af.get_object()

    def walk(ref, parent, inh, depth):
        if depth > 64:
            return
        d = ref.get_object()
        t = d.get("/T")
        name = parent
        if t is not None:
            t = str(t.get_object())
            name = f"{parent}.{t}" if parent else t
        inh = dict(inh)
        for k in ("/FT", "/V", "/Ff"):
            if k in d:
                inh[k] = d[k]
        kids = d.get("/Kids")
        kids = list(kids.get_object()) if kids is not None else []
        field_kids = [k for k in kids if "/T" in k.get_object() or "/Kids" in k.get_object()]
        if field_kids:
            for k in field_kids:
                walk(k, name, inh, depth + 1)
            return
        widgets = kids if kids else [ref]
        ft = norm_val(inh.get("/FT"))
        value = norm_val(inh.get("/V"))
        ws = []
        for wref in widgets:
            wd = wref.get_object()
            page = None
            if isinstance(wref, IndirectObject):
                page = annot_page.get(wref.idnum)
            if page is None and "/P" in wd:
                page = page_index.get(wd.raw_get("/P").idnum)
            ws.append({"page": page, "rect": rect(wd.get("/Rect")), "as": norm_val(wd.get("/AS"))})
        out.append({"name": name, "type": ft, "value": value, "export": export_of(d, ft, value), "flags": int(inh.get("/Ff", 0)), "widgets": ws})

    for f in af.get("/Fields", []):
        walk(f, "", {}, 0)
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(src.glob("*.pdf")):
        fields = extract(str(pdf))
        (dst / (pdf.stem + ".json")).write_text(json.dumps({"file": pdf.name, "fields": fields}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(pdf.name, len(fields))
