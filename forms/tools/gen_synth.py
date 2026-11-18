import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    ByteStringObject,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
    TextStringObject,
)
from reportlab.lib.colors import black
from reportlab.pdfgen import canvas

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "pdfs")


def n(x):
    return NameObject(x)


def num(x):
    return FloatObject(x) if isinstance(x, float) else NumberObject(x)


def arr(*xs):
    return ArrayObject(list(xs))


def utf16(s):
    return ByteStringObject(b"\xfe\xff" + s.encode("utf-16-be"))


def reportlab_form(path):
    c = canvas.Canvas(str(path), pagesize=(612, 792))
    f = c.acroForm
    c.setFont("Helvetica", 11)
    c.drawString(72, 740, "Name:")
    f.textfield(name="rl_name", x=160, y=730, width=250, height=20, value="Zoë Ñúñez", borderColor=black)
    c.drawString(72, 700, "Notes:")
    f.textfield(name="rl_notes", x=160, y=640, width=250, height=70, value="Line one\nLine two\nLine three", fieldFlags="multiline", borderColor=black)
    c.drawString(72, 600, "Agree:")
    f.checkbox(name="rl_agree", x=160, y=595, size=16, checked=True, buttonStyle="check")
    c.drawString(72, 570, "Spam:")
    f.checkbox(name="rl_spam", x=160, y=565, size=16, checked=False, buttonStyle="check")
    c.drawString(72, 540, "Colour:")
    for i, col in enumerate(["red", "green", "blue"]):
        f.radio(name="rl_color", value=col, selected=(col == "green"), x=160 + i * 40, y=535, size=16, buttonStyle="circle")
    c.drawString(72, 500, "Country:")
    f.choice(name="rl_country", value="Norway", options=["Germany", "Norway", "Sweden"], x=160, y=495, width=150, height=20)
    c.drawString(72, 460, "Language:")
    f.listbox(name="rl_lang", value="Rust", options=["Go", "Python", "Rust"], x=160, y=400, width=150, height=60)
    c.showPage()
    c.setFont("Helvetica", 11)
    c.drawString(72, 740, "Page two text:")
    f.textfield(name="rl_p2_text", x=160, y=730, width=250, height=20, value="On page two", borderColor=black)
    c.drawString(72, 700, "Page two check:")
    f.checkbox(name="rl_p2_check", x=160, y=695, size=16, checked=True, buttonStyle="check")
    c.showPage()
    c.save()


def content(w, lines):
    s = DecodedStreamObject()
    parts = []
    for x, y, t in lines:
        parts.append("BT /F1 11 Tf %d %d Td (%s) Tj ET" % (x, y, t))
    s.set_data(("\n".join(parts)).encode("latin-1"))
    return w._add_object(s)


def lowlevel_form(path, simple=False):
    w = PdfWriter()
    font = w._add_object(DictionaryObject({n("/Type"): n("/Font"), n("/Subtype"): n("/Type1"), n("/BaseFont"): n("/Helvetica"), n("/Encoding"): n("/WinAnsiEncoding")}))
    npages = 1 if simple else 2
    pages = []
    for i in range(npages):
        p = w.add_blank_page(612, 792)
        p[n("/Resources")] = DictionaryObject({n("/Font"): DictionaryObject({n("/F1"): font})})
        pages.append(p)
    page_refs = [p.indirect_reference for p in pages]
    annots = [[] for _ in range(npages)]
    fields = []

    def add(d):
        return w._add_object(d)

    def widget(page, rect, extra=None, with_p=True, parent=None):
        d = DictionaryObject({n("/Type"): n("/Annot"), n("/Subtype"): n("/Widget"), n("/F"): NumberObject(4), n("/Rect"): arr(*[num(v) for v in rect])})
        if with_p:
            d[n("/P")] = page_refs[page]
        if parent is not None:
            d[n("/Parent")] = parent
        for k, v in (extra or {}).items():
            d[n(k)] = v
        r = add(d)
        annots[page].append(r)
        return r

    def merged(page, rect, keys, with_p=True, parent=None):
        return widget(page, rect, keys, with_p, parent)

    da = TextStringObject("/Helv 11 Tf 0 g")
    if simple:
        labels = [(72, 745, "Full name:"), (72, 705, "City:"), (72, 665, "Subscribe:"), (72, 625, "Country:")]
        fields.append(merged(0, [160, 738, 420, 758], {"/FT": n("/Tx"), "/T": TextStringObject("full_name"), "/V": TextStringObject("Ada Lovelace"), "/DA": da}))
        fields.append(merged(0, [160, 698, 420, 718], {"/FT": n("/Tx"), "/T": TextStringObject("city"), "/V": utf16("Zoë Ñúñez"), "/DA": da}))
        fields.append(merged(0, [160, 660, 176, 676], {"/FT": n("/Btn"), "/T": TextStringObject("subscribe"), "/V": n("/Yes"), "/AS": n("/Yes"), "/DA": TextStringObject("/ZaDb 0 Tf 0 g"), "/MK": DictionaryObject({n("/CA"): TextStringObject("4")})}))
        fields.append(merged(0, [160, 618, 320, 638], {"/FT": n("/Ch"), "/T": TextStringObject("country"), "/Ff": NumberObject(131072), "/Opt": arr(TextStringObject("Norway"), TextStringObject("Sweden")), "/V": TextStringObject("Norway"), "/DA": da}))
        pages[0][n("/Contents")] = content(w, labels)
    else:
        labels1 = [(72, 745, "Person name / notes / a.b.c / inh / pdfdoc / caf / check / size / country / langs")]
        person = DictionaryObject({n("/T"): TextStringObject("person")})
        person_ref = add(person)
        name_ref = merged(0, [160, 738, 420, 758], {"/FT": n("/Tx"), "/T": TextStringObject("name"), "/V": utf16("Zoë Ñúñez 日本"), "/DA": da}, parent=person_ref)
        notes_ref = merged(0, [160, 660, 420, 730], {"/FT": n("/Tx"), "/T": TextStringObject("notes"), "/Ff": NumberObject(4096), "/V": TextStringObject("Line one\nLine two\nLine three"), "/DA": da}, parent=person_ref)
        person[n("/Kids")] = arr(name_ref, notes_ref)
        fields.append(person_ref)

        a = DictionaryObject({n("/T"): TextStringObject("a")})
        a_ref = add(a)
        b = DictionaryObject({n("/T"): TextStringObject("b"), n("/FT"): n("/Tx"), n("/DA"): da, n("/Parent"): a_ref})
        b_ref = add(b)
        c_ref = merged(0, [160, 620, 420, 640], {"/T": TextStringObject("c"), "/V": TextStringObject("deep value")}, parent=b_ref)
        b[n("/Kids")] = arr(c_ref)
        a[n("/Kids")] = arr(b_ref)
        fields.append(a_ref)

        inh = DictionaryObject({n("/T"): TextStringObject("inh"), n("/FT"): n("/Tx"), n("/V"): TextStringObject("inherited value"), n("/DA"): da})
        inh_ref = add(inh)
        x_ref = merged(0, [160, 590, 280, 610], {"/T": TextStringObject("x")}, parent=inh_ref)
        y_ref = merged(0, [300, 590, 420, 610], {"/T": TextStringObject("y"), "/V": TextStringObject("own y")}, parent=inh_ref)
        inh[n("/Kids")] = arr(x_ref, y_ref)
        fields.append(inh_ref)

        fields.append(merged(0, [160, 560, 420, 580], {"/FT": n("/Tx"), "/T": TextStringObject("pdfdoc"), "/V": ByteStringObject(b"Bullet \x80 dash \x84 \xe9t\xe9"), "/DA": da}))
        fields.append(merged(0, [160, 530, 420, 550], {"/FT": n("/Tx"), "/T": utf16("café"), "/V": TextStringObject("x"), "/DA": da}))
        fields.append(merged(0, [160, 500, 420, 520], {"/FT": n("/Tx"), "/T": TextStringObject("empty_text"), "/DA": da}))

        mw = DictionaryObject({n("/T"): TextStringObject("multi_check"), n("/FT"): n("/Btn"), n("/V"): n("/Yes")})
        mw_ref = add(mw)
        w1 = widget(0, [160, 470, 176, 486], {"/AS": n("/Yes")}, parent=mw_ref)
        w2 = widget(1, [160, 470, 176, 486], {"/AS": n("/Yes")}, with_p=False, parent=mw_ref)
        mw[n("/Kids")] = arr(w1, w2)
        fields.append(mw_ref)

        fields.append(merged(0, [200, 470, 216, 486], {"/FT": n("/Btn"), "/T": TextStringObject("agree_off"), "/V": n("/Off"), "/AS": n("/Off")}))

        size = DictionaryObject({n("/T"): TextStringObject("size"), n("/FT"): n("/Btn"), n("/Ff"): NumberObject(49152), n("/V"): n("/1"), n("/Opt"): arr(TextStringObject("Small"), TextStringObject("Medium"), TextStringObject("Large"))})
        size_ref = add(size)
        kids = []
        for i in range(3):
            kids.append(widget(0, [160 + 30 * i, 440, 176 + 30 * i, 456], {"/AS": n("/1") if i == 1 else n("/Off")}, parent=size_ref))
        size[n("/Kids")] = arr(*kids)
        fields.append(size_ref)

        fields.append(merged(0, [160, 400, 320, 420], {"/FT": n("/Ch"), "/T": TextStringObject("country"), "/Ff": NumberObject(131072), "/Opt": arr(TextStringObject("Norway"), TextStringObject("Sweden")), "/V": TextStringObject("Sweden"), "/DA": da}))
        fields.append(merged(0, [160, 320, 320, 390], {"/FT": n("/Ch"), "/T": TextStringObject("langs"), "/Ff": NumberObject(2097152), "/Opt": arr(arr(TextStringObject("rs"), TextStringObject("Rust")), arr(TextStringObject("py"), TextStringObject("Python")), arr(TextStringObject("go"), TextStringObject("Go"))), "/V": arr(TextStringObject("rs"), TextStringObject("go")), "/DA": da}))

        fields.append(merged(1, [420, 700, 160, 680], {"/FT": n("/Tx"), "/T": TextStringObject("p2_text"), "/V": TextStringObject("Page two text"), "/DA": da}, with_p=False))
        pages[0][n("/Contents")] = content(w, labels1)
        pages[1][n("/Contents")] = content(w, [(72, 745, "Page two")])

    for i, p in enumerate(pages):
        p[n("/Annots")] = ArrayObject(annots[i])
    helv = w._add_object(DictionaryObject({n("/Type"): n("/Font"), n("/Subtype"): n("/Type1"), n("/BaseFont"): n("/Helvetica"), n("/Encoding"): n("/WinAnsiEncoding")}))
    zadb = w._add_object(DictionaryObject({n("/Type"): n("/Font"), n("/Subtype"): n("/Type1"), n("/BaseFont"): n("/ZapfDingbats")}))
    w._root_object[n("/AcroForm")] = w._add_object(DictionaryObject({
        n("/Fields"): ArrayObject(fields),
        n("/NeedAppearances"): BooleanObject(True),
        n("/DA"): TextStringObject("/Helv 0 Tf 0 g"),
        n("/DR"): DictionaryObject({n("/Font"): DictionaryObject({n("/Helv"): helv, n("/ZaDb"): zadb})}),
    }))
    with open(path, "wb") as fh:
        w.write(fh)


def fill_irs(src, dst, values):
    r = PdfReader(src)
    w = PdfWriter(clone_from=r)
    for page in w.pages:
        w.update_page_form_field_values(page, values, auto_regenerate=False)
    with open(dst, "wb") as fh:
        w.write(fh)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    reportlab_form(OUT / "synth_reportlab.pdf")
    lowlevel_form(OUT / "synth_lowlevel_noap.pdf")
    lowlevel_form(OUT / "synth_simple_noap.pdf", simple=True)
    if len(sys.argv) > 3:
        p1 = "topmostSubform[0].Page1[0]."
        fill_irs(sys.argv[2], OUT / "irs_fw9_filled.pdf", {
            p1 + "f1_01[0]": "Zoë Ñúñez",
            p1 + "f1_02[0]": "Ñúñez Consulting LLC",
            p1 + "Boxes3a-b_ReadOrder[0].c1_1[0]": "/1",
            p1 + "Address_ReadOrder[0].f1_07[0]": "123 Main St., Apt 4",
            p1 + "Address_ReadOrder[0].f1_08[0]": "Springfield, IL 62701",
            p1 + "f1_11[0]": "123",
            p1 + "f1_12[0]": "45",
            p1 + "f1_13[0]": "6789",
        })
        fill_irs(sys.argv[3], OUT / "irs_fw4_filled.pdf", {
            p1 + "Step1a[0].f1_01[0]": "Ada",
            p1 + "Step1a[0].f1_02[0]": "Lovelace",
            p1 + "Step1a[0].f1_03[0]": "12 Analytical Engine Rd",
            p1 + "Step1a[0].f1_04[0]": "London, ZZ 00000",
            p1 + "f1_05[0]": "000-00-0000",
            p1 + "c1_1[1]": "/2",
            p1 + "c1_2[0]": "/1",
            p1 + "Step3_ReadOrder[0].f1_06[0]": "2000",
            "topmostSubform[0].Page3[0].f3_01[0]": "15000",
        })
