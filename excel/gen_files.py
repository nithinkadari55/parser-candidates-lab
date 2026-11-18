import os
import sys
import zipfile
from xml.sax.saxutils import escape

CT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>"""
RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"""
WB_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>"""
NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'

CUSTOM = [(164, "yyyy-mm-dd hh:mm:ss"), (165, "[h]:mm:ss"), (166, '"$"#,##0.00'), (167, "mmm-yy"), (168, "yyyy-mm-dd"), (169, "0.0%")]
XF = [0, 14, 164, 20, 165, 10, 166, 44, 11, 49, 12, 167, 4, 168, 169]
S = {"general": 0, "d14": 1, "dt": 2, "time": 3, "dur": 4, "pct": 5, "usd": 6, "acct": 7, "sci": 8, "text": 9, "frac": 10, "mmmyy": 11, "num2": 12, "ymd": 13, "pct1": 14}


def styles_xml():
    nf = "".join(f'<numFmt numFmtId="{i}" formatCode="{escape(c, {chr(34): "&quot;"})}"/>' for i, c in CUSTOM)
    apply = ' applyNumberFormat="1"'
    xfs = "".join(f'<xf numFmtId="{n}" fontId="0" fillId="0" borderId="0" xfId="0"{apply if n else ""}/>' for n in XF)
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<styleSheet {NS}><numFmts count="{len(CUSTOM)}">{nf}</numFmts>'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            f'<cellXfs count="{len(XF)}">{xfs}</cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')


def workbook_xml(date1904, name):
    pr = '<workbookPr date1904="1"/>' if date1904 else '<workbookPr/>'
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<workbook {NS}>{pr}'
            f'<sheets><sheet name="{name}" sheetId="1" r:id="rId1"/></sheets></workbook>')


class SST:
    def __init__(self):
        self.items, self.index = [], {}

    def add(self, s, raw=None):
        key = raw or s
        if key not in self.index:
            self.index[key] = len(self.items)
            self.items.append(raw or f'<si><t xml:space="preserve">{escape(s)}</t></si>')
        return self.index[key]

    def xml(self):
        return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<sst {NS.split(" xmlns:r")[0]} count="{len(self.items)}" '
                f'uniqueCount="{len(self.items)}">{"".join(self.items)}</sst>')


def write_package(path, date1904, sheet_name, sheet_xml_parts, sst):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CT)
        z.writestr("_rels/.rels", RELS)
        z.writestr("xl/workbook.xml", workbook_xml(date1904, sheet_name))
        z.writestr("xl/_rels/workbook.xml.rels", WB_RELS)
        z.writestr("xl/styles.xml", styles_xml())
        with z.open("xl/worksheets/sheet1.xml", "w") as f:
            for part in sheet_xml_parts:
                f.write(part.encode("utf-8"))
        z.writestr("xl/sharedStrings.xml", sst.xml())


def edge(path, date1904):
    sst = SST()

    def s(txt):
        return f't="s"><v>{sst.add(txt)}</v>'

    rich = '<si><r><rPr><b/></rPr><t>bold</t></r><r><t xml:space="preserve"> plain</t></r></si>'
    rows = [
        ("int", '><v>42</v>'),
        ("float", '><v>3.14159</v>'),
        ("neg_fmt", f's="{S["num2"]}"><v>-1234.5</v>'),
        ("bigint", '><v>12345678901</v>'),
        ("bool_true", 't="b"><v>1</v>'),
        ("bool_false", 't="b"><v>0</v>'),
        ("err_na", 't="e"><v>#N/A</v>'),
        ("err_div0", 't="e"><v>#DIV/0!</v>'),
        ("empty_sst", s("")),
        ("inline", 't="inlineStr"><is><t>inline text</t></is>'),
        ("date14", f's="{S["d14"]}"><v>45658</v>'),
        ("datetime", f's="{S["dt"]}"><v>45658.5</v>'),
        ("time", f's="{S["time"]}"><v>0.75</v>'),
        ("duration", f's="{S["dur"]}"><v>1.5</v>'),
        ("percent", f's="{S["pct"]}"><v>0.125</v>'),
        ("currency", f's="{S["usd"]}"><v>12.5</v>'),
        ("accounting", f's="{S["acct"]}"><v>1234.56</v>'),
        ("scientific", f's="{S["sci"]}"><v>0.000123</v>'),
        ("text_fmt", f's="{S["text"]}" ' + s("00123")),
        ("fraction", f's="{S["frac"]}"><v>0.5</v>'),
        ("mmm_yy", f's="{S["mmmyy"]}"><v>45708</v>'),
        ("f_num", '><f>B1*2</f><v>84</v>'),
        ("f_str", 't="str"><f>A1&amp;"x"</f><v>intx</v>'),
        ("f_bool", 't="b"><f>B1&gt;1</f><v>1</v>'),
        ("f_err", 't="e"><f>NA()</f><v>#N/A</v>'),
        ("f_date", f's="{S["d14"]}"><f>B11+1</f><v>45659</v>'),
        ("f_nocache", '><f>1+1</f>'),
        ("float_whole", '><v>1.0</v>'),
        ("rich_sst", f't="s"><v>{sst.add(None, rich)}</v>'),
        ("iso_date", f't="d" s="{S["ymd"]}"><v>2025-03-04T00:00:00</v>'),
        ("f_empty_str", 't="str"><f>""</f><v></v>'),
        ("pct1", f's="{S["pct1"]}"><v>0.0525</v>'),
    ]
    out = [f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<worksheet {NS}><sheetData>']
    for i, (label, frag) in enumerate(rows, start=1):
        cells = f'<c r="A{i}" t="s"><v>{sst.add(label)}</v></c><c r="B{i}" {frag}</c>'
        if i == 1:
            cells += f'<c r="D1" t="s"><v>{sst.add("merged_hdr")}</v></c><c r="F1" s="{S["d14"]}"/>'
        if i == 4:
            cells += '<c r="D4"><v>7</v></c>'
        out.append(f'<row r="{i}">{cells}</row>')
    out.append('</sheetData><mergeCells count="2"><mergeCell ref="D1:E2"/><mergeCell ref="D4:D6"/></mergeCells></worksheet>')
    write_package(path, date1904, "Edge", out, sst)


def big(path, nrows=100_000, ncols=20):
    sst = SST()
    for k in range(1000):
        sst.add(f"item_{k}")

    def col(c):
        return chr(ord("A") + c)

    def parts():
        yield f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<worksheet {NS}><sheetData>'
        hdr = "".join(f'<c r="{col(c)}1" t="s"><v>{sst.add(f"col_{c}")}</v></c>' for c in range(ncols))
        yield f'<row r="1">{hdr}</row>'
        for i in range(nrows):
            r = i + 2
            cs = []
            for c in range(ncols):
                ref = f"{col(c)}{r}"
                m = c % 7
                if m == 0:
                    cs.append(f'<c r="{ref}"><v>{i}</v></c>')
                elif m == 1:
                    cs.append(f'<c r="{ref}"><v>{i * 1.5 + c}</v></c>')
                elif m == 2:
                    cs.append(f'<c r="{ref}" t="s"><v>{(i * 7 + c) % 1000}</v></c>')
                elif m == 3:
                    cs.append(f'<c r="{ref}" s="{S["ymd"]}"><v>{45000 + (i % 3650)}</v></c>')
                elif m == 4:
                    cs.append(f'<c r="{ref}" t="b"><v>{i % 2}</v></c>')
                elif m == 5:
                    cs.append(f'<c r="{ref}" s="{S["usd"]}"><v>{(i % 10000) / 4}</v></c>')
                else:
                    cs.append(f'<c r="{ref}" s="{S["pct"]}"><v>{(i % 100) / 100}</v></c>')
            yield f'<row r="{r}">{"".join(cs)}</row>'
        yield '</sheetData></worksheet>'

    write_package(path, False, "Big", parts(), sst)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "gen"
    edge(os.path.join(outdir, "x_edge.xlsx"), False)
    edge(os.path.join(outdir, "x_edge_1904.xlsx"), True)
    if "--no-big" not in sys.argv:
        big(os.path.join(outdir, "big_100k.xlsx"))
